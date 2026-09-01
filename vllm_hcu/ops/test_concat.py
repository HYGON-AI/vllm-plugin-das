# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Manual GPU correctness/performance harness for the concat kernels.

This module predates the repository pytest suite.  Its ``test_*`` callables
take benchmark inputs supplied by the manual driver below, not pytest
fixtures, so exclude the module from automatic collection.
"""

__test__ = False

import triton
import triton.language as tl
import torch
from functools import reduce
import pytest
import torch
import math
import vllm.envs as envs

from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_ds_cat():
    try:
        from lightop.tensor import ds_cat
    except (ImportError, AttributeError):
        logger.warning_once(
            "LightOp ds_cat is unavailable; using torch.cat."
        )
        return None
    return ds_cat


def test_concat_Acc_prefill(shape_pair, dim):

    torch.manual_seed(1)
    shape1, shape2 = shape_pair
    M = shape1[0]
    N = shape1[1]
    x_sizes = [M, N, 128]
    x_strides = [N//8 * 2048, 256, 1]
    x_max_index = N//8 * 2048 * M
    x_required_length = x_max_index
    x_data = torch.arange(x_required_length,device='cuda').bfloat16()  
    x = torch.as_strided(x_data, size=x_sizes, stride=x_strides)

    
    y_sizes = [M, N, 64]
    y_strides = [576, 0, 1]
    y_max_index = 576 * M
    y_required_length = y_max_index
    y_data = torch.arange(y_required_length,device='cuda').bfloat16()  
    y = torch.as_strided(y_data, size=y_sizes, stride=y_strides)
    
    expected = torch.cat([x,y], dim=dim)
    result = concat_prefill_helper_Triton(x, y, dim=dim)
    result_lightop = lightop_concat_prefill_helper(x, y, dim=dim)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), "Mismatch"
    # print("精度验证通过")
    # print("expected",expected)
    # print("result_lightop",result_lightop)
    assert torch.allclose(result, result_lightop, rtol=1e-5, atol=1e-5), "result_lightop Mismatch Triton error"
    assert torch.allclose(expected, result_lightop, rtol=1e-5, atol=1e-5), "result_lightop Mismatch torch error"
    print("prefill 精度验证通过")

def test_concat_Acc_decode(shape_pair, dim):

    torch.manual_seed(1)
    shape1, shape2 = shape_pair
    M = shape1[0]
    N = shape1[1]
    x_sizes = [M, N, 512]
    x_strides = [512, 512*M, 1]
    x_max_index = M * N * 512
    x_required_length = x_max_index
    x_data = torch.arange(x_required_length,device='cuda').bfloat16()  
    x = torch.as_strided(x_data, size=x_sizes, stride=x_strides)
    # print("形状:", x.shape)      
    # print("步幅:", x.stride())  
    y_sizes = [M, N, 64]
    y_strides = [1536*(N//8), 192, 1]
    y_max_index = 1536*(N//8) * M
    y_required_length = y_max_index
    y_data = torch.arange(y_required_length,device='cuda').bfloat16()  
    y = torch.as_strided(y_data, size=y_sizes, stride=y_strides)
    # print("形状:", y.shape)     
    # print("步幅:", y.stride())   
    
    
    expected = torch.cat([x,y], dim=dim)
    result = concat_helper_decode(x, y, dim=dim)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), "Mismatch"

    print("decode 精度正常")



@triton.jit
def concat_kernel(
    A_ptr, B_ptr, C_ptr,
    A_section_numel, B_section_numel, C_section_numel,
    Per_block_A,
    Per_block_B, 
    section_numA,
    section_numB,
    M,
    N,
    Astride_0,
    Astride_1,
    Astride_2,
    Bstride_0,
    Bstride_1,
    Bstride_2,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    numA = section_numA // Per_block_A
    if (block_idx < numA):
        #处理A的block
        for sub_section_index in range(Per_block_A):
            sub_offset = block_idx * Per_block_A + sub_section_index
            if sub_offset <= section_numA-1:
                M_idx = sub_offset // N
                N_idx = sub_offset % N
                C_ptr_block_start = C_ptr + sub_offset * C_section_numel
                A_ptr_block_start = A_ptr + M_idx * Astride_0 + N_idx * Astride_1 
                for offset in range(0, A_section_numel, BLOCK_SIZE):
                    offset_idx = offset + tl.arange(0, BLOCK_SIZE)
                    mask = offset_idx < A_section_numel
                    val_from_A = tl.load(A_ptr_block_start + offset_idx, mask=mask)
                    tl.store(C_ptr_block_start + offset_idx, val_from_A, mask=mask)
    else:
        #处理B的block
        #shape是1024*8*64，实际上只有1024 * 64 块数据，开了1024/4=256个线程块来处理。每个线程块处理1块连续的数据 
        #需要注意C的分块也是有M * N 大小的，而这里只有M大小个线程块，每个线程块需要写入N次数据到C中。
        for sub_section_index in range(Per_block_B):
            sub_offset = (block_idx - numA) * Per_block_B + sub_section_index 
            if sub_offset <= section_numB-1:
                C_ptr_block_start = C_ptr + sub_offset * N * C_section_numel
                B_ptr_block_start = B_ptr + sub_offset * Bstride_0
                for offset in range(0, B_section_numel, BLOCK_SIZE):
                    offset_idx = offset + tl.arange(0, BLOCK_SIZE)
                    mask = offset_idx < B_section_numel
                    val_from_B = tl.load(B_ptr_block_start + offset_idx, mask=mask)
                    for idx in range(0,N,1):
                        tl.store(C_ptr_block_start + idx * C_section_numel + A_section_numel + offset_idx, val_from_B, mask=mask)                
    
def concat_prefill_helper_Triton(A:torch.Tensor, B:torch.Tensor, dim:int):

    output_shape = list(A.shape)
    output_shape[dim] = A.shape[dim] + B.shape[dim]  #128+64=192
    C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
    
    if dim!=0 :
        #分开计算block块A需要
        Per_block_A = 64
        Per_block_B = 1
        #128 \64 \192
        unit_offset_A, unit_offset_B, unit_offset_C = A.shape[dim],B.shape[dim],C.shape[dim]
        #A的分块数是：M * N 这里的demo是1024 * 8
        block_numA = reduce(lambda x, y: x * y, output_shape[:dim])
        #B的分块数是：M 这里的demo是1024
        block_numB = output_shape[0]
        #A的每个分块可以处理多份数据的读取和写入，这是因为单次的任务量太小。假设这里Per_block = 8 那么A就开启了1024个线程块，每个线程块处理8份数据的读取和写入
        #B的每个分块处理1次B的读取和8次C的写入，L2 cache复用率高
        block_num = block_numA // Per_block_A + block_numB // Per_block_B

        num_blocks = math.ceil(block_num)
        concat_kernel[(num_blocks,)](
                A, B, C,
                unit_offset_A, unit_offset_B, unit_offset_C,
                Per_block_A,
                Per_block_B, 
                block_numA,
                block_numB,
                output_shape[0],
                output_shape[1],
                A.stride(0),
                A.stride(1),
                A.stride(2),
                B.stride(0),
                B.stride(1),
                B.stride(2),                 
                BLOCK_SIZE=1024)
        return C

    assert False, "not support"


def concat_helper_decode(A:torch.Tensor, B:torch.Tensor, dim:int):
    assert dim==2 , "tensor dim must be 3 and concat dim must be 2"
    ds_cat = _get_ds_cat()
    if ds_cat is None:
        return torch.cat((A, B), dim=dim)
    output_shape = list(A.shape)
    output_shape[dim] = A.shape[dim] + B.shape[dim]  
    C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
    mode = 0
    if dim!=0 :
        ds_cat( A, B, C, mode)
        return C
    assert False, "not support"

def lightop_concat_prefill_helper(A:torch.Tensor, B:torch.Tensor, dim:int):
    assert dim==2 , "tensor dim must be 3 and concat dim must be 2"
    ds_cat = _get_ds_cat()
    if ds_cat is None:
        return torch.cat((A, B), dim=dim)
    output_shape = list(A.shape)
    output_shape[dim] = A.shape[dim] + B.shape[dim]  
    C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
    mode = 6
    if dim!=0 :
        ds_cat( A, B, C, mode)
        return C
    assert False, "not support"


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=['M','N'],
        x_vals=[(1024,8),(2048,8),(3072,8),(4096,8),(6144,8),(8192,8),\
                (1024,16),(2048,16),(3072,16),(4096,16),(6144,16),(8192,16),\
                    (1024,32),(2048,32),(3072,32),(4096,32),(6144,32),(8192,32)
                    ],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch', 'lightop'],
        line_names=['Triton', 'Torch','Lightop'],
        styles=[('blue', '-'), ('green', '-'), ('yellow', '-')],
        ylabel='s',
        plot_name='concat-dim2',
        args={"dim":2},
    ),
)


configs_decode = []
configs_decode.append(
    triton.testing.Benchmark(
        x_names=['M','N'],
        x_vals=[(4,8),(8,8),(16,8),(32,8),(64,8),(96,8),(128,8),(256,8),(512,8),(768,8),(767,8),(765,8),(766,8), \
                (4,16),(8,16),(16,16),(32,16),(64,16),(96,16),(128,16),(256,16),(512,16),(768,16),(767,16),(765,16),(766,16), \
                (4,32),(8,32),(16,32),(32,32),(64,32),(96,32),(128,32),(256,32),(512,32),(768,32),(767,32),(765,32),(766,32)],
        x_log=True,
        line_arg='provider',
        line_vals=['lightop', 'torch'],
        line_names=['Lightop', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='s',
        plot_name='concat-dim2',
        args={"dim":2},
    ),
)

@triton.testing.perf_report(configs)
def benchmark_prefill(M, N, provider, dim):

    x_sizes = [M, N, 128]
    x_strides = [N//8 * 2048, 256, 1]
    x_max_index = N//8 * 2048 * M
    x_required_length = x_max_index
    x_data = torch.arange(x_required_length,device='cuda').bfloat16()  
    x = torch.as_strided(x_data, size=x_sizes, stride=x_strides)
    
    y_sizes = [M, N, 64]
    y_strides = [576, 0, 1]
    y_max_index = 576 * M
    y_required_length = y_max_index
    y_data = torch.arange(y_required_length,device='cuda').bfloat16()  
    y = torch.as_strided(y_data, size=y_sizes, stride=y_strides)
  
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.cat([x,y],dim=dim), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: concat_prefill_helper_Triton(x, y,dim=dim), quantiles=quantiles)
    if provider == 'lightop':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: lightop_concat_prefill_helper(x, y,dim=dim), quantiles=quantiles)
    return (ms*1000), (max_ms*1000), (min_ms*1000)

@triton.testing.perf_report(configs_decode)
def benchmark_decode(M, N, provider, dim):

    x_sizes = [M, N, 512]
    x_strides = [512, 512*M, 1]
    x_max_index = M * N * 512
    x_required_length = x_max_index
    x_data = torch.arange(x_required_length,device='cuda').bfloat16()  
    x = torch.as_strided(x_data, size=x_sizes, stride=x_strides)

    # print("形状:", x.shape)      
    # print("步幅:", x.stride())  
    
    y_sizes = [M, N, 64]
    y_strides = [1536*(N//8), 192, 1]
    y_max_index = 1536*(N//8) * M
    y_required_length = y_max_index 
    y_data = torch.arange(y_required_length,device='cuda').bfloat16()  
    y = torch.as_strided(y_data, size=y_sizes, stride=y_strides)

    # print("形状:", y.shape)     
    # print("步幅:", y.stride())   

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.cat([x,y],dim=dim), quantiles=quantiles)
    if provider == 'lightop':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: concat_helper_decode(x, y,dim=dim), quantiles=quantiles)
    return (ms*1000), (max_ms*1000), (min_ms*1000)


if __name__ == '__main__':
    benchmark_prefill.run(save_path="./triton_test",print_data=True)
    test_concat_Acc_prefill(((1024, 8, 128), (1024, 8, 64)),  2)  
    test_concat_Acc_prefill(((2048, 8, 128), (2048, 8, 64)),  2)  
    test_concat_Acc_prefill(((4096, 8, 128), (4096, 8, 64)),  2)  
    test_concat_Acc_prefill(((8192, 8, 128), (8192, 8, 64)),  2)
    test_concat_Acc_prefill(((1024, 16, 128), (1024, 16, 64)),  2)  
    test_concat_Acc_prefill(((2048, 16, 128), (2048, 16, 64)),  2)  
    test_concat_Acc_prefill(((4096, 16, 128), (4096, 16, 64)),  2)  
    test_concat_Acc_prefill(((8192, 16, 128), (8192, 16, 64)),  2)  
    test_concat_Acc_prefill(((1024, 32, 128), (1024, 32, 64)),  2)  
    test_concat_Acc_prefill(((2048, 32, 128), (2048, 32, 64)),  2)  
    test_concat_Acc_prefill(((4096, 32, 128), (4096, 32, 64)),  2)  
    test_concat_Acc_prefill(((8192, 32, 128), (8192, 32, 64)),  2)

    benchmark_decode.run(save_path="./cat_triton_test",print_data=True)
    test_concat_Acc_decode(((16, 8, 512), (16, 8, 64)),  2)
    test_concat_Acc_decode(((32, 8, 512), (32, 8, 64)),  2)
    test_concat_Acc_decode(((128, 8, 512), (128, 8, 64)),  2)
    test_concat_Acc_decode(((768, 8, 512), (768, 8, 64)),  2)    
    test_concat_Acc_decode(((32, 16, 512), (32, 16, 64)),  2)
    test_concat_Acc_decode(((32, 32, 512), (32, 32, 64)),  2)
    test_concat_Acc_decode(((768, 32, 512), (768, 32, 64)),  2)
    test_concat_Acc_decode(((128, 32, 512), (128, 32, 64)),  2)
    test_concat_Acc_decode(((512, 32, 512), (512, 32, 64)),  2)

    test_concat_Acc_decode(((765, 8, 512), (765, 8, 64)),  2)
    test_concat_Acc_decode(((766, 8, 512), (766, 8, 64)),  2)
    test_concat_Acc_decode(((767, 8, 512), (767, 8, 64)),  2)
    test_concat_Acc_decode(((765, 16, 512), (765, 16, 64)),  2)
    test_concat_Acc_decode(((766, 16, 512), (766, 16, 64)),  2)
    test_concat_Acc_decode(((765, 32, 512), (765, 32, 64)),  2)
    test_concat_Acc_decode(((767, 32, 512), (767, 32, 64)), 2)
