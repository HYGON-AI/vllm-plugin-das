#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip/hip_bf16.h"
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_hcc.h>
#include <vector>
#include <algorithm>

#include "ops.h"
#include "step5_w4a8_intrinsics.h"


#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define DG_HIP_CHECK(call)                                                   \
  do {                                                                       \
    hipError_t err = (call);                                                  \
    TORCH_CHECK(                                                              \
        err == hipSuccess, #call " failed: ", hipGetErrorString(err));       \
  } while (0)


namespace step5_w4a8_native {

void validate_metadata(const torch::Tensor& input, const torch::Tensor& weight,
                       const torch::Tensor& output, const torch::Tensor& a_scale,
                       const torch::Tensor& w_scale, const torch::Tensor& ids) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3 && output.dim() == 2,
              "Step5 W4A8 expects input[M,K], packed weight[E,N,K/2], output[M,N]");
  const auto m = input.size(0), k = input.size(1), n = output.size(1);
  TORCH_CHECK((k == 4096 && n == 3072) || (k == 1536 && n == 4096),
              "Step5 W4A8 supports gate/up4096x3072 and down1536x4096 only");
  TORCH_CHECK(input.scalar_type() == at::kChar && weight.scalar_type() == at::kChar &&
              output.scalar_type() == at::kBFloat16, "Step5 W4A8 requires INT8 input/packed weights and BF16 output");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat && w_scale.scalar_type() == at::kFloat &&
              ids.scalar_type() == at::kInt, "Step5 W4A8 requires FP32 scales and INT32 expert ids");
  TORCH_CHECK((a_scale.dim() == 1 || (a_scale.dim() == 2 && a_scale.size(1) == 1)) &&
              (w_scale.dim() == 2 || (w_scale.dim() == 3 && w_scale.size(2) == 1)), "Invalid channel scale layout");
  TORCH_CHECK(ids.dim() == 1 && ids.size(0) == m && a_scale.size(0) == m && output.size(0) == m,
              "Step5 W4A8 M dimension mismatch");
  TORCH_CHECK(weight.size(0) > 0 && weight.size(1) == n && weight.size(2) == k / 2 &&
              w_scale.size(0) == weight.size(0) && w_scale.size(1) == n, "Step5 W4A8 weight/scale shape mismatch");
  for (const auto& tensor : {weight, output, a_scale, w_scale, ids})
    TORCH_CHECK(tensor.device() == input.device(), "Step5 W4A8 tensors must share one device");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && output.is_contiguous() &&
              a_scale.is_contiguous() && w_scale.is_contiguous() && ids.is_contiguous(), "Step5 W4A8 requires contiguous tensors");
}


template <typename T, typename T_hidden = bhalf_t>
struct GemmParams_w4a8_contiguous {
    GemmParams_w4a8_contiguous(
        const T* ptr_A,
        const T* ptr_B0,
        T_hidden* ptr_C,
        float* ptr_A_scale,
        float* ptr_B_scale,
        const int32_t* m_indices,
        uint32_t size_m,
        uint32_t size_n,
        uint32_t size_k,
        uint32_t stride_asm,
        uint32_t stride_bse,
        uint32_t stride_bsn,
        uint32_t stride_bsk)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
          ptr_B_scale(ptr_B_scale),
          m_indices(m_indices),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          stride_asm(stride_asm),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk) {}

    const T* ptr_A;
    const T* ptr_B0;
    T_hidden* ptr_C;
    float* ptr_A_scale;
    float* ptr_B_scale;
    const int32_t* m_indices;
    uint32_t size_m;
    uint32_t size_n;
    uint32_t size_k;
    uint32_t stride_asm;
    uint32_t stride_bse;
    uint32_t stride_bsn;
    uint32_t stride_bsk;
};
template<bool Is_store_A,
    int A_prefetch_level,
    int B_prefetch_level,
    int WARP_NUM,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int STAGE,
    int GROUP_N,
    int GROUP_K,
    int SIZE_K,
    typename Element,
    typename ElementAccum = int32_t,
    int N_LOOP_NUM = 4,
    bool ABC_ORDER = false>
__forceinline__ __device__ void  gemm_nt_marlin_prefill_w4a8(
    const Element* input_ptr,
    const Element* weight_ptr,
    Element* A_lds,
    Element* B_lds,
    float* input_scale_ptr,
    float* weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element,WARP_K/4> A_reg[][STAGE],
    union_vec_opt<Element,WARP_K/4> B_reg[][2][STAGE],
    intx4 C_reg[][(WARP_M/16)*(WARP_N/16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    const int bidx

    ) {

        constexpr int n_loop_num = N_LOOP_NUM;

    const int size_k = (SIZE_K == 0) ? seqlen_A_stride : SIZE_K;
    const int k_loop_count = size_k / 128;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2 ) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);

    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        int s_index = (warp_n_id  * (WARP_N / 32)  + n_tile ) * 32 * 32;
        int index = s_index + row_id  * 2 * 8  + col_id * 32 * 8;

            g_row_B[n_tile ] = index ;

    }

    int A_index = warp_id;


    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){



            int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index*4;
            int row = std::min(s_offset + col_id, max_n_len_offset - 1);
            g_row_A[m_tile] = row * size_k + row_id * 4;


    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K ) + kloop * (128);

    int i =0;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){

                inline_buffer_load_dword_lds( A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K ) / 4 +  (0) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4 , (g_row_A[m_tile] )/4);


        }
    }


    i = 1;


    k_start += stage_offset;

    #pragma unroll
    for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){

                inline_buffer_load_dword_lds(  A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 +  (1) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4, (g_row_A[m_tile] )/4);

        }
    }

for(kloop  ;kloop < k_loop_count ;kloop++){

    __builtin_hcu_sched_barrier(0);
    vmcnt_wait(0);

    int k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (128 / 2) * size_n;





















    union {
        vec<int, 4> v;
        uint32_t i[4];
    } weight_reg_tmp[ WARP_N / 32][2][STAGE];


if constexpr (STAGE == 2){

const Element* cur_weight_ptr = weight_ptr ;

int i =0;
int stage_b_flag =0;
int n_loop =0;

__builtin_hcu_sched_barrier(0);
#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){


        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ] );

        }

}

__builtin_hcu_sched_barrier(0);

i = 1;

k_start_b += stage_offset_b;


__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){


        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile ]   );

        }

}
__builtin_hcu_sched_barrier(0);

vmcnt_wait(2 *( WARP_N / 32 ) );

    #pragma unroll
    for(int i =0;i<2;i++){
        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){
                int s_index = m_tile * (16) * 64 + + (i) * WARP_M / 16 * (16) * WARP_K;
                A_reg[m_tile][i].int4_array[k_tile] = *( vec<Element, 16> *) (&A_lds[ s_index + row_id * 64 + col_id * 16      ]);

            }
        }
    }

k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (WARP_K * 2 / 2) * size_n;

n_loop ++;

for(n_loop;n_loop < n_loop_num;n_loop++)
{

stage_b_flag ^= 1;
cur_weight_ptr  +=   BLOCK_N * 32;
int i =0;

__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){


        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ]);


        }

}
__builtin_hcu_sched_barrier(0);

i = 1;

k_start_b += stage_offset_b;

__builtin_hcu_sched_barrier(0);
#pragma unroll
for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){

        #pragma unroll
        for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){


        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile ][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b , g_row_B[n_tile ] );

        }

}
__builtin_hcu_sched_barrier(0);



__builtin_hcu_sched_barrier(0);
vmcnt_only_wait( 3* ( WARP_N / 32 ));


n_loop--;
i = 0;
stage_b_flag ^= 1;

__builtin_hcu_sched_barrier(0);

__builtin_hcu_sched_barrier(0);



    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){

            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            }
        }
    }

__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int m_tile = 0; m_tile <  WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}

__builtin_hcu_sched_barrier(0);

vmcnt_only_wait( 2* ( WARP_N / 32 ));

i = 1;

__builtin_hcu_sched_barrier(0);





    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){

            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            }
        }
    }

__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}


__builtin_hcu_sched_barrier(0);
stage_b_flag ^= 1;
n_loop ++;

k_start_b = warp_k_id * (WARP_K / 2 ) * size_n  + kloop * (WARP_K * 2 / 2) * size_n;

}

    n_loop--;
    wave_barrier();
    if(kloop <  k_loop_count - 1 ) {

        kloop++;
        k_start = warp_k_id * (WARP_K ) + kloop * (128);

        i =0;

        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){

                    inline_buffer_load_dword_lds( A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K ) / 4 +  (0) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4 , (g_row_A[m_tile] )/4);


            }
        }


        i = 1;


        k_start += stage_offset;

        #pragma unroll
        for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++){

                    inline_buffer_load_dword_lds(  A_lds,  g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 +  (1) * (WARP_M / 16) * (16) * WARP_K / 4  ,  ( k_tile * READ_K + k_start) / 4, (g_row_A[m_tile] )/4);

            }
        }
        kloop --;
    }



    vmcnt_only_wait(  ( WARP_N / 32 ) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)) );

if(kloop == k_loop_count - 1 ){
    vmcnt_wait(  ( WARP_N / 32 ) );
}

i = 0;

__builtin_hcu_sched_barrier(0);



    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){

            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            }

        }
    }

__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}

__builtin_hcu_sched_barrier(0);

    vmcnt_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));

    if(kloop == k_loop_count - 1 ){
        vmcnt_wait(  0 );
    }


i = 1;

__builtin_hcu_sched_barrier(0);





    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){

            uint32_t packed_val[2];
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){

                packed_val[k_tile] = *(uint32_t*)(&(weight_reg_tmp[n_tile ][stage_b_flag][i].i[it * 2 +k_tile])) ;

            }
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                uint32_t packed_val_second = (packed_val[k_tile] << 4) &  0xf0f0f0f0;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int*)&packed_val_first;
                B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int*)&packed_val_second;
            }
        }
    }



__builtin_hcu_sched_barrier(0);

#pragma unroll
for(int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++){
    #pragma unroll
    for(int n_tile = 0; n_tile < WARP_N / 32; n_tile++){
        #pragma unroll
        for(int it = 0 ;it < 2 ; it++){
            #pragma unroll
            for(int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++){
                if constexpr (ABC_ORDER) {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                } else {
                    C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *( vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *( vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile*(WARP_N/MFMA_N) + n_tile * 2 + it]);
                }
            }
        }
    }
}
__builtin_hcu_sched_barrier(0);

__builtin_hcu_sched_barrier(0);

    }
else if constexpr (STAGE == 1){
;
}

}
    }

template<
  typename scalar_t,
  typename Element,
  int WARP_NUM,
  int BLOCK_SIZE_M,
  int BLOCK_SIZE_N,
  int BLOCK_SIZE_K,
  int WARP_M,
  int WARP_N,
  int WARP_K,
  int GROUP_N,
  int GROUP_K,
  int STAGES,
  int N_LOOP_NUM = 4,
  bool DIRECT_DWORDX4 = true>
__global__ void __launch_bounds__(256,1) W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_CONTIGUOUS(
  const Element* __restrict__ input,
  const Element* __restrict__ qweight,
  scalar_t* __restrict__ output,
  float* __restrict__ input_scale,
  float* __restrict__ weight_scale,
  const int32_t* __restrict__ m_indices,
  uint32_t size_m,
  uint32_t size_n,
  uint32_t size_k,
  uint32_t stride_asm,
  uint32_t stride_bse,
  uint32_t stride_bsn,
  uint32_t stride_bsk
) {
  const int bidx = blockIdx.z;
  const int bidy = blockIdx.y;
  const int block_m_base = bidx * BLOCK_SIZE_M;
  if (block_m_base >= size_m) {
    return;
  }

  const int expert_id = m_indices[block_m_base];
  if (expert_id < 0) {
    return;
  }
  const bool full_contiguous_block =
      block_m_base + BLOCK_SIZE_M <= size_m &&
      m_indices[block_m_base + BLOCK_SIZE_M - 1] == expert_id;

  constexpr int mfma_m = 16;
  constexpr int mfma_n = 16;
  constexpr int n_loop_num = N_LOOP_NUM;
  int warp_id_vec = threadIdx.x / 64;
  int warp_id = __builtin_hcu_readfirstlane(warp_id_vec);
  int lane_id = threadIdx.x & 63;
  int row_id = lane_id % 16;
  int col_id = lane_id / 16;

  extern __shared__ Element smem[];
  Element* input_lds = (Element*)&(smem);
  Element* qweight_lds = input_lds;

  union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];
  intx4 C_reg[n_loop_num][(WARP_M/16)*(WARP_N/16)] = {0,0,0,0};

  const uint64_t expert_offset = static_cast<uint64_t>(size_n) * size_k / 2 * expert_id;
  const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num;
  const uint64_t weight_scale_offset = stride_bse * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;
  auto g_qweight = qweight + qweight_offset;
  auto g_weight_scale = weight_scale + weight_scale_offset;

  float b_scale_dwordx4[n_loop_num][8];
  __builtin_hcu_sched_barrier(0);
  constexpr int SIZE_K = 0;
  gemm_nt_marlin_prefill_w4a8<
      false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
      WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element,
      scalar_t, N_LOOP_NUM, DIRECT_DWORDX4>(
      input, g_qweight, input_lds, qweight_lds, input_scale, g_weight_scale,
      size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, 1,
      stride_bse, stride_bsn, stride_bsk, bidx);
  __builtin_hcu_sched_barrier(0);

  if constexpr (DIRECT_DWORDX4) {
    #pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {
      const int scale_base = weight_scale_offset + BLOCK_SIZE_N * n_loop +
                             warp_id * WARP_N + col_id * 8;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        b_scale_dwordx4[n_loop][i] = weight_scale[scale_base + i];
      }
    }
    vmcnt_wait(0);

    auto store_contiguous_row = [&](int n_loop, int min_tile_m, int row) {
      const float a_scale_value = input_scale[row * stride_asm];
      vec_element_8<scalar_t> packed_store_value;
      const int tile_idx = min_tile_m * (WARP_N / mfma_n);
      #pragma unroll
      for (int reg_id = 0; reg_id < 4; reg_id++) {
        scalar_t store_pair[2];
        #pragma unroll
        for (int pair_id = 0; pair_id < 2; pair_id++) {
          const int acc = C_reg[n_loop][tile_idx + pair_id][reg_id];
          const float scaled =
              static_cast<float>(acc) * a_scale_value *
              b_scale_dwordx4[n_loop][reg_id * 2 + pair_id];
          store_pair[pair_id] = b32_to_b16<scalar_t>(scaled);
        }
        #pragma unroll
        for (int pair_id = 0; pair_id < 2; pair_id++) {
          packed_store_value.data[reg_id * 2 + pair_id] = store_pair[pair_id];
        }
      }
      const int n_index = bidy * BLOCK_SIZE_N * n_loop_num +
                          BLOCK_SIZE_N * n_loop +
                          warp_id * WARP_N + col_id * 8;
      *reinterpret_cast<vec_element_8<scalar_t>*>(
          &output[static_cast<int64_t>(row) * size_n + n_index]) =
          packed_store_value;
    };

    if (full_contiguous_block) {
      #pragma unroll
      for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {
          const int row = block_m_base + min_tile_m * mfma_m + row_id;
          store_contiguous_row(n_loop, min_tile_m, row);
        }
      }
    } else {
      #pragma unroll
      for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {
          const int row = block_m_base + min_tile_m * mfma_m + row_id;
          if (row < size_m && m_indices[row] == expert_id) {
            store_contiguous_row(n_loop, min_tile_m, row);
          }
        }
      }
    }
  }
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden, int N_LOOP_NUM, bool DIRECT_DWORDX4>
void launch_w4a8_hipc_contiguous_direct(const GemmParams_w4a8_contiguous<T,T_hidden>& params) {
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    dim3 blockDim(WARP_NUM * 64, 1, 1);
    dim3 gridDim(1, DIVIDE(params.size_n, BLOCK_SIZE_N * N_LOOP_NUM),
                 DIVIDE(params.size_m, BLOCK_SIZE_M));
    const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_CONTIGUOUS<
        T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        WARP_M, WARP_N, WARP_K, GROUP_N, GROUP_K, STAGES, N_LOOP_NUM,
        DIRECT_DWORDX4><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.m_indices,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk);
}

constexpr int kW4A8HipcBlockM = 64;
template <typename scalar_t>
void launch_w4a8_hipc_contiguous_prefill(
    const torch::Tensor& input,
    const torch::Tensor& b_qweight,
    torch::Tensor& output,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    const torch::Tensor& m_indices) {
  const int size_m = input.size(0);
  const int size_k = input.size(1);
  const int size_n = output.size(1);
  const int stride_asm = a_scale.dim() == 1 ? 1 : a_scale.stride(0);
  const int stride_bsk = b_scale.dim() == 2 ? 1 : b_scale.stride(2);

  GemmParams_w4a8_contiguous<char, scalar_t> params(
      reinterpret_cast<const char*>(input.data_ptr<int8_t>()),
      reinterpret_cast<const char*>(b_qweight.data_ptr<int8_t>()),
      reinterpret_cast<scalar_t*>(output.data_ptr()),
      a_scale.data_ptr<float>(),
      b_scale.data_ptr<float>(),
      m_indices.data_ptr<int32_t>(),
      size_m,
      size_n,
      size_k,
      stride_asm,
      b_scale.stride(0),
      b_scale.stride(1),
      stride_bsk);

  launch_w4a8_hipc_contiguous_direct<
      64, 128, 64, 64, 32, 64, 2, char, scalar_t, 4, true>(params);
}

torch::Tensor m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
  torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor a_scale,
  torch::Tensor b_scale,
  torch::Tensor m_indices,
  int mode) {

  validate_metadata(input, b_qweight, output, a_scale, b_scale, m_indices);
  TORCH_CHECK(mode == 1002, "Step5 W4A8 supports MODE1002 only");
  TORCH_CHECK(input.is_cuda(), "Step5 W4A8 execution requires an HCU tensor");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  if (input.size(0) == 0) return output;
  if (output.scalar_type() == at::ScalarType::BFloat16) {
    launch_w4a8_hipc_contiguous_prefill<bhalf_t>(
        input, b_qweight, output, a_scale, b_scale, m_indices);
  } else { TORCH_CHECK(false, "This private Step5 build supports BF16 output only"); }
  DG_HIP_CHECK(hipGetLastError());
  return output;
}

} // namespace step5_w4a8_native

void step5_w4a8_gemm(const torch::Tensor& input, const torch::Tensor& weight,
                      torch::Tensor& output, const torch::Tensor& a_scale,
                      const torch::Tensor& w_scale, const torch::Tensor& ids) {
  step5_w4a8_native::m_grouped_marlin_w4a8_gemm_nt_contiguous_hipc(
      input, weight, output, a_scale, w_scale, ids, 1002);
}

void step5_w4a8_gemm_meta(const torch::Tensor& input, const torch::Tensor& weight,
                           torch::Tensor& output, const torch::Tensor& a_scale,
                           const torch::Tensor& w_scale, const torch::Tensor& ids) {
  // Shape/storage is caller-owned. Do not concretize symbolic M or strides
  // during Fake/Meta dispatch; execution validates the complete CUDA contract.
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3 && output.dim() == 2 && ids.dim() == 1,
              "Step5 W4A8 rank mismatch");
  TORCH_CHECK(input.scalar_type() == at::kChar && weight.scalar_type() == at::kChar &&
              output.scalar_type() == at::kBFloat16 && a_scale.scalar_type() == at::kFloat &&
              w_scale.scalar_type() == at::kFloat && ids.scalar_type() == at::kInt,
              "Step5 W4A8 dtype mismatch");
}
