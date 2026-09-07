// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "ops.h"

#include <torch/library.h>
#include <torch/version.h>
#include <Python.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// A version of the TORCH_LIBRARY_IMPL macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("init_custom_ar(int[] ipc_tensors, Tensor rank_data, int rank, bool fully_connected) -> int");
  ops.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  ops.def("all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, int reg_buffer_sz_bytes) -> ()");
  ops.impl("all_reduce", torch::kCUDA, &all_reduce);
  ops.def("dispose", &dispose);
  ops.def("meta_size", &meta_size);
  ops.def("register_buffer", &register_buffer);
  ops.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  ops.def("register_graph_buffers", &register_graph_buffers);
  ops.def("allocate_shared_buffer_and_handle", &allocate_shared_buffer_and_handle);
  ops.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  ops.def("free_shared_buffer", &free_shared_buffer);

  ops.def(
      "reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, "
      "Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, "
      "Tensor k_scale, Tensor v_scale) -> ()");
  ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache_hcu);
  ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, "
      "Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, "
      "Tensor k_scale, Tensor v_scale) -> ()");
  ops.impl("reshape_and_cache_flash", torch::kCUDA,
           &reshape_and_cache_flash_hcu);
  ops.def("concat_and_cache_mla", &concat_and_cache_mla_hcu);
  ops.def(
      "deepseek_v4_inv_rope(Tensor! rope, Tensor position_ids, "
      "Tensor cos_sin_cache) -> ()");
  ops.impl("deepseek_v4_inv_rope", torch::kCUDA, &deepseek_v4_inv_rope);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
