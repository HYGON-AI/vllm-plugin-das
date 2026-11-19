// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

void deepseek_v4_inv_rope(torch::Tensor& rope, const torch::Tensor& positions,
                          const torch::Tensor& cos_sin_cache);

TORCH_LIBRARY_FRAGMENT(hcu_vllm_das, m) {
  m.def("deepseek_v4_inv_rope(Tensor(a!) rope, Tensor position_ids, Tensor cos_sin_cache) -> ()");
}

TORCH_LIBRARY_IMPL(hcu_vllm_das, CUDA, m) {
  m.impl("deepseek_v4_inv_rope", &deepseek_v4_inv_rope);
}
