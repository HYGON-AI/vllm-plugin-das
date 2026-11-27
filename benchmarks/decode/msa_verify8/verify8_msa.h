// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

// Owns one fixed-width graph workspace on the caller's execution stream.
// Inputs may change in place. Consume selected pages before the next select.
class Verify8PageSelector {
public:
    Verify8PageSelector(at::Tensor ends, at::Tensor page_ids, at::Tensor scale,
                        int64_t pages_per_request, int mode = 0, int query_width = 8);

    at::Tensor select(const at::Tensor& query, const at::Tensor& cache);
    at::Tensor select_scores(const at::Tensor& scores);

private:
    at::Tensor ends_, page_ids_, scale_, metadata_, work_prefix_, scores_, selected_;
    int pages_, queries_, batch_, padded_pages_, work_, mode_;
    at::Tensor select_scores_impl(const at::Tensor& scores, int score_layout);
};
