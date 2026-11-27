// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include "verify8_msa.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<Verify8PageSelector>(module, "Verify8PageSelector")
        .def(pybind11::init<at::Tensor, at::Tensor, at::Tensor, int64_t, int, int>(),
             pybind11::arg("ends"), pybind11::arg("page_ids"), pybind11::arg("scale"),
             pybind11::arg("pages_per_request"), pybind11::arg("mode") = 0, pybind11::arg("query_width") = 8)
        .def("select", &Verify8PageSelector::select)
        .def("select_scores", &Verify8PageSelector::select_scores);
}
