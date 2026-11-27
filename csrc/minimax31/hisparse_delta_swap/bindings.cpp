#include <torch/extension.h>
using Tensor = at::Tensor;
void plan_delta(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&);
void swap_delta(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("plan_delta", &plan_delta);
    module.def("swap_delta", &swap_delta);
}
