#include <torch/extension.h>

using Tensor = at::Tensor;
void plan_delta(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                int, int, int);
void commit_mirror(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                   const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                   int, int, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("plan", &plan_delta);
    module.def("commit", &commit_mirror);
}
