#include <torch/extension.h>
using Tensor = at::Tensor;
void commit_mirror(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                   const Tensor&, const Tensor&, const Tensor&, const Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("commit_mirror", &commit_mirror);
}
