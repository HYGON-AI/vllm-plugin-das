#include <torch/extension.h>
void plan_spans(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                const at::Tensor&,const at::Tensor&,const at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("plan_spans", &plan_spans);
}
