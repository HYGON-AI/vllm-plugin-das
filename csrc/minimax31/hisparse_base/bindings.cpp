#include <torch/extension.h>
void plan_headwise(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
void swap_headwise(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
void commit_all(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("plan_headwise",&plan_headwise);m.def("swap_headwise",&swap_headwise);m.def("commit_all",&commit_all);}
