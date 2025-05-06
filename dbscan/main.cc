#include <torch/extension.h>

torch::Tensor forward(torch::Tensor data, float eps, int min_pts);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
