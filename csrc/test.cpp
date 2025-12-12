#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include <tuple>

// ==========================================
// Part 1: NPU Kernel 实现 (模拟)
// ==========================================

// 正向：输入 x, y -> 输出 out1, out2
std::tuple<torch::Tensor, torch::Tensor> my_npu_forward_impl(const torch::Tensor& x, const torch::Tensor& y) {
    auto out1 = x + y;
    auto out2 = x * y;
    return std::make_tuple(out1, out2);
}

// 反向：输入 grad_out1, grad_out2, 以及正向的 x, y -> 输出 grad_x, grad_y
std::tuple<torch::Tensor, torch::Tensor> my_npu_backward_impl(
    const torch::Tensor& grad_out1, 
    const torch::Tensor& grad_out2, 
    const torch::Tensor& x, 
    const torch::Tensor& y) {
    
    // out1 = x + y  => d(out1)/dx = 1, d(out1)/dy = 1
    // out2 = x * y  => d(out2)/dx = y, d(out2)/dy = x
    
    // grad_x = grad_out1 * 1 + grad_out2 * y
    auto grad_x = grad_out1 + grad_out2 * y;
    
    // grad_y = grad_out1 * 1 + grad_out2 * x
    auto grad_y = grad_out1 + grad_out2 * x;
    
    return std::make_tuple(grad_x, grad_y);
}

// ==========================================
// Part 2: Autograd Function
// ==========================================

class MultiIOFunction : public torch::autograd::Function<MultiIOFunction> {
public:
    // Forward: 接收多个 Tensor，返回 variable_list
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx, 
        torch::Tensor x, 
        torch::Tensor y) {
        
        // 1. 保存需要用于反向计算的 Tensor
        ctx->save_for_backward({x, y});
        
        // 2. 调用 NPU 实现
        // 这里的 tuple 需要解包放入 vector 返回给 autograd 系统
        auto result = my_npu_forward_impl(x, y);
        
        return {std::get<0>(result), std::get<1>(result)};
    }

    // Backward: 接收 variable_list (对应 forward 的输出个数)
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx, 
        torch::autograd::variable_list grad_outputs) {
        
        // grad_outputs 的大小等于 Forward 输出的个数 (这里是 2)
        auto grad_out1 = grad_outputs[0];
        auto grad_out2 = grad_outputs[1];

        // 1. 获取保存的 Tensor
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto y = saved[1];

        // 2. 调用 NPU 反向实现
        auto grads = my_npu_backward_impl(grad_out1, grad_out2, x, y);

        // 3. 返回 variable_list (大小必须等于 Forward 输入的个数，这里是 2)
        // 顺序对应 x, y
        return {std::get<0>(grads), std::get<1>(grads)};
    }
};

// ==========================================
// Part 3: Wrapper & 注册
// ==========================================

// 包装器：连接 Autograd 和 Schema
// Schema 期望返回 (Tensor, Tensor)，但 Function::apply 返回 vector<Tensor>
std::tuple<torch::Tensor, torch::Tensor> multi_io_autograd_wrapper(const torch::Tensor& x, const torch::Tensor& y) {
    auto result = MultiIOFunction::apply(x, y);
    return std::make_tuple(result[0], result[1]);
}

TORCH_LIBRARY(my_npu_lib, m) {
    // Schema 定义：使用括号表示多输入和多输出
    // (Tensor x, Tensor y) -> (Tensor, Tensor)
    m.def("multi_op(Tensor x, Tensor y) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(my_npu_lib, AutogradPrivateUse1, m) {
    m.impl("multi_op", multi_io_autograd_wrapper);
}

// 如果需要，也可以单独注册不带 Autograd 的 NPU 版本
TORCH_LIBRARY_IMPL(my_npu_lib, PrivateUse1, m) {
    m.impl("multi_op", my_npu_forward_impl);
}