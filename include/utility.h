#pragma once
#include <memory>
#include <functional>
#include <vector>

class Tensor;
float accuracy(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
bool check_one_tensor(std::shared_ptr<Tensor> target, std::function<std::shared_ptr<Tensor>()> forward_fn, float eps, float tolerance,float atol = 1e-3f);
bool gradient_check_add(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_matmul(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_subtract(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_divide(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_multiply(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_unary(std::shared_ptr<Tensor> t,
                          std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)> op,
                          float eps = 1e-3, float tolerance = 1e-2);

size_t normalize_axis(int axis, size_t rank);
 std::shared_ptr<Tensor> attention(std::shared_ptr<Tensor> Q, std::shared_ptr<Tensor> K, std::shared_ptr<Tensor> V,bool causal=false);
 std::shared_ptr<Tensor> causal_mask(size_t T);
