#pragma once
#include <memory>
class Tensor;
float accuracy(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);
bool check_one_tensor(std::shared_ptr<Tensor> target, std::function<std::shared_ptr<Tensor>()> forward_fn, float eps, float tolerance);
bool gradient_check_add(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_matmul(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_subtract(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_divide(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_multiply(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-3, float tolerance = 2e-2);
bool gradient_check_unary(std::shared_ptr<Tensor> t,
                          std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)> op,
                          float eps = 1e-3, float tolerance = 1e-2);