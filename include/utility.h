#pragma once
#include <memory>
class Tensor;
float accuracy(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target);