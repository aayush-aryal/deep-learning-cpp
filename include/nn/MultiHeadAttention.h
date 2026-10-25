#pragma once 
#include <memory>
#include <vector>
#include "nn/Linear.h"
#include "utility.h"


class Tensor;

class MultiHeadAttention{
    public:
    MultiHeadAttention(size_t d_model, size_t num_heads);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x, bool causal=false);
    std::vector<std::shared_ptr<Tensor>> parameters();

    private:
    size_t d_model_;
    size_t num_heads_;

    Linear wq_;
    Linear wk_;
    Linear wv_;
    Linear wo_;
};