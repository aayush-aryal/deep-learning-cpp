#pragma once 
#include <memory>
#include <vector>


class Tensor; 

class LayerNorm{
    public:
    LayerNorm(size_t  d_model, float eps= 1e-5f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::vector<std::shared_ptr<Tensor>> parameters();

    private:
        size_t d_model_;
        float eps_;
        std::shared_ptr<Tensor> gamma_;
        std::shared_ptr<Tensor> beta_;
};