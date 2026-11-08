#pragma once
#include <memory>
#include <vector>

class Tensor;

/**
 * @brief Adam optimizer for updating tensor parameters.
 *
 */
class Adam{
    public:
    /**
     * @brief Creates an Adam optimizer for a vector of tensors
     * 
     * @param tensor Tensors who's value will be updated during optimization
     * @param lr Learning rate used for parameter updates 
     */
    Adam(std::vector<std::shared_ptr<Tensor>> tensors, float lr=1e-3f, float beta1=0.9f ,float beta2=0.999f,  float eps=1e-8f);
    void step();
    void zero_grad();

    private:
        std::vector<std::shared_ptr<Tensor>> tensors_;
        float lr_;
        float beta1_;
        float beta2_; 
        float eps_;
        std::vector<std::vector<float>> m_, v_;
        size_t t_;

};