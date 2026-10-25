#pragma once 
#include "autograd/BackwardNode.hpp"
#include <memory>
#include "vector"

class Tensor;


/**
 * @brief Backward operation for the permutation operation.
 *
 * Propagates gradients through the permutation operation during backpropagation.
 */
class PermuteBackward:public BackwardNode{
    public:
        /**
         * @brief Creates a backward node for a permute operation.
         *
         * @param pA Input tensor to the permute operation.
         * @param rG Gradient associated with the permute result.
         */
        PermuteBackward(std::shared_ptr<const Tensor>pA, std::shared_ptr<std::vector<float>> rG, std::vector<size_t> perm, std::vector<size_t> result_shape){
            parentA_=pA;
            result_grad_=rG;
            permutation_=perm;
            result_shape_=result_shape;
        }
        std::vector<std::shared_ptr<const Tensor>> get_parents() override;
        std::shared_ptr<std::vector<float>> get_gradient() override;
        void apply(const std::vector<float>& incoming_grad_) override;
        
    private:
        std::shared_ptr<const Tensor> parentA_;
        std::shared_ptr<std::vector<float>> result_grad_;
        std::vector<size_t> permutation_;
        std::vector<size_t> result_shape_;
    
};