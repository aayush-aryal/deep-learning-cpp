#pragma once 
#include "autograd/BackwardNode.hpp"
#include <memory>

class Tensor;


/**
 * @brief Backward operation for the Reshape  function.
 *
 * Propagates gradients through the Reshape operation during backpropagation.
 */
class ReshapeBackward:public BackwardNode{
    public:
        /**
         * @brief Creates a backward node for a Reshape operation.
         *
         * @param pA Input tensor to the Reshape operation.
         * @param rG Gradient associated with the Reshape result.
         */
        ReshapeBackward(std::shared_ptr<const Tensor>pA, std::shared_ptr<std::vector<float>> rG){
            parentA_=pA;
            result_grad_=rG;
        }
        std::vector<std::shared_ptr<const Tensor>> get_parents() override;
        std::shared_ptr<std::vector<float>> get_gradient() override;
        void apply(const std::vector<float>& incoming_grad_) override;
        
    private:
        std::shared_ptr<const Tensor> parentA_;
        std::shared_ptr<std::vector<float>> result_grad_;
    
};