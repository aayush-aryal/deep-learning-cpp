#pragma once 
#include "autograd/BackwardNode.hpp"
#include <memory>
#include <vector>

class Tensor;


/**
 * @brief Backward operation for the Gather  function.
 *
 * Propagates gradients through the Gather operation during backpropagation.
 */
class GatherBackward:public BackwardNode{
    public:
        /**
         * @brief Creates a backward node for a Gather operation.
         *
         * @param pA Input tensor to the Gather operation.
         * @param rG Gradient associated with the Gather result.
         */
        GatherBackward(std::shared_ptr<const Tensor>pA, std::shared_ptr<std::vector<float>> rG, std::vector<size_t> ids, size_t D){
            parentA_=pA;
            result_grad_=rG;
            ids_=ids;
            D_=D;
        }
        std::vector<std::shared_ptr<const Tensor>> get_parents() override;
        std::shared_ptr<std::vector<float>> get_gradient() override;
        void apply(const std::vector<float>& incoming_grad_) override;
        
    private:
        std::shared_ptr<const Tensor> parentA_;
        std::shared_ptr<std::vector<float>> result_grad_;
        std::vector<size_t> ids_;
        size_t D_;
    
};