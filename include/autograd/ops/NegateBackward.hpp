#pragma once
#include "autograd/BackwardNode.hpp"
#include <memory>
#include "vector"

class Tensor;

/**
 * @brief Backward operation for tensor negation.
 *
 * Propagates gradients from the result of an addition operation
 * to its parent tensors, including any necessary broadcasting.
 */
class NegateBackward: public BackwardNode{
    public:
        /**
         * @brief Creates a backward node for a tensor addition operation.
         *
         * @param pA Tensor involved in the negation.
         * @param rG Gradient associated with the result tensor.
         */
        NegateBackward(std::shared_ptr<const Tensor> pA, std::shared_ptr<std::vector<float>> rG){
            parentA_=pA;
            result_grad_=rG;
        }
        void apply(const std::vector<float>& incoming_grad) override;
        std::vector<std::shared_ptr<const Tensor>>get_parents() override;
        std::shared_ptr<std::vector<float>> get_gradient() override;

    private:
    // shared pointers are used for reference counting to automatically delete out of scope references
        std::shared_ptr<const Tensor> parentA_;
        std::shared_ptr<std::vector<float>> result_grad_;
        std::vector<size_t> result_shape_;
};

