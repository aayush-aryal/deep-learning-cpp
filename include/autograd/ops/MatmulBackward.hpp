#pragma once 
#include "autograd/BackwardNode.hpp"
#include <memory>
#include "vector"

/**
 * @brief Backwards Node for tensor multiplication
 * 
 * Propagates gradients from the result of an a multiplication operation
 * to its parent tensors, including any necessary broadcasting.
 */
class MatmulBackward: public BackwardNode{

    public:
    /**
         * @brief Creates a backward node for a tensor multiplication operation.
         *
         * @param pA First tensor involved in the multiplication.
         * @param pB Second tensor involved in the multiplication.
         * @param rG Gradient associated with the result tensor.
         * @param resultShape Shape of the result tensor.
    */  
    MatmulBackward(std::shared_ptr<const Tensor>pA, std::shared_ptr<const Tensor>pB,std::shared_ptr<std::vector<float>> rG, std::vector<size_t> rS){
        parentA_=pA;
        parentB_=pB;
        result_grad_=rG;
        result_shape_=rS;
    }
    void apply(const std::vector<float>&incoming_grad) override;
    std::vector<std::shared_ptr<const Tensor>> get_parents() override;
    std::shared_ptr<std::vector<float>> get_gradient() override;

    private:
    std::shared_ptr<const Tensor> parentA_;
    std::shared_ptr<const Tensor>parentB_;
    std::shared_ptr<std::vector<float>> result_grad_;
    std::vector<size_t> result_shape_;

};