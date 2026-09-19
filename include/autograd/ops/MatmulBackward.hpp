#pragma once 
#include "autograd/BackwardNode.hpp"
#include <memory>
#include "vector"

class MatmulBackward: public BackwardNode{

    public:
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