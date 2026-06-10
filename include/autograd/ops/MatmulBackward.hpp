#pragma once 
#include "autograd/BackwardNode.hpp"
#include <memory>


class MatmulBackward: public BackwardNode{

    public:
    MatmulBackward(std::shared_ptr<const Tensor>pA, std::shared_ptr<const Tensor>pB,std::shared_ptr<std::vector<float>> rG){
        parentA_=pA;
        parentB_=pB;
        result_grad_=rG;
    }
    void apply(const std::vector<float>&incoming_grad) override;
    std::vector<std::shared_ptr<const Tensor>> get_parents() override;
    std::shared_ptr<std::vector<float>> get_gradient() override;

    private:
    std::shared_ptr<const Tensor> parentA_;
    std::shared_ptr<const Tensor>parentB_;
    std::shared_ptr<std::vector<float>> result_grad_;

};