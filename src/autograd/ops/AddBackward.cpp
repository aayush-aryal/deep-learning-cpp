#include "autograd/ops/AddBackward.hpp"
#include "Tensor.h"
#include <memory>

void AddBackward::apply(const std::vector<float>& incoming_grad){
    // parent grad is a vector of float
    std::vector<float>&gradA=this->parentA_->get_grad();
    std::vector<float>& gradB=this->parentB_->get_grad();

    for (size_t i=0; i<incoming_grad.size();i++){
        gradA[i]+=incoming_grad[i];
        gradB[i]+=incoming_grad[i];
    }
}

std::vector<std::shared_ptr<const Tensor>>AddBackward::get_parents(){
    return {this->parentA_,this->parentB_};
}

std::shared_ptr<std::vector<float>> AddBackward::get_gradient(){
    return this->result_grad_;
}