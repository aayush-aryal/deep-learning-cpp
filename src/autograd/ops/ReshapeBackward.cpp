#include "autograd/ops/ReshapeBackward.hpp"
#include "Tensor.h"
#include <memory>



std::vector<std::shared_ptr<const Tensor>> ReshapeBackward::get_parents(){
    return {this->parentA_};
}


std::shared_ptr<std::vector<float>> ReshapeBackward::get_gradient(){
    return this->result_grad_;
}

void ReshapeBackward::apply(const std::vector<float>& incoming_grad){

    
    std::vector<float>& gradA= this->parentA_->get_grad();
    for (size_t i=0; i<incoming_grad.size();i++){
        gradA[i]+=incoming_grad[i];
    }
}
