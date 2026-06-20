#include <memory>
#include "Tensor.h"
#include "autograd/ops/SoftmaxBackward.hpp"

std::vector<std::shared_ptr<const Tensor>> SoftmaxBackward::get_parents(){
    return {this->parentA_};
}


std::shared_ptr<std::vector<float>> SoftmaxBackward::get_gradient(){
    return this->result_grad_;
}


void SoftmaxBackward::apply(const std::vector<float>& incoming_grad){
    // how to backpropagate softmax?

    std::vector<float>& gradA=this->parentA_->get_grad();
    std::vector<float>& target= this->target_->get_data_ref();

  

    for (size_t i=0; i<gradA.size();i++){
        gradA[i]+=((*this->softmax_prob_)[i]-target[i])/target_->get_shape()[0];
    }
}

