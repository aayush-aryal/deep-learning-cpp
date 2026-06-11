#include <memory>
#include "autograd/ops/MSEBackward.hpp"
#include "Tensor.h"





void MSEBackward::apply(const std::vector<float>& incoming_grad){
    const std::vector<float>& dataA=this->parentA_->get_data();
    std::vector<float>& gradA= this->parentA_->get_grad();

    const std::vector<float>& target=this->target_->get_data();

    int N= target.size();
    for (size_t i=0; i<dataA.size(); i++){
        gradA[i]+= (2.0f/N)* (dataA[i]-target[i]) * incoming_grad[0];
    }
}


std::vector<std::shared_ptr< const Tensor>> MSEBackward::get_parents(){
    return {parentA_,target_};
}


std::shared_ptr<std::vector<float>> MSEBackward::get_gradient(){
    return this->result_grad_;
}