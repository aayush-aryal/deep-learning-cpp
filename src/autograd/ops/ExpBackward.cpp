#include "autograd/ops/ExpBackward.hpp"
#include <memory>
#include "Tensor.h"
#include <cmath>


std::vector<std::shared_ptr<const Tensor>> ExpBackward::get_parents(){
    return {this->parentA_};
}



std::shared_ptr<std::vector<float>> ExpBackward::get_gradient(){
    return this->result_grad_;
}

void ExpBackward::apply(const std::vector<float>& incoming_grad){
    std::vector<float>& gradA= this->parentA_->get_grad();
    const std::vector<float>& x= this->parentA_->get_data();

    for (int i=0;i<incoming_grad.size();i++){
        gradA[i]+=incoming_grad[i]*exp(x[i]);
    }
}