#include "autograd/ops/SqrtBackward.hpp"
#include "Tensor.h"
#include <memory>
#include <cmath>


std::vector<std::shared_ptr<const Tensor>> SqrtBackward::get_parents(){
    return {this->parentA_};
}



std::shared_ptr<std::vector<float>> SqrtBackward::get_gradient(){
    return this->result_grad_;
}


void SqrtBackward::apply(const std::vector<float>& incoming_grad){
    std::vector<float>& gradA= this->parentA_->get_grad();
    const std::vector<float>& x= this->parentA_->get_data();
    for (int i=0; i<gradA.size();i++){
        gradA[i]+=incoming_grad[i]/(2.0f*std::sqrt(x[i]));
    }

}