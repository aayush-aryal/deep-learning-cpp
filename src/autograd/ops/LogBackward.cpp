#include "autograd/ops/LogBackward.hpp"
#include "Tensor.h"
#include <memory>



std::vector<std::shared_ptr<const Tensor>> LogBackward::get_parents(){
    return {this->parentA_};
}



std::shared_ptr<std::vector<float>> LogBackward::get_gradient(){
    return this->result_grad_;
}


void LogBackward::apply(const std::vector<float>& incoming_grad ){
    std::vector<float>& gradA= this->parentA_->get_grad();
    const std::vector<float>& x= this->parentA_->get_data();
    for (int i=0; i<gradA.size();i++){
       gradA[i]+=incoming_grad[i]/x[i];
    }
}


