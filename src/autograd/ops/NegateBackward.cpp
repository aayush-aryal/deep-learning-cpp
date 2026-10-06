#include "autograd/ops/NegateBackward.hpp"
#include "Tensor.h"
#include <memory>



std::vector<std::shared_ptr<const Tensor>> NegateBackward::get_parents(){
    return {this->parentA_};
}



std::shared_ptr<std::vector<float>> NegateBackward::get_gradient(){
    return this->result_grad_;
}


void NegateBackward::apply(const std::vector<float>& incoming_grad ){
    std::vector<float>& gradA= this->parentA_->get_grad();
    for (int i=0; i<gradA.size();i++){
        gradA[i]-=incoming_grad[i];
    }

}


