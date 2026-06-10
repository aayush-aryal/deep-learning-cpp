#include "autograd/ops/ReluBackward.hpp"
#include "Tensor.h"
#include <memory>



std::vector<std::shared_ptr<const Tensor>> ReluBackward::get_parents(){
    return {this->parentA_};
}


std::shared_ptr<std::vector<float>> ReluBackward::get_gradient(){
    return this->result_grad_;
}

void ReluBackward::apply(const std::vector<float>& incoming_grad){

    // for relu add incoming_grad to parent if its greater than 0 if not put 0
    std::vector<float>& gradA= this->parentA_->get_grad();
    const std::vector<float>& dataA= this->parentA_->get_data();

    for (size_t i=0; i<incoming_grad.size();i++){
        if (dataA[i]>0.0f){
            gradA[i]+=incoming_grad[i];
        }
    }

}
