#include "autograd/ops/GatherBackward.hpp"
#include "Tensor.h"
#include <memory>



std::vector<std::shared_ptr<const Tensor>> GatherBackward::get_parents(){
    return {this->parentA_};
}


std::shared_ptr<std::vector<float>> GatherBackward::get_gradient(){
    return this->result_grad_;
}


void GatherBackward::apply(const std::vector<float>& incoming_grad){
    std::vector<float>& gradA= this->parentA_->get_grad();
    for (size_t p=0; p<this->ids_.size();p++){
        for (size_t d=0; d<this->D_;d++){
            gradA[ids_[p]*D_+d]+= incoming_grad[p*D_+d];
        }
    }
}