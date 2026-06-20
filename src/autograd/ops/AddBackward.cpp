#include "autograd/ops/AddBackward.hpp"
#include "Tensor.h"
#include <memory>

void AddBackward::apply(const std::vector<float>& incoming_grad){
    // parent grad is a vector of float
    std::vector<float>&gradA=this->parentA_->get_grad();
    std::vector<float>& gradB=this->parentB_->get_grad();

    if (this->parentA_->get_shape()[0]==this->parentB_->get_shape()[0] && this->parentA_->get_shape()[1]==this->parentA_->get_shape()[0]){
            for (size_t i=0; i<incoming_grad.size();i++){
                gradA[i]+=incoming_grad[i];
                gradB[i]+=incoming_grad[i];
            }
    }else{
        // special case for broadcasting
        size_t rows = this->parentA_->get_shape()[0];
        size_t cols = this->parentA_->get_shape()[1];

        //dA= incoming grad
        for (size_t i=0; i<incoming_grad.size();i++){
            gradA[i]+=incoming_grad[i];
        }
        // broadcasting other from [1, C] to [R, C]
        for (size_t r=0; r<rows;r++){
            for (size_t c=0; c<cols;c++){
                gradB[c]+=incoming_grad[r*cols+c];
            }
        }
    }

}

std::vector<std::shared_ptr<const Tensor>>AddBackward::get_parents(){
    return {this->parentA_,this->parentB_};
}

std::shared_ptr<std::vector<float>> AddBackward::get_gradient(){
    return this->result_grad_;
}