#include "autograd/ops/ReciprocalBackward.hpp"
#include "Tensor.h"
#include <memory>



std::vector<std::shared_ptr<const Tensor>> ReciprocalBackward::get_parents(){
    return {this->parentA_};
}



std::shared_ptr<std::vector<float>> ReciprocalBackward::get_gradient(){
    return this->result_grad_;
}


void ReciprocalBackward::apply(const std::vector<float>& incoming_grad){
    std::vector<float>& gradA= this->parentA_->get_grad();
    const std::vector<float>& x= this->parentA_->get_data();
    for (int i=0; i<gradA.size();i++){
        gradA[i]+=incoming_grad[i]*(-1.0f/(x[i]*x[i]));
    }

}