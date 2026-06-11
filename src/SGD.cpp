#include <memory>
#include "SGD.h"
#include "Tensor.h"
#include <vector>



void SGD::step(){
    for (std::shared_ptr<Tensor> t: this->tensors_){
        // go through each tensors data
        std::vector<float>& tensor_data= t->get_data_ref();
        std::vector<float>& gradient_data=t->get_grad();
        for (size_t i=0; i<tensor_data.size();i++){
            // most important thing in sgd
            // data = data-  gradient*lr
            tensor_data[i]-=gradient_data[i]*this->lr_;
        }
    }
}

// prevents gradient from accumulating for next forward pass
void SGD::zero_grad(){
    for (std::shared_ptr<Tensor> t: this->tensors_){
        std::vector<float>& gradient_data=t->get_grad();
        for (size_t i=0; i<gradient_data.size();i++){
            gradient_data[i]=0.0f;
        }
    }
}