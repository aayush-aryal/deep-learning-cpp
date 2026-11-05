#include "nn/PositionalEncoding.h"
#include "Tensor.h"
#include <vector>
#include <memory>
#include <cmath>




PositionalEncoding::PositionalEncoding(size_t max_T,size_t d_model){
    max_T_=max_T;
    d_model_=d_model;


    auto pe= std::make_shared<Tensor>(std::vector<size_t>{max_T_,d_model_});
    
    pe_= pe;  

    // loop through rows and cols 
    // how does the pe work for sinusoidal ?
    for (size_t p=0; p<max_T_;p++){
        for (size_t i=0; i<d_model_/2; i++){
            float angle= static_cast<float>(p)/std::pow(10000.0f,static_cast<float>((2.0f*i)/d_model_));
            float s= std::sin(angle);
            float c= std::cos(angle);
            // get the exact flat index
            size_t flat_index= p*d_model_+2*i;
            pe_->get_data_ref()[flat_index]=s;
            pe_->get_data_ref()[flat_index+1]=c;
        }  
    }
}


std::shared_ptr<Tensor> PositionalEncoding::forward(std::shared_ptr<Tensor>x){
    size_t T= x->get_shape()[x->get_shape().size()-2];
    size_t D= x->get_shape()[x->get_shape().size()-1];

    if (T!= max_T_ || D!= d_model_){
        throw std::invalid_argument("shape of the tensor does not match");
    }
    return x->add(pe_);
   
}

std::vector<std::shared_ptr<Tensor>> PositionalEncoding::parameters(){
   return {};
}
