#include "autograd/ops/MultiplyBackward.hpp"
#include "Tensor.h"
#include <memory>

bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape);
size_t get_correct_index(std::vector<size_t> index, std::vector<size_t> strides);
std::vector<size_t> compute_strides_with_shape(std::vector<size_t> shape);


void MultiplyBackward::apply(const std::vector<float>& incoming_grad){
    std::vector<float>&gradA=this->parentA_->get_grad();
    std::vector<float>& gradB=this->parentB_->get_grad();
    if(this->parentA_->get_shape()== this->parentB_->get_shape()){
        for (int i=0; i<gradA.size();i++){
            gradA[i]+=parentB_->get_data()[i]*incoming_grad[i];
            gradB[i]+=parentA_->get_data()[i]*incoming_grad[i];
        }
    }else{
        size_t target_length= std::max(this->parentA_->get_shape().size(),this->parentB_->get_shape().size());

        // pad shapes and strides 
        std::vector<size_t> parentA_shape= this->parentA_->get_shape();
        std::vector<size_t> parentB_shape= this->parentB_->get_shape();

        std::vector<size_t> padded_shape_parentA= this->parentA_->pad_shape(parentA_shape,target_length);
        std::vector<size_t> padded_shape_parentB= this->parentB_->pad_shape(parentB_shape,target_length);

        std::vector<size_t> padded_stride_parentA=this->parentA_->pad_strides(parentA_shape,this->parentA_->get_strides(),target_length);
        std::vector<size_t> padded_stride_parentB=this->parentB_->pad_strides(parentB_shape,this->parentB_->get_strides(),target_length);

        std::vector<size_t> result_strides= compute_strides_with_shape(result_shape_);
        std::vector<size_t> idx(target_length,0);
        
        do{
            size_t flat_index_parentA=get_correct_index(idx,padded_stride_parentA);
            size_t flat_index_parentB=get_correct_index(idx,padded_stride_parentB);
            size_t flat_index_result= get_correct_index(idx,result_strides);

            gradA[flat_index_parentA]+=parentB_->get_data()[flat_index_parentB]*incoming_grad[flat_index_result];
            gradB[flat_index_parentB]+=parentA_->get_data()[flat_index_parentA]*incoming_grad[flat_index_result];

        }while(increment_index(idx,this->result_shape_));
    }
}

std::vector<std::shared_ptr<const Tensor>>MultiplyBackward::get_parents(){
    return {this->parentA_,this->parentB_};
}

std::shared_ptr<std::vector<float>> MultiplyBackward::get_gradient(){
    return this->result_grad_;
}


