#include "autograd/ops/PermuteBackward.hpp"
#include "Tensor.h"
#include <memory>


bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape);
size_t get_correct_index(std::vector<size_t> index, std::vector<size_t> strides);
std::vector<size_t> compute_strides_with_shape(std::vector<size_t> shape);


std::vector<std::shared_ptr<const Tensor>> PermuteBackward::get_parents(){
    return {this->parentA_};
}


std::shared_ptr<std::vector<float>> PermuteBackward::get_gradient(){
    return this->result_grad_;
}

void PermuteBackward::apply(const std::vector<float>& incoming_grad){
    std::vector<float>& gradA= this->parentA_->get_grad();
    std::vector<size_t> idx(this->result_shape_.size(),0);

    std::vector<size_t> result_strides= compute_strides_with_shape(result_shape_);
    do{
        size_t result_index_flat= get_correct_index(idx,result_strides);
        std::vector<size_t> in_idx(result_shape_.size(),0);
        for (size_t i=0; i<in_idx.size();i++){
            in_idx[permutation_[i]]=idx[i];
        }
        size_t flat_index_in= get_correct_index(in_idx,this->parentA_->get_strides());
        gradA[flat_index_in]+=incoming_grad[result_index_flat];
    }while(increment_index(idx,this->result_shape_));
}
