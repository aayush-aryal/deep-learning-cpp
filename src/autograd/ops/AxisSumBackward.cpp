#include "autograd/ops/AxisSumBackward.hpp"
#include "Tensor.h"
#include <memory>

bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape);
size_t get_correct_index(std::vector<size_t> index, std::vector<size_t> strides);
std::vector<size_t> compute_strides_with_shape(std::vector<size_t> shape);


std::vector<std::shared_ptr<const Tensor>> AxisSumBackward::get_parents(){
    return {this->parentA_};
}



std::shared_ptr<std::vector<float>> AxisSumBackward::get_gradient(){
    return this->result_grad_;
}


void AxisSumBackward::apply(const std::vector<float>& incoming_grad ){
    std::vector<float>& gradA= this->parentA_->get_grad();
    std::vector<size_t>idx(this->parentA_->get_shape().size(),0);
    do{
        size_t flat_axis_A= get_correct_index(idx,this->parentA_->get_strides());
        std::vector<size_t> out_idx=idx;
        out_idx[this->axis_]=0;
        size_t flat_axis_output= get_correct_index(out_idx,result_strides_);
        gradA[flat_axis_A]+=incoming_grad[flat_axis_output];
    }while(increment_index(idx,this->parentA_->get_shape()));

}


