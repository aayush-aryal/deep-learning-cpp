#include "autograd/ops/AddBackward.hpp"
#include "Tensor.h"
#include <memory>

bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape);
size_t get_correct_index(std::vector<size_t> index, std::vector<size_t> strides);
std::vector<size_t> compute_strides_with_shape(std::vector<size_t> shape);

// void AddBackward::apply(const std::vector<float>& incoming_grad){
//     // parent grad is a vector of float
//     std::vector<float>&gradA=this->parentA_->get_grad();
//     std::vector<float>& gradB=this->parentB_->get_grad();

//     if (this->parentA_->get_shape()[0]==this->parentB_->get_shape()[0] && this->parentA_->get_shape()[1]==this->parentA_->get_shape()[0]){
//             for (size_t i=0; i<incoming_grad.size();i++){
//                 gradA[i]+=incoming_grad[i];
//                 gradB[i]+=incoming_grad[i];
//             }
//     }else{
//         // special case for broadcasting
//         size_t rows = this->parentA_->get_shape()[0];
//         size_t cols = this->parentA_->get_shape()[1];

//         //dA= incoming grad
//         for (size_t i=0; i<incoming_grad.size();i++){
//             gradA[i]+=incoming_grad[i];
//         }
//         // broadcasting other from [1, C] to [R, C]
//         for (size_t r=0; r<rows;r++){
//             for (size_t c=0; c<cols;c++){
//                 gradB[c]+=incoming_grad[r*cols+c];
//             }
//         }
//     }

// }

// now we need to make our backwards compatible with n-dimensional vectors
void AddBackward::apply(const std::vector<float>& incoming_grad){
    // if the size of both parent vectors are the same then its a simple flow of gradients 
    std::vector<float>&gradA=this->parentA_->get_grad();
    std::vector<float>& gradB=this->parentB_->get_grad();
    if(this->parentA_->get_shape()== this->parentB_->get_shape()){
        for (size_t i=0; i<incoming_grad.size();i++){
            gradA[i]+=incoming_grad[i];
            gradB[i]+=incoming_grad[i];
        }
    }else{
        // what to do for broadcasting?
        std::vector<size_t> parentA_shape= this->parentA_->get_shape();
        std::vector<size_t> parentB_shape= this->parentB_->get_shape();

        size_t target_length= std::max(parentA_shape.size(),parentB_shape.size());
        std::vector<size_t> padded_shape_parentA= this->parentA_->pad_shape(parentA_shape,target_length);
        std::vector<size_t> padded_shape_parentB= this->parentB_->pad_shape(parentB_shape,target_length);
        std::vector<size_t> idx(padded_shape_parentA.size(),0);

        // since we padded A and B potentially their strides might be stale so calculate new strides
        std::vector<size_t> padded_stride_parentA= this->parentA_->pad_strides(parentA_shape,this->parentA_->get_strides(),target_length);
        std::vector<size_t> padded_stride_parentB= this->parentB_->pad_strides(parentB_shape,this->parentB_->get_strides(),target_length);

        std::vector<size_t> result_strides= compute_strides_with_shape(result_shape_);

        std::cout << "padded_stride_parentB: ";
for (auto s : padded_stride_parentB) std::cout << s << " ";
std::cout << std::endl;
std::cout << "gradB size: " << gradB.size() << std::endl;

        do{
            // accumulate in the parents gradient?
            size_t flat_index_parentA= get_correct_index(idx,padded_stride_parentA);
            size_t flat_index_parentB= get_correct_index(idx,padded_stride_parentB);
            size_t flat_index_result= get_correct_index(idx,result_strides);

            gradA[flat_index_parentA]+=incoming_grad[flat_index_result];
            gradB[flat_index_parentB]+=incoming_grad[flat_index_result];

        }while (increment_index(idx,this->result_shape_));
    }

}

std::vector<std::shared_ptr<const Tensor>>AddBackward::get_parents(){
    return {this->parentA_,this->parentB_};
}

std::shared_ptr<std::vector<float>> AddBackward::get_gradient(){
    return this->result_grad_;
}

// redundant helpers for now


