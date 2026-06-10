#include "autograd/ops/MatmulBackward.hpp"
#include <memory>
#include "Tensor.h"


// nede a helped function for matrix multiplication
// here it uses transposed matrix as the final shape has to match

std::vector<float> matrix_multiply(const std::vector<float>& ma,const std::vector<float>& mb, std::vector<size_t> shape1, std::vector<size_t> shape2, bool transpose_ma, bool transpose_mb){
    std::vector<float>result;
    // assuming vector ma in shape MxN vector  
    int M= transpose_ma?shape1[1]:shape1[0];
    int N= transpose_ma?shape1[0]:shape1[1];

    // assuming vector mb in vector NxP vector

    int P=transpose_mb?shape2[0]:shape2[1];

    // resulting matrix is MxP

    result.resize(M*P,0.0f);
    // resulting 
    for (size_t r=0; r<M; r++){
        for (size_t c=0; c<P; c++){
            for (size_t k=0; k<N; k++){
                size_t indexA;
                size_t indexB;

                // ma only cares about r,k
                if (transpose_ma){
                    // row k, col r
                    indexA=(k*shape1[1])+r;
                }else{
                    // row r , col k
                    indexA=(r*shape1[1])+k;
                }
                // mb only cares about k,c
                if (transpose_mb){
                    // row c, col k
                    indexB=(c*shape2[1])+k;
                }else{
                    indexB=(k*shape2[1])+c;
                }
                size_t resultIndex=(r*P)+c;
                result[resultIndex]+=ma[indexA] *mb[indexB];  
            }
        }
    }
    return result;
}


void MatmulBackward::apply(const std::vector<float>& incoming_grad){
    // get parent gradient 
    std::vector<float>& gradA= this->parentA_->get_grad();
    std::vector<float>& gradB=this->parentB_->get_grad();


    // get actual data since it is multiplied by incoming gradients for backprop
    const std::vector<float>& dataA=this->parentA_->get_data();
    const std::vector<float>& dataB= this-> parentB_->get_data();

    std::vector<size_t> shapeA = this->parentA_->get_shape();
    std::vector<size_t> shapeB = this->parentB_->get_shape();
    std::vector<size_t> shape_incoming = {shapeA[0], shapeB[1]}; // M x P

    // now we need to know how to restribute incoming gradient in betweeen the parent gradient for matrix multiplication
    std::vector<float> resB=matrix_multiply(dataA,incoming_grad,shapeA,shape_incoming,false,true);
    std::vector<float> resA=matrix_multiply(incoming_grad,dataB,shape_incoming,shapeB,true,false);

    for (size_t i=0; i<gradA.size();i++){
        gradA[i]+=resA[i];
    }
    for (size_t i=0; i<gradB.size();i++){
        gradB[i]+=resB[i];
    }
}



std::vector<std::shared_ptr<const Tensor>> MatmulBackward::get_parents(){
    return {this->parentA_, this->parentB_};
}

std::shared_ptr<std::vector<float>> MatmulBackward::get_gradient(){
    return this->result_grad_;
}


