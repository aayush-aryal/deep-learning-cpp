#include "autograd/ops/MatmulBackward.hpp"
#include <memory>
#include "Tensor.h"


// helper func might put in a utulities folder later
bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape);
size_t get_correct_index(std::vector<size_t> index, std::vector<size_t> strides);
std::vector<size_t> compute_strides_with_shape(std::vector<size_t> shape);


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
    std::vector<float> resB=matrix_multiply(dataA,incoming_grad,shapeA,shape_incoming,true,false);
    std::vector<float> resA=matrix_multiply(incoming_grad,dataB,shape_incoming,shapeB,false,true);

    for (size_t i=0; i<gradA.size();i++){
        gradA[i]+=resA[i];
    }
    for (size_t i=0; i<gradB.size();i++){
        gradB[i]+=resB[i];
    }
}


// good thing we have matrix multiplication helper 
// can use that as (a,b,c,m,n) as only the last two axes are what that gets multiplied
// create a counter like and increment and for each find its index and then multiply the matrices?

void MatmulBackward::apply(const std::vector<float>& incoming_grad){
    std::vector<float>& gradA= this->parentA_->get_grad();
    std::vector<float>& gradB= this->parentB_->get_grad();


    // since the gradient that flows through is dependent on the actual values of the other matrix ie the incoming grad is multiplied by the transpose of the other matix
    std::vector<size_t> shapeA= this->parentA_->get_shape();
    std::vector<size_t> shapeB= this->parentB_->get_shape();


    size_t target_length=std::max(shapeA.size(),shapeB.size());

    // if there is broadcasting then we need to pad the shapes so they are uniform
    std::vector<size_t> padded_shapeA= this->parentA_->pad_shape(shapeA,target_length);
    std::vector<size_t> padded_shapeB= this->parentB_->pad_shape(shapeB, target_length);

    // now create an index and for each index get the offsets to get to the correct 2 matrices to multiply 
    // based on those matrices we change the gradients accordingly!!

    // now that we padded shapes the strides that the parents have are of diff size so we need to pad the strides as well
    std::vector<size_t> padded_strides_shapeA= this->parentA_->pad_strides(shapeA, this->parentA_->get_strides(),target_length);
    std::vector<size_t> padded_strides_shapeB= this->parentB_->pad_strides(shapeB, this->parentB_->get_strides(),target_length);
    std::vector<size_t> res_strides=compute_strides_with_shape(resultShape_);


    // get teh last 2 dimensions since that is where we will be doing our multiplication

    // the matrices are A=(m,k) and B=(p,n) and final matrix is Res=(m,n)
     size_t k= padded_shapeA[target_length-1];
    size_t m= padded_shapeA[target_length-2];

    size_t n= padded_shapeB[target_length-1];
    size_t p= padded_shapeB[target_length-2];

    // we need the actual data for a and b as it is used to calculate gradients
    const std::vector<float>& dataA=this->parentA_->get_data();
    const std::vector<float>& dataB= this-> parentB_->get_data();

    // now that we have the correct strides, we can use them to index when needed
    std::vector<size_t> broadcast_dim;
    for (int i=0; i<target_length-2;i++){
        broadcast_dim.push_back(std::max(padded_shapeA[i],padded_shapeB[i]));
    }
    
    std::vector<size_t> idx(broadcast_dim.size(),0);
    do{
        size_t flat_index_res=get_correct_index(idx, res_strides);
        size_t flat_index_parentA= get_correct_index(idx,padded_strides_shapeA);
        size_t flat_index_parentB= get_correct_index(idx,padded_strides_shapeB);

        // now loop through the last two dimensions and matrix multiply the transpose to get the correct shape

        // we know to get parent A we have to transpose parentB and then res * parentB to get the parentA grad

        // gradA = res* transpose of b 
        for (int row=0; row<m; row++){
            for (int col=0; col<p;col++){
                float dotProduct=0.0;
                // since we swapped row and cols to transpose b we also have to swap strides 
                for (int j=0; j<n;j++){
                    dotProduct+=incoming_grad[flat_index_res+row*res_strides[target_length-2]+j*res_strides[target_length-1]]* dataB[flat_index_parentB+j*padded_strides_shapeB[target_length-1]+col*padded_strides_shapeB[target_length-2]];
                }
                gradA[flat_index_parentA+row*padded_strides_shapeA[target_length-2]+col*padded_strides_shapeA[target_length-1]]+=dotProduct;
            }
        }

        // gradB= transposeA * res 
        for (int row=0; row<k;row++){
            for (int col=0; col<n;col++){
                float dotProduct=0.0;
                for (int kk=0; kk<m;kk++){
                    dotProduct+=dataA[flat_index_parentA+row*padded_strides_shapeA[target_length-1]+kk*padded_strides_shapeA[target_length-2]]*incoming_grad[flat_index_res+kk*res_strides[target_length-2]+col*res_strides[target_length-1]];
                }
                gradB[flat_index_parentB+row*padded_strides_shapeB[target_length-2]+col*padded_strides_shapeB[target_length-1]]+=dotProduct;
            }
        }   
    }while(increment_index(idx,broadcast_dim));

}



std::vector<std::shared_ptr< const Tensor>> MatmulBackward::get_parents(){
    return {this->parentA_, this->parentB_};
}

std::shared_ptr<std::vector<float>> MatmulBackward::get_gradient(){
    return this->result_grad_;
}


