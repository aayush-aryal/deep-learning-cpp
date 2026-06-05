#include "Tensor.h"
#include <iostream>
#include <stdexcept>
#include <random>

Tensor::Tensor(std::vector<int> shape){
    shape_=shape;
    // intuition said to use nested vectors
    // but need to use flat vector because nested vectors are scattered all over RAM
    // use strides to mimic the vector of row, col
    rows_= shape_[0];
    cols_= shape_[1];

    data_.resize(rows_*cols_,0.0f);
    // i want to change initialize to have random values upon initialization?
}

void Tensor::set(int r, int c, float value){
    int index=r*cols_+c;
    data_[index]=value;

}

float Tensor::get(int r, int c) const{
    int index=r*cols_+c;
    return data_[index];
}

void::Tensor::printShape(){
    std::cout<< "Shape dimensions: " << rows_ << "x"<< cols_ << std::endl;

}

float Tensor::operator()(int r, int c) const{
    int index=(r*cols_)+c;
    return data_[index];

}

Tensor Tensor::add(const Tensor& other)const{
    // matrix must have equal rows and columns to be added or equal col and one must be 1xC

    // special case for add is broadcasting
    // if 32*64 matrix is added to 1*64 it should be broadcasted to every row
    if (other.rows_==1 && this->cols_==other.cols_){
        // broadcast
        Tensor result(this->shape_);
        for (int r=0; r<this->rows_;r++){
            for (int c=0; c<other.cols_;c++){
                float sum=(*this).get(r,c)+other.get(0,c);
                result.set(r,c,sum);
            }
        }
        return result;
    }
    if (this->rows_==other.rows_ && this->cols_==other.cols_){
            Tensor result(shape_);
            int total_elements=rows_*cols_;
            for (int i=0;i<total_elements;i++){
                float sum=data_[i]+other.data_[i];
                result.data_[i]=sum;
            };
        return result;
    }
    throw std::runtime_error("Cannot add: dimensions mismatch and not broadcastable.");
}

std::ostream& operator<<(std::ostream& os,const Tensor& t){
    // overload << so it knows how to print tensors
    os<< "Shape ("<<t.shape_[0]<< "x"<< t.shape_[1]<< ")"<<std::endl;
    int rows=t.rows_;
    int cols=t.cols_;
    for (int i=0;i<rows;i++){
        os<<"[";
        for (int j=0; j<cols;j++){
            os<<t(i,j);
            if (j!=cols-1){
                os<<",";
            }
        }
        os<<"]"<<std::endl;
    }
    return os;
}

void Tensor::randomize(){
    // use random library instead of rand because it cannot give enough random values for large NNs
    // std::default_random_engine generator;
    // std::uniform_real_distribution<float> distribution(-1.0,1.0);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    int total=rows_*cols_;
    for (int i=0; i<total;i++){
        data_[i]=distribution(generator);
    }
}

Tensor Tensor::matmul(const Tensor& other) const{
    // matrix a can be multiplied with matrix b if
    // column of matrix A must be equal to row of Matrix B
    if (this->cols_!=other.rows_){
        throw std::runtime_error("Cannot multiply matrices because dimension mismatch");
    }

    // resulting matrix will be (Rows of A, Columns of B)

    std::vector<int>result_shape={this->rows_,other.cols_};
    Tensor result_tensor=Tensor(result_shape);
    for (int r=0; r<this->rows_; r++){
        for(int c=0; c<other.cols_;c++){
            float dotProduct=0.0f;
            for (int k=0;k<this->cols_;k++){
                dotProduct+=(*this)(r,k)*other(k,c);
            }
            result_tensor.set(r,c,dotProduct);   
        }
    }
    return result_tensor;
}