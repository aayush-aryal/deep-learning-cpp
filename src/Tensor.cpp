#include "Tensor.h"
#include <iostream>
#include <stdexcept>

Tensor::Tensor(std::vector<int> shape){
    shape_=shape;
    // intuition said to use nested vectors
    // but need to use flat vector because nested vectors are scattered all over RAM
    // use strides to mimic the vector of row, col
    rows_= shape_[0];
    cols_= shape_[1];

    data_.resize(rows_*cols_,0.0f);
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
    // matrix must have equal rows and columns to be added
    int rows= other.shape_[0];
    int cols= other.shape_[1];

    if (rows_!=rows || cols_!= cols){
        // throw an exception
        throw std::runtime_error("Cannot add two Tensors with different dimensions.");
    };

    Tensor result(shape_);
    int total_elements=rows_*cols_;
    for (int i=0;i<total_elements;i++){
        int sum=data_[i]+other.data_[i];
        result.data_[i]=sum;
    };
     return result;
};

std::ostream& operator<<(std::ostream& os,const Tensor& t){
    os<< "Shape ("<<t.shape_[0]<< "x"<< t.shape_[1]<< ")"<<std::endl;
    int rows=t.shape_[0];
    int cols=t.shape_[1];
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