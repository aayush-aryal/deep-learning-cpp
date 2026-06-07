#include "Tensor.h"
#include <iostream>
#include <stdexcept>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "autograd/ops/AddBackward.hpp"


Tensor::Tensor(std::vector<size_t> shape){
    shape_=shape;
    // intuition said to use nested vectors
    // but need to use flat vector because nested vectors are scattered all over RAM
    // use strides to mimic the vector of row, col
    // rows_= shape_[0];
    // cols_= shape_[1];

    size_t total=1;
    for (size_t i=0; i<shape_.size();i++){
        total*=shape[i];
    }

    data_.resize(total,0.0f);
    grad_=std::make_shared<std::vector<float>>(total,0.0f);
    this->requires_grad_=false;
    
}

void Tensor::set(int r, int c, float value){
    int index=r*shape_[1]+c;
    data_[index]=value;

}

float Tensor::get(int r, int c) const{
    if (shape_.size() != 2) throw std::runtime_error("get(r,c) only works on 2D Tensors!");
    int index=r*shape_[1]+c;
    return data_[index];
}

void::Tensor::printShape(){
    std::cout<< "Shape dimensions: " << shape_[0] << "x"<< shape_[1] << std::endl;

}

float Tensor::operator()(int r, int c) const{
    int index=(r*shape_[1])+c;
    return data_[index];

}

Tensor Tensor::add(Tensor& other){
    // matrix must have equal rows and columns to be added or equal col and one must be 1xC

    // special case for add is broadcasting
    // if 32*64 matrix is added to 1*64 it should be broadcasted to every row
    if (other.shape_[0]==1 && this->shape_[1]==other.shape_[1]){
        // broadcast
        Tensor result(this->shape_);
        for (int r=0; r<this->shape_[0];r++){
            for (int c=0; c<other.shape_[1];c++){
                float sum=(*this).get(r,c)+other.get(0,c);
                result.set(r,c,sum);
            }
        }
        auto node=std::make_shared<AddBackward>(std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other),result.grad_);
        result.grad_fn_=node;
        return result;
    }
    if (this->shape_[0]==other.shape_[0] && this->shape_[1]==other.shape_[1]){
            Tensor result(shape_);
            int total_elements=shape_[0]*shape_[1];
            for (int i=0;i<total_elements;i++){
                float sum=data_[i]+other.data_[i];
                result.data_[i]=sum;
            };
        auto node=std::make_shared<AddBackward>(std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other),result.grad_);
        result.grad_fn_=node;
        return result;
    }
    throw std::runtime_error("Cannot add: dimensions mismatch and not broadcastable.");
}

std::ostream& operator<<(std::ostream& os,const Tensor& t){
    // overload << so it knows how to print tensors
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

void Tensor::randomize(){
    // use random library instead of rand because it cannot give enough random values for large NNs
    // std::default_random_engine generator;
    // std::uniform_real_distribution<float> distribution(-1.0,1.0);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    int total=shape_[0]*shape_[1];
    for (int i=0; i<total;i++){
        data_[i]=distribution(generator);
    }
}

Tensor Tensor::matmul(const Tensor& other) const{
    // matrix a can be multiplied with matrix b if
    // column of matrix A must be equal to row of Matrix B
    if (this->shape_[1]!=other.shape_[0]){
        throw std::runtime_error("Cannot multiply matrices because dimension mismatch");
    }

    // resulting matrix will be (Rows of A, Columns of B)

    std::vector<size_t>result_shape={this->shape_[0],other.shape_[1]};
    Tensor result_tensor=Tensor(result_shape);
    for (int r=0; r<this->shape_[0]; r++){
        for(int c=0; c<other.shape_[1];c++){
            float dotProduct=0.0f;
            for (int k=0;k<this->shape_[1];k++){
                dotProduct+=(*this)(r,k)*other(k,c);
            }
            result_tensor.set(r,c,dotProduct);   
        }
    }
    return result_tensor;
}


void Tensor::relu(){
    int max_data=this->data_.size();
    for (int i=0;i<max_data;i++){
        if (this->data_[i]<0){
            this->data_[i]=0;
        }
    }
}


void Tensor::backward(){
    // if its not created through a mathematical operation return
    if (!this->grad_fn_)return;

    for (size_t i = 0; i < this->grad_->size(); i++) {
        (*this->grad_)[i] = 1.0f; // The derivative of the output w.r.t itself
    }

    std::queue<BackwardNode*> queueP;
    std::unordered_map<BackwardNode*, int> dependencies;
    std::unordered_set<BackwardNode*> visited;

    queueP.push(this->grad_fn_.get());
    visited.insert(this->grad_fn_.get());

    while (!queueP.empty()){
        BackwardNode* curr= queueP.front();
        queueP.pop();
        for (std::shared_ptr<Tensor> parent: curr->get_parents()){
            if (parent->grad_fn_){
                BackwardNode* parent_node=parent->grad_fn_.get();
                dependencies[parent_node]++;
                if (visited.find(parent_node)==visited.end()){
                    visited.insert(parent_node);
                    queueP.push(parent_node);
                }
            }
        }
    }

    std::queue<BackwardNode*> orderedExec;
    std::vector<BackwardNode*> executionList;

    // since the dependencies of first parent node is 0 push it
    orderedExec.push(this->grad_fn_.get());
    while (!orderedExec.empty()){
        BackwardNode* curr= orderedExec.front();
        orderedExec.pop();

        executionList.push_back(curr);
        for (std::shared_ptr<Tensor> parent:curr->get_parents()){
            if (parent->grad_fn_){
                BackwardNode* parent_node=parent->grad_fn_.get();
                dependencies[parent_node]--;

                if (dependencies[parent_node]==0){
                    orderedExec.push(parent_node);
                }
            }
        }
    }

    // now our execution list contains the perfect order to execute our nodes in
    for (BackwardNode* node: executionList){
        std::shared_ptr<std::vector<float>>incoming_grad= node->get_gradient();
        node->apply(*incoming_grad);
    }
    
}