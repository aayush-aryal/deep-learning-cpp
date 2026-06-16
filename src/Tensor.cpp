#include "Tensor.h"
#include <iostream>
#include <stdexcept>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "autograd/ops/AddBackward.hpp"
#include "autograd/ops/MatmulBackward.hpp"
#include "autograd/ops/ReluBackward.hpp"
#include "autograd/ops/MSEBackward.hpp"
#include "autograd/ops/SoftmaxBackward.hpp"
#include <memory>


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

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other){
    // matrix must have equal rows and columns to be added or equal col and one must be 1xC

    // special case for add is broadcasting
    // if 32*64 matrix is added to 1*64 it should be broadcasted to every row
    if (other->shape_[0]==1 && this->shape_[1]==other->shape_[1]){
        // broadcast
        auto result= std::make_shared<Tensor>(this->shape_);
        for (int r=0; r<this->shape_[0];r++){
            for (int c=0; c<other->shape_[1];c++){
                float sum=(*this).get(r,c)+other->get(0,c);
                result->set(r,c,sum);
            }
        }
        if (this->requires_grad_|| other->requires_grad_){
            auto node=std::make_shared<AddBackward>(this->shared_from_this(),other,result->grad_);
            result->grad_fn_=node;
            result->requires_grad_=true;
        }
        
        return result;
    }
    if (this->shape_[0]==other->shape_[0] && this->shape_[1]==other->shape_[1]){
            auto result= std::make_shared<Tensor>(this->shape_);
            int total_elements=shape_[0]*shape_[1];
            for (int i=0;i<total_elements;i++){
                float sum=data_[i]+other->data_[i];
                result->data_[i]=sum;
            };

        if (this->requires_grad_|| other->requires_grad_){
            auto node=std::make_shared<AddBackward>(this->shared_from_this(),other,result->grad_);
            result->grad_fn_=node;
            result->requires_grad_=true;
        }
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

void Tensor::randomize(size_t input){
    // use random library instead of rand because it cannot give enough random values for large NNs
    // std::default_random_engine generator;
    // std::uniform_real_distribution<float> distribution(-1.0,1.0);

    std::random_device rd;
    std::mt19937 generator(rd());
    float stdev=std::sqrt(2.0f/input);
    std::normal_distribution<float> distribution(0.0f,stdev);
    int total=shape_[0]*shape_[1];
    for (int i=0; i<total;i++){
        data_[i]=distribution(generator);
    }
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) const{
    // matrix a can be multiplied with matrix b if
    // column of matrix A must be equal to row of Matrix B
    if (this->shape_[1]!=other->shape_[0]){
        throw std::runtime_error("Cannot multiply matrices because dimension mismatch");
    }

    // resulting matrix will be (Rows of A, Columns of B)

    std::vector<size_t>result_shape={this->shape_[0],other->shape_[1]};
    auto result_tensor=std::make_shared<Tensor>(result_shape);
    for (int r=0; r<this->shape_[0]; r++){
        for(int c=0; c<other->shape_[1];c++){
            float dotProduct=0.0f;
            for (int k=0;k<this->shape_[1];k++){
                dotProduct+=(*this)(r,k)*(*other)(k,c);
            }
            result_tensor->set(r,c,dotProduct);   
        }
    }

    if (this->requires_grad_|| other->requires_grad_){
            auto node=std::make_shared<MatmulBackward>(this->shared_from_this(),other,result_tensor->grad_);
            result_tensor->grad_fn_=node;
            result_tensor->requires_grad_=true;
        }
        
    return result_tensor;
}


// to work on probably a new tensor reference after relu that is created that has the output
// add the shared backward node 
std::shared_ptr<Tensor> Tensor::relu(){

    int max_data=this->data_.size();
    auto result_tensor=std::make_shared<Tensor>(this->shape_);
    for (size_t i=0;i<max_data;i++){
        if (this->data_[i]<0){
            result_tensor->data_[i]=0;
        }else{
            result_tensor->data_[i]=data_[i];
        }
    }
    if (this->requires_grad_){
        auto node= std::make_shared<ReluBackward>(this->shared_from_this(),result_tensor->grad_);
        result_tensor->grad_fn_=node;
        result_tensor->requires_grad_=true;
    }

    return result_tensor;
}


// since in other operations output at index (r,c) depends on the input at index (r,c)
// but in softmax since we sum all exponents, each output is dependent on all other input
// if you implement softmax and backprpop on its own with crossentropy and its backward the back propagation becomes very complex and numerically unstable
// so if you fuse the two operations the resulting grad is easily calculated as it is (pred-target) for each


std::shared_ptr<Tensor> Tensor::softmax_crossentropy(std::shared_ptr<Tensor>target){
    auto softmax_prob=std::make_shared<std::vector<float>>();
    (*softmax_prob).resize(this->shape_[0]* this->shape_[1]);
    float total=0.0f;

        // exp(x - max) / sum(exp(xi - max))
        // = exp(x) * exp(-max) / (sum(exp(xi)) * exp(-max))
        // = exp(x) / sum(exp(xi))

        // we subtract by max of the given row in order to prevent overflow due to exp of large numbers
    for (size_t r=0; r<this->shape_[0];r++){
        total=0.0f;
        float maximum=this->get(r,0);
        for (size_t c=0; c<this->shape_[1];c++){
            if (this->get(r,c)>maximum){
                maximum=this->get(r,c);
            }     
        } 
        for (size_t c=0; c<this->shape_[1];c++){
            total+=(std::exp(this->get(r,c)-maximum));   
        }
        for (size_t c=0; c<this->shape_[1];c++){
            float curr_val=std::exp(this->get(r,c)-maximum);
            size_t index= (r*this->shape_[1])+c;
            (*softmax_prob)[index]=curr_val/total;
        }
    }
    // after calculating the result we need to do crossentropy loss which can be given as 
    //-log(pred_probability_of_true_class)
    float total_loss=0.0f;

    for (size_t row=0; row<this->shape_[0]; row++){
        
        for (size_t col=0; col<this->shape_[1];col++){
            // we use log to determine loss, after we use exp for softmax to create probabilities
            size_t index= (row*this->shape_[1])+col;
            float prob=std::log((*softmax_prob)[index]);
            total_loss+= -1.0f * prob* target->get(row,col);
        }

    }
    // this will store the average loss across all images
    auto loss_tensor=std::make_shared<Tensor>(std::vector<size_t>{1,1});
    loss_tensor->set(0,0,total_loss/this->shape_[0]);
    if (this->get_requires_grad()){
        loss_tensor->set_requires_grad(true);
        // one more step is to add backward node and set it as grad fn
        auto node= std::make_shared<SoftmaxBackward>(this->shared_from_this(),softmax_prob,loss_tensor->grad_,target);
        loss_tensor->grad_fn_=node;
    }
    return loss_tensor;
}
std::shared_ptr<std::vector<float>> Tensor::softmax(std::shared_ptr<Tensor>target){
    auto softmax_prob=std::make_shared<std::vector<float>>();
    (*softmax_prob).resize(this->shape_[0]* this->shape_[1]);
    float total=0.0f;

        // exp(x - max) / sum(exp(xi - max))
        // = exp(x) * exp(-max) / (sum(exp(xi)) * exp(-max))
        // = exp(x) / sum(exp(xi))

        // we subtract by max of the given row in order to prevent overflow due to exp of large numbers
    for (size_t r=0; r<this->shape_[0];r++){
        total=0.0f;
        float maximum=this->get(r,0);
        for (size_t c=0; c<this->shape_[1];c++){
            if (this->get(r,c)>maximum){
                maximum=this->get(r,c);
            }     
        } 
        for (size_t c=0; c<this->shape_[1];c++){
            total+=(std::exp(this->get(r,c)-maximum));   
        }
        for (size_t c=0; c<this->shape_[1];c++){
            float curr_val=std::exp(this->get(r,c)-maximum);
            size_t index= (r*this->shape_[1])+c;
            (*softmax_prob)[index]=curr_val/total;
        }
    }
    return softmax_prob;
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
        for (std::shared_ptr<const Tensor> parent: curr->get_parents()){
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
        for (std::shared_ptr<const Tensor> parent:curr->get_parents()){
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

std::shared_ptr<Tensor> Tensor::mse_loss(std::shared_ptr<Tensor>target){

    // target data
    std::vector<float> data= target->get_data();
    // the resulting Tensor must be size of target
    auto result=std::make_shared<Tensor>(std::vector<size_t>{1,1});
    float total_sum=0.0f;
    for (size_t i=0; i<data.size();i++){
        float actual=data[i];
        float prediction=this->data_[i];
        float curr_loss=(prediction-actual)*(prediction-actual);
        total_sum+= curr_loss;
        
    }

    float mean_loss= total_sum/ data.size();

    result->set(0,0,mean_loss);

    if (this->requires_grad_){
        auto node= std::make_shared<MSEBackward>(target,this->shared_from_this(),result->grad_);
        result->grad_fn_=node;
        result->set_requires_grad(true);
    }
    return result;

}




// void Tensor::backward(){

//     if (!this->grad_fn_){return;}
//     std::unordered_set<BackwardNode*>visited;
//     std::unordered_map<BackwardNode*,int> dependencies;
//     std::queue<BackwardNode*> queueP;

//     // since it has a history add it to the queue 
//     queueP.push(this->grad_fn_.get());
//     while(!queueP.empty()){
//         BackwardNode* curr=queueP.front();
//         queueP.pop();
//         // loop through its parents
//         for(std::shared_ptr<Tensor>parent:curr->get_parents()){
//             BackwardNode* parent_node= parent->grad_fn_.get();
//             dependencies[parent_node]++;
//             if (visited.find(parent_node)==visited.end()){
//                 visited.insert(parent_node);
//                 queueP.push(parent_node);
//             }
//         }
//     }
//     std::queue<BackwardNode*> executionOrd;
//     std::vector<BackwardNode*> executionList;

//     executionOrd.push(this->grad_fn_.get());
//     while (!executionOrd.empty()){
//         BackwardNode* curr= executionOrd.front();
//         executionOrd.pop();

//         executionList.push_back(curr);
//         for( std::shared_ptr<Tensor>parent:curr->get_parents()){
//             BackwardNode* parent_node=parent->grad_fn_.get();
//             dependencies[parent_node]--;
//             if (dependencies[parent_node]==0){
//                 executionOrd.push(parent_node);
//             }
//         }
//     }
//     for (BackwardNode* node:executionList){
//         std::shared_ptr<std::vector<float>> incoming_grad=node->get_gradient();
//         node->apply(*incoming_grad);
//     }


// }