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
#include <assert.h>


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

    // set strides
    set_stride(); 
}

void Tensor::set(int r, int c, float value){
    int index=r*shape_[1]+c;
    data_[index]=value;

}

void Tensor::set(std::vector<size_t>idx, float value){
    int index= this->flat_index(idx);
    data_[index]=value;
}

float Tensor::get(int r, int c) const{
    if (shape_.size() != 2) throw std::runtime_error("get(r,c) only works on 2D Tensors!");
    int index=r*shape_[1]+c;
    return data_[index];
}

float Tensor::get(std::vector<size_t> index)const{
    size_t flat_index=this->flat_index(index);
    return data_[flat_index];
}

void Tensor::printShape(){
    std::cout<< "Shape dimensions: ";
    for (int i=0; i<this->shape_.size();i++){
        std::cout<< this->shape_[i]<< ",";
    }
    std::cout<< std::endl;
}

float Tensor::operator()(int r, int c) const{
    int index=(r*shape_[1])+c;
    return data_[index];

}


size_t get_correct_index(std::vector<size_t>index, std::vector<size_t> strides){
    size_t correct_index=0;
    if (index.size()!=strides.size()){
        return -1;
    }
    for (int i=0; i<index.size();i++){
        correct_index+=strides[i]*index[i];
    }
    return correct_index;
}

// returns true if you can increment index and false if it overflowed
bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape){
    int p= idx.size()-1;
    while (p>=0){
        idx[p]=idx[p]+1;
        if (idx[p]<shape[p]){
            return true;
        }else{
            idx[p]=0;
            p--;
        }
            
    }
    return false;
}

// std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other){
//     // matrix must have equal rows and columns to be added or equal col and one must be 1xC

//     // special case for add is broadcasting
//     // if 32*64 matrix is added to 1*64 it should be broadcasted to every row
//     if (other->shape_[0]==1 && this->shape_[1]==other->shape_[1]){
//         // broadcast
//         auto result= std::make_shared<Tensor>(this->shape_);
//         for (int r=0; r<this->shape_[0];r++){
//             for (int c=0; c<other->shape_[1];c++){
//                 float sum=(*this).get(r,c)+other->get(0,c);
//                 result->set(r,c,sum);
//             }
//         }
//         if (this->requires_grad_|| other->requires_grad_){
//             auto node=std::make_shared<AddBackward>(this->shared_from_this(),other,result->grad_);
//             result->grad_fn_=node;
//             result->requires_grad_=true;
//         }
        
//         return result;
//     }
//     if (this->shape_[0]==other->shape_[0] && this->shape_[1]==other->shape_[1]){
//             auto result= std::make_shared<Tensor>(this->shape_);
//             int total_elements=shape_[0]*shape_[1];
//             for (int i=0;i<total_elements;i++){
//                 float sum=data_[i]+other->data_[i];
//                 result->data_[i]=sum;
//             };

//         if (this->requires_grad_|| other->requires_grad_){
//             auto node=std::make_shared<AddBackward>(this->shared_from_this(),other,result->grad_);
//             result->grad_fn_=node;
//             result->requires_grad_=true;
//         }
//         return result;
//     }
//     throw std::runtime_error("Cannot add: dimensions mismatch and not broadcastable.");
// }

// adapting addition operation for n-dimensional tensors
std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> other){
    // two ways we can add two tensors are
    // if all of their dimensions match
    // if their dimensions are equal or one of them is 1
    // if they are equal
    if (this->shape_== other->shape_){
        // we can just sum individual flat index for each tensor
        auto result= std::make_shared<Tensor>(this->shape_);
        for (int i=0; i<this->data_.size();i++){
            result->data_[i]=this->data_[i]+other->data_[i];
        }
        auto node=std::make_shared<AddBackward>(this->shared_from_this(),other,result->grad_);
        result->grad_fn_=node;
        result->requires_grad_=true;
        return result;
    }

    // if not loop through each dimension if they are 1 or equal do sum
    size_t target_length=std::max(this->shape_.size(),other->shape_.size());
    std::vector<size_t> padded_shape_this=this->pad_shape(this->shape_,target_length);
    std::vector<size_t> padded_shape_other=other->pad_shape(other->shape_,target_length);

    // we need to pad strides as well
    std::vector<size_t> padded_strides_this=this->pad_strides(this->shape_,this->strides_,target_length);
     std::vector<size_t> padded_strides_other=other->pad_strides(other->shape_,other->strides_,target_length);


    std::vector<size_t> res_shape;
    for(int i=0; i<padded_shape_this.size();i++){
        if (padded_shape_this[i]!=padded_shape_other[i] && (padded_shape_other[i]!=1 && padded_shape_this[i]!=1)){
             throw std::runtime_error("Cannot add: dimensions mismatch and not broadcastable");
        }
        if (padded_shape_other[i]>padded_shape_this[i]){
            res_shape.push_back(padded_shape_other[i]);
        }else{
            res_shape.push_back(padded_shape_this[i]);
        }
    }
    auto result=std::make_shared<Tensor>(res_shape);

    // now use this loop to actually broadcast
    std::vector<size_t> idx(res_shape.size(),0);
    do{
        size_t flat_index_this=get_correct_index(idx,padded_strides_this);
        size_t flat_index_others=get_correct_index(idx,padded_strides_other);
        size_t flat_index_res=get_correct_index(idx,result->strides_);

        result->data_[flat_index_res]=this->data_[flat_index_this]+other->data_[flat_index_others];

    }while(increment_index(idx,res_shape));   
    auto node=std::make_shared<AddBackward>(this->shared_from_this(),other,result->grad_);
    result->grad_fn_=node;
    result->requires_grad_=true;
    return result;
}
std::ostream& operator<<(std::ostream& os,const Tensor& t){
    // overload << so it knows how to print tensors
    os << "Shape (";
    for (int i=0; i<t.shape_.size();i++){
        os<<t.shape_[i];
        if (i!=t.shape_.size()-1){
            os<<"x";
        }
    }
    os << ")" << std::endl;

    std::vector<size_t>idx(t.shape_.size(),0);

    do{
        size_t index= t.flat_index(idx);
        os<< t.data_[index]<< " ";

    }while (increment_index(idx,t.shape_));

    os<<std::endl;
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
    int total=1;
    for (int i=0; i<this->shape_.size();i++){
        total*=shape_[i];
    }

    for (int i=0; i<total;i++){
        data_[i]=distribution(generator);
    }
}

// std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) const{
//     // matrix a can be multiplied with matrix b if
//     // column of matrix A must be equal to row of Matrix B
//     if (this->shape_[1]!=other->shape_[0]){
//         throw std::runtime_error("Cannot multiply matrices because dimension mismatch");
//     }

//     // resulting matrix will be (Rows of A, Columns of B)

//     std::vector<size_t>result_shape={this->shape_[0],other->shape_[1]};
//     auto result_tensor=std::make_shared<Tensor>(result_shape);
//     for (int r=0; r<this->shape_[0]; r++){
//         for(int c=0; c<other->shape_[1];c++){
//             float dotProduct=0.0f;
//             for (int k=0;k<this->shape_[1];k++){
//                 dotProduct+=(*this)(r,k)*(*other)(k,c);
//             }
//             result_tensor->set(r,c,dotProduct);   
//         }
//     }

//     if (this->requires_grad_|| other->requires_grad_){
//             auto node=std::make_shared<MatmulBackward>(this->shared_from_this(),other,result_tensor->grad_);
//             result_tensor->grad_fn_=node;
//             result_tensor->requires_grad_=true;
//         }
        
//     return result_tensor;
// }

std::vector<size_t>compute_strides_with_shape(std::vector<size_t> shape){
    std::vector<size_t> strides;
    size_t acc_stride=1;

    for (int i=shape.size()-1;i>=0;i--){
        strides.insert(strides.begin(),acc_stride);
        acc_stride=shape[i]*acc_stride;
    }

    return strides;

}


std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other){
    // implement matrix multiplication for n-dimensional arrays
    // if th inner dimensions do not match then you cannot multiply them
    // the core is always the last 2 dimension multiplication
    // which part of the inner most matrices we multiply with each other depends on the outer dimensions and all of their comvination
    

    // first of all we need the shape of final res
    std::vector<size_t> res_shape;

    // if 1d matrix given it should still work, we can pad another dimension to the shape if it is
     bool this_1d= this->shape_.size()==1;
     bool other_1d= other->shape_.size()==1;

     std::vector<size_t> norm_this=this->shape_;
     std::vector<size_t> norm_other=other->shape_;

     // we guarantee atleast a 2d matrix with this

     if (this_1d) norm_this=this_1d?std::vector<size_t>{1,this->shape_[0]}:this->shape_ ;
     if (other_1d) norm_other=other_1d?std::vector<size_t>{other->shape_[0],1}:other->shape_;


     // m*k matrix multiplied by p*n matrix k must equal p 

     // now we can pad both tensors 
    size_t target_length= std::max(norm_this.size(),norm_other.size());

     std::vector<size_t> this_shape_padded= this->pad_shape(norm_this,target_length);
     std::vector<size_t> other_shape_padded=other->pad_shape(norm_other,target_length);

     

     std::vector<size_t> this_strides_padded=this->pad_strides(norm_this,this_1d?compute_strides_with_shape(norm_this):this->strides_,target_length);
     std::vector<size_t> other_strides_padded=other->pad_strides(norm_other,other_1d?compute_strides_with_shape(norm_other):other->strides_,target_length);





    size_t k= this_shape_padded[target_length-1];
    size_t m= this_shape_padded[target_length-2];

    size_t n= other_shape_padded[target_length-1];
    size_t p= other_shape_padded[target_length-2];

    if (k!=p){
        throw std::invalid_argument("Incompatible shapes for matrix multiplication");
     }


     // now check the padded dimensions if any dont match or oen of them arent 1 
     std::vector<size_t> broadcast_dim;

     for (int i=0;i<target_length-2;i++){
        if (this_shape_padded[i]!=other_shape_padded[i] && (this_shape_padded[i]!=1  && other_shape_padded[i]!=1)){
            throw std::invalid_argument("Incompatible shapes for matrix multiplication");
        }
        res_shape.push_back(std::max(this_shape_padded[i],other_shape_padded[i]));
        broadcast_dim.push_back(std::max(this_shape_padded[i],other_shape_padded[i]));
     }
     // res shape must include the innermost dimensions as well that we established 
     res_shape.push_back(m);
     res_shape.push_back(n);
     

     // now we know the res shape create a res tensor 
     auto res_tensor=std::make_shared<Tensor>(res_shape);

    
     std::vector<size_t> idx(broadcast_dim.size(),0);
     // now we index over everything and do 2d matrix multiplication and save it on the result
     
     // since to get correct index the size of idx and size of padded strides dont match we can append 0,0 so it matches
     // which would give us the top left of whichever matrix we are looking for then we can use that as an offset to get to the correct 2d matrix m,n spot

     do{
    
     std::vector<size_t> full_idx= idx;
     full_idx.insert(full_idx.end(),{0,0});

     int base_offset_this= get_correct_index(full_idx,this_strides_padded);
     int base_offset_other= get_correct_index(full_idx,other_strides_padded);
     int base_offset_res= get_correct_index(full_idx,res_tensor->strides_);

        for (int i=0; i<m; i++){
            for (int j=0; j<n; j++){
                float dotProduct=0.0f;
                int row_strides_res=res_tensor->strides_[target_length-2];
                int col_strides_res=res_tensor->strides_[target_length-1];
                for (int kk=0; kk<p;kk++){
                   dotProduct+=this->data_[base_offset_this+i*this_strides_padded[target_length-2]+kk*this_strides_padded[target_length-1]]*other->data_[base_offset_other+kk*other_strides_padded[target_length-2]+j*other_strides_padded[target_length-1]];
                }
                res_tensor->data_[base_offset_res+i*row_strides_res+j*col_strides_res]=dotProduct;
            }
        }
     }while (increment_index(idx,broadcast_dim));

     // todo: connect backward node , squeeze dimensions if we extended it originally
     // note i need to handle erasing both 1d dimensions 
     // for squeezing dimensions 


    // now loop through each of the outermost dimensions
     // take every combination 
     if (this_1d && other_1d){
        res_shape.erase(res_shape.begin()+target_length-1);
        res_shape.erase(res_shape.begin()+target_length-2);
     }
     else if (this_1d){
        // that means that n has been expanded to (1,n)
        // this means the padded this has a target_length-2 position that i have to drop
        res_shape.erase(res_shape.begin()+target_length-2);
    
     }
     else if (other_1d){
        //that neans that n has been expanded to (n,1)
        // drop the last dimension and then recalculate strides for it?
        res_shape.erase(res_shape.begin()+target_length-1);
     }
     std::vector<size_t> final_strides=compute_strides_with_shape(res_shape);
     res_tensor->strides_=final_strides;
     res_tensor->shape_=res_shape;

    if (this->requires_grad_|| other->requires_grad_){
            auto node=std::make_shared<MatmulBackward>(this->shared_from_this(),other,res_tensor->grad_);
            res_tensor->grad_fn_=node;
            res_tensor->requires_grad_=true;
    }
     return res_tensor;
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


//implementaiton for a n-dimensional array in order to support transformer architecture

std::vector<size_t>Tensor::compute_strides(){

    std::vector<size_t> strides;
    size_t acc_stride=1;

    for (int i=this->shape_.size()-1;i>=0;i--){
        // stride indicates how much you need to jump for each dimension for the given shape, first stride is always one
        strides.insert(strides.begin(),acc_stride);
        acc_stride=this->shape_[i]*acc_stride;

    }
    return strides;
}




void Tensor::set_stride(){
    std::vector<size_t>strides=this->compute_strides();
    this->strides_=strides;
}


// flat index returns the index for the 1 dimensional vector for an n dimensional index
size_t Tensor::flat_index(std::vector<size_t>index) const{
    if (index.size()!= this->shape_.size()){
        throw std::invalid_argument("flat_index: index size does not match tensor rank");
    }
    // now each stride* each index gives you the correct flat index of the element you are looking for 
    size_t correct_index=0;
    for (int i=0; i<index.size();i++){
        correct_index+=this->strides_[i]*index[i];
    }
    return correct_index;
}


// for broadcasting n dimensional array (2,3,4)+(4) we should be able to broadcast
// it should pad its shape (1,1,4) and its strides we calculate as well

std::vector<size_t> Tensor::pad_shape(std::vector<size_t> shape, size_t target_length){

    std::vector<size_t> copy= shape;
    for (int i=target_length-shape.size()-1;i>=0;i--){
        copy.insert(copy.begin(),1);
    }
    return copy;
}

std::vector<size_t> Tensor::pad_strides(std::vector<size_t> shape, std::vector<size_t> strides, size_t target_length){
    std::vector<size_t> copy=strides;
    // pad dimensiosn that have been stretched as 0
    for (int i=target_length-shape.size()-1;i>=0;i--){
        copy.insert(copy.begin(),0);
    }
    // check original dimensions if any of them is 1 then set the stride to 0
    for (int j=0;j<=shape.size()-1;j++){
        int offset=copy.size()-shape.size();
        if (shape[j]==1){
            copy[j+offset]=0;
        }
    }
    return copy;

}




