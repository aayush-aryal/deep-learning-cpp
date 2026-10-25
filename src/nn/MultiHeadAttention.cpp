#include "nn/MultiHeadAttention.h"
#include "Tensor.h"
#include <vector>
#include <memory>


MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    :d_model_(d_model),
    num_heads_(num_heads),
    wq_(d_model,d_model),
    wk_(d_model,d_model),
    wv_(d_model,d_model),
    wo_(d_model,d_model)
    {
        if (d_model% num_heads!=0){
            throw std::invalid_argument("d_model must be divisible by num_heads");
        }
    }

std::vector<std::shared_ptr<Tensor>>MultiHeadAttention::parameters(){
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto* layer: {&wq_, &wk_, &wv_, &wo_}){
        auto p= layer->parameters();
        params.insert(params.end(), p.begin(),p.end());
    }
    return params;
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(std::shared_ptr<Tensor> x, bool causal){
    // input must have shape (Batch Size ,T (Sequence Length ),D (Number of features))
    std::vector<size_t> x_shape= x->get_shape();
    if (x_shape.size()!=3){
        throw std::invalid_argument("input shape must be size of 3");
    }

    if (x_shape[2]!=d_model_){
        throw std::invalid_argument("number of features (d_model) do not match");
    }
    size_t B= x_shape[0];
    size_t T= x_shape[1];


    int Bi= static_cast<int>(B);
    int Ti= static_cast<int>(T);


    size_t H= num_heads_;
    int Hi= static_cast<int>(H);

    auto Q= wq_.forward(x);
    auto K= wk_.forward(x);
    auto V= wv_.forward(x);

    // multiheaded attention splits attention calculation so each head can attend to d/h features over all t tokens ?
    // basically allows multiple single headed attention to occur quicker by batching attention mechanism 
    // batching attention is just reshaping and making sure the features and token length are the ones that do attention from input that contins many features

    auto q_reshaped=Q->reshape(std::vector<int>{Bi,Ti,Hi,-1});
    auto k_reshaped=K->reshape(std::vector<int>{Bi,Ti,Hi,-1});
    auto v_reshaped=V->reshape(std::vector<int>{Bi,Ti,Hi,-1});

    // now we permute so the actual attention ahppens between T and D/H 

    std::vector<int> permutation= {0,2,1,3};

    auto q_permuted= q_reshaped->permute(permutation);
    auto k_permuted= k_reshaped->permute(permutation);
    auto v_permuted= v_reshaped->permute(permutation);

    auto attended= attention(q_permuted,k_permuted,v_permuted,causal);

    // after this permute the shape back to its original

    auto attended_permuted= attended->permute(permutation);

    auto attended_reshaped= attended_permuted->reshape({Bi,Ti,-1});

    // this goes through the output so the attention heads can influence each other

    return wo_.forward(attended_reshaped);

}