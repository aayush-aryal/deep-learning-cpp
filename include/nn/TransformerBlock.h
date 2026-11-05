#pragma once 
#include <memory>
#include <vector>
#include "nn/LayerNorm.h"
#include "nn/MultiHeadAttention.h"
#include "nn/FeedForwardBlock.h"







class Tensor; 

class TransformerBlock{

    public:
    TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff);
    std::vector<std::shared_ptr<Tensor>> parameters() ;
    std::shared_ptr<Tensor> forward( std::shared_ptr<Tensor>x);
    
    private:
    LayerNorm ln1_;
    MultiHeadAttention mha_;
    LayerNorm ln2_;
    FeedForwardBlock ffn_;
};