#pragma once 
#include <memory>
#include <vector>
#include "nn/TransformerBlock.h"
#include "nn/LayerNorm.h"
#include "nn/Linear.h"
#include "nn/Embedding.h"
#include "nn/PositionalEncoding.h"


class Tensor; 

class TransformerLM{
    public:
    TransformerLM(size_t vocab_size, size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers, size_t max_T);
    std::vector<std::shared_ptr<Tensor>> parameters() ;
    std::shared_ptr<Tensor> forward( std::shared_ptr<Tensor>ids);
    
    private:
    Embedding em_;
    PositionalEncoding pe_;
    std::vector<TransformerBlock> transformer_blocks_;
    LayerNorm ln_;
    Linear out_;
    
};