#pragma once 
#include <memory>
#include <vector>
#include "nn/Linear.h"


class Tensor; 

class FeedForwardBlock{

    public:
    FeedForwardBlock(size_t d_model, size_t d_ff);
    std::vector<std::shared_ptr<Tensor>> parameters() ;
    std::shared_ptr<Tensor> forward( std::shared_ptr<Tensor>x);
    
    private:
    size_t d_model_;
    size_t d_ff_;
    Linear fc1_;
    Linear fc2_;
};