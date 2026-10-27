#pragma once 
#include <memory>
#include <vector>


class Tensor; 

class Embedding{

    public:
    Embedding(size_t vocab_size,size_t d_model);
    std::vector<std::shared_ptr<Tensor>> parameters() ;
    std::shared_ptr<Tensor> forward( std::shared_ptr<Tensor>ids);
    
    private:
    size_t vocab_size_;
    size_t d_model_;
    std::shared_ptr<Tensor> table_;
};