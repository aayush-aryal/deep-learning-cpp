#pragma once 
#include <memory>
#include <vector>


class Tensor; 

class PositionalEncoding{

    public:
    PositionalEncoding(size_t max_T,size_t d_model);
    std::vector<std::shared_ptr<Tensor>> parameters() ;
    std::shared_ptr<Tensor> forward( std::shared_ptr<Tensor>x);
    
    private:
    size_t max_T_;
    size_t d_model_;
    std::shared_ptr<Tensor> pe_;
};