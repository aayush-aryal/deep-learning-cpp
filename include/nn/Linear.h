#pragma once
#include "Module.h"
#include <memory>


class Tensor;

class Linear: public Module{
    public:
    Linear(size_t in_features, size_t out_features);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    std::shared_ptr<Tensor> forward( std::shared_ptr<Tensor>)override;


    private:
    size_t input_features_;
    size_t output_features_;

    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;


};