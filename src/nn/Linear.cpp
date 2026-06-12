#include "nn/Linear.h"
#include "Tensor.h"
#include <memory>

Linear::Linear(size_t in_features, size_t out_features){

    // since in a neural network, every single input node must be connected to evry single output node
    // initialize tensor of size input * output

    input_features_=in_features;
    output_features_=out_features;

    weight_= std::make_shared<Tensor>(std::vector<size_t>{input_features_,output_features_});
    bias_= std::make_shared<Tensor>(std::vector<size_t>{1,out_features});

    weight_->randomize();
    bias_->randomize();

    weight_->set_requires_grad(true);
    bias_->set_requires_grad(true);

}

std::vector<std::shared_ptr<Tensor>> Linear::parameters(){
    return {weight_,bias_};
}


std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input){
    return input->matmul(weight_)->add(bias_);
}
