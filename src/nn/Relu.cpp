#include <memory>
#include "Tensor.h"
#include "nn/Relu.h"

std::vector<std::shared_ptr<Tensor>> Relu::parameters(){
    return {};
}

std::shared_ptr<Tensor>Relu::forward(std::shared_ptr<Tensor> input_tensor){
    return input_tensor->relu();
}