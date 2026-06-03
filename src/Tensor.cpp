#include "Tensor.h"
#include <iostream>

Tensor::Tensor(std::vector<int> shape){
    shape_=shape;
}

void::Tensor::printShape(){
    std::cout<< "Shape dimensions: " << shape_.size() << std::endl;

}