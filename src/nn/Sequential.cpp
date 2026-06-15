#include <memory>
#include "Tensor.h"
#include "nn/Sequential.h"
#include "nn/Module.h"




std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input){
    // loop through all your internal modules 
    // curr tensor = input then gets modified after each module call

    std::shared_ptr<Tensor>& curr_tensor=input;

    for (std::unique_ptr<Module>& m: this->modules_){
        curr_tensor=m->forward(curr_tensor);

    }
    return curr_tensor;
    // return the tensor
}


std::vector<std::shared_ptr<Tensor>> Sequential::parameters(){
    std::vector<std::shared_ptr<Tensor>> tensors;
        for (std::unique_ptr<Module>& m: this->modules_){
            std::vector<std::shared_ptr<Tensor>> curr_tensors= m->parameters();
            tensors.insert(tensors.begin(),curr_tensors.begin(),curr_tensors.end());
    };

    return tensors;
}

