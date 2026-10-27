#include "nn/Embedding.h"
#include "Tensor.h"
#include <vector>
#include <memory>



Embedding::Embedding(size_t vocab_size,size_t d_model){
    vocab_size_=vocab_size;
    d_model_=d_model;

    // build a embedding tensor of this size (V,D)
    // this needs to be initialized with a specific type of numbers as well

    auto embedding_tensor= std::make_shared<Tensor>(std::vector<size_t>{vocab_size_,d_model_});
    embedding_tensor->randomize(d_model_);
    embedding_tensor->set_requires_grad(true);

    table_=embedding_tensor;
}


std::shared_ptr<Tensor> Embedding::forward(std::shared_ptr<Tensor>ids){
    return this->table_->gather(ids);
}

std::vector<std::shared_ptr<Tensor>> Embedding::parameters(){
    return {table_};
}
