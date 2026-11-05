#include "nn/FeedForwardBlock.h"
#include "Tensor.h"



// FeedForwardBlock::FeedForwardBlock(size_t d_model, size_t d_ff){
//     d_model_=d_model;
//     d_ff_=d_ff;

//     // now i need to initialize two Linear layers

//     auto l1= Linear(d_model_,d_ff_);
//     auto l2= Linear(d_ff_,d_model_);

//     fc1_= l1;
//     fc2_= l2;
// }


FeedForwardBlock::FeedForwardBlock(size_t d_model, size_t d_ff)
    : d_model_(d_model),
      d_ff_(d_ff),
      fc1_(d_model, d_ff),
      fc2_(d_ff, d_model)
{}


std::vector<std::shared_ptr<Tensor>> FeedForwardBlock::parameters(){
    std::vector<std::shared_ptr<Tensor>> params;

    auto p1= fc1_.parameters();
    params.insert(params.end(),p1.begin(),p1.end());

    auto p2=fc2_.parameters();
    params.insert(params.end(), p2.begin(), p2.end());

    return params;
}


std::shared_ptr<Tensor> FeedForwardBlock::forward(std::shared_ptr<Tensor>x){
    auto r1=  fc1_.forward(x)->relu();
    return fc2_.forward(r1);
    
}