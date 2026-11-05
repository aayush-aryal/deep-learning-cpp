#include "nn/TransformerLM.h"
#include "Tensor.h"



TransformerLM:: TransformerLM(size_t vocab_size, size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers, size_t max_T)
    :em_(vocab_size, d_model),
    pe_(max_T,d_model),
    ln_(d_model),
    out_(d_model,vocab_size)
{
    transformer_blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        transformer_blocks_.emplace_back(d_model, num_heads, d_ff);
    }
}



std::shared_ptr<Tensor> TransformerLM::forward(std::shared_ptr<Tensor> ids){
    auto  x= em_.forward(ids);
    // now that we have a representation of the different ids into a vector each with d_model features
    x= pe_.forward(x);

    // now that we have the positional information pass it to the transformer blocks 
    for (int i=0; i<transformer_blocks_.size(); i++){
        // does each block work off the others output 
        x= transformer_blocks_[i].forward(x);
    }

    // after that pass it to a layer norm 
    x= ln_.forward(x);

    // after that pass it thorugh linear layer 
    x=  out_.forward(x);

    return x;

}


std::vector<std::shared_ptr<Tensor>> TransformerLM::parameters(){
    std::vector<std::shared_ptr<Tensor>> params;
    auto p1= em_.parameters();

    params.insert(params.end(), p1.begin(), p1.end());




    for (size_t i=0; i<transformer_blocks_.size();i++){
        auto pb = transformer_blocks_[i].parameters();
        params.insert(params.end(),pb.begin(), pb.end());
    }
    
    auto p2= ln_.parameters();
    params.insert(params.end(), p2.begin(),p2.end());

    auto p3= out_.parameters();

    params.insert(params.end(), p3.begin(),p3.end());

    return params;

}

