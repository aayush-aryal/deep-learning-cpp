#include "nn/TransformerBlock.h"
#include "Tensor.h"


TransformerBlock::TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff)
                    : ln1_(d_model),
                    mha_(d_model,num_heads),
                    ln2_(d_model),
                    ffn_(d_model, d_ff)
{}


std::vector<std::shared_ptr<Tensor>> TransformerBlock::parameters(){

    std::vector<std::shared_ptr<Tensor>>params;

    auto p1= ln1_.parameters();
    params.insert(params.end(), p1.begin(),p1.end());

    auto p2= mha_.parameters();

    params.insert(params.end(), p2.begin(), p2.end());

    auto p3= ln2_.parameters();

    params.insert(params.end(), p3.begin(), p3.end());

    auto p4= ffn_.parameters();

    params.insert(params.end(),p4.begin(),p4.end());

    return params;

}


std::shared_ptr<Tensor> TransformerBlock::forward(std::shared_ptr<Tensor> x){
    // normalize
    auto h= ln1_.forward(x);
    // multi head 
    auto a= mha_.forward(h, true);
    // residual connection from input to the attention
    auto x1= x->add(a);
    // normalize the residual connection?
    auto h2= ln2_.forward(x1);
    // feed forward network
    auto ff=ffn_.forward(h2);

    return x1->add(ff);
}