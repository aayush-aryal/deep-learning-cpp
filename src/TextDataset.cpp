#include "TextDataset.h"
#include "Tensor.h"
#include <stdexcept>

TextDataset::TextDataset(const std::string& input, float train_size)
    :tokenizer_(input),
    rng_(1234)
{
    if (train_size<=0 || train_size>=1){
        throw std::invalid_argument("train size must be between 0 and 1");
    }

    auto ids= tokenizer_.encode(input);
    size_t n_train= static_cast<size_t>(train_size*ids.size());
    // assign 90% of text to train and 10% to val and sample chunks from them
    train_ids_.assign(ids.begin(), ids.begin()+n_train);
    val_ids_.assign(ids.begin()+n_train,ids.end());
}


std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> TextDataset::get_batch(size_t B, size_t T, bool train){
    const auto& ids= train?train_ids_: val_ids_;
    if (ids.size()<=T+1){
        throw std::invalid_argument("not enough text for this T");
    }
    std::shared_ptr<Tensor> x=std::make_shared<Tensor>(std::vector<size_t>{B,T});
    std::shared_ptr<Tensor> y=std::make_shared<Tensor>(std::vector<size_t>{B,T});


    // a window needs T characters for x, plus one more for the final target in y. So it uses T + 1 characters in total, and the last valid start is the one where that window just fits:
    std::uniform_int_distribution<size_t> dist(0,ids.size()-T-1);
    for (size_t i=0; i<B; i++){
        auto s= dist(rng_);
        auto& xd=x->get_data_ref();
        auto& yd= y->get_data_ref();
        for (size_t j=0 ; j<T; j++){
            xd[i*T+j]=ids[s+j];
            yd[i*T+j]=ids[s+j+1];
        }
    }
    return {x,y};
}

const Tokenizer& TextDataset::tokenizer() const { return tokenizer_; }