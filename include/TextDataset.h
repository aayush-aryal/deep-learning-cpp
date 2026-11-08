#pragma once 
#include "Tokenizer.h"
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <utility>
#include <cstddef>

#include "Tensor.h"



class TextDataset {

    public:
    TextDataset(const std::string& input, float train_size=0.9);
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> get_batch(size_t B, size_t T, bool train);
    const Tokenizer& tokenizer() const;


    private:
    Tokenizer tokenizer_;
    std::vector<int> train_ids_;
    std::vector<int> val_ids_;
    std::mt19937 rng_;
};