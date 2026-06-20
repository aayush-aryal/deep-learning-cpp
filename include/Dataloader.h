#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <utility>
// stores entire DataSet, keeps track of batches 

class Tensor;

class DataLoader{
    public:
        DataLoader(const std::string& filePath,size_t batchSize){
            batch_size_=batchSize;
            file_path_=filePath;
            load_from_csv();
        };

        void reset_and_shuffle();
        bool has_next_batch();
        std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>get_next_batch();

    
    private:
        size_t batch_size_;
        size_t curr_index_=0;
        size_t num_samples=0;
        std::string file_path_;
        std::vector<std::size_t> index_order_;
        std::vector<float> targets_;
        std::vector<float> inputs_;
        void load_from_csv();
};


