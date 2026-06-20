#include "Dataloader.h"
#include "Tensor.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>



void DataLoader::load_from_csv(){
    // so what needs to happen here?
    // open the file path first
    std::ifstream fin;
    fin.open(this->file_path_);   
    if (!fin){
        std::cerr<<"Could not read the file path!"<< std::endl;
        return;
    }
    // this line string is overwritten in each new line
    std::string line;
    // while there is line get line and write it in line
    // skips first line can make this a bool optional thing later
    std::getline(fin,line);
    while (std::getline(fin,line)){
        // parse line as a stream of tokens, write each token into "token"
        std::stringstream ss(line);
        std::string token;
        int count=0;
        // while there is a token
        while (std::getline(ss,token,',')){
            if (count==0){
                this->targets_.push_back(std::stoi(token));
            }else{
                this->inputs_.push_back(std::stof(token)/255.0f);
            }
            count++;
        }
        this->index_order_.push_back(this->num_samples);
        this->num_samples++;
    }
}

bool DataLoader::has_next_batch(){
    return this->batch_size_<= (this->num_samples-this->curr_index_);
}

void DataLoader::reset_and_shuffle(){
    this->curr_index_=0;
    // shuffle index_order_
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(this->index_order_.begin(), this->index_order_.end(), generator);
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::get_next_batch(){
    // to keep in mind 784 is hardcoded to match total pixels for one image in mnist dataset
    auto input_tensor= std::make_shared<Tensor>(std::vector<size_t>{this->batch_size_,784});
    // also output tensor is hardcoded as well lol i will have to modify this if i feel like it
    auto output_tensor= std::make_shared<Tensor>(std::vector<size_t>{this->batch_size_,10});
    


    // to do --> loop from current_index to (curr_index+batch_size)*784
    // each row will be pixel values for an image
    // for output tensor just loop from curr_index to curr_index+batch_size
    // push into output tensor
    // after you correctly finish this write your training loop, create a utility function folder where you can track accuracy --> get it done--> update resume , add readme

    for (int b=0; b<batch_size_;b++){
        // for each batch
        // start at curr_index+batch_size
        size_t sample_index=this->index_order_[curr_index_+b];
        // next 784 values are the pixel values we need
        for (int i=0;i<784;i++){
            input_tensor->set(b,i,this->inputs_[sample_index*784+i]);
        }
        // update output tensor, one hot encode the target
        float target= this->targets_[sample_index];
        output_tensor->set(b,target,1.0f);
    }
    this->curr_index_+=this->batch_size_;
    return {input_tensor,output_tensor};
}