#include <memory>
#include "Tensor.h"
#include "utility.h"

float accuracy(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target){
    size_t rows=input->get_shape()[0];
    size_t cols= input-> get_shape()[1];

    float total_correct=0;


    for (size_t r=0; r<rows;r++){
        size_t max_index=0;
        float max_prob=input->get(r,0);
        for (size_t c=0; c<cols; c++){
            // for each col find the index with the highest softmax probability 
            if (input->get(r,c)>max_prob){
                max_index=(r*cols)+c;
                max_prob=input->get(r,c);
            }
        }
        // after this iteration we know what the max index for the thing is now we check if the same index in target is 1, if so we increase "total_correct"
        if (target->get_data()[max_index]==1.0f){
            total_correct+=1;
        }
    }

    return total_correct/rows;
}