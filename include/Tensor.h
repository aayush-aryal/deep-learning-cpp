#ifndef TENSOR_H
#define TENSOR_H 
#include <vector>

class Tensor{
    public:
    // constructor 
    Tensor(std::vector<int>shape);
    void printShape();

    private:
    std::vector<float> data_;
    std::vector<int> shape_;
};

#endif