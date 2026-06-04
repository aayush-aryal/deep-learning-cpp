#ifndef TENSOR_H
#define TENSOR_H 
#include <vector>
#include <iostream>

class Tensor{
    public:
    // constructor 
    Tensor(std::vector<int>shape);
    void printShape();
    void set( int r, int c, float value);
    float get(int r, int c) const;
    float operator()(int r, int c) const;
    Tensor add(const Tensor& other)const;
    friend std::ostream& operator<< (std::ostream& os,const Tensor& t);

    private:
    std::vector<float> data_;
    std::vector<int> shape_;
    int rows_;
    int cols_;
};

#endif