#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <memory>
#include "autograd/BackwardNode.hpp"

class Tensor{
    public:
    // constructor 
    Tensor(std::vector<size_t>shape);
    void printShape();
    void set( int r, int c, float value);
    float get(int r, int c) const;
    float operator()(int r, int c) const;
    Tensor add(Tensor& other);
    friend std::ostream& operator<< (std::ostream& os,const Tensor& t);
    void randomize();
    Tensor matmul(const Tensor& a) const;
    void relu();
    void zero_grad();

    std::vector<float>&get_grad(){
        return *grad_;
    };
    std::shared_ptr<BackwardNode> grad_fn_;
    size_t size() const{
        return data_.size();
    };

    std::vector<size_t> get_shape(){
        return shape_;
    }

    const std::vector<float>& get_data()const{
        return data_;
    }
    
    void backward();


    private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    // for autograd
    std::shared_ptr<std::vector<float>> grad_;
    bool requires_grad_;
};