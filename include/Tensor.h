#pragma once
#include <vector>
#include <iostream>
#include <functional>
#include <memory>
#include "autograd/BackwardNode.hpp"


/**
 * @brief Represents a tensor containing multidimensional numerical data.
 *
 * The Tensor class provides basic operations for manipulating
 * multidimensional data used throughout the neural network library.
 */
class Tensor:public std::enable_shared_from_this<Tensor>{
    public:


    /** 
     * @brief Creates a tensor with the specified shape.
     * 
     * @param shape Dimensions of the tensor
    */
    Tensor(std::vector<size_t>shape);

    /**
     * @brief Prints the shape of the tensor
     */
    void printShape();
    void set( int r, int c, float value);

    /**
     * @brief Sets the value at the specified tensor dimensions.
     *
     * @param dim The indices specifying the position in the tensor.
     * @param value The value to assign at that position.
     */
    void set(std::vector<size_t> dim, float value);

    // im going to overload get to include a n-dimensional parameter 
    float get(int r, int c) const;
    /**
     * @brief Get teh value at a specified tensor index
     * 
     * @param index The indices specifying the position in the tensor.
     */
    float get(std::vector<size_t> index) const;

    /**
     * @brief Returns the strides of the tensor.
     *
     * @return A vector containing the stride for each dimension of the tensor.
     */
    std::vector<size_t> get_strides()const;

    float operator()(int r, int c) const;

    void set_stride();
    /**
     * @brief Adds two tensors together and returns the result tensor.
     * 
     * @return A tensor containing the result of the addition between the two tensors.
     */
    std::shared_ptr<Tensor> add(std::shared_ptr<Tensor>other);
    friend std::ostream& operator<< (std::ostream& os,const Tensor& t);

    /**
     * @brief Randomizes the initial values of a tensor to random values following a normal distribution.
     */
    void randomize(size_t input=1);

    /**
     * @brief Performs matrix multiplication with another tensor.
     *
     * Supports matrix multiplication on tensors with arbitrary numbers of
     * dimensions, following standard broadcasting rules for the leading
     * dimensions.
     *
     * @param a The tensor to multiply with. The inner dimensions must match,
     * and corresponding broadcast dimensions must either be equal or one
     * of them must be 1.
     * @return A shared pointer to the resulting tensor.
    */
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor>a);

    /**
     * @brief Performs Relu activation function for all of the tensor values for this tensor.
     */
    std::shared_ptr<Tensor> relu();

    /**
     * @brief Computes the softmax cross-entropy loss.
     *
     * @param target Target tensor containing the expected class labels.
     * @return Computed loss.
    */
    std::shared_ptr<Tensor> softmax_crossentropy(std::shared_ptr<Tensor> target);

    std::shared_ptr<std::vector<float>> softmax(std::shared_ptr<Tensor> target);
    /**
     * @brief Zeros the gradient vector for the given tensor.
     */
    void zero_grad();

    std::vector<float>&get_grad()const{
        return *grad_;
    };
    std::shared_ptr<BackwardNode> grad_fn_;
    size_t size() const{
        return data_.size();
    };

    std::vector<size_t> get_shape()const{
        return shape_;
    }

    const std::vector<float>& get_data()const{
        return data_;
    }

    std::vector<float>& get_data_ref(){
        return data_;
    }

    /**
     * @brief Returns a 1x1 tensor that stores the mean squared error between the target and the given tensor.
     * 
     * @param target Target tensor containing the actual values
     */
    std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor>target);


    /**
     * @brief Enables or disables gradient tracking for this tensor.
     *
     * When enabled, operations involving this tensor can be tracked for
     * backpropagation.
     *
     * @param val Whether gradient tracking should be enabled.
    */
    void set_requires_grad(bool val){
        requires_grad_=val;
    }

    bool get_requires_grad()const{
        return requires_grad_;
    }
    
    /**
     * @brief Performs backpropagation starting from the current tensor all the way to all connected tensors
     */
    void backward();

    /**
     * Computes the strides for the N-dimensional tensor using its own shape
     */
    std::vector<size_t> compute_strides();


    /**
     * @brief Returns the index for 1-dimensional vector corresponsing to the N-dimensional tensor index.
     * @param index The N-dimensional tensor index
     */
    size_t flat_index(std::vector<size_t> index) const;


    /**
     * @brief Pads the shape of the given tensor to the target length and returns the new shape
     * @param shape The original shape of the tensor.
     * @param target_length The target length of the new shape tensor.
     * @return The new padded shape
     */
    std::vector<size_t> pad_shape(std::vector<size_t> shape, size_t target_length)const;


    /**
     * @brief Pads the strides of a given tensor to the target size
     * @param strides The original strides of the tensor
     * @param shape The shape of the original tensor
     * @param target_length The length of the new padded strides.
     * @return Vector storing the new padded strides.
     */
    std::vector<size_t> pad_strides(std::vector<size_t> strides, std::vector<size_t>shape, size_t target_length)const;

    private:
    /// @brief  stores the actual tensor data
    std::vector<float> data_;
    /// @brief stores the shape of the tensor
    std::vector<size_t> shape_;

    // for autograd

    /// @brief stores the pointer that points towards a vector of float that stroes the gradient for the tensor if requires_grad_ is set to true
    std::shared_ptr<std::vector<float>> grad_;
    /// @brief stores the boolean value if gradient is enabled or not
    bool requires_grad_;
    /// @brief stores teh strides for the current shape of the tensor
    std::vector<size_t> strides_;
};