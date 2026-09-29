#pragma once 
#include <cstddef>

extern "C"{
    typedef void* TensorHandle;

    TensorHandle tensor_create(const size_t* shape, size_t shape_len);
    void tensor_set_data(TensorHandle t, const float* data, size_t len);
    void tensor_get_data(TensorHandle t, float* out_buffer, size_t len);
    size_t tensor_size(TensorHandle t);

    // tensor operations
    TensorHandle tensor_relu(TensorHandle t);
    TensorHandle tensor_matmul(TensorHandle a, TensorHandle b);
    TensorHandle tensor_addition(TensorHandle a, TensorHandle b);
    void tensor_free(TensorHandle t);
}