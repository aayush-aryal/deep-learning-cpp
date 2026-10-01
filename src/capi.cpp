#include "capi.h"
#include "Tensor.h"
#include <vector>
#include <memory>


extern "C"{
    TensorHandle tensor_create(const size_t* shape, size_t shape_len){
        // create vector of appropriate size
        std::vector<size_t> shape_vec(shape, shape+shape_len);
        // create a shared tensor object and store it as t
        auto t= std::make_shared<Tensor>(shape_vec);
        // we need a pointer to this obj that is last beyonf this func so 
        auto* handle= new std::shared_ptr<Tensor>(t);
        // static cast and return the tensor since c does not understand other types
        return static_cast<void*>(handle);
    }

    void tensor_set_data(TensorHandle t, const float* data, size_t len){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        auto& data_ref= (*tensor_ptr)->get_data_ref();
        for (size_t i = 0; i < len; i++) data_ref[i] = data[i];
    }

    void tensor_set_elem(TensorHandle t, size_t* idx, size_t idx_len, float value){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        std::vector<size_t>idx_vec(idx, idx+idx_len);
        (*tensor_ptr)->set(idx_vec,value);
    }

    void tensor_get_data(TensorHandle t, float* out_buffer, size_t len){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        const auto& data= (*tensor_ptr)->get_data();
        for(size_t i=0; i<len; i++) out_buffer[i]= data[i];
    }

    void tensor_get_shape(TensorHandle t, size_t* out_shape, size_t out_len){
        auto* tensor_ptr=static_cast<std::shared_ptr<Tensor>*>(t);
        auto shape= (*tensor_ptr)->get_shape();
        for (int i=0; i<out_len;i++)out_shape[i]=shape[i];
    }

    size_t tensor_ndim(TensorHandle t){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        return (*tensor_ptr)->get_shape().size();
    }

    size_t tensor_size(TensorHandle t){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        return (*tensor_ptr)->get_data().size();  
    }

    TensorHandle tensor_relu(TensorHandle t){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        auto result= (*tensor_ptr)->relu();
        auto* handle= new std::shared_ptr<Tensor>(result);
        return static_cast<void*>(handle);
    }

    void tensor_free(TensorHandle t){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        delete tensor_ptr;
    }

    TensorHandle tensor_matmul(TensorHandle a, TensorHandle b){
        auto* a_ptr= static_cast<std::shared_ptr<Tensor>*>(a);
        auto* b_ptr= static_cast<std::shared_ptr<Tensor>*>(b);
        auto result =(*a_ptr)->matmul(*b_ptr);
        auto* handle= new std::shared_ptr<Tensor>(result);
        return static_cast<void*>(handle);
    }

    TensorHandle tensor_addition(TensorHandle a, TensorHandle b){
        auto* a_ptr= static_cast<std::shared_ptr<Tensor>*>(a);
        auto* b_ptr= static_cast<std::shared_ptr<Tensor>*>(b);
        auto result=(*a_ptr)->add(*b_ptr);
        auto* handle= new std::shared_ptr<Tensor>(result);
        return static_cast<void*>(handle);
    }

    void tensor_backward(TensorHandle t){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        (*tensor_ptr)->backward();
    }

    void tensor_set_requires_grad(TensorHandle t, bool requires){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        (*tensor_ptr)->set_requires_grad(requires);
    }

    void tensor_get_grad(TensorHandle t, float* out_buffer, size_t out_len){
        auto* tensor_ptr= static_cast<std::shared_ptr<Tensor>*>(t);
        std::vector<float>& grad= (*tensor_ptr)->get_grad();
        for (int i=0; i<out_len;i++){out_buffer[i]=grad[i];}
    }

}