#include <memory>
#include "Tensor.h"
#include "utility.h"

bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape);
size_t get_correct_index(std::vector<size_t> index, std::vector<size_t> strides);
std::vector<size_t> compute_strides_with_shape(std::vector<size_t> shape);

float accuracy(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target){
    size_t rows=input->get_shape()[0];
    size_t cols= input-> get_shape()[1];

    float total_correct=0;


    for (size_t r=0; r<rows;r++){
        size_t max_col=0;
        float max_prob=input->get(r,0);
        for (size_t c=0; c<cols; c++){
            // for each col find the index with the highest softmax probability 
            if (input->get(r,c)>max_prob){
                max_col=c;
                max_prob=input->get(r,c);
            }
        }
        size_t flat_index=r*cols+max_col;
        // after this iteration we know what the max index for the thing is now we check if the same index in target is 1, if so we increase "total_correct"
        if (target->get_data()[flat_index]==1.0f){
            total_correct+=1;
        }
    }

    return total_correct/rows;
}

// add a gradient checker here to prevent hand written derivations and un

// bool gradient_check(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-4, float tolerance=1e-3){
    
//     auto result= t1->add(t2);
//     result->backward();

//     // gradient check is basically how much does loss vary with tiny changes in input
//     std::vector<float>&t1_data= t1->get_data_ref();

//     float L=0.0f;


//     for (int i=0; i<result->get_data().size(); i++){
//         L+=result->get_data()[i];
//     }

//     // now check what happens to the loss if eps is added

//     for (int i=0; i<t1_data.size();i++){
//         t1_data[i]+=eps;
//     }

//     auto result_plus= t1->add(t2);
//     result_plus->backward();

//     float L_plus=0.0f;

//     for (int i=0; i<result_plus->get_data().size();i++){
//         L_plus+=result_plus->get_data()[i];
//     }

//     for (int i=0; i<t1_data.size();i++){
//         t1_data[i]-=2*eps;
//     }

//     auto result_minus= t1->add(t2);
//     result_minus->backward();

//     float L_minus=0.0f;

//     for (int i=0; i<result_plus->get_data().size();i++){
//         L_minus+=result_plus->get_data()[i];
//     }

//     float numerical_grad = (L_plus - L_minus) / (2*eps);

// }

// bool gradient_check(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps=1e-4, float tolerance=1e-3){
//     auto result= t1->add(t2);
//     result->backward();


//     bool all_true=true;
//     std::vector<float>& t1_data= t1->get_data_ref();
//     for (int i=0; i<t1_data.size();i++){
//         float L_plus=0.0f;
//         float L_minus=0.0f;
//         // for each index increase the value by eps 
//         t1_data[i]+=eps;
//         // after this calculate the new loss
//         auto res_plus= t1->add(t2);
//         // for each res value add to get a final loss
//         for (int j=0; j<res_plus->get_data().size();j++){
//             L_plus+=res_plus->get_data()[j];
//         }
//         // after this get L_minus as well
//         t1_data[i]-=2*eps;
//         auto res_minus=t1->add(t2);

//         for (int k=0; k<res_minus->get_data().size();k++){
//             L_minus+=res_minus->get_data()[k];
//         }
//         // reset original data
//         t1_data[i]+=eps;
//         float numerical_grad = (L_plus - L_minus) / (2*eps);
//         float diff= std::abs(numerical_grad-t1->get_grad()[i]);
//         all_true=all_true&& (diff<=tolerance);
//     }
//     return all_true;

// }

bool check_one_tensor(std::shared_ptr<Tensor> target, std::function<std::shared_ptr<Tensor>()> forward_fn, float eps, float tolerance) {
    auto result = forward_fn();
    result->backward();

    bool all_true = true;
    std::vector<float>& data = target->get_data_ref();
    for (int i = 0; i < data.size(); i++) {
        float orig = data[i];
        data[i] = orig + eps;
        auto res_plus = forward_fn();
        float L_plus = 0.0f;
        for (float v : res_plus->get_data()) L_plus += v;

        data[i] = orig - eps;
        auto res_minus = forward_fn();
        float L_minus = 0.0f;
        for (float v : res_minus->get_data()) L_minus += v;

        data[i] = orig;
        float numerical_grad = (L_plus - L_minus) / (2*eps);
        float diff = std::abs(numerical_grad - target->get_grad()[i]);
        float relative_diff= diff/std::abs(target->get_grad()[i]+eps);
        all_true = all_true && (relative_diff <= tolerance);
    }
    return all_true;
}

bool gradient_check_add(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps, float tolerance){
    bool a_ok = check_one_tensor(t1, [&](){ return t1->add(t2); }, eps, tolerance);
    t1->zero_grad();
    t2->zero_grad();
    bool b_ok = check_one_tensor(t2, [&](){ return t1->add(t2); }, eps, tolerance);
    return a_ok && b_ok;
}


bool gradient_check_matmul(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps, float tolerance){
    bool a_ok = check_one_tensor(t1, [&](){ return t1->matmul(t2); }, eps, tolerance);
    t1->zero_grad();
    t2->zero_grad();
    bool b_ok = check_one_tensor(t2, [&](){ return t1->matmul(t2); }, eps, tolerance);
    return a_ok && b_ok;
}


bool gradient_check_unary(std::shared_ptr<Tensor> t,
                          std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)> op,
                          float eps, float tolerance){
    return check_one_tensor(t, [&](){ return op(t); }, eps, tolerance);
}


bool gradient_check_subtract(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps, float tolerance){
    bool a_ok = check_one_tensor(t1, [&](){ return t1->subtract(t2); }, eps, tolerance);
    t1->zero_grad();
    t2->zero_grad();
    bool b_ok = check_one_tensor(t2, [&](){ return t1->subtract(t2); }, eps, tolerance);
    return a_ok && b_ok;
}

bool gradient_check_divide(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps, float tolerance){
    bool a_ok = check_one_tensor(t1, [&](){ return t1->divide(t2); }, eps, tolerance);
    t1->zero_grad();
    t2->zero_grad();
    bool b_ok = check_one_tensor(t2, [&](){ return t1->divide(t2); }, eps, tolerance);
    return a_ok && b_ok;
}

bool gradient_check_multiply(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2, float eps, float tolerance){
    bool a_ok = check_one_tensor(t1, [&](){ return t1->multiply(t2); }, eps, tolerance);
    t1->zero_grad();
    t2->zero_grad();
    bool b_ok = check_one_tensor(t2, [&](){ return t1->multiply(t2); }, eps, tolerance);
    return a_ok && b_ok;
}