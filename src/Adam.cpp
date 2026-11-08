#include "Adam.h"
#include "Tensor.h"
#include <utility>
#include <algorithm>



Adam::Adam(std::vector<std::shared_ptr<Tensor>> tensors, float lr, float beta1, float beta2, float eps):
        tensors_(std::move(tensors)),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        t_(0)
{
    for (auto& t: tensors_){
        size_t size= t->get_data().size();
        m_.emplace_back(size, 0.0f);
        v_.emplace_back(size, 0.0f);
    }

}


void Adam::step(){
    // adam uses two optimization techniques 
    // momentum and rmsprop

    // momentum: accelerates gradient descent by using a moving average of past gradients
    // this reduces oscillations and speeds up convergence
    // rule is "wt+1= wt- alpha*mt"
    // updated recursively as 
    //  mt= b1*m(t-1)+(1-b1)* gradient of loss function with respect to the weights at time t


    // rmsprop (root mean square propagation)
    // is adaptive learning rate opt method that uses exponentially weighted moving average of squared gradient
    // prevents lr from dropping too quickly
    //    //  vt= b1*v(t-1)+(1-b1)* (gradient of loss function with respect to the weights at time t)^2
    // vt is the exponentially weighted average of squared gradients

    t_++;
    float bc1_= 1.0f- std::pow(beta1_, static_cast<float>(t_));
    float bc2_= 1.0f- std::pow(beta2_,static_cast<float>(t_));

    for (size_t i=0; i<tensors_.size();i++){
        auto& data= tensors_[i]->get_data_ref();
        const auto& grad= tensors_[i]->get_grad();
        for (size_t j=0; j<data.size(); j++){
            float g= grad[j];
            m_[i][j]=beta1_*m_[i][j]+(1-beta1_)*g;
            v_[i][j]= beta2_*v_[i][j]+(1-beta2_)*g*g;
            float m_hat= m_[i][j]/bc1_;
            float v_hat= v_[i][j]/bc2_;
            data[j]-=lr_*m_hat/(std::sqrt(v_hat)+eps_);
        }

    }
}

void Adam::zero_grad(){
    for (auto& t : tensors_){
        auto& g = t->get_grad();
        std::fill(g.begin(), g.end(), 0.0f);
    }
}