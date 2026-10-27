#include "nn/LayerNorm.h"
#include "Tensor.h"
#include <vector>
#include <memory>


LayerNorm::LayerNorm(size_t d_model, float eps){
    d_model_=d_model;
    eps_=eps;
    gamma_= std::make_shared<Tensor>(std::vector<size_t>{1,d_model_});
    beta_= std::make_shared<Tensor>(std::vector<size_t>{1,d_model_});

    for (int i=0; i<gamma_->get_data().size();i++){
        gamma_->get_data_ref()[i]=1.0f;
    }

    
    gamma_->set_requires_grad(true);
    beta_->set_requires_grad(true);
}


std::vector<std::shared_ptr<Tensor>> LayerNorm::parameters(){
    return {gamma_,beta_};
}


std::shared_ptr<Tensor> LayerNorm::forward(std::shared_ptr<Tensor> x){
    size_t r= x->get_shape().size();

    auto eps_ten= std::make_shared<Tensor>(std::vector<size_t>(r,1));
    eps_ten->set(std::vector<size_t>(r,0),eps_);

    auto inv_d= std::make_shared<Tensor>(std::vector<size_t>(r,1));
    inv_d->set(std::vector<size_t>(r,0),1.0f/static_cast<float>(d_model_));
    
    auto mean= x->axiswise_sum(-1)->multiply(inv_d);
    auto centered= x->subtract(mean);
    auto var= centered->multiply(centered)->axiswise_sum(-1)->multiply(inv_d);
    auto inv_std= var->add(eps_ten)->sqrt()->reciprocal();
    auto normalized= centered->multiply(inv_std);
    return normalized->multiply(gamma_)->add(beta_);
}

