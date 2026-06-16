#include <memory>
#include "autograd/BackwardNode.hpp"


class Tensor;

class SoftmaxBackward: public BackwardNode{
    public:
    SoftmaxBackward(std::shared_ptr<Tensor> pA, std::shared_ptr<std::vector<float>>softmaxProb, std::shared_ptr<std::vector<float>> rG, std::shared_ptr<Tensor>target){
        parentA_=pA;
        result_grad_=rG;
        softmax_prob_=softmaxProb;
        target_=target;
    }
    void apply(const std::vector<float>& incoming_grad) override;
    std::vector<std::shared_ptr<const Tensor>> get_parents() override;
    std::shared_ptr<std::vector<float>> get_gradient() override;



    private:
    std::shared_ptr<Tensor> parentA_;
    std::shared_ptr<std::vector<float>> result_grad_;
    std::shared_ptr<std::vector<float>> softmax_prob_;
    std::shared_ptr<Tensor> target_;
};