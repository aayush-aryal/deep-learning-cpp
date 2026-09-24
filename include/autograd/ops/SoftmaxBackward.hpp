#include <memory>
#include "autograd/BackwardNode.hpp"


class Tensor;

/**
 * @brief Backward operation for the softmax activation and cross-entropy loss.
 *
 * Stores the softmax probabilities and target values needed to compute
 * and propagate gradients during backpropagation.
 */
class SoftmaxBackward: public BackwardNode{
    public:
    /**
     * @brief Creates a backward node for a softmax cross-entropy operation.
     *
     * @param pA Input tensor to the softmax operation.
     * @param softmaxProb Computed softmax probabilities.
     * @param rG Gradient associated with the result.
     * @param target Target tensor containing the expected class values.
     */
    SoftmaxBackward(std::shared_ptr<const Tensor> pA, std::shared_ptr<std::vector<float>>softmaxProb, std::shared_ptr<std::vector<float>> rG, std::shared_ptr<Tensor>target){
        parentA_=pA;
        result_grad_=rG;
        softmax_prob_=softmaxProb;
        target_=target;
    }
    void apply(const std::vector<float>& incoming_grad) override;
    std::vector<std::shared_ptr<const Tensor>> get_parents() override;
    std::shared_ptr<std::vector<float>> get_gradient() override;



    private:
    std::shared_ptr<const Tensor> parentA_;
    std::shared_ptr<std::vector<float>> result_grad_;
    std::shared_ptr<std::vector<float>> softmax_prob_;
    std::shared_ptr<Tensor> target_;
};