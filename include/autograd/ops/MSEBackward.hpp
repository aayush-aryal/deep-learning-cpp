
#include "autograd/BackwardNode.hpp"
#include <memory>

/**
 * @brief Backward operation for mean squared error loss.
 *
 * Propagates the gradient of the MSE loss to the prediction tensor
 * during backpropagation.
 */
class MSEBackward: public BackwardNode{
    public:
        // mse needs to know its parent value and predicted value

        /**
         * @brief Creates a backward node for a mean squared error operation.
         *
         * @param target Target tensor containing the expected values.
         * @param pA Prediction tensor used to compute the loss.
         * @param rG Gradient associated with the loss result.
        */
        MSEBackward(std::shared_ptr<const Tensor> target, std::shared_ptr< const Tensor> pA , std::shared_ptr<std::vector<float>> rG){
            target_=target;
            parentA_=pA;
            result_grad_=rG;
        };
        void apply(const std::vector<float>& incoming_grad) override;
        std::vector<std::shared_ptr<const Tensor>> get_parents() override;
        std::shared_ptr<std::vector<float>> get_gradient() override;


    private:
        std::shared_ptr< const Tensor> target_;
        std::shared_ptr< const Tensor> parentA_;
        std::shared_ptr<std::vector<float>> result_grad_;
};