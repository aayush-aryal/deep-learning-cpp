#include <memory>
#include <vector>

class Tensor;

/**
 * @brief Stochastic Gradient Descent optimizer for updating tensor parameters.
 *
 * Stores a collection of tensors and updates their values using their
 * computed gradients and the specified learning rate.
 */
class SGD{
    public:
    /**
     * @brief Creates an SGD optimizer for a vector of tensors
     * 
     * @param tensor Tensors who's value will be updated during optimization
     * @param lr Learning rate used for parameter updates 
     */
    SGD(std::vector<std::shared_ptr<Tensor>> tensors, float lr){
        tensors_=tensors;
        lr_=lr;
    }

    /**
     * @brief Updates the tensors using their stored gradients
     */
    void step();
    /**
     * @brief Zeros the gradients for all tensors 
     */
    void zero_grad();

    private:
        std::vector<std::shared_ptr<Tensor>> tensors_;
        float lr_;

};