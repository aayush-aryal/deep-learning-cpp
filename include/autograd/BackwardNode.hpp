#pragma once // only reads this file once
#include <memory>
// it prevents compiler from needing to know Tensor size
class Tensor;
/**
 * @brief Represents a node in the computational graph required to do backpropagation
 */
class BackwardNode {
    public:
    virtual ~BackwardNode()=default;
    /**
     * @brief Applies the backpropagation function using the gradient for the output tensor
     * 
     * @param incoming_grad Gradient vector of the output tensor
     */
    virtual void apply(const std::vector<float>& incoming_grad)=0;

    /**
     * @brief Gets the parent tensors that created the backward node
     *  @return The parent tensors in the computational graph
     */
    virtual std::vector<std::shared_ptr<const Tensor>>get_parents()=0;
    /**
     * @brief Returns the gradient associated with this node.
     *
     * @return A shared pointer to the stored gradient.
     */
    virtual std::shared_ptr<std::vector<float>> get_gradient()=0;
};