#pragma once // only reads this file once
#include <memory>
// it prevents compiler from needing to know Tensor size
class Tensor;
class BackwardNode {
    public:
    virtual ~BackwardNode()=default;
    virtual void apply(const std::vector<float>& incoming_grad)=0;
    virtual std::vector<std::shared_ptr<const Tensor>>get_parents()=0;
    virtual std::shared_ptr<std::vector<float>> get_gradient()=0;
};