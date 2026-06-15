#include "Module.h"
#include <memory>


class Tensor;

class Relu: public Module{
    public:
    Relu(){};
    std::vector<std::shared_ptr<Tensor>>parameters() override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;


};