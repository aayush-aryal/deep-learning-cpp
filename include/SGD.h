#include <memory>
#include <vector>

class Tensor;
class SGD{
    public:
    SGD(std::vector<std::shared_ptr<Tensor>> tensors, float lr){
        tensors_=tensors;
        lr_=lr;
    }

    void step();
    void zero_grad();

    private:
        std::vector<std::shared_ptr<Tensor>> tensors_;
        float lr_;

};