#include <memory>
#include "Module.h"


class Tensor;

class Sequential:public Module{
    public:
    Sequential(std::vector<std::unique_ptr<Module>> modules){
        modules_=std::move(modules);
    };

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor>input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;


    private:
    std::vector<std::unique_ptr<Module>> modules_;
};