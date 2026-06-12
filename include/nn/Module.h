#pragma once
#include <memory>
#include <vector>


class Tensor;

class Module{
    public:
        virtual ~Module()=default;
        virtual std::vector<std::shared_ptr<Tensor>> parameters()=0;
        virtual std::shared_ptr<Tensor> forward( std::shared_ptr<Tensor> input)=0;
};