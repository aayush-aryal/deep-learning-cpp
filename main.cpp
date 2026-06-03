#include <iostream>
#include "Tensor.h"


int main(){
    std::cout << "Engine starting..." <<std::endl;

    std::vector<int> my_shape={2,3};
    Tensor t(my_shape);
    t.printShape();

    return 0;
}