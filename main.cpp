#include <iostream>
#include "Tensor.h"


int main(){
    std::cout << "Engine starting..." <<std::endl;
    std::vector<int> my_shape={2,3};
    Tensor t1(my_shape);
    t1.printShape();
    t1.set(1,2,3.0);
    std::cout<< t1(1,2)<<std::endl;
    Tensor t2(my_shape);
    t2.set(1,1,1.0);
    std::cout<< t1.add(t2);
    return 0;
}