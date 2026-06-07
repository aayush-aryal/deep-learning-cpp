#include <iostream>
#include "Tensor.h"


int main(){
    std::cout << "Engine starting..." <<std::endl;
    std::vector<size_t> my_shape={2,3};
    Tensor t1(my_shape);
    t1.printShape();
    t1.set(1,2,3.0);
    std::cout<< t1(1,2)<<std::endl;
    Tensor t2(my_shape);
    t2.set(1,1,1.0);
    std::cout<< t1.add(t2);
    // std::cout<<t2.matmul(t1);
    t2.randomize();
    std::cout<<t2;
    Tensor t3(std::vector<size_t>{3,2});
    t3.randomize();
    t1.randomize();
    std::cout<<"--------------Matmul-------------"<<std::endl;
    std::cout<<t1<<std::endl;
    std::cout<<t3<<std::endl;
    std::cout<<t3.matmul(t1);

    std::cout<<"--------------Broadcasting-------------"<<std::endl;
    Tensor t4(std::vector<size_t>{1,3});
    t4.randomize();
    std::cout<<t4;
    std::cout<<t1;
    std::cout<<t1.add(t4);

    t1.relu();
    std::cout<<t1;
    Tensor A({3});
    Tensor B({3});

    Tensor C = A.add(B);
    C.backward();

    std::vector<float>& gradA = A.get_grad();
    std::vector<float>& gradB = B.get_grad();

    std::cout << "Gradient A: ";
    for (float g : gradA) std::cout << g << " ";
    std::cout << "\n";

    std::cout << "Gradient B: ";
    for (float g : gradB) std::cout << g << " ";
    std::cout << "\n";

    return 0;
}