#include <iostream>
#include <vector>
#include "Tensor.h"
#include "SGD.h"

int main() {
    std::cout << "Engine starting..." << std::endl;

    {
        auto t1 = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        t1->printShape();
        t1->set(1, 2, 3.0);
        std::cout << (*t1)(1, 2) << std::endl;

        auto t2 = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        t2->set(1, 1, 1.0);
        std::cout << *(t1->add(t2));

        t2->randomize();
        std::cout << *t2;
    }

    std::cout << "--------------Matmul-------------" << std::endl;
    {
        auto t1 = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        auto t3 = std::make_shared<Tensor>(std::vector<size_t>{3, 2});
        t1->randomize();
        t3->randomize();

        std::cout << *t1 << std::endl;
        std::cout << *t3 << std::endl;
        std::cout << *(t3->matmul(t1));
    }

    std::cout << "--------------Broadcasting-------------" << std::endl;
    {
        auto t1 = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        auto t4 = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
        t1->randomize();
        t4->randomize();

        std::cout << *t4;
        std::cout << *t1;
        std::cout << *(t1->add(t4));
    }

    std::cout << "--------------Forward Relu-------------" << std::endl;
    {
        auto t1 = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        t1->randomize();
        auto out = t1->relu();
        std::cout << *out;
    }

    std::cout << "--------------Autograd Add-------------" << std::endl;
    {
        auto A = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
        auto B = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
        A->set_requires_grad(true);
        B->set_requires_grad(true);

        A->randomize();
        B->randomize();

        auto C = A->add(B);
        C->backward();

        std::cout << "Gradient A: ";
        for (float g : A->get_grad()) std::cout << g << " ";
        std::cout << "\n";

        std::cout << "Gradient B: ";
        for (float g : B->get_grad()) std::cout << g << " ";
        std::cout << "\n";
    }

    std::cout << "--------------Multiplication-------------" << std::endl;
    {
        auto t5 = std::make_shared<Tensor>(std::vector<size_t>{3, 2});
        auto t6 = std::make_shared<Tensor>(std::vector<size_t>{2, 5});

        t5->set_requires_grad(true);
        t6->set_requires_grad(true);
        t5->randomize();
        t6->randomize();

        auto t7 = t5->matmul(t6);
        t7->backward();

        std::cout << *t7;
        std::cout << "Gradient T7: ";
        for (float g : t7->get_grad()) std::cout << g << " ";
        std::cout << "\n";

        std::cout << *t5;
        std::cout << "Gradient T5: ";
        for (float g : t5->get_grad()) std::cout << g << " ";
        std::cout << "\n";

        std::cout << *t6;
        std::cout << "Gradient T6: ";
        for (float g : t6->get_grad()) std::cout << g << " ";
        std::cout << "\n";
    }

    std::cout << "--------------Relu Autograd-------------" << std::endl;
    {
        auto t6 = std::make_shared<Tensor>(std::vector<size_t>{2, 5});
        t6->set_requires_grad(true);
        t6->randomize();

        auto t8 = t6->relu();
        t8->backward();

        std::cout << *t8;
        std::cout << "Gradient T6: ";
        for (float g : t6->get_grad()) std::cout << g << " ";
        std::cout << "\n";
    }

    std::cout << "----------" << std::endl;
    {
        auto t = std::make_shared<Tensor>(std::vector<size_t>{1, 5});
        t->set_requires_grad(true);
        t->randomize();

        auto out = t->relu();
        out->backward();

        std::cout << *t;
        for (float g : t->get_grad()) std::cout << g << " ";
        std::cout << "\n";
    }

    std::cout << "----------MSE Loss-----------" << std::endl;
    {
        auto pred = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
        auto target = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
        
        pred->set_requires_grad(true);
        pred->set(0, 0, 0.0f);
        pred->set(0, 1, 1.0f);
        pred->set(0, 2, 2.0f);

        target->set(0, 0, 0.0f);
        target->set(0, 1, 0.0f);
        target->set(0, 2, 0.0f);

        std::cout << "Predictions: " << *pred;
        std::cout << "Targets: " << *target;

        auto loss = pred->mse_loss(target);
        std::cout << "Loss Output: " << (*loss)(0,0) << "\n";

        loss->backward();

        std::cout << "Gradient Prediction: ";
        for (float g : pred->get_grad()){
            std::cout << g << " ";
        }
        std::cout << "\n";
    }

    std::cout << "----------SGD Optimizer-----------" << std::endl;
    {
        auto weight = std::make_shared<Tensor>(std::vector<size_t>{1,1});
        auto target = std::make_shared<Tensor>(std::vector<size_t>{1,1});

        weight->set_requires_grad(true);
        weight->set(0,0,2.0f);
        target->set(0,0,0.0f);

        std::vector<std::shared_ptr<Tensor>> params = {weight};
        SGD optimizer(params, 0.1f);
        std::cout << "Initial Weight: " << weight->get(0, 0) << "\n";

        auto loss = weight->mse_loss(target);

        loss->backward();
        std::cout << "Gradient after backward (Expected 4.0): " << weight->get_grad()[0] << "\n";
        optimizer.step();
        std::cout << "Weight after step (Expected 1.6): " << weight->get(0, 0) << "\n";
        optimizer.zero_grad();
        std::cout << "Gradient after zero_grad (Expected 0.0): " << weight->get_grad()[0] << "\n";
    }

    return 0;
}