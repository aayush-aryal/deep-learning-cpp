#include <iostream>
#include <vector>
#include "Tensor.h"
#include "SGD.h"
#include "nn/Linear.h"
#include "nn/Relu.h"
#include "nn/Sequential.h"
#include "nn/Module.h"
#include "nn/MultiHeadAttention.h"
#include "vector"
#include "Dataloader.h"
#include "utility.h"

int main() {
    std::cout << "Engine starting..." << std::endl;

    {
        auto t1 = std::make_shared<Tensor>(std::vector<size_t>{2,2,3});
        t1->printShape();
        t1->set({0,0,1}, 3.0);
        std::cout << t1->get({0,0,1}) << std::endl;

        auto t2 = std::make_shared<Tensor>(std::vector<size_t>{3});
        t2->set({1}, 1.0);
        auto res= *(t1->add(t2));
        std::cout << res << std::endl;


        auto t3= std::make_shared<Tensor>(std::vector<size_t>{2,2,3});
        t3->set({0,0,1},4.0);
        res=*(t3->add(t1));
        std::cout<<res<<std::endl;

    }

    std::cout << "--------------Matmul-------------" << std::endl;
    {
        auto t1 = std::make_shared<Tensor>(std::vector<size_t>{1,2, 3});
        auto t3 = std::make_shared<Tensor>(std::vector<size_t>{2,3, 2});
        t1->randomize();
        t3->randomize();

        std::cout << *t1 << std::endl;
        std::cout << *t3 << std::endl;
        std::cout << *(t3->matmul(t1));


        auto t4= std::make_shared<Tensor>(std::vector<size_t>{3});
        t4->randomize();
        std::cout << *(t1->matmul(t4));

    }

    // std::cout << "--------------Broadcasting-------------" << std::endl;
    // {
    //     auto t1 = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
    //     auto t4 = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
    //     t1->randomize();
    //     t4->randomize();

    //     std::cout << *t4;
    //     std::cout << *t1;
    //     std::cout << *(t1->add(t4));
    // }

    // std::cout << "--------------Forward Relu-------------" << std::endl;
    // {
    //     auto t1 = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
    //     t1->randomize();
    //     auto out = t1->relu();
    //     std::cout << *out;
    // }

    std::cout << "--------------Autograd Add-------------" << std::endl;
    {
        auto A = std::make_shared<Tensor>(std::vector<size_t>{1,2,3, 3});
        auto B = std::make_shared<Tensor>(std::vector<size_t>{3});
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

    std::cout << "--------------Matmul backward-------------" << std::endl;
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
{
    std::cout << "-------- Gradient checks --------" << std::endl;

    // fresh tensor with positive, well-behaved values
    auto make = [](std::vector<size_t> shape){
        auto t = std::make_shared<Tensor>(shape);
        auto& d = t->get_data_ref();
        for (size_t i = 0; i < d.size(); i++) d[i] = 0.5f + 0.3f * (i % 7) + 0.05f * i;
        t->set_requires_grad(true);
        return t;
    };
    auto report = [](const std::string& name, bool ok){
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << std::endl;
    };

    // unary
    report("negate",     gradient_check_unary(make({2,3}), [](auto x){ return x->negate(); }));
    report("exp",        gradient_check_unary(make({2,3}), [](auto x){ return x->exp(); }));
    report("sqrt",       gradient_check_unary(make({2,3}), [](auto x){ return x->sqrt(); }));
    report("reciprocal", gradient_check_unary(make({2,3}), [](auto x){ return x->reciprocal(); }));
    report("log",        gradient_check_unary(make({2,3}), [](auto x){ return x->log(); }));

    // binary, same shape
    report("add",      gradient_check_add(make({2,3}), make({2,3})));
    report("subtract", gradient_check_subtract(make({2,3}), make({2,3})));
    report("multiply", gradient_check_multiply(make({2,3}), make({2,3})));
    report("divide",   gradient_check_divide(make({2,3}), make({2,3})));

    // binary, broadcasting
    report("add broadcast",      gradient_check_add(make({2,2,3}), make({3})));
    report("subtract broadcast", gradient_check_subtract(make({2,2,3}), make({3})));
    report("multiply broadcast", gradient_check_multiply(make({2,2,3}), make({1,3})));
    report("divide broadcast",   gradient_check_divide(make({2,2,3}), make({1,3})));

    // matmul
    report("matmul 2D",      gradient_check_matmul(make({3,4}),   make({4,2})));
    report("matmul batched", gradient_check_matmul(make({2,3,4}), make({2,4,2})));
    report("matmul broadcast", gradient_check_matmul(make({2,3,4}), make({4,2})));
}
{
    std::cout<<"Axiswise sum"<< std::endl;
    auto t1= std::make_shared<Tensor>(std::vector<size_t>{2,2,3});
    t1->randomize();
    t1->set_requires_grad(true);
    auto result= t1->multiply(t1)->axiswise_sum(2);
    std::cout<<*t1;
    std::cout<<*result;

    result->backward();
        for (float g : t1->get_grad()) std::cout << g << " ";
        std::cout << "\n";

    for (int axis : {0, 1, 2, -1}) {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2,2,3});
    t->randomize();
    t->set_requires_grad(true);
    auto fwd = [&]() { return t->multiply(t->axiswise_sum(axis)); };
    std::cout << "axis " << axis << ": "
              << (check_one_tensor(t, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;
}
}

{
    std::cout << "Axiswise max" << std::endl;

    std::vector<float> data={1,5,3,-5,-2,-9};
    auto t= std::make_shared<Tensor>(std::vector<size_t>{2,3});
    std::vector<float>& t_ref=t->get_data_ref();
    t_ref=data;

    auto m1 = t->axiswise_max(1);    // expect shape (2x1): 5 -2
    auto m0 = t->axiswise_max(0);    // expect shape (1x3): 1 5 3
    auto mn = t->axiswise_max(-1);   // should match m1

    std::cout << *t << *m1 << *m0 << *mn;
}
{
    std::cout<<"Softmax function"<<std::endl;
    std::vector<float> data={1,5,3,-5,-2,-9};
    auto t= std::make_shared<Tensor>(std::vector<size_t>{2,3});
    t->set_requires_grad(true);
    std::vector<float>& t_ref=t->get_data_ref();
    t_ref=data;

    auto sm = t->softmax(-1);
    std::cout << *sm;                         // row 0 expect ~0.090 0.245 0.665
    std::cout << *sm->axiswise_sum(-1);

    
    for (int axis:{-1,0,1}){
        auto t2= std::make_shared<Tensor>(std::vector<size_t>{2,3,4});
        t2->randomize();
        t2->set_requires_grad(true);


        auto w=  std::make_shared<Tensor>(std::vector<size_t>{2,3,4});

        w->randomize();
        auto fwd=[&](){return t2->softmax(axis)->multiply(w);};

        std::cout<<"softmax axis" << axis << ":" 
        << (check_one_tensor(t,fwd,1e-3f,2e-2f)? "PASS":"FAIL")
        << std::endl;
    }
    auto big = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
    big->get_data_ref() = {1000, 1001, 1002};
    std::cout << *big->softmax(-1);
}
{
    std::cout<<"Permute"<<std::endl;
    auto t= std::make_shared<Tensor>(std::vector<size_t>{2,3,4});
    t->randomize();
    t->set({1,2,3},23.0);
    std::vector<int> p{2,0,1};
    auto t2= t->permute(p);
    std::cout<<*t2<<std::endl;
    std::cout << t2->get({3,1,2}) << std::endl;

    std::cout<<"Backward"<<std::endl;
    std::vector<std::vector<int>> perms = {{1,2,0}, {2,0,1}, {0,2,1}, {-1,0,1}};

    for(auto&p:perms){
        auto ten=std::make_shared<Tensor>(std::vector<size_t>{2,3,4});
        ten->randomize();
        ten->set_requires_grad(true);

        auto permed= ten->permute(p);

        auto w= std::make_shared<Tensor>(permed->get_shape());
        w->randomize();

        auto fwd=[&](){return ten->permute(p)->multiply(w);};
        std::cout << "perm (" << p[0] << "," << p[1] << "," << p[2] << "): "
                  << (check_one_tensor(ten, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL")
                  << std::endl;

    }

    {
        std::cout<< "Transpose"<< std::endl;
        auto m = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
        m->get_data_ref() = {1, 2, 3, 4, 5, 6};
        auto mt = m->transpose(0, 1);
        std::cout<<*mt;
        auto x = std::make_shared<Tensor>(std::vector<size_t>{2, 4, 3});
        x->randomize();
        auto a = x->transpose(-2, -1);
        auto b = x->transpose(1, 2);
        std::cout << *a;    // expect Shape (2x3x4)
        bool same = (a->get_data() == b->get_data()) && (a->get_shape() == b->get_shape());
        std::cout << "neg vs pos axes: " << (same ? "PASS" : "FAIL") << std::endl;

        auto rt = x->transpose(-2, -1)->transpose(-2, -1);
        bool rt_ok = (rt->get_data() == x->get_data()) && (rt->get_shape() == x->get_shape());
        std::cout << "round trip: " << (rt_ok ? "PASS" : "FAIL") << std::endl;
        auto g = std::make_shared<Tensor>(std::vector<size_t>{2, 4, 3});
        g->randomize();
        g->set_requires_grad(true);
        auto w = std::make_shared<Tensor>(std::vector<size_t>{2, 3, 4});
        w->randomize();   // constant, no grad
        auto fwd = [&]() { return g->transpose(-2, -1)->multiply(w); };
        std::cout << "transpose grad: "
                << (check_one_tensor(g, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;
        
        try {
        x->transpose(0, 3);
        std::cout << "range check: FAIL (no throw)" << std::endl;
    } catch (const std::out_of_range&) {
        std::cout << "range check: PASS" << std::endl;
    }
    }

    {
    std::cout << "Attention" << std::endl;
    size_t d = 3;

    auto Q = std::make_shared<Tensor>(std::vector<size_t>{2,4,3});
    auto K = std::make_shared<Tensor>(std::vector<size_t>{2,4,3});
    auto V = std::make_shared<Tensor>(std::vector<size_t>{2,4,3});
    Q->randomize(); K->randomize(); V->randomize();
    Q->set_requires_grad(true);
    K->set_requires_grad(true);
    V->set_requires_grad(true);

    auto w= std::make_shared<Tensor>(std::vector<size_t>{2,4,3});


    w->randomize();

    auto out= attention(Q,K,V);
    std::cout << "out shape: ";
    for (size_t s : out->get_shape()) std::cout << s << " ";
    std::cout << std::endl;

    auto fwd = [&]() { return attention(Q, K, V)->multiply(w); };

    std::cout << "Q: " << (check_one_tensor(Q, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;
    Q->zero_grad(); K->zero_grad(); V->zero_grad();
    std::cout << "K: " << (check_one_tensor(K, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;
    Q->zero_grad(); K->zero_grad(); V->zero_grad();
    std::cout << "V: " << (check_one_tensor(V, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;

    
    }
    {
        std::cout<<"Causal Mask" <<std::endl;
        std::cout<<*causal_mask(4);
    }

    {
        std::cout<<"Reshape"<< std::endl;
        t= std::make_shared<Tensor>(std::vector<size_t>{2,3,4});
        t->randomize();
        auto t1= t->reshape({2,-1});
        std::cout<<*t<<std::endl;
        std::cout<<*t1<<std::endl;
        auto t2= t1->reshape({2,3,4});
        std::cout<<*t2<<std::endl;

        // testing backward 
         auto x = std::make_shared<Tensor>(std::vector<size_t>{2,3,4});
        x->randomize();
        x->set_requires_grad(true);
        auto w = std::make_shared<Tensor>(std::vector<size_t>{6,4});
        w->randomize();   // constant, no grad
        auto fwd = [&]() { return x->reshape({6,4})->multiply(w); };
        std::cout << "reshape grad: "
                << (check_one_tensor(x, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;

        // testing split merge

        auto y = std::make_shared<Tensor>(std::vector<size_t>{2,4,6});
        y->randomize();
        y->set_requires_grad(true);
        auto split_merge = [&]() {
            return y->reshape({2,4,2,-1})
                    ->permute(std::vector<int>{0,2,1,3})
                    ->permute(std::vector<int>{0,2,1,3})
                    ->reshape({2,4,-1});
        };
        auto rt = split_merge();
        std::cout << "split/merge round trip: "
                << ((rt->get_data() == y->get_data() && rt->get_shape() == y->get_shape()) ? "PASS" : "FAIL")
                << std::endl;

        auto w2 = std::make_shared<Tensor>(std::vector<size_t>{2,4,6});
        w2->randomize();
        auto fwd2 = [&]() { return split_merge()->multiply(w2); };
        std::cout << "split/merge grad: "
                << (check_one_tensor(y, fwd2, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;

        }
    } 
    {
    std::cout << "MultiHeadAttention" << std::endl;
    const size_t B = 2, T = 4, D = 6;

    auto make_x = [&]() {
        auto x = std::make_shared<Tensor>(std::vector<size_t>{B, T, D});
        x->randomize();
        x->set_requires_grad(true);
        return x;
    };
    auto w = std::make_shared<Tensor>(std::vector<size_t>{B, T, D});
    w->randomize();   // constant, no grad

    // check shape
    {
        MultiHeadAttention mha(D, 2);
        auto out = mha.forward(make_x(), false);
        std::cout << "shape: ";
        for (size_t s : out->get_shape()) std::cout << s << " ";
        std::cout << "(expect 2 4 6)" << std::endl;
        std::cout << "param count: " << mha.parameters().size() << " (expect 8)" << std::endl;
    }

    //  d_model (number of features for each token) not divisible by heads must throw
    try {
        MultiHeadAttention bad(6, 4);
        std::cout << "divisibility check: FAIL (no throw)" << std::endl;
    } catch (const std::invalid_argument&) {
        std::cout << "divisibility check: PASS" << std::endl;
    }

    // h=1 should give same output as single headed attention
    {
        MultiHeadAttention mha(D, 1);
        auto x = make_x();
        auto p = mha.parameters();   // Wq,bq,Wk,bk,Wv,bv,Wo,bo
        auto Q = x->matmul(p[0])->add(p[1]);
        auto K = x->matmul(p[2])->add(p[3]);
        auto V = x->matmul(p[4])->add(p[5]);
        auto ref = attention(Q, K, V, false)->matmul(p[6])->add(p[7]);
        auto got = mha.forward(x, false);
        float maxdiff = 0.0f;
        for (size_t i = 0; i < ref->get_data().size(); i++)
            maxdiff = std::max(maxdiff, std::abs(ref->get_data()[i] - got->get_data()[i]));
        std::cout << "H=1 consistency: " << (maxdiff < 1e-5f ? "PASS" : "FAIL")
                  << " (maxdiff " << maxdiff << ")" << std::endl;
    }

    // checking causal false and true with gradient checker
    for (bool causal : {false, true}) {
        MultiHeadAttention mha(D, 2);
        auto x = make_x();
        auto fwd = [&]() { return mha.forward(x, causal)->multiply(w); };
        std::cout << "grad x (causal=" << causal << "): "
                  << (check_one_tensor(x, fwd, 1e-3f, 2e-2f) ? "PASS" : "FAIL") << std::endl;
    }

    //  check gradients on each of teh sub vectors
    const char* names[8] = {"Wq","bq","Wk","bk","Wv","bv","Wo","bo"};
    for (size_t i = 0; i < 8; i++) {
        MultiHeadAttention mha(D, 2);
        auto x = make_x();
        auto target = mha.parameters()[i];
        auto fwd = [&]() { return mha.forward(x, false)->multiply(w); };
        std::cout << "grad " << names[i] << ": "
                  << (check_one_tensor(target, fwd, 1e-3f, 5e-2f) ? "PASS" : "FAIL") << std::endl;
    }

    // changing the last token must not affect earlier outputs
    {
        MultiHeadAttention mha(D, 2);
        auto x = make_x();
        auto out1 = mha.forward(x, true);
        std::vector<float> before = out1->get_data();   // copy

        std::vector<float>& xd = x->get_data_ref();
        for (size_t b = 0; b < B; b++)
            for (size_t d = 0; d < D; d++)
                xd[b * T * D + (T - 1) * D + d] += 5.0f;   // perturb last token only

        auto out2 = mha.forward(x, true);
        const std::vector<float>& after = out2->get_data();

        float early_diff = 0.0f, last_diff = 0.0f;
        for (size_t b = 0; b < B; b++)
            for (size_t t = 0; t < T; t++)
                for (size_t d = 0; d < D; d++) {
                    size_t i = b * T * D + t * D + d;
                    float diff = std::abs(before[i] - after[i]);
                    if (t < T - 1) early_diff = std::max(early_diff, diff);
                    else last_diff = std::max(last_diff, diff);
                }
        std::cout << "no-leak: " << (early_diff < 1e-6f ? "PASS" : "FAIL")
                  << " (early diff " << early_diff << ")" << std::endl;
        std::cout << "last token did change: " << (last_diff > 1e-3f ? "PASS" : "FAIL") << std::endl;
    }
}      

    // std::cout << "--------------Relu Autograd-------------" << std::endl;
    // {
    //     auto t6 = std::make_shared<Tensor>(std::vector<size_t>{2, 5});
    //     t6->set_requires_grad(true);
    //     t6->randomize();

    //     auto t8 = t6->relu();
    //     t8->backward();

    //     std::cout << *t8;
    //     std::cout << "Gradient T6: ";
    //     for (float g : t6->get_grad()) std::cout << g << " ";
    //     std::cout << "\n";
    // }

    // std::cout << "----------" << std::endl;
    // {
    //     auto t = std::make_shared<Tensor>(std::vector<size_t>{1, 5});
    //     t->set_requires_grad(true);
    //     t->randomize();

    //     auto out = t->relu();
    //     out->backward();

    //     std::cout << *t;
    //     for (float g : t->get_grad()) std::cout << g << " ";
    //     std::cout << "\n";
    // }

    // std::cout << "----------MSE Loss-----------" << std::endl;
    // {
    //     auto pred = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
    //     auto target = std::make_shared<Tensor>(std::vector<size_t>{1, 3});
        
    //     pred->set_requires_grad(true);
    //     pred->set(0, 0, 0.0f);
    //     pred->set(0, 1, 1.0f);
    //     pred->set(0, 2, 2.0f);

    //     target->set(0, 0, 0.0f);
    //     target->set(0, 1, 0.0f);
    //     target->set(0, 2, 0.0f);

    //     std::cout << "Predictions: " << *pred;
    //     std::cout << "Targets: " << *target;

    //     auto loss = pred->mse_loss(target);
    //     std::cout << "Loss Output: " << (*loss)(0,0) << "\n";

    //     loss->backward();

    //     std::cout << "Gradient Prediction: ";
    //     for (float g : pred->get_grad()){
    //         std::cout << g << " ";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "----------SGD Optimizer-----------" << std::endl;
    // {
    //     auto weight = std::make_shared<Tensor>(std::vector<size_t>{1,1});
    //     auto target = std::make_shared<Tensor>(std::vector<size_t>{1,1});

    //     weight->set_requires_grad(true);
    //     weight->set(0,0,2.0f);
    //     target->set(0,0,0.0f);

    //     std::vector<std::shared_ptr<Tensor>> params = {weight};
    //     SGD optimizer(params, 0.1f);
    //     std::cout << "Initial Weight: " << weight->get(0, 0) << "\n";

    //     auto loss = weight->mse_loss(target);

    //     loss->backward();
    //     std::cout << "Gradient after backward (Expected 4.0): " << weight->get_grad()[0] << "\n";
    //     optimizer.step();
    //     std::cout << "Weight after step (Expected 1.6): " << weight->get(0, 0) << "\n";
    //     optimizer.zero_grad();
    //     std::cout << "Gradient after zero_grad (Expected 0.0): " << weight->get_grad()[0] << "\n";
    // }

    // std::cout<< "--------Linear Layer--------"<<std::endl;
    // {
    //     Linear new_layer(3,2);
    //     auto input= std::make_shared<Tensor>(std::vector<size_t>{1,3});
    //     input->randomize();
    //     SGD optimizer(new_layer.parameters(),0.01f);

    //     auto target= std::make_shared<Tensor>(std::vector<size_t>{1,2});
    //     target->set(0,0,1.0f);
    //     target->set(0,1,0.0f);


    //     for (int i=0; i<50; i++){
    //         //  the training process is 
    //         // clear old gradients
    //         optimizer.zero_grad();
    //         // do a new forward pass
    //         auto pred= new_layer.forward(input);
    //         // calculate the loss
    //         auto loss= pred->mse_loss(target);
    //         // backward pass based on the loss
    //         loss->backward();
    //         // update the weights 
    //         optimizer.step();
    //         std::cout << "Epoch " << i << " Loss: " << (*loss)(0, 0) << std::endl;
    //     }   

    //     std::cout<<"Final pred" << *new_layer.forward(input);
    // }

    // std::cout<< " Sequential Layer-----" <<std::endl;
    // {
    //     std::unique_ptr<Linear> layer1 =std::make_unique<Linear>(3,2);
    //     std::unique_ptr<Relu> activation= std::make_unique<Relu>();
    //     std::unique_ptr<Linear> layer2=std::make_unique<Linear>(2,1);

    //     std::vector<std::unique_ptr<Module>> layers;

    //     // because unique pointers cannot be copied std::move needs to be used to move their reference
    //     layers.push_back(std::move(layer1));
    //     layers.push_back(std::move(activation));
    //     layers.push_back(std::move(layer2));


    //     Sequential combined_layers(std::move(layers));
    //     auto input= std::make_shared<Tensor>(std::vector<size_t>{1,3});
    //     input->set(0,0,1);
    //     std::cout<<"Input tensor :"<<*input<< std::endl;
    //     auto target= std::make_shared<Tensor>(std::vector<size_t>{1,1});
    //     target->set(0,0,1.0f);

    //     SGD optimizer(combined_layers.parameters(),0.1f);
    //     for (int i=0; i<30; i++){
    //         optimizer.zero_grad();
    //         // forward pass
    //         auto pred= combined_layers.forward(input);
    //         auto loss= pred->mse_loss(target);
    //         // back propagate based on loss ie update gradients
    //         loss->backward();
    //         // update weights based on updated grad
    //         optimizer.step();
    //         std::cout << "Epoch " << i << " Loss: " << (*loss)(0, 0) << std::endl;
    //     }

    //     std::cout<<"Final pred" << *combined_layers.forward(input);
    // }
    // std::cout<< "----- Softmax Crossentropy -----" <<std::endl;
    // {
    //     std::unique_ptr<Linear> layer1 =std::make_unique<Linear>(3,6);
    //     std::unique_ptr<Linear> layer2=std::make_unique<Linear>(6,3);

    //     std::vector<std::unique_ptr<Module>> layers;

    //     // because unique pointers cannot be copied std::move needs to be used to move their reference
    //     layers.push_back(std::move(layer1));
    //     layers.push_back(std::move(layer2));


    //     Sequential combined_layers(std::move(layers));
    //     auto input= std::make_shared<Tensor>(std::vector<size_t>{1,3});
    //     input->set(0,0,0.2f);
    //     input->set(0,1,0.8f);
    //     input->set(0,2,0.1f);

    //     std::cout<<"Input tensor :"<<*input<< std::endl;
    //     auto target= std::make_shared<Tensor>(std::vector<size_t>{1,3});
    //     target->set(0,2,1.0f);

    //     SGD optimizer(combined_layers.parameters(),0.1f);
    //     for (int i=0; i<30; i++){
    //         optimizer.zero_grad();
    //         // forward pass
    //         auto pred= combined_layers.forward(input);
    //         auto loss= pred->softmax_crossentropy(target);
    //         // back propagate based on loss ie update gradients
    //         loss->backward();
    //         // update weights based on updated grad
    //         optimizer.step();
    //         std::cout << "Epoch " << i << " Loss: " << (*loss)(0, 0) << std::endl;
    //     }

    //     auto pred= *combined_layers.forward(input)->softmax(target);
    //     std::cout<<"[";
    //     for (size_t j=0; j<pred.size(); j++){
    //         std::cout<< pred[j]<<",";
    //     }
    //     std::cout<<"]"<<std::endl;
    // }

    // std::cout<< "----MNIST test----" << std::endl;
    // {
    //     std::unique_ptr<Linear> layer1 =std::make_unique<Linear>(784,128);
    //     std::unique_ptr<Relu> layer2= std::make_unique<Relu>();
    //     std::unique_ptr<Linear> layer3=std::make_unique<Linear>(128,64);
    //     std::unique_ptr<Relu>   layer4 = std::make_unique<Relu>();
    //     std::unique_ptr<Linear> layer5=std::make_unique<Linear>(64,10);

    //     std::vector<std::unique_ptr<Module>> layers;

    //     // because unique pointers cannot be copied std::move needs to be used to move their reference
    //     layers.push_back(std::move(layer1));
    //     layers.push_back(std::move(layer2));
    //     layers.push_back(std::move(layer3));
    //     layers.push_back(std::move(layer4));
    //     layers.push_back(std::move(layer5));

    //     Sequential combined_layers(std::move(layers));
    //     SGD optimizer(combined_layers.parameters(),0.01f);
    //     DataLoader d("/Users/aayush-aryal/Documents/deep-learning-cpp/datasets/mnist_train.csv",32);
    //     DataLoader t("/Users/aayush-aryal/Documents/deep-learning-cpp/datasets/mnist_test.csv",32);
    //     for (size_t k=0; k<20; k++){
    //         d.reset_and_shuffle();
    //         t.reset();

    //         // this is for train 
    //         float total_train_loss_sum = 0.0f;
    //         int total_train_samples_processed = 0;
    //         int train_batch_idx = 0;
    //         float total_train_accuracy_sum=0.0f;

    //         // this is for test
    //         float total_test_loss_sum = 0.0f;
    //         int total_test_samples_processed = 0;
    //         int test_batch_idx = 0;
    //         float total_test_accuracy_sum=0.0f;

    //         while (d.has_next_batch()){
    //             // get target and output for next batch
    //             auto [input,target]=d.get_next_batch();
    //             optimizer.zero_grad();
    //             auto pred=combined_layers.forward(input);
    //             auto loss= pred->softmax_crossentropy(target);
    //             loss->backward();
    //             optimizer.step();
    //             int current_batch_size = input->get_shape()[0];
    //             total_train_loss_sum+=(*loss)(0,0)*current_batch_size;
    //             total_train_samples_processed+=current_batch_size;
    //             // calculate accuracy
    //             total_train_accuracy_sum+=accuracy(pred,target)*current_batch_size;
    //             // std::cout << "Epoch " << i << " Loss: " << (*loss)(0, 0) << std::endl;
    //         }
    //         float average_train_epoch_loss = total_train_loss_sum / total_train_samples_processed;
    //         float average_train_epoch_accuracy=total_train_accuracy_sum/total_train_samples_processed;
    //         std::cout << "=======================================" << std::endl;
    //         std::cout << "Average Train Dataset Loss: " << average_train_epoch_loss << std::endl;
    //         std::cout << "Average Train Dataset Accuracy: " << average_train_epoch_accuracy << std::endl;

    //         while (t.has_next_batch()){
    //             // get target and output for next batch
    //             auto [input,target]=t.get_next_batch();
    //             auto pred=combined_layers.forward(input);
    //             auto loss= pred->softmax_crossentropy(target);
    //             // loss->backward();
    //             // optimizer.step();
    //             int current_batch_size = input->get_shape()[0];
    //             total_test_loss_sum+=(*loss)(0,0)*current_batch_size;
    //             total_test_samples_processed+=current_batch_size;
    //             // calculate accuracy
    //             total_test_accuracy_sum+=accuracy(pred,target)*current_batch_size;
    //             // std::cout << "Epoch " << i << " Loss: " << (*loss)(0, 0) << std::endl;
    //         }
    //             float average_test_epoch_loss = total_test_loss_sum / total_test_samples_processed;
    //             float average_test_epoch_accuracy=total_test_accuracy_sum/total_test_samples_processed;
    //             std::cout << "Average Test Dataset Loss: " << average_test_epoch_loss << std::endl;
    //             std::cout << "Average Test Dataset Accuracy: " << average_test_epoch_accuracy << std::endl;
    //             std::cout << "Epoch " << k+1 << " Finished." << std::endl;
    //             std::cout << "=======================================" << std::endl;
    //     }    
    // }
    return 0;
}