cmake_minimum_required(VERSION 3.10)
project(deep-learning-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
add_executable(run_engine 
                main.cpp 
                src/Tensor.cpp
                src/autograd/ops/AddBackward.cpp
                src/autograd/ops/MatmulBackward.cpp
                src/autograd/ops/ReluBackward.cpp
                src/autograd/ops/MSEBackward.cpp
                src/autograd/ops/Softmax.cpp
                src/SGD.cpp
                src/nn/Linear.cpp
                src/nn/Relu.cpp
                src/nn/Sequential.cpp

                src/Dataloader.cpp
                src/utility.cpp
                
                )
