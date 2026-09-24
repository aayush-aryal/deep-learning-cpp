cmake_minimum_required(VERSION 3.10)
project(deep-learning-cpp)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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


find_package(Doxygen)

if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen"
        VERBATIM
    )
endif()