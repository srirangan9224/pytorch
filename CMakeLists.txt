cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(pytorch transfer.cpp)
target_link_libraries(pytorch "${TORCH_LIBRARIES}")
set_property(TARGET pytorch PROPERTY CXX_STANDARD 17)