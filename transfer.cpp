#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]){
    // if the user does not input two parameters while running raise an error
    if (argc != 2){
        std::cerr << "usage: example-app <path-to-exported-module>\n";
        return -1;
    }
    // declare a pytorch module instance and assign the value 
    torch::jit::script::Module module;
    try{
        module = torch::jit::load(argv[1]);
    }
    // raise an error if the module does not load
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
    }
    std::cout << "ok\n";
}