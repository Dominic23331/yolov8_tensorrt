#include <iostream>
#include <NvInfer.h>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << nvinfer1::kNV_TENSORRT_VERSION_IMPL << std::endl;
    return 0;
}
