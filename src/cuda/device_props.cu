#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main(int argc, char** argv) {
    int currentDeviceIdx = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, currentDeviceIdx);

    cout << "-------------------------------------------------------------------"  << endl;
    cout << "CUDA Device Properties:" << endl;
    cout << "-------------------------------------------------------------------"  << endl;
    cout << "Device Name: \t\t\t" << deviceProp.name << endl;
    cout << "Total Global Memory: \t\t" << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << "Shared Memory per Block: \t" << deviceProp.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "Number of Multiprocessors: \t" << deviceProp.multiProcessorCount << " SMs" <<  endl;
    cout << "Maximum Threads per Block: \t" << deviceProp.maxThreadsPerBlock << endl;
    cout << "Maximum Grid Size: \t\t(" << deviceProp.maxGridSize[0] << ", " 
    << deviceProp.maxGridSize[1] << ", " 
    << deviceProp.maxGridSize[2] << ")" << endl;
    cout << "Compute Capability: \t\t" << deviceProp.major << "." << deviceProp.minor << endl;
    cout << "Warp Size: \t\t\t" << deviceProp.warpSize << " threads" << endl;
    cout << "-------------------------------------------------------------------"  << endl;

    return 0;
}

