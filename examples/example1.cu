#include <gpuStencil.cuh>

#include <iostream>

int main () {
    const unsigned N = 128;
    float *u_in, *u_out;
    cudaMallocManaged(&u_in, sizeof(float)*N);
    cudaMallocManaged(&u_out, sizeof(float)*N);
    for (unsigned i = 0; i < N; i++) {
        u_in[i] = i;
    }
    kernel<<<1, N>>>(u_out, u_in);
    cudaDeviceSynchronize();
    std::cout << "Yaaay" << std::endl;
}
