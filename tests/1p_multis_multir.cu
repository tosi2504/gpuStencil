#include "error.h"
#include "gpuStencil.cuh"
#include "errorcheck.h"
#include "datatypes.cuh"

#include <iostream>

constexpr unsigned N = 64;
constexpr unsigned nRhs = 32;
constexpr unsigned nSites = 128;

int main () {
    // matrix field management
    complexF *u_A;
    complexF ***u_u_u_A;
    CCE(    cudaMallocManaged(&u_A, sizeof(complexF)*N*N*nSites)   );
    CCE(    cudaMallocManaged(&u_u_u_A, sizeof(complexF**)*nSites) );
    for (unsigned s = 0; s < nSites; s++) {
        CCE(    cudaMallocManaged(u_u_u_A+s, sizeof(complexF*))    );
        u_u_u_A[s][0] = u_A;
    }
    
    // vector field management
    complexF **u_u_X, **u_u_Y;
    CCE(    cudaMallocManaged(&u_u_X, sizeof(complexF*)*nRhs)      );
    CCE(    cudaMallocManaged(&u_u_Y, sizeof(complexF*)*nRhs)      );
    for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
        CCE(    cudaMallocManaged(u_u_X + iRhs, sizeof(complexF)*N*nSites)    );
        CCE(    cudaMallocManaged(u_u_Y + iRhs, sizeof(complexF)*N*nSites)    );
    }
    
    for (unsigned s = 0; s < nSites; s++) {
        for (unsigned i = 0; i < N*N; i++) {
            if (i/N == i%N) {
                u_A[s*N*N+i] = 1;
            } else {
                u_A[s*N*N+i] = 0;
            }
        }
    }
    
    for (unsigned s = 0; s < nSites; s++) {
        for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
            for (unsigned i = 0; i < N; i++) {
                u_u_X[iRhs][s*N+i] = i;
            }
        }
    }
    
    ret_status_t res = interface1<N, nRhs, N/2, N/2, 2, 2>(u_u_Y, u_u_u_A, u_u_X, 1, nSites);
    if (res != OK) {
        std::cout << "Bad return value" << std::endl;
        return 1;
    }
    for (unsigned s = 0; s < nSites; s++) {
        for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
            for (unsigned i = 0; i < N; i++) {
                if (std::abs(u_u_X[iRhs][s*N+i]-u_u_Y[iRhs][s*N+i]) > 0.000001) {
                    printf("Bad diff -> s: %u, iRhs: %u, i: %u", s, iRhs, i);
                    std::cout << "X: " << u_u_X[iRhs][s*N+i] << std::endl;
                    std::cout << "Y: " << u_u_Y[iRhs][s*N+i] << std::endl;
                    return 2;
                }
            }
        }
    }

    // LETS LEAK ALL THE MEMORY YAY
    
    return 0;
}
