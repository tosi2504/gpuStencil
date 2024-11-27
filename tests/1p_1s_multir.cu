#include "error.h"
#include "gpuStencil.cuh"
#include "errorcheck.h"
#include "datatypes.cuh"

#include <iostream>

constexpr unsigned N = 64;
constexpr unsigned nRhs = 32;

int main () {
    complexF *u_A;
    complexF **u_u_A, **u_u_X, **u_u_Y;
    complexF ***u_u_u_A;
    
    CCE(    cudaMallocManaged(&u_A, sizeof(complexF)*N*N)       );
    CCE(    cudaMallocManaged(&u_u_A, sizeof(complexF*))        );
    CCE(    cudaMallocManaged(&u_u_u_A, sizeof(complexF**))     );
    CCE(    cudaMallocManaged(&u_u_X, sizeof(complexF*)*nRhs)   );
    CCE(    cudaMallocManaged(&u_u_Y, sizeof(complexF*)*nRhs)   );
    for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
        CCE(    cudaMallocManaged(u_u_X + iRhs, sizeof(complexF)*N)    );
        CCE(    cudaMallocManaged(u_u_Y + iRhs, sizeof(complexF)*N)    );
    }
    
    *u_u_A = u_A;
    *u_u_u_A = u_u_A;
    
    for (unsigned i = 0; i < N*N; i++) {
        if (i/N == i%N) {
            u_A[i] = 1;
        } else {
            u_A[i] = 0;
        }
    }
    
    for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
        for (unsigned i = 0; i < N; i++) {
            u_u_X[iRhs][i] = i;
        }
    }
    
    ret_status_t res = interface1<N, nRhs, N/2, N/2, 2, 2>(u_u_Y, u_u_u_A, u_u_X, 1, 1);
    if (res != OK) {
        std::cout << "Bad return value" << std::endl;
        return 1;
    }
    for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
        for (unsigned i = 0; i < N; i++) {
            if (std::abs(u_u_X[iRhs][i]-u_u_Y[iRhs][i]) > 0.00001) {
                std::cout << "Bad diff" << std::endl;
                return 2;
            }
        }
    }
    
    return 0;
}
