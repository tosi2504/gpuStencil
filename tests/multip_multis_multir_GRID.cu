#include "Accelerator.h"
#include "error.h"
#include "gpuStencil.cuh"
#include "errorcheck.h"
#include "datatypes.cuh"

#include <iostream>

constexpr unsigned N = 32;
// constexpr unsigned N = 2;
constexpr unsigned nRhs = 16;
// constexpr unsigned nRhs = 1;
constexpr unsigned nSites = 128;
// constexpr unsigned nSites = 3; // why tf for 3??? why not for 2 or 1
constexpr unsigned nPnts = 9;
// constexpr unsigned nPnts = 1;

int main () {
    acceleratorInit();

    // matrix field management
    complexF ***u_u_u_A;
    CCE(    cudaMallocManaged(&u_u_u_A, sizeof(complexF**)*nSites) );

    complexF ** h_u_Atemp = new complexF*[nPnts];
    for (unsigned p = 0; p < nPnts; p++) {
        CCE(    cudaMallocManaged(h_u_Atemp + p, sizeof(complexF)*N*N*nSites)   );
    }
    for (unsigned s = 0; s < nSites; s++) {
        CCE(    cudaMallocManaged(u_u_u_A+s, sizeof(complexF*))    );
        for (unsigned p = 0; p < nPnts; p++) {
            u_u_u_A[s][p] = &h_u_Atemp[p][((s+p) % nSites)*N*N];
        }
    }
    
    // vector field management
    complexF **u_u_X, **u_u_Y;
    CCE(    cudaMallocManaged(&u_u_X, sizeof(complexF*)*nRhs)      );
    CCE(    cudaMallocManaged(&u_u_Y, sizeof(complexF*)*nRhs)      );
    for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
        CCE(    cudaMallocManaged(u_u_X + iRhs, sizeof(complexF)*N*nSites)    );
        CCE(    cudaMallocManaged(u_u_Y + iRhs, sizeof(complexF)*N*nSites)    );
    }
    
    for (unsigned p = 0; p < nPnts; p++) {
        for (unsigned s = 0; s < nSites; s++) {
            for (unsigned i = 0; i < N*N; i++) {
                if (i/N == i%N) {
                    h_u_Atemp[p][s*N*N+i] = p+1; // CONTINUE HERE
                } else {
                    h_u_Atemp[p][s*N*N+i] = 0;
                }
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

    ret_status_t res = interface2<N, nRhs, N/2, N/2, 2, 2>(u_u_Y, u_u_u_A, u_u_X, nPnts, nSites);

    accelerator_barrier()

    if (res != OK) {
        std::cout << "Bad return value" << std::endl;
        return 1;
    }

    // check
    unsigned factor = (nPnts*(nPnts+1))/2;
    for (unsigned s = 0; s < nSites; s++) {
        for (unsigned iRhs = 0; iRhs < nRhs; iRhs++) {
            for (unsigned i = 0; i < N; i++) {
                if (std::abs(factor*u_u_X[iRhs][s*N+i]-u_u_Y[iRhs][s*N+i]) > 0.000001) {
                    printf("Bad diff -> s: %u, iRhs: %u, i: %u\n", s, iRhs, i);
                    std::cout << "X: " << u_u_X[iRhs][s*N+i] << std::endl;
                    std::cout << "Y: " << u_u_Y[iRhs][s*N+i] << std::endl;
                    return 2;
                }
            }
        }
    }

    std::cout << "Everything okay" << std::endl;

    // LETS LEAK ALL THE MEMORY YAY
    return 0;
}
