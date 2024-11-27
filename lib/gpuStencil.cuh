#pragma once

#include "datatypes.cuh"
#include "error.h"
#include "errorcheck.h"
#include "kernels.cuh"

template <
    unsigned N, unsigned nRhs,
    unsigned numRowTiles,  // in units of tiles
    unsigned numRhsTiles,  // in units of tiles
    unsigned rowTileSize, // in units of index
    unsigned rhsTileSize
> ret_status_t interface1 (
    complexF * const * d_d_Y, // d_d_Y[iRhs][s*N + i]
    const complexF * const * const * d_d_d_A, // d_d_d_A[s][p][i*N + j]
    const complexF * const * d_d_X, // d_d_X[iRhs][s*N + i]
    unsigned nPnts, unsigned nSites
) {
    const unsigned sizeShmem = sizeof(complexF)*(2*nRhs*N + N*numRowTiles*rowTileSize);
    auto func = ker_stencil2DBlocktilingV2<
        complexF, N, nRhs,
        numRowTiles, numRhsTiles,
        rowTileSize, rhsTileSize
    >;
    
    cudaError_t err = cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeShmem);
    if (err != cudaSuccess) {
        return GENERIC_ERROR;
    }
    
    func <<< nSites, numRowTiles*numRhsTiles, sizeShmem >>> 
        (d_d_Y, d_d_d_A, d_d_X, nPnts);
    CLCE();
    if (cudaGetLastError() != cudaSuccess) {
        return GENERIC_ERROR;
    }
    
    CCE(cudaDeviceSynchronize());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return GENERIC_ERROR;
    }
    
    return OK;
}

