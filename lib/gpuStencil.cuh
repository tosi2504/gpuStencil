#pragma once

__global__ void kernel(float * res, const float * in) {
    res[threadIdx.x] = 2 * in[threadIdx.x];
}
