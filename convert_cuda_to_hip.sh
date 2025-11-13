#!/bin/bash
# Script to convert CUDA API calls to HIP in a given file

if [ $# -eq 0 ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    echo "File $FILE not found!"
    exit 1
fi

# Create backup
cp "$FILE" "${FILE}.bak"

# Perform replacements
sed -i 's/cudaError_t/hipError_t/g' "$FILE"
sed -i 's/cudaSuccess/hipSuccess/g' "$FILE"
sed -i 's/cudaMalloc/hipMalloc/g' "$FILE"
sed -i 's/cudaMemcpy/hipMemcpy/g' "$FILE"
sed -i 's/cudaMemcpyHostToDevice/hipMemcpyHostToDevice/g' "$FILE"
sed -i 's/cudaMemcpyDeviceToHost/hipMemcpyDeviceToHost/g' "$FILE"
sed -i 's/cudaMemcpyDeviceToDevice/hipMemcpyDeviceToDevice/g' "$FILE"
sed -i 's/cudaFree/hipFree/g' "$FILE"
sed -i 's/cudaDeviceSynchronize/hipDeviceSynchronize/g' "$FILE"
sed -i 's/cudaGetLastError/hipGetLastError/g' "$FILE"
sed -i 's/cudaGetErrorString/hipGetErrorString/g' "$FILE"
sed -i 's/cudaSetDevice/hipSetDevice/g' "$FILE"
sed -i 's/cudaGetDeviceProperties/hipGetDeviceProperties/g' "$FILE"
sed -i 's/cudaDeviceProp/hipDeviceProp_t/g' "$FILE"
sed -i 's/cudaMemset/hipMemset/g' "$FILE"
sed -i 's/"support\/cuda-setup\.h"/"support\/hip-setup.h"/g' "$FILE"
sed -i 's/#include <cuda\.h>/#include <hip\/hip_runtime.h>/g' "$FILE"
sed -i 's/#include <cuda_runtime\.h>/#include <hip\/hip_runtime.h>/g' "$FILE"
sed -i 's/"cuda_runtime\.h"/<hip\/hip_runtime.h>/g' "$FILE"
sed -i 's/#define _CUDA_COMPILER_/#define _HIP_COMPILER_/g' "$FILE"

echo "Converted $FILE to HIP"
