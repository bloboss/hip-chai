# CHAI Benchmark Suite - HIP Porting Plan

## Overview
This document outlines the comprehensive plan for porting the CHAI CUDA-D benchmarks to HIP (Heterogeneous-compute Interface for Portability), enabling execution on AMD GPUs and other alternative computing platforms.

## Current Status
- ✅ HIP-D directory created (copied from CUDA-D)
- ✅ Docker container with hipify-clang utility configured
- ⏳ CUDA to HIP conversion pending
- ⏳ Build system adaptation pending
- ⏳ Testing and validation pending

## Benchmarks to Port
The following benchmarks need to be ported from CUDA to HIP:

1. **BFS** - Breadth-First Search
2. **BS** - Binary Search
3. **CEDD** - Color-based Edge and Corner Detection (Dynamic)
4. **CEDT** - Color-based Edge and Corner Detection (Threshold)
5. **HSTI** - Histogram (Input-aware)
6. **HSTO** - Histogram (Output-aware)
7. **PAD** - Padding
8. **RSCD** - Reduction Scan (Dynamic)
9. **RSCT** - Reduction Scan (Threshold)
10. **SC** - Scan
11. **SSSP** - Single-Source Shortest Path
12. **TQ** - Task Queue
13. **TQH** - Task Queue with Histogram
14. **TRNS** - Transpose

## Phase 1: Environment Setup

### Docker Container Usage

Build the Docker container:
```bash
docker build -f Dockerfile.hipify -t chai-hipify:latest .
```

Run the container:
```bash
docker run -it --rm -v $(pwd):/workspace chai-hipify:latest
```

For AMD GPU access (when testing):
```bash
docker run -it --rm --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  -v $(pwd):/workspace chai-hipify:latest
```

## Phase 2: Automated CUDA to HIP Conversion

### Using hipify-clang

The `hipify-clang` tool automatically converts CUDA source code to HIP. Key conversion mappings:

#### CUDA → HIP API Mappings
- `cuda` → `hip`
- `CUDA` → `HIP`
- `cudaError_t` → `hipError_t`
- `cudaSuccess` → `hipSuccess`
- `cudaMalloc` → `hipMalloc`
- `cudaMemcpy` → `hipMemcpy`
- `cudaFree` → `hipFree`
- `cudaDeviceSynchronize` → `hipDeviceSynchronize`
- `cudaGetLastError` → `hipGetLastError`
- `cudaMemcpyHostToDevice` → `hipMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost` → `hipMemcpyDeviceToHost`
- `__global__` → `__global__` (unchanged)
- `__device__` → `__device__` (unchanged)
- `__shared__` → `__shared__` (unchanged)
- `__syncthreads()` → `__syncthreads()` (unchanged)

### Conversion Process

#### Option 1: Using the helper script
```bash
# Inside Docker container
cd /workspace
./usr/local/bin/hipify_directory.sh CUDA-D HIP-D
```

#### Option 2: Manual conversion per benchmark
```bash
# Example for BFS benchmark
cd HIP-D/BFS
hipify-clang kernel.cu -o kernel.cpp -- -std=c++11 -x cuda

# For each .cu file in each benchmark
for dir in HIP-D/*/; do
    cd "$dir"
    for cufile in *.cu; do
        if [ -f "$cufile" ]; then
            hipify-clang "$cufile" -o "${cufile%.cu}.cpp" -- -std=c++11 -x cuda
        fi
    done
    cd /workspace
done
```

## Phase 3: Build System Modifications

Each benchmark's Makefile needs to be updated:

### Changes Required:

1. **Compiler Change**
   ```makefile
   # Old (CUDA)
   CXX=nvcc

   # New (HIP)
   CXX=hipcc
   ```

2. **Environment Variables**
   ```makefile
   # Old (CUDA)
   ifndef CHAI_CUDA_LIB
       $(error CHAI_CUDA_LIB not defined...)
   endif
   LIB=-L/usr/lib/ -L$(CHAI_CUDA_LIB) -lm

   # New (HIP)
   ifndef CHAI_HIP_LIB
       $(error CHAI_HIP_LIB not defined. This environment variable must be defined to point to the location of the HIP library)
   endif
   LIB=-L/usr/lib/ -L$(CHAI_HIP_LIB) -L$(ROCM_PATH)/lib -lm -lamdhip64
   ```

3. **Include Paths**
   ```makefile
   # Old (CUDA)
   ifndef CHAI_CUDA_INC
       $(error CHAI_CUDA_INC not defined...)
   endif
   INC=-I$(CHAI_CUDA_INC)

   # New (HIP)
   ifndef CHAI_HIP_INC
       $(error CHAI_HIP_INC not defined. This environment variable must be defined to point to the location of the HIP header files)
   endif
   INC=-I$(CHAI_HIP_INC) -I$(ROCM_PATH)/include
   ```

4. **Source File Extensions**
   ```makefile
   # Old (CUDA)
   SRC=main.cpp kernel.cpp kernel.cu

   # New (HIP)
   SRC=main.cpp kernel.cpp kernel.cpp  # .cu converted to .cpp
   ```

5. **Compiler Flags**
   ```makefile
   # Keep existing C++11 standard
   CXX_FLAGS=-std=c++11

   # Optional: Add HIP-specific flags if needed
   CXX_FLAGS=-std=c++11 --offload-arch=gfx90a  # Example for MI200 series
   ```

## Phase 4: Header File Updates

### Support Headers Modifications

1. **cuda-setup.h → hip-setup.h**
   - Replace CUDA device management with HIP equivalents
   - Update error checking macros

2. **common.h**
   - Update preprocessor directives:
     ```cpp
     // Old
     #define _CUDA_COMPILER_

     // New
     #define _HIP_COMPILER_
     ```

3. **Include Guards**
   - Ensure proper HIP headers are included:
     ```cpp
     #ifdef _HIP_COMPILER_
     #include <hip/hip_runtime.h>
     #endif
     ```

## Phase 5: Code-Level Adjustments

### Areas Requiring Manual Review

1. **Atomic Operations**
   - Most atomic operations translate directly
   - Verify `atomicAdd`, `atomicMax`, `atomicExch` behavior

2. **Shared Memory**
   - Dynamic shared memory allocation syntax remains the same
   - Verify bank conflict patterns if optimized for NVIDIA

3. **Launch Configuration**
   - Grid and block dimensions translate directly
   - Consider AMD-specific occupancy optimization later

4. **Warp/Wavefront Size**
   - NVIDIA: 32 threads per warp
   - AMD: 64 threads per wavefront
   - Review any warp-specific optimizations

5. **Synchronization**
   - `__syncthreads()` translates directly
   - Verify memory fence semantics if used

6. **Error Handling**
   - Update error checking macros for HIP

## Phase 6: Testing and Validation

### Build Testing
```bash
# Set environment variables
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include
export ROCM_PATH=/opt/rocm

# Build each benchmark
cd HIP-D/BFS
make clean && make

# Test execution
./bfs -h
./bfs  # Run with default parameters
```

### Validation Steps
1. **Compilation**: Ensure all benchmarks compile without errors
2. **Execution**: Verify benchmarks run without runtime errors
3. **Correctness**: Compare output with CUDA-D reference results
4. **Performance**: Baseline performance measurements (optional for initial port)

### Test Matrix
- [ ] BFS - Build, Run, Verify
- [ ] BS - Build, Run, Verify
- [ ] CEDD - Build, Run, Verify
- [ ] CEDT - Build, Run, Verify
- [ ] HSTI - Build, Run, Verify
- [ ] HSTO - Build, Run, Verify
- [ ] PAD - Build, Run, Verify
- [ ] RSCD - Build, Run, Verify
- [ ] RSCT - Build, Run, Verify
- [ ] SC - Build, Run, Verify
- [ ] SSSP - Build, Run, Verify
- [ ] TQ - Build, Run, Verify
- [ ] TQH - Build, Run, Verify
- [ ] TRNS - Build, Run, Verify

## Phase 7: Documentation Updates

### Update README.md
Add HIP-D instructions section:
```markdown
## HIP-D Implementation

Export environment variables:
  ```
  export CHAI_HIP_LIB=/opt/rocm/lib
  export CHAI_HIP_INC=/opt/rocm/include
  export ROCM_PATH=/opt/rocm
  ```

Select HIP implementation:
  ```
  cd HIP-D
  ```

Select desired benchmark and compile:
  ```
  cd BFS
  make
  ./bfs
  ```
```

### Create HIP-D/README.md
Document HIP-specific considerations, dependencies, and tested platforms.

## Phase 8: Optimization (Future Work)

After successful porting, consider AMD-specific optimizations:

1. **Wavefront-aware optimizations** (64 threads vs 32)
2. **LDS (Local Data Share) optimization** - AMD's shared memory
3. **GCN/CDNA architecture-specific tuning**
4. **ROCm profiler (rocprof) analysis**
5. **Memory coalescing patterns for AMD architecture**

## Key Differences: CUDA vs HIP

### Compatibility
- HIP code can run on both AMD (via ROCm) and NVIDIA (via CUDA backend)
- Most CUDA code ports with minimal changes
- Performance characteristics may differ between platforms

### Development Tools
- **CUDA**: nvcc, cuda-gdb, nvprof/Nsight
- **HIP**: hipcc, rocgdb, rocprof

### Memory Model
- Largely compatible
- Minor differences in cache hierarchy

### Compute Capabilities
- NVIDIA: Compute Capability (e.g., 8.0 for A100)
- AMD: GFX architecture (e.g., gfx908 for MI100, gfx90a for MI200)

## Expected Challenges

1. **Platform-specific optimizations**: Code optimized for NVIDIA may need tuning for AMD
2. **Library dependencies**: Ensure any CUDA-specific libraries have HIP equivalents
3. **Precision differences**: Minor numerical differences possible due to hardware
4. **Build system complexity**: Managing dual CUDA/HIP builds if needed

## Success Criteria

- ✅ All 14 benchmarks compile successfully with hipcc
- ✅ All benchmarks execute without runtime errors
- ✅ Output correctness verified against CUDA-D reference
- ✅ Documentation updated
- ✅ Docker container functional for automated conversion

## Automation Script

A complete automation script (`port_to_hip.sh`) should be created to:
1. Run hipify-clang on all .cu files
2. Update all Makefiles
3. Update header files
4. Run test builds
5. Generate report

## Timeline Estimate

- Phase 1-2 (Setup & Conversion): 1-2 days
- Phase 3-4 (Build & Headers): 2-3 days
- Phase 5 (Code Review): 3-5 days
- Phase 6 (Testing): 3-5 days
- Phase 7 (Documentation): 1-2 days

**Total Estimated Time**: 2-3 weeks for complete port and validation

## References

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [hipify-clang Documentation](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/)
- [HIP API Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/index.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
