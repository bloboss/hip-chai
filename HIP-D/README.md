# CHAI Benchmarks - HIP-D Implementation

## Overview

This directory contains the HIP (Heterogeneous-compute Interface for Portability) implementations of the CHAI benchmarks. These implementations enable execution on AMD GPUs and other alternative computing platforms while maintaining compatibility with the original CUDA functionality.

## Prerequisites

### Hardware Requirements
- AMD GPU with ROCm support (e.g., MI50, MI100, MI200 series)
- OR NVIDIA GPU (HIP can use CUDA backend)
- Minimum 8GB GPU memory recommended

### Software Requirements
- ROCm 5.0 or later
- HIP runtime and development libraries
- hipcc compiler (included with ROCm)
- Make and GCC/G++ compiler toolchain

### Installing ROCm (Ubuntu/Debian)

```bash
# Add ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 jammy main' | \
  sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install ROCm
sudo apt update
sudo apt install rocm-hip-sdk hipify-clang

# Add user to render and video groups
sudo usermod -a -G render,video $LOGNAME
```

## Quick Start

### 1. Set Environment Variables

```bash
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include
export ROCM_PATH=/opt/rocm
```

Add these to your `~/.bashrc` or `~/.profile` for persistence:

```bash
echo 'export CHAI_HIP_LIB=/opt/rocm/lib' >> ~/.bashrc
echo 'export CHAI_HIP_INC=/opt/rocm/include' >> ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
source ~/.bashrc
```

### 2. Build a Benchmark

```bash
# Navigate to a benchmark directory
cd HIP-D/BFS

# Clean and build
make clean
make

# Run the benchmark
./bfs -h        # Show help
./bfs           # Run with default parameters
```

### 3. Run All Benchmarks

```bash
# Script to build and test all benchmarks
cd HIP-D
for dir in */; do
    benchmark=$(basename "$dir")
    echo "Building ${benchmark}..."
    cd "$benchmark"
    if make clean && make; then
        echo "✓ ${benchmark} built successfully"
    else
        echo "✗ ${benchmark} build failed"
    fi
    cd ..
done
```

## Available Benchmarks

| Benchmark | Description | Status |
|-----------|-------------|--------|
| BFS | Breadth-First Search | ⏳ Ported |
| BS | Binary Search | ⏳ Ported |
| CEDD | Color Edge/Corner Detection (Dynamic) | ⏳ Ported |
| CEDT | Color Edge/Corner Detection (Threshold) | ⏳ Ported |
| HSTI | Histogram (Input-aware) | ⏳ Ported |
| HSTO | Histogram (Output-aware) | ⏳ Ported |
| PAD | Padding | ⏳ Ported |
| RSCD | Reduction Scan (Dynamic) | ⏳ Ported |
| RSCT | Reduction Scan (Threshold) | ⏳ Ported |
| SC | Scan | ⏳ Ported |
| SSSP | Single-Source Shortest Path | ⏳ Ported |
| TQ | Task Queue | ⏳ Ported |
| TQH | Task Queue with Histogram | ⏳ Ported |
| TRNS | Transpose | ⏳ Ported |

## Benchmark Usage Examples

### BFS (Breadth-First Search)
```bash
cd BFS
./bfs -f input/graph_1M.txt -s 0
```

### SSSP (Single-Source Shortest Path)
```bash
cd SSSP
./sssp -f input/graph_1M.txt -s 0
```

### HSTO (Histogram)
```bash
cd HSTO
./hsto -f input/image.txt
```

## Docker Support

A Docker container with all dependencies is available:

```bash
# Build the container
docker build -f ../Dockerfile.hipify -t chai-hipify:latest ..

# Run with GPU access
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v $(pwd)/..:/workspace \
  chai-hipify:latest

# Inside container
cd /workspace/HIP-D/BFS
make && ./bfs
```

## Differences from CUDA-D Implementation

### Code Changes
- `.cu` files converted to `.cpp` files
- CUDA API calls replaced with HIP equivalents
- `nvcc` replaced with `hipcc` compiler

### API Mapping
- `cudaMalloc` → `hipMalloc`
- `cudaMemcpy` → `hipMemcpy`
- `cudaFree` → `hipFree`
- `cudaDeviceSynchronize` → `hipDeviceSynchronize`
- `cudaGetLastError` → `hipGetLastError`

### Performance Considerations
- Wavefront size: AMD GPUs use 64-thread wavefronts (vs 32-thread warps on NVIDIA)
- Shared memory: Called LDS (Local Data Share) on AMD
- Occupancy: May differ from NVIDIA due to architectural differences

## Troubleshooting

### Build Errors

**Error**: `CHAI_HIP_LIB not defined`
```bash
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include
```

**Error**: `hipcc: command not found`
```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH
```

**Error**: `cannot find -lamdhip64`
```bash
# Ensure ROCm libraries are in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Runtime Errors

**Error**: `HSA error: Unable to open device`
```bash
# Add user to render and video groups
sudo usermod -a -G render,video $USER
# Log out and log back in
```

**Error**: Performance is lower than expected
- Check GPU clock settings with `rocm-smi`
- Verify power/performance profile: `rocm-smi --setpoweroverdrive 20`
- Monitor GPU utilization: `rocm-smi --showuse`

## Verification

To verify correctness, compare outputs with CUDA-D reference:

```bash
# Run CUDA version
cd ../CUDA-D/BFS
./bfs -f input/test.txt > ../../cuda_output.txt

# Run HIP version
cd ../../HIP-D/BFS
./bfs -f input/test.txt > ../../hip_output.txt

# Compare outputs
diff ../../cuda_output.txt ../../hip_output.txt
```

## Performance Profiling

### Using rocprof

```bash
# Profile a benchmark
rocprof --stats ./bfs

# Generate detailed trace
rocprof --timestamp on --hip-trace ./bfs
```

### Using rocm-smi

```bash
# Monitor GPU during execution
watch -n 1 rocm-smi

# Show GPU utilization
rocm-smi --showuse
```

## Known Issues and Limitations

1. **Initial Port**: This is the initial port from CUDA. Further optimization for AMD architecture is ongoing.
2. **Numerical Precision**: Minor differences in floating-point results may occur due to hardware differences.
3. **Platform Support**: Tested on AMD MI100/MI200 series. Other GPUs may require tuning.

## Contributing

When contributing improvements:
1. Maintain compatibility with the original CUDA-D behavior
2. Document any AMD-specific optimizations
3. Test on multiple GPU architectures if possible
4. Update this README with new findings

## Additional Resources

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [HIP API Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/index.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD GPU Architecture](https://www.amd.com/en/technologies/cdna)

## Citation

If you use these HIP implementations, please cite the original CHAI paper:

```bibtex
@inproceedings{chai2017,
  author = {Gómez-Luna, Juan and El Hajj, Izzat and Chang, Li-Wen and
            Garcia-Flores, Victor and Garcia de Gonzalo, Simon and
            Jablin, Thomas and Peña, Antonio J. and Hwu, Wen-mei},
  title = {Chai: Collaborative Heterogeneous Applications for Integrated-architectures},
  booktitle = {Proceedings of IEEE International Symposium on Performance Analysis
               of Systems and Software (ISPASS)},
  year = {2017}
}
```

## Support

For issues specific to the HIP implementation:
1. Check the main [HIP-PORTING-PLAN.md](../HIP-PORTING-PLAN.md) document
2. Review ROCm documentation
3. Open an issue in the repository

For general CHAI benchmark questions, refer to the main repository README.
