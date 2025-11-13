# ROCm Testing Environment Setup Guide

Complete guide for setting up AMD ROCm environment for testing CHAI HIP benchmarks.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Method 1: Docker (Recommended)](#method-1-docker-recommended)
  - [Method 2: Native ROCm Installation](#method-2-native-rocm-installation)
  - [Method 3: Cloud GPU Instances](#method-3-cloud-gpu-instances)
- [Verification](#verification)
- [Running Benchmarks](#running-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [Additional Resources](#additional-resources)

---

## Overview

This guide covers setting up an AMD ROCm environment for:
- Compiling CHAI HIP benchmarks
- Running performance tests
- Debugging HIP applications
- Profiling GPU workloads

**Recommended Path**: Use Docker containers (no GPU hardware required for compilation testing)

**Production Path**: Native ROCm installation on AMD GPU hardware

---

## System Requirements

### Minimum Requirements (Compilation Only)

- **OS**: Ubuntu 20.04/22.04, RHEL 8/9, or SLES 15
- **CPU**: x86-64 processor with SSE4.2 support
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space for ROCm + build artifacts
- **Software**: Docker 20.10+ or native ROCm 5.7+

### Recommended Requirements (GPU Execution)

- **OS**: Ubuntu 22.04 LTS (best supported)
- **GPU**: AMD Radeon Instinct MI50/MI100/MI200 series or Radeon RX 6000/7000 series
- **CPU**: Modern x86-64 multi-core processor
- **RAM**: 32GB+ for large benchmarks
- **Disk**: 50GB+ free space
- **Kernel**: Linux kernel 5.15+ with AMDGPU driver support

### Supported AMD GPUs

**Data Center GPUs** (Instinct):
- MI300 series (CDNA3) - ROCm 6.0+
- MI200 series (CDNA2) - ROCm 5.0+
- MI100 (CDNA1) - ROCm 4.0+
- MI50, MI60 (GCN5) - ROCm 3.0+

**Consumer GPUs** (Radeon):
- RX 7900 XT/XTX (RDNA3) - ROCm 5.7+
- RX 6000 series (RDNA2) - ROCm 5.0+
- Radeon VII (GCN5) - ROCm 3.0+

**Compatibility Check**:
```bash
# Check if your GPU is supported
lspci | grep -i vga
# Look for AMD/ATI entries
```

---

## Installation Methods

### Method 1: Docker (Recommended)

**Advantages**:
- ✅ No system-wide ROCm installation required
- ✅ Isolated environment, no dependency conflicts
- ✅ Works on any Linux distribution
- ✅ Easy to switch between ROCm versions
- ✅ Compilation testing without GPU hardware

**Disadvantages**:
- ❌ Requires Docker
- ❌ GPU access requires additional configuration
- ❌ Slight performance overhead

#### Step 1: Install Docker

**Ubuntu/Debian**:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world
```

**RHEL/CentOS**:
```bash
# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### Step 2: Build CHAI Docker Images

```bash
# Clone the repository
git clone https://github.com/yourusername/hip-chai.git
cd hip-chai

# Build test container (for compilation)
./scripts/docker-build.sh test

# Build all containers (dev, test, runtime)
./scripts/docker-build.sh all

# Verify images
docker images | grep chai
```

#### Step 3: Configure GPU Access (Optional)

For GPU execution, Docker needs access to AMD GPU devices:

```bash
# Check GPU devices
ls -la /dev/kfd /dev/dri/

# Ensure current user has access
sudo usermod -aG video $USER
sudo usermod -aG render $USER
newgrp video
```

Test GPU access:
```bash
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --group-add render \
    chai-test:latest \
    rocm-smi
```

#### Step 4: Run Validation

```bash
# Quick validation (no builds)
./validate.sh --quick

# Full validation (builds + compilation)
./validate.sh --full

# With GPU testing
./validate.sh --full --gpu
```

---

### Method 2: Native ROCm Installation

**Advantages**:
- ✅ Best performance (no containerization overhead)
- ✅ Full system integration
- ✅ Easier debugging with system tools

**Disadvantages**:
- ❌ Complex installation
- ❌ Potential dependency conflicts
- ❌ Distribution-specific
- ❌ Requires AMD GPU hardware

#### Ubuntu 22.04 Installation

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm,hip,hiplibsdk

# Install development tools
sudo apt install -y \
    rocm-dev \
    rocm-libs \
    hipcc \
    hip-dev \
    rocm-utils \
    rocm-cmake \
    rocprofiler-dev \
    roctracer-dev

# Add ROCm to PATH
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
echo 'export HIP_PATH=/opt/rocm/hip' >> ~/.bashrc
source ~/.bashrc

# Add user to video/render groups
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Reboot required for GPU access
sudo reboot
```

#### Ubuntu 20.04 Installation

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/focal/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb

# Install ROCm (same as above)
sudo amdgpu-install --usecase=rocm,hip,hiplibsdk

# Continue with same steps as Ubuntu 22.04
```

#### RHEL/CentOS 8/9 Installation

```bash
# Add ROCm repository
sudo tee /etc/yum.repos.d/amdgpu.repo <<EOF
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/6.0/rhel/8.9/main/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel8/6.0/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

# Install ROCm
sudo yum install -y rocm-dev rocm-utils hip-dev hipcc

# Set up environment
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
source ~/.bashrc

# Add user to video group
sudo usermod -aG video $USER
sudo reboot
```

#### Post-Installation Configuration

```bash
# Verify installation
rocm-smi
hipcc --version
hip-config --full

# Check GPU detection
rocminfo | grep "Name:" | head -5

# Test HIP functionality
cat > test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " HIP device(s)" << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }
    return 0;
}
EOF

hipcc test_hip.cpp -o test_hip
./test_hip
```

Expected output:
```
Found 1 HIP device(s)
Device 0: AMD Radeon RX 7900 XTX
```

#### Build CHAI Benchmarks Natively

```bash
# Clone repository
git clone https://github.com/yourusername/hip-chai.git
cd hip-chai

# Set environment variables
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include

# Build a single benchmark
cd HIP-D/BFS
make -f Makefile.hip

# Build all benchmarks
cd ../..
for benchmark in HIP-D/*/; do
    echo "Building $(basename $benchmark)..."
    cd "$benchmark"
    make -f Makefile.hip clean
    make -f Makefile.hip
    cd ../..
done
```

---

### Method 3: Cloud GPU Instances

For testing without local AMD GPU hardware.

#### AWS EC2 (AMD GPUs)

AWS doesn't currently offer AMD GPU instances. Consider:
- Use Docker method on any EC2 instance (compilation only)
- Use on-premises or data center AMD GPUs

#### Google Cloud Platform (AMD GPUs)

Limited AMD GPU availability. Best approach:
- Use Docker for compilation testing
- Contact AMD for cloud partner options

#### Azure (AMD GPUs)

Azure has limited AMD GPU instances:

```bash
# Launch NVv4 series (AMD Radeon Instinct MI25)
az vm create \
    --resource-group myResourceGroup \
    --name myAMDVM \
    --size Standard_NV4as_v4 \
    --image UbuntuLTS \
    --admin-username azureuser \
    --generate-ssh-keys

# SSH into instance
ssh azureuser@<PUBLIC_IP>

# Follow native installation steps above
```

#### AMD Cloud Partners

Contact AMD for access to:
- **Penguin Computing**: MI200-based cloud instances
- **Nimbix**: AMD GPU cloud access
- **Oracle Cloud**: MI100 instances

---

## Verification

### Step 1: Verify ROCm Installation

```bash
# Check ROCm version
rocm-smi --showproductname
rocm-smi --showdriverversion

# Verify hipcc
hipcc --version
which hipcc

# Check environment
echo $ROCM_PATH
echo $HIP_PATH
```

### Step 2: Verify GPU Detection

```bash
# List GPUs
rocm-smi

# Detailed GPU info
rocminfo

# Check GPU topology
rocm-smi --showtopo
```

### Step 3: Test HIP Compilation

```bash
# Create simple test
cat > simple_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    printf("HIP SAXPY test\n");
    return 0;
}
EOF

# Compile
hipcc simple_hip.cpp -o simple_hip

# Run
./simple_hip
```

### Step 4: Run CHAI Validation

```bash
# Using Docker
cd hip-chai
./validate.sh --full

# Native installation
cd hip-chai
./validate.sh --full --gpu
```

---

## Running Benchmarks

### Using Docker

```bash
# Compile all benchmarks
./scripts/docker-test.sh compile

# Run smoke tests (no GPU required)
./scripts/docker-test.sh smoke

# Run with GPU
docker-compose run --rm chai-runtime bash
# Inside container:
cd /workspace/HIP-D/BFS
./bfs -n 100 -t 256
```

### Native Execution

```bash
# Run BFS benchmark
cd HIP-D/BFS
./bfs -h  # Show help
./bfs -n 100 -t 256  # Run with 100 iterations, 256 threads

# Run SSSP benchmark
cd ../SSSP
./sssp -n 50 -t 512

# Run all benchmarks with default parameters
for benchmark in HIP-D/*/; do
    name=$(basename "$benchmark")
    echo "Running $name..."
    cd "$benchmark"
    ./"${name,,}" -n 10 -t 256 2>&1 | tee "results-${name}.log"
    cd ../..
done
```

### Batch Execution

```bash
# Create benchmark runner script
cat > run_all_benchmarks.sh << 'EOF'
#!/bin/bash

RESULTS_DIR="benchmark-results-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

for benchmark in BFS BS CEDD CEDT HSTI HSTO PAD RSCD RSCT SC SSSP TQ TQH TRNS; do
    echo "=== Running $benchmark ==="
    cd "HIP-D/$benchmark"

    binary=$(find . -maxdepth 1 -type f -executable | head -1)
    if [ -n "$binary" ]; then
        $binary -n 100 -t 256 2>&1 | tee "../../$RESULTS_DIR/$benchmark.log"
    fi

    cd ../..
done

echo "Results saved to $RESULTS_DIR/"
EOF

chmod +x run_all_benchmarks.sh
./run_all_benchmarks.sh
```

---

## Troubleshooting

### ROCm Not Detecting GPU

**Symptoms**: `rocm-smi` shows no GPUs

**Solutions**:
```bash
# Check kernel module
lsmod | grep amdgpu

# If not loaded, load manually
sudo modprobe amdgpu

# Check dmesg for errors
dmesg | grep -i amdgpu | tail -20

# Reinstall AMDGPU driver
sudo amdgpu-install --usecase=rocm --uninstall
sudo amdgpu-install --usecase=rocm
sudo reboot
```

### Permission Denied Errors

**Symptoms**: Cannot access `/dev/kfd` or `/dev/dri`

**Solutions**:
```bash
# Check device permissions
ls -la /dev/kfd /dev/dri/

# Add user to groups
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Set permissions (if needed)
sudo chmod 666 /dev/kfd
sudo chmod -R 755 /dev/dri

# Logout and login (or reboot)
```

### hipcc Not Found

**Symptoms**: `hipcc: command not found`

**Solutions**:
```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm

# Make permanent
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
source ~/.bashrc

# Verify
which hipcc
hipcc --version
```

### Compilation Errors

**Symptoms**: Errors during `make -f Makefile.hip`

**Solutions**:
```bash
# Check HIP installation
hip-config --full

# Verify include paths
ls -la /opt/rocm/include/hip/

# Check library paths
ls -la /opt/rocm/lib/

# Set variables explicitly
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include

# Try verbose build
make -f Makefile.hip VERBOSE=1
```

### Runtime Errors

**Symptoms**: Binary runs but GPU errors occur

**Solutions**:
```bash
# Enable HIP debugging
export HIP_VISIBLE_DEVICES=0
export AMD_LOG_LEVEL=3
export HSA_ENABLE_SDMA=0

# Check GPU status
rocm-smi
rocm-smi --showmeminfo vram

# Reset GPU if needed
sudo rocm-smi --gpureset

# Check for compute process
rocm-smi --showpids
```

### Docker GPU Access Issues

**Symptoms**: GPU not accessible in Docker container

**Solutions**:
```bash
# Verify host GPU access
rocm-smi

# Check device ownership
ls -la /dev/kfd /dev/dri/

# Run with all necessary flags
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --group-add render \
    -v /opt/rocm:/opt/rocm:ro \
    chai-test:latest \
    rocm-smi

# If still failing, try privileged mode (less secure)
docker run --rm --privileged chai-test:latest rocm-smi
```

---

## Performance Tuning

### GPU Clock Settings

```bash
# Check current clocks
rocm-smi --showclocks

# Set performance mode
sudo rocm-smi --setperflevel high

# Set specific clock speeds (advanced)
sudo rocm-smi --setsclk 7  # Set to highest SCLK level
```

### Memory Settings

```bash
# Check memory usage
rocm-smi --showmeminfo vram

# Set memory clock
sudo rocm-smi --setmclk 3  # Set to highest MCLK level
```

### Environment Optimization

```bash
# Enable performance counters
export HSA_ENABLE_SDMA=1

# Disable CPU affinity (can help performance)
export HIP_FORCE_DEV_KERNARG=0

# Set number of compute units
export HIP_VISIBLE_DEVICES=0

# Optimize for specific GPU
export HSA_OVERRIDE_GFX_VERSION=9.0.0  # For MI100
# or
export HSA_OVERRIDE_GFX_VERSION=9.0.6  # For MI200
```

### Profiling

```bash
# Profile with rocprof
rocprof --stats ./bfs -n 100 -t 256

# Detailed profiling
rocprof --timestamp on --hip-trace ./bfs -n 100 -t 256

# View results
rocprof --input results.csv --stats

# Use rocprofiler for advanced profiling
rocprofiler --kernel-trace --hip-trace ./bfs -n 100 -t 256
```

---

## Additional Resources

### Official Documentation

- **ROCm Documentation**: https://rocm.docs.amd.com/
- **HIP Programming Guide**: https://rocm.docs.amd.com/projects/HIP/
- **ROCm Installation Guide**: https://rocm.docs.amd.com/en/latest/deploy/linux/
- **hipcc Compiler Guide**: https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/hipcc.html

### Tutorials and Examples

- **HIP Examples**: https://github.com/ROCm-Developer-Tools/HIP-Examples
- **ROCm Examples**: https://github.com/amd/rocm-examples
- **CHAI Benchmarks**: https://github.com/chai-benchmarks/chai

### Community

- **ROCm GitHub**: https://github.com/RadeonOpenCompute/ROCm
- **ROCm Issues**: https://github.com/RadeonOpenCompute/ROCm/issues
- **AMD GPUOpen**: https://gpuopen.com/
- **ROCm Developer Forum**: https://community.amd.com/t5/rocm/ct-p/amd-rocm

### CHAI HIP Specific

- **HIP-D Benchmarks**: See individual README files in `HIP-D/*/`
- **Docker Guide**: [DOCKER-GUIDE.md](DOCKER-GUIDE.md)
- **HIP Porting Plan**: [HIP-PORTING-PLAN.md](HIP-PORTING-PLAN.md)
- **Conversion Summary**: [HIP-CONVERSION-SUMMARY.md](HIP-CONVERSION-SUMMARY.md)
- **Quick Start**: [QUICKSTART-HIP.md](QUICKSTART-HIP.md)

### Debugging Tools

- **rocgdb**: ROCm debugger
  ```bash
  rocgdb ./bfs
  ```

- **roctracer**: HIP/HSA tracing
  ```bash
  export ROCTRACER_LOG=1
  ./bfs -n 10 -t 256
  ```

- **roofline**: Performance analysis
  ```bash
  rocprof --hsa-trace ./bfs -n 100 -t 256
  ```

---

## Quick Reference

### Essential Commands

```bash
# Check ROCm version
rocm-smi --showdriverversion

# List GPUs
rocm-smi

# Compile HIP code
hipcc source.cpp -o binary

# Profile application
rocprof --stats ./binary

# Debug application
rocgdb ./binary

# Check GPU utilization
watch -n 1 rocm-smi
```

### Environment Variables

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export PATH=/opt/rocm/bin:$PATH
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include
export HIP_VISIBLE_DEVICES=0
```

### Common Issues Quick Fix

```bash
# GPU not found
sudo modprobe amdgpu && sudo usermod -aG video $USER

# hipcc not found
export PATH=/opt/rocm/bin:$PATH

# Permission denied
sudo usermod -aG video,render $USER && newgrp video

# Reset GPU
sudo rocm-smi --gpureset
```

---

**Last Updated**: 2025-01-XX
**ROCm Version**: 6.0
**Maintainer**: CHAI HIP Development Team
