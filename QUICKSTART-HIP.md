# CHAI HIP Quick Start Guide

This guide will help you quickly get started with the HIP implementation of CHAI benchmarks.

## Prerequisites Check

Before starting, verify you have:

```bash
# Check if ROCm is installed
rocm-smi --version

# Check if hipcc is available
which hipcc

# Check if hipify-clang is available (for porting)
which hipify-clang
```

If these commands fail, you need to install ROCm first or use the Docker container.

## Option 1: Using Docker (Easiest)

### Step 1: Build the Container

```bash
docker build -f Dockerfile.hipify -t chai-hipify:latest .
```

### Step 2: Run the Container

For conversion/porting (no GPU needed):
```bash
docker run -it --rm -v $(pwd):/workspace chai-hipify:latest
```

For testing with AMD GPU:
```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v $(pwd):/workspace \
  chai-hipify:latest
```

### Step 3: Port the Benchmarks (Inside Container)

```bash
cd /workspace
./port_to_hip.sh
```

### Step 4: Build and Test

```bash
cd HIP-D/BFS
make
./bfs -h
./bfs
```

## Option 2: Using Local ROCm Installation

### Step 1: Install ROCm

On Ubuntu 22.04:
```bash
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 jammy main' | \
  sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-hip-sdk hipify-clang
```

Add your user to the required groups:
```bash
sudo usermod -a -G render,video $USER
# Log out and log back in for changes to take effect
```

### Step 2: Set Environment Variables

Add to your `~/.bashrc`:
```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include
export ROCM_PATH=/opt/rocm
```

Then reload:
```bash
source ~/.bashrc
```

### Step 3: Port the Benchmarks

```bash
./port_to_hip.sh
```

### Step 4: Build and Test

```bash
cd HIP-D/BFS
make
./bfs -h
./bfs
```

## Manual Porting (For Understanding)

If you want to understand the porting process:

### 1. Convert a Single File

```bash
cd HIP-D/BFS
hipify-clang kernel.cu -o kernel.cpp -- -std=c++11 -x cuda
```

### 2. Update the Makefile

Change:
```makefile
CXX=nvcc
```
to:
```makefile
CXX=hipcc
```

### 3. Build

```bash
make clean
make
```

## Testing Your Setup

### Quick Test

```bash
# Test BFS benchmark
cd HIP-D/BFS
make clean && make
./bfs

# If successful, you should see output like:
# Reading input...
# Initializing...
# Running GPU kernel...
# Time: X.XX ms
```

### Test All Benchmarks

```bash
cd HIP-D
for dir in BFS BS CEDD CEDT HSTI HSTO PAD RSCD RSCT SC SSSP TQ TQH TRNS; do
    echo "Testing $dir..."
    cd $dir
    if make clean && make && ./${dir,,}; then
        echo "✓ $dir passed"
    else
        echo "✗ $dir failed"
    fi
    cd ..
done
```

## Common Issues

### Issue: "CHAI_HIP_LIB not defined"

**Solution**: Set environment variables:
```bash
export CHAI_HIP_LIB=/opt/rocm/lib
export CHAI_HIP_INC=/opt/rocm/include
```

### Issue: "hipcc: command not found"

**Solution**: Add ROCm to PATH:
```bash
export PATH=/opt/rocm/bin:$PATH
```

### Issue: "cannot find -lamdhip64"

**Solution**: Set library path:
```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Issue: "HSA error: Unable to open device"

**Solution**: Add user to correct groups:
```bash
sudo usermod -a -G render,video $USER
# Log out and log back in
```

### Issue: Docker container can't access GPU

**Solution**: Ensure you're using the correct devices:
```bash
ls /dev/kfd /dev/dri  # Should list devices
# If not, your GPU may not be supported or drivers not installed
```

## Performance Tips

### Check GPU Status
```bash
rocm-smi  # Shows GPU utilization, temperature, etc.
```

### Monitor During Execution
```bash
watch -n 1 rocm-smi
# In another terminal, run your benchmark
```

### Profile Your Application
```bash
rocprof --stats ./bfs
```

## Next Steps

1. Read the full [HIP-PORTING-PLAN.md](HIP-PORTING-PLAN.md) for detailed information
2. Review [HIP-D/README.md](HIP-D/README.md) for usage details
3. Check out the [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
4. Explore performance optimization opportunities

## Getting Help

- HIP Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/
- ROCm Documentation: https://rocm.docs.amd.com/
- AMD GPU Support: Check [ROCm hardware support](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)

## Success Checklist

- [ ] ROCm or Docker installed
- [ ] Environment variables set
- [ ] HIP-D directory created
- [ ] Benchmarks converted (if using hipify-clang)
- [ ] At least one benchmark builds successfully
- [ ] At least one benchmark runs successfully
- [ ] Output verified against CUDA version (optional)

If you've checked all boxes, congratulations! You're ready to use CHAI with HIP.
