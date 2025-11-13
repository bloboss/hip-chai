# CHAI HIP Docker Guide

Complete guide for using Docker containers to build, test, and run CHAI HIP benchmarks.

## Table of Contents

- [Quick Start](#quick-start)
- [Container Types](#container-types)
- [Building Images](#building-images)
- [Running Containers](#running-containers)
- [Testing](#testing)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Docker 20.10+ installed
- Docker Compose 1.29+ (optional but recommended)
- 10GB+ free disk space for images
- AMD GPU (optional, for execution testing)

### Fastest Path to Testing

```bash
# Clone the repository
git clone https://github.com/your-org/hip-chai.git
cd hip-chai

# Build test container
./scripts/docker-build.sh test

# Run compilation test
./scripts/docker-test.sh compile

# If successful, all 14 benchmarks compiled!
```

---

## Container Types

### 1. Development Container (`chai-hipify`)

**Purpose:** CUDA to HIP conversion and development

**Dockerfile:** `Dockerfile.hipify`

**Features:**
- hipify-clang for CUDA→HIP conversion
- ROCm development tools
- Git, vim, development utilities

**Use Cases:**
- Converting CUDA code to HIP
- Exploring the codebase
- Development work

**Build:**
```bash
docker build -f Dockerfile.hipify -t chai-hipify:latest .
```

**Run:**
```bash
docker run -it --rm -v $(pwd):/workspace chai-hipify:latest
```

### 2. Test Container (`chai-test`)

**Purpose:** Compilation and testing

**Dockerfile:** `Dockerfile.test`

**Features:**
- Full ROCm development suite
- hipcc compiler
- Testing utilities (pytest, benchmark tools)
- Pre-configured build scripts
- Profiling tools (rocprof)

**Use Cases:**
- Compiling benchmarks
- Running smoke tests
- CI/CD pipelines
- Development with full toolchain

**Build:**
```bash
docker build -f Dockerfile.test -t chai-test:latest .
```

**Run:**
```bash
docker run -it --rm -v $(pwd):/workspace chai-test:latest
# Inside container:
compile_all.sh
smoke_tests.sh
```

### 3. Runtime Container (`chai-runtime`)

**Purpose:** Production execution of compiled benchmarks

**Dockerfile:** `Dockerfile.rocm` (multi-stage)

**Features:**
- Minimal runtime environment
- Pre-compiled benchmarks
- ROCm runtime only (no dev tools)
- Optimized for size

**Use Cases:**
- Running benchmarks on AMD GPUs
- Production deployments
- Performance testing

**Build:**
```bash
docker build -f Dockerfile.rocm --target runtime -t chai-runtime:latest .
```

**Run (requires AMD GPU):**
```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  chai-runtime:latest
# Inside container:
run_tests.sh
```

### 4. Builder Container (`chai-builder`)

**Purpose:** Build stage for multi-stage builds

**Dockerfile:** `Dockerfile.rocm` (builder stage)

**Features:**
- Compilation environment
- All benchmarks compiled
- Intermediate build artifacts

**Use Cases:**
- Part of multi-stage builds
- Extracting compiled binaries
- CI/CD build stage

---

## Building Images

### Using Build Script (Recommended)

```bash
# Build all images
./scripts/docker-build.sh all

# Build specific image
./scripts/docker-build.sh dev      # Development only
./scripts/docker-build.sh test     # Testing only
./scripts/docker-build.sh runtime  # Runtime only

# Specify ROCm version
./scripts/docker-build.sh all 5.7
```

### Using Docker Compose

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build chai-test
docker-compose build chai-runtime
```

### Manual Build

```bash
# Development container
docker build -f Dockerfile.hipify -t chai-hipify:latest .

# Test container
docker build -f Dockerfile.test -t chai-test:latest .

# Runtime container (multi-stage)
docker build -f Dockerfile.rocm -t chai-runtime:latest .

# With specific ROCm version
docker build --build-arg ROCM_VERSION=5.7 -f Dockerfile.test -t chai-test:5.7 .
```

---

## Running Containers

### Interactive Development

```bash
# Start development container
docker-compose up -d chai-dev
docker exec -it chai-hip-dev bash

# Or directly with docker run
docker run -it --rm -v $(pwd):/workspace chai-hipify:latest
```

### Compilation Testing

```bash
# Start test container
docker run -it --rm -v $(pwd):/workspace chai-test:latest

# Inside container, compile all benchmarks
compile_all.sh

# Or compile specific benchmark
cd HIP-D/BFS
make -f Makefile.hip
./bfs -h
```

### Running with AMD GPU

```bash
# Check GPU availability
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  chai-test:latest \
  rocm-smi

# Run benchmarks with GPU
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v $(pwd):/workspace \
  chai-test:latest

# Inside container
cd HIP-D/BFS
./bfs -f input/NYR_input.dat
```

### Using Docker Compose

```bash
# Start test environment
docker-compose up -d chai-test
docker exec -it chai-hip-test bash

# Run CI tests
docker-compose run chai-ci

# With GPU (uncomment devices in docker-compose.yml first)
docker-compose up -d chai-runtime
```

---

## Testing

### Automated Testing Scripts

#### Compilation Test

```bash
./scripts/docker-test.sh compile
```

This will:
1. Build the test container
2. Compile all 14 benchmarks
3. Report success/failure for each

#### Smoke Tests

```bash
./scripts/docker-test.sh smoke
```

This will:
1. Check if binaries exist
2. Test help command for each benchmark
3. Report results

#### Full CI Test

```bash
./scripts/docker-test.sh ci
```

Runs complete CI/CD test suite.

#### Verify Images

```bash
./scripts/docker-test.sh verify
```

Checks that all required Docker images are built.

### Manual Testing

#### Inside Container

```bash
# Start container
docker run -it --rm -v $(pwd):/workspace chai-test:latest

# Compile a benchmark
cd HIP-D/BFS
make -f Makefile.hip clean
make -f Makefile.hip

# Test execution
./bfs -h
./bfs -f input/NYR_input.dat

# Profile with rocprof (if GPU available)
profile_benchmark.sh BFS
```

#### Testing All Benchmarks

```bash
# Inside container
compile_all.sh    # Compile all
smoke_tests.sh    # Quick tests
run_tests.sh      # Full tests (GPU required)
```

---

## CI/CD Integration

### GitHub Actions

Example workflow (`.github/workflows/rocm-test.yml`):

```yaml
name: ROCm Compilation Test

on: [push, pull_request]

jobs:
  compile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build test container
        run: docker build -f Dockerfile.test -t chai-test:latest .

      - name: Compile benchmarks
        run: |
          docker run --rm -v $(pwd):/workspace chai-test:latest compile_all.sh

      - name: Smoke tests
        run: |
          docker run --rm -v $(pwd):/workspace chai-test:latest smoke_tests.sh
```

### GitLab CI

Example `.gitlab-ci.yml`:

```yaml
image: docker:latest

services:
  - docker:dind

stages:
  - build
  - test

build:
  stage: build
  script:
    - docker build -f Dockerfile.test -t chai-test:latest .

test:
  stage: test
  script:
    - docker run --rm -v $(pwd):/workspace chai-test:latest compile_all.sh
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'docker build -f Dockerfile.test -t chai-test:latest .'
            }
        }
        stage('Test') {
            steps {
                sh 'docker run --rm -v $(pwd):/workspace chai-test:latest compile_all.sh'
            }
        }
    }
}
```

---

## Troubleshooting

### Image Build Failures

**Problem:** ROCm repository keys fail to import

```bash
# Solution: Clear Docker cache
docker builder prune -a
docker build --no-cache -f Dockerfile.test -t chai-test:latest .
```

**Problem:** Out of disk space

```bash
# Check Docker disk usage
docker system df

# Clean up
docker system prune -a
docker volume prune
```

### Compilation Failures

**Problem:** `hipcc: command not found`

```bash
# Verify ROCm installation in container
docker run --rm chai-test:latest which hipcc
docker run --rm chai-test:latest hipcc --version

# Check environment variables
docker run --rm chai-test:latest env | grep ROCM
```

**Problem:** Include files not found

```bash
# Check CHAI_HIP_INC is set correctly
docker run --rm chai-test:latest bash -c 'echo $CHAI_HIP_INC; ls -la $CHAI_HIP_INC'

# Verify ROCm headers
docker run --rm chai-test:latest ls -la /opt/rocm/include
```

### GPU Access Issues

**Problem:** GPU not detected in container

```bash
# Check host GPU
rocm-smi

# Verify device nodes
ls -la /dev/kfd /dev/dri

# Check user groups
groups

# Add user to video/render groups
sudo usermod -a -G video,render $USER
# Log out and back in
```

**Problem:** Permission denied for GPU devices

```bash
# Run with correct devices and security options
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  chai-test:latest \
  rocm-smi
```

### Container Size Issues

**Problem:** Images are too large

```bash
# Use multi-stage builds
docker build -f Dockerfile.rocm --target runtime -t chai-runtime:latest .

# Check image sizes
docker images | grep chai

# Clean intermediate images
docker image prune
```

---

## Advanced Usage

### Custom ROCm Version

```bash
# Build with specific ROCm version
docker build --build-arg ROCM_VERSION=5.7 -f Dockerfile.test -t chai-test:5.7 .
```

### Volume Mounts for Development

```bash
# Mount source code for live editing
docker run -it --rm \
  -v $(pwd)/HIP-D:/workspace/HIP-D \
  -v $(pwd)/scripts:/workspace/scripts \
  chai-test:latest
```

### Extracting Compiled Binaries

```bash
# Build builder stage
docker build -f Dockerfile.rocm --target builder -t chai-builder:latest .

# Copy binaries from container
docker create --name temp-container chai-builder:latest
docker cp temp-container:/workspace/HIP-D/BFS/bfs ./bfs-binary
docker rm temp-container
```

### Network Configuration

```bash
# Run with custom network
docker network create chai-network
docker run --network chai-network chai-test:latest
```

---

## Environment Variables

### Required

- `ROCM_PATH=/opt/rocm`
- `CHAI_HIP_LIB=/opt/rocm/lib`
- `CHAI_HIP_INC=/opt/rocm/include`

### Optional

- `HIP_PLATFORM=amd` (or `nvidia` for CUDA backend)
- `ROCM_VERSION=6.0` (for builds)
- `CI=true` (for CI/CD mode)

---

## Performance Tips

1. **Use BuildKit** for faster builds:
   ```bash
   DOCKER_BUILDKIT=1 docker build -f Dockerfile.test -t chai-test:latest .
   ```

2. **Layer caching**: Structure Dockerfiles to cache expensive operations

3. **Multi-stage builds**: Use for production to reduce image size

4. **Volume mounts**: For development, avoid copying large datasets

---

## Security Considerations

1. **Minimal runtime images**: Production images only include necessary runtime

2. **Non-root user**: Consider adding non-root user for production

3. **Scan images**: Regularly scan for vulnerabilities
   ```bash
   docker scan chai-test:latest
   ```

4. **Limit capabilities**: Use minimal security options
   ```bash
   --cap-drop=ALL --cap-add=SYS_ADMIN
   ```

---

## Container Maintenance

### Regular Updates

```bash
# Update base images
docker pull ubuntu:22.04

# Rebuild with latest
./scripts/docker-build.sh all

# Update ROCm version
./scripts/docker-build.sh all 6.1
```

### Cleanup

```bash
# Remove old images
docker image prune -a

# Remove unused volumes
docker volume prune

# Complete cleanup
docker system prune -a --volumes
```

---

## Support

For issues with containers:
1. Check [Troubleshooting](#troubleshooting) section
2. Review Docker logs: `docker logs <container>`
3. Check ROCm documentation: https://rocm.docs.amd.com/
4. Open an issue in the repository

---

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
