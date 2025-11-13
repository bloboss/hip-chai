# CHAI HIP Validation and Testing Guide

Complete guide for validating and testing CHAI HIP benchmarks locally and in CI/CD.

## Quick Start

```bash
# Quick validation (no Docker builds)
./validate.sh --quick

# Full validation with compilation
./validate.sh --full

# Full validation with GPU testing
./validate.sh --full --gpu
```

## Overview

This guide covers three validation approaches:

1. **Local Validation** - Using `validate.sh` script on your machine
2. **Docker Validation** - Testing in containerized environments
3. **CI/CD Validation** - Automated testing via GitHub Actions

---

## Files Added

This validation framework includes:

### Scripts

- **`validate.sh`** - Comprehensive local validation script
  - Quick mode: Code structure validation only
  - Full mode: Complete build and test cycle
  - GPU mode: Hardware testing with AMD GPUs
  - Generates detailed reports

### Documentation

- **`ROCM-SETUP.md`** - Complete ROCm installation and setup guide
  - Docker installation (recommended)
  - Native ROCm installation (Ubuntu, RHEL, etc.)
  - Cloud GPU instance setup
  - Troubleshooting guide
  - Performance tuning

- **`VALIDATION-GUIDE.md`** - This file
  - Validation workflow documentation
  - Usage examples
  - Best practices

### Docker

- **`Dockerfile.comprehensive`** - Fully documented multi-stage Dockerfile
  - Stage 1: ROCm base environment
  - Stage 2: Development environment with source code
  - Stage 3: Build all benchmarks
  - Stage 4: Minimal runtime image
  - Stage 5: Full development image with tools
  - Includes helper scripts: compile_all.sh, smoke_tests.sh, etc.

### CI/CD

- **`.github/workflows/comprehensive-rocm-ci.yml`** - Enhanced GitHub Actions workflow
  - Code validation (CUDA remnants, file structure)
  - Multi-version Docker builds (ROCm 5.7, 6.0, 6.1)
  - Compilation testing (all benchmarks + individual)
  - Smoke tests
  - Report generation
  - Optional GPU benchmarking

---

## Local Validation with validate.sh

### Prerequisites

```bash
# Check prerequisites
docker --version
python3 --version
git --version
```

### Usage

```bash
./validate.sh [options]

Options:
  --quick          Quick validation (no Docker builds)
  --full           Full validation (Docker builds + compilation)
  --compile-only   Build and compile only
  --gpu            Include GPU tests (requires AMD GPU)
  --rocm-version   Specify ROCm version (default: 6.0)
  --help           Show help message
```

### Examples

#### Quick Validation (30 seconds)

Validates code structure without building Docker images:

```bash
./validate.sh --quick
```

Checks:
- ✅ Prerequisites (Docker, Python, Git)
- ✅ CUDA to HIP conversion completeness
- ✅ File structure (Makefile.hip, hip-setup.h)
- ✅ Docker configuration syntax
- ✅ Script syntax

#### Full Validation (15-30 minutes)

Complete build and compilation testing:

```bash
./validate.sh --full
```

Performs:
- All quick validation checks
- Docker image builds (test image)
- Container verification
- Compile all 14 benchmarks
- Smoke tests
- Generate comprehensive report

#### GPU Validation (20-40 minutes)

Full validation plus GPU execution tests:

```bash
./validate.sh --full --gpu
```

Requires:
- AMD GPU hardware
- ROCm drivers installed
- User in `video` and `render` groups

Additional steps:
- GPU detection and status
- Run actual benchmarks on GPU
- Profile execution

#### Compilation Only

Build Docker and compile without running tests:

```bash
./validate.sh --compile-only
```

Useful for:
- CI/CD pipelines
- Quick build verification
- Pre-commit checks

#### Custom ROCm Version

```bash
./validate.sh --full --rocm-version 6.1
```

Tests against ROCm 6.1 instead of default 6.0.

### Output

Validation creates timestamped log directory:

```
validation-logs-20250113_143052/
├── validation-report.md          # Comprehensive report
├── cuda-remnants.log             # CUDA API search results
├── docker-compose-validation.log # YAML validation
├── build-test.log                # Docker build logs
├── compilation/                  # Per-benchmark logs
│   ├── BFS.log
│   ├── BS.log
│   └── ...
├── smoke-*.log                   # Smoke test results
└── gpu-*.log                     # GPU test results (if --gpu)
```

### Report Example

```markdown
# CHAI HIP Validation Report

**Date**: 2025-01-13 14:30:52 UTC
**ROCm Version**: 6.0
**Host**: development-machine
**User**: developer

## Validation Results

### Code Structure
- ✅ CUDA to HIP conversion verified
- ✅ All benchmark files present
- ✅ Makefile configuration correct
- ✅ HIP headers properly included

### Docker Images
test: 4.2GB

### Compilation Results
- ✅ BFS: SUCCESS
- ✅ BS: SUCCESS
- ✅ CEDD: SUCCESS
...

## Recommendations
- Run GPU benchmarks with `--gpu` flag to test actual execution
```

---

## Docker Validation

### Using Comprehensive Dockerfile

#### Build Complete Development Environment

```bash
# Build full development image (default)
docker build -f Dockerfile.comprehensive -t chai-comprehensive:latest .

# Run interactively
docker run --rm -it chai-comprehensive:latest
```

Inside container:
```bash
# Check GPU (if available)
check_gpu.sh

# Compile all benchmarks
compile_all.sh

# Run smoke tests
smoke_tests.sh

# Run a specific benchmark
cd HIP-D/BFS
./bfs -n 100 -t 256
```

#### Build Specific Stages

```bash
# Runtime only (minimal image)
docker build -f Dockerfile.comprehensive --target runtime -t chai-runtime:latest .

# Development environment (with source)
docker build -f Dockerfile.comprehensive --target development -t chai-dev:latest .

# Pre-compiled (all benchmarks built)
docker build -f Dockerfile.comprehensive --target builder -t chai-builder:latest .
```

#### Custom ROCm Version

```bash
docker build -f Dockerfile.comprehensive \
  --build-arg ROCM_VERSION=6.1 \
  -t chai-comprehensive:rocm6.1 .
```

### Using Existing Docker Scripts

```bash
# Build test container
./scripts/docker-build.sh test

# Build all containers
./scripts/docker-build.sh all

# Run compilation tests
./scripts/docker-test.sh compile

# Run smoke tests
./scripts/docker-test.sh smoke

# Run all tests
./scripts/docker-test.sh all
```

### Docker Compose

```bash
# Start development container
docker-compose up -d chai-dev

# Compile in test container
docker-compose run --rm chai-test compile_all.sh

# Run with GPU access
docker-compose run --rm chai-runtime bash
```

---

## CI/CD Validation

### GitHub Actions Workflow

The comprehensive CI/CD workflow (`.github/workflows/comprehensive-rocm-ci.yml`) runs automatically on:

- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### Workflow Jobs

#### 1. Code Validation (2-3 minutes)

Static analysis:
- CUDA API remnants check
- File structure verification
- Makefile configuration
- HIP header inclusion
- Docker/YAML syntax validation

#### 2. Docker Build (20-40 minutes)

Matrix build strategy:
- Images: dev, test, runtime
- ROCm versions: 5.7, 6.0, 6.1
- Total: 7 image combinations

Each build:
- Uses layer caching
- Verifies hipcc availability
- Checks image size
- Exports artifacts for later jobs

#### 3. Compilation Test (15-25 minutes)

Sequential compilation:
- All 14 benchmarks in single container
- Detailed logging per benchmark
- Binary size reporting
- Build logs uploaded on failure

#### 4. Individual Benchmark Compilation (10-15 minutes)

Parallel matrix build:
- Each benchmark in separate job
- Better failure isolation
- Faster overall execution
- 14 jobs run in parallel

#### 5. Smoke Tests (5-10 minutes)

Quick validation:
- Help output for each benchmark
- Input data access verification
- No GPU required

#### 6. Test Report Generation (2-3 minutes)

Generates markdown report:
- Job status summary
- Image sizes
- Compilation results
- Posted as PR comment

#### 7. GPU Benchmarks (Optional, 30-60 minutes)

Requires self-hosted AMD GPU runner:
- Actual benchmark execution
- Performance measurement
- Results uploaded as artifacts

### Manual Workflow Trigger

```bash
# Via GitHub CLI
gh workflow run comprehensive-rocm-ci.yml \
  -f rocm_version=6.1 \
  -f run_benchmarks=true

# Via GitHub web interface:
# Actions → Comprehensive ROCm CI/CD → Run workflow
```

### Viewing Results

```bash
# List workflow runs
gh run list --workflow=comprehensive-rocm-ci.yml

# View specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

### CI/CD Best Practices

**For Contributors**:

1. Run `./validate.sh --quick` before committing
2. Ensure all code validation checks pass
3. Test locally with Docker if possible
4. Review CI/CD logs for failures

**For Maintainers**:

1. Review test reports in PRs
2. Check artifact sizes (image bloat)
3. Monitor build times
4. Update caching strategies as needed

**For GPU Testing**:

1. Set up self-hosted runner with AMD GPU
2. Configure runner labels: `[self-hosted, amd-gpu]`
3. Enable workflow dispatch with `run_benchmarks=true`
4. Monitor GPU utilization during runs

---

## Troubleshooting

### validate.sh Issues

#### Docker Not Found

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

#### Permission Denied

```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

#### Compilation Failures

Check logs in `validation-logs-*/compilation/`:

```bash
# View specific benchmark log
cat validation-logs-*/compilation/BFS.log

# Search for errors
grep -i error validation-logs-*/compilation/*.log
```

Common issues:
- Missing HIP headers → Rebuild Docker image
- hipcc not found → Check PATH in container
- Library linking errors → Verify CHAI_HIP_LIB

### Docker Build Issues

#### Out of Disk Space

```bash
# Clean up Docker
docker system prune -a

# Check disk usage
docker system df
```

#### Build Timeouts

```bash
# Increase timeout (validate.sh doesn't have built-in timeout)
# For docker-build.sh:
timeout 3600 ./scripts/docker-build.sh test
```

#### Network Issues During Build

```bash
# Use build arg for mirrors
docker build --build-arg HTTP_PROXY=http://proxy:port \
  -f Dockerfile.comprehensive -t chai:latest .
```

### CI/CD Issues

#### Workflow Not Triggering

Check:
1. Branch name matches pattern (`main`, `develop`, `claude/**`)
2. Workflow file syntax is valid
3. Repository has Actions enabled

```bash
# Validate workflow locally
act -l  # Requires 'act' tool
```

#### Build Cache Issues

GitHub Actions cache can become stale:

```bash
# Clear cache via API
gh api -X DELETE /repos/:owner/:repo/actions/caches/:cache_id

# Or force rebuild without cache
# Edit workflow: remove cache-from parameter
```

#### Self-Hosted Runner Not Working

```bash
# Check runner status
./run.sh  # On runner machine

# View runner logs
tail -f _diag/Runner_*.log

# Verify GPU access
rocm-smi
docker run --device=/dev/kfd --device=/dev/dri --rm ubuntu:22.04 ls -la /dev/kfd
```

---

## Performance Benchmarking

### Quick Performance Test

```bash
# Run BFS benchmark with various sizes
for n in 10 50 100 500; do
    echo "Testing with n=$n"
    cd HIP-D/BFS
    ./bfs -n $n -t 256 2>&1 | grep "Time"
    cd ../..
done
```

### Profiling with rocprof

Using comprehensive Dockerfile:

```bash
# Run container with GPU
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  chai-comprehensive:latest

# Inside container, profile a benchmark
profile_benchmark.sh BFS -n 100 -t 256

# View results
cat /workspace/profiles/BFS-*/results.csv
```

### Batch Profiling

```bash
# Run all benchmarks and collect results
run_all_benchmarks.sh

# Results saved in timestamped directory
ls /workspace/results/run-*/
```

---

## Best Practices

### Development Workflow

1. **Before Making Changes**:
   ```bash
   ./validate.sh --quick
   ```

2. **After Code Changes**:
   ```bash
   ./validate.sh --compile-only
   ```

3. **Before Committing**:
   ```bash
   ./validate.sh --full
   ```

4. **Before Production**:
   ```bash
   ./validate.sh --full --gpu
   ```

### Docker Workflow

1. **Use multi-stage builds** for different purposes:
   - `runtime`: Production deployment
   - `development`: Interactive development
   - `full`: Complete environment

2. **Layer caching**: Order Dockerfile commands from least to most frequently changing

3. **Volume mounts** for development:
   ```bash
   docker run -v $(pwd):/workspace chai-dev:latest
   ```

### CI/CD Workflow

1. **Matrix testing** for coverage:
   - Multiple ROCm versions
   - Multiple Ubuntu versions
   - Multiple benchmark variants

2. **Artifact management**:
   - Upload logs on failure
   - Keep binaries for successful builds
   - Limit retention to save space

3. **Caching strategy**:
   - Cache Docker layers
   - Cache compiled objects
   - Invalidate when dependencies change

---

## Additional Resources

- [ROCm Setup Guide](ROCM-SETUP.md) - Complete ROCm installation
- [Docker Guide](DOCKER-GUIDE.md) - Docker usage details
- [HIP Porting Plan](HIP-PORTING-PLAN.md) - CUDA to HIP conversion
- [Quick Start](QUICKSTART-HIP.md) - Getting started guide

### External Links

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

## Summary

**Quick validation**: `./validate.sh --quick` (30 sec)

**Full local test**: `./validate.sh --full` (20 min)

**GPU testing**: `./validate.sh --full --gpu` (30 min)

**Docker development**: `docker build -f Dockerfile.comprehensive -t chai:latest . && docker run --rm -it chai:latest`

**CI/CD**: Push to trigger automatic testing

All validation approaches work together to ensure CHAI HIP benchmarks are properly converted, compile correctly, and execute successfully on AMD GPUs.

---

**Last Updated**: 2025-01-13
**Version**: 1.0
**Maintainer**: CHAI HIP Development Team
