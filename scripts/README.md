# CHAI HIP Scripts

Helper scripts for building and testing CHAI HIP benchmarks in Docker containers.

## Scripts

### docker-build.sh

Build Docker images for CHAI HIP benchmarks.

**Usage:**
```bash
./scripts/docker-build.sh [target] [rocm_version]
```

**Targets:**
- `dev` - Build development container (hipify-clang, conversion tools)
- `test` - Build testing container (full ROCm dev suite)
- `runtime` - Build production runtime container (multi-stage)
- `all` - Build all containers (default)

**Examples:**
```bash
# Build all containers with ROCm 6.0
./scripts/docker-build.sh all 6.0

# Build only test container
./scripts/docker-build.sh test

# Build with specific ROCm version
./scripts/docker-build.sh all 5.7
```

### docker-test.sh

Run tests in Docker containers.

**Usage:**
```bash
./scripts/docker-test.sh [test_type]
```

**Test Types:**
- `compile` - Test compilation of all benchmarks
- `smoke` - Run smoke tests (quick validation)
- `ci` - Run full CI/CD test suite
- `verify` - Verify Docker images are built
- `all` - Run all tests (default)

**Examples:**
```bash
# Test compilation
./scripts/docker-test.sh compile

# Run smoke tests
./scripts/docker-test.sh smoke

# Verify images
./scripts/docker-test.sh verify

# Run everything
./scripts/docker-test.sh all
```

## Quick Start

```bash
# 1. Build containers
./scripts/docker-build.sh all

# 2. Verify images
./scripts/docker-test.sh verify

# 3. Test compilation
./scripts/docker-test.sh compile

# 4. Run smoke tests
./scripts/docker-test.sh smoke
```

## CI/CD Integration

These scripts are designed to work in CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Build and test
  run: |
    ./scripts/docker-build.sh test
    ./scripts/docker-test.sh compile
```

## Environment Variables

The scripts use these environment variables:

- `ROCM_VERSION` - ROCm version to use (default: 6.0)
- `COMPOSE_CMD` - Docker compose command (auto-detected)

## Requirements

- Docker 20.10+
- Bash 4.0+
- 10GB+ free disk space

## Troubleshooting

**Script not executable:**
```bash
chmod +x scripts/*.sh
```

**Docker permission denied:**
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

**Out of disk space:**
```bash
docker system prune -a
```

## See Also

- [DOCKER-GUIDE.md](../DOCKER-GUIDE.md) - Complete Docker documentation
- [HIP-PORTING-PLAN.md](../HIP-PORTING-PLAN.md) - HIP porting strategy
- [QUICKSTART-HIP.md](../QUICKSTART-HIP.md) - HIP quick start guide
