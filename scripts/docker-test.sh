#!/bin/bash
# Test script for CHAI HIP Docker containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cd "$PROJECT_ROOT"

log_info "CHAI HIP Docker Test Script"
log_info "============================"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    log_warn "docker-compose not found, using 'docker compose' instead"
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Parse arguments
TEST_TYPE=${1:-compile}

run_compile_test() {
    log_info "Running compilation test..."
    log_info "Building test container..."

    docker build -f Dockerfile.test -t chai-test:latest . || {
        log_error "Failed to build test container"
        exit 1
    }

    log_info "Running compilation test in container..."
    docker run --rm \
        -v "$(pwd)":/workspace \
        -e CHAI_HIP_LIB=/opt/rocm/lib \
        -e CHAI_HIP_INC=/opt/rocm/include \
        -e ROCM_PATH=/opt/rocm \
        chai-test:latest \
        compile_all.sh

    log_info "✓ Compilation test completed"
}

run_smoke_test() {
    log_info "Running smoke tests..."

    docker run --rm \
        -v "$(pwd)":/workspace \
        chai-test:latest \
        smoke_tests.sh

    log_info "✓ Smoke tests completed"
}

run_ci_test() {
    log_info "Running CI/CD test suite..."

    $COMPOSE_CMD run --rm chai-ci

    log_info "✓ CI tests completed"
}

run_verify_images() {
    log_info "Verifying Docker images..."

    REQUIRED_IMAGES=(
        "chai-test:latest"
        "chai-hipify:latest"
    )

    ALL_PRESENT=true
    for img in "${REQUIRED_IMAGES[@]}"; do
        if docker images | grep -q "$(echo $img | cut -d: -f1).*$(echo $img | cut -d: -f2)"; then
            log_info "  ✓ $img present"
        else
            log_warn "  ✗ $img missing"
            ALL_PRESENT=false
        fi
    done

    if $ALL_PRESENT; then
        log_info "✓ All required images present"
    else
        log_warn "Some images are missing. Run: ./scripts/docker-build.sh all"
    fi
}

case "$TEST_TYPE" in
    compile)
        run_compile_test
        ;;
    smoke)
        run_smoke_test
        ;;
    ci)
        run_ci_test
        ;;
    verify)
        run_verify_images
        ;;
    all)
        run_verify_images
        run_compile_test
        run_smoke_test
        ;;
    *)
        log_error "Unknown test type: $TEST_TYPE"
        echo "Usage: $0 [compile|smoke|ci|verify|all]"
        echo "  compile  - Test compilation of all benchmarks"
        echo "  smoke    - Run smoke tests"
        echo "  ci       - Run full CI/CD test suite"
        echo "  verify   - Verify Docker images"
        echo "  all      - Run all tests"
        exit 1
        ;;
esac

echo ""
log_info "Test completed successfully!"
