#!/bin/bash
# Build script for CHAI HIP Docker images

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
BUILD_TARGET=${1:-all}
ROCM_VERSION=${2:-6.0}

cd "$PROJECT_ROOT"

log_info "CHAI HIP Docker Build Script"
log_info "============================="
log_info "Build Target: $BUILD_TARGET"
log_info "ROCm Version: $ROCM_VERSION"
echo ""

build_dev() {
    log_info "Building development container..."
    docker build \
        --build-arg ROCM_VERSION=$ROCM_VERSION \
        -f Dockerfile.hipify \
        -t chai-hipify:latest \
        -t chai-hipify:$ROCM_VERSION \
        .
    log_info "✓ Development container built: chai-hipify:latest"
}

build_test() {
    log_info "Building test container..."
    docker build \
        --build-arg ROCM_VERSION=$ROCM_VERSION \
        -f Dockerfile.test \
        -t chai-test:latest \
        -t chai-test:$ROCM_VERSION \
        .
    log_info "✓ Test container built: chai-test:latest"
}

build_runtime() {
    log_info "Building runtime container (multi-stage)..."

    # Build builder stage
    log_info "  Building builder stage..."
    docker build \
        --build-arg ROCM_VERSION=$ROCM_VERSION \
        -f Dockerfile.rocm \
        --target builder \
        -t chai-builder:latest \
        -t chai-builder:$ROCM_VERSION \
        .
    log_info "  ✓ Builder stage complete"

    # Build runtime stage
    log_info "  Building runtime stage..."
    docker build \
        --build-arg ROCM_VERSION=$ROCM_VERSION \
        -f Dockerfile.rocm \
        --target runtime \
        -t chai-runtime:latest \
        -t chai-runtime:$ROCM_VERSION \
        .
    log_info "✓ Runtime container built: chai-runtime:latest"
}

case "$BUILD_TARGET" in
    dev)
        build_dev
        ;;
    test)
        build_test
        ;;
    runtime)
        build_runtime
        ;;
    all)
        build_dev
        build_test
        build_runtime
        ;;
    *)
        log_error "Unknown build target: $BUILD_TARGET"
        echo "Usage: $0 [dev|test|runtime|all] [rocm_version]"
        echo "  dev      - Build development container"
        echo "  test     - Build testing container"
        echo "  runtime  - Build production runtime container"
        echo "  all      - Build all containers (default)"
        exit 1
        ;;
esac

echo ""
log_info "============================="
log_info "Build complete!"
log_info "============================="
echo ""
log_info "Available images:"
docker images | grep chai- | head -10

echo ""
log_info "Next steps:"
log_info "  docker-compose up -d chai-test    # Start test container"
log_info "  docker exec -it chai-hip-test bash  # Enter container"
log_info "  compile_all.sh                     # Compile benchmarks"
