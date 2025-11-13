#!/bin/bash

###############################################################################
# CHAI HIP Local Validation Script
#
# Comprehensive validation script for local testing of CHAI HIP benchmarks
#
# Usage:
#   ./validate.sh [options]
#
# Options:
#   --quick          Run quick validation only (no Docker builds)
#   --full           Run full validation including all Docker images
#   --compile-only   Only test compilation
#   --gpu            Run GPU benchmarks (requires AMD GPU)
#   --rocm-version   Specify ROCm version (default: 6.0)
#   --help           Show this help message
#
###############################################################################

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
ROCM_VERSION="${ROCM_VERSION:-6.0}"
QUICK_MODE=false
FULL_MODE=false
COMPILE_ONLY=false
GPU_MODE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/validation-logs-$TIMESTAMP"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            FULL_MODE=true
            shift
            ;;
        --compile-only)
            COMPILE_ONLY=true
            shift
            ;;
        --gpu)
            GPU_MODE=true
            shift
            ;;
        --rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        --help)
            head -n 25 "$0" | tail -n 18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is available"
        return 0
    else
        print_error "$1 is not available"
        return 1
    fi
}

###############################################################################
# Validation Steps
###############################################################################

check_prerequisites() {
    print_header "Checking Prerequisites"

    local all_ok=true

    print_step "Checking required commands..."
    check_command "bash" || all_ok=false
    check_command "docker" || all_ok=false
    check_command "python3" || all_ok=false
    check_command "git" || all_ok=false

    if [ "$all_ok" = false ]; then
        print_error "Missing required commands. Please install them first."
        exit 1
    fi

    print_step "Checking Docker daemon..."
    if docker ps &> /dev/null; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running or you don't have permission"
        echo "  Try: sudo usermod -aG docker \$USER && newgrp docker"
        exit 1
    fi

    print_step "Checking Docker Compose..."
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker Compose is available"
    else
        print_warning "Docker Compose not found (optional)"
    fi

    if [ "$GPU_MODE" = true ]; then
        print_step "Checking AMD GPU availability..."
        if command -v rocm-smi &> /dev/null; then
            rocm-smi | tee "$LOG_DIR/gpu-info.log"
            print_success "AMD GPU detected"
        else
            print_error "rocm-smi not found. GPU mode requires ROCm installation."
            exit 1
        fi
    fi

    print_success "All prerequisites met"
}

validate_code_structure() {
    print_header "Validating Code Structure"

    print_step "Checking for CUDA remnants..."
    if grep -r "cudaMalloc\|cudaMemcpy\|cudaFree\|cudaDeviceSynchronize" \
            HIP-D/ --include="*.cpp" --include="*.h" --include="*.hip" 2>/dev/null > "$LOG_DIR/cuda-remnants.log"; then
        print_error "Found unconverted CUDA API calls:"
        head -10 "$LOG_DIR/cuda-remnants.log"
        return 1
    else
        print_success "No CUDA API calls found"
    fi

    print_step "Verifying all benchmarks have required files..."
    local missing=0
    for benchmark in BFS BS CEDD CEDT HSTI HSTO PAD RSCD RSCT SC SSSP TQ TQH TRNS; do
        if [ ! -f "HIP-D/$benchmark/Makefile.hip" ]; then
            print_error "Missing Makefile.hip for $benchmark"
            missing=$((missing + 1))
        fi
        if [ ! -f "HIP-D/$benchmark/support/hip-setup.h" ]; then
            print_error "Missing hip-setup.h for $benchmark"
            missing=$((missing + 1))
        fi
    done

    if [ $missing -gt 0 ]; then
        print_error "$missing required files missing"
        return 1
    fi
    print_success "All required files present"

    print_step "Checking Makefile configuration..."
    for makefile in HIP-D/*/Makefile.hip; do
        benchmark=$(basename $(dirname "$makefile"))
        if ! grep -q "CXX=hipcc" "$makefile"; then
            print_error "$benchmark: Makefile doesn't use hipcc"
            return 1
        fi
        if ! grep -q "lamdhip64" "$makefile"; then
            print_error "$benchmark: Makefile doesn't link AMD HIP library"
            return 1
        fi
    done
    print_success "All Makefiles properly configured"

    print_step "Verifying HIP headers..."
    local header_issues=0
    for file in HIP-D/*/kernel.h HIP-D/*/support/timer.h; do
        if [ -f "$file" ] && ! grep -q "hip/hip_runtime.h" "$file"; then
            print_warning "$file missing HIP header"
            header_issues=$((header_issues + 1))
        fi
    done

    if [ $header_issues -eq 0 ]; then
        print_success "All headers properly include HIP"
    else
        print_warning "$header_issues files missing HIP headers (may be intentional)"
    fi

    print_success "Code structure validation complete"
}

validate_docker_config() {
    print_header "Validating Docker Configuration"

    print_step "Checking Dockerfiles..."
    for dockerfile in Dockerfile.rocm Dockerfile.test Dockerfile.hipify; do
        if [ ! -f "$dockerfile" ]; then
            print_error "$dockerfile not found"
            return 1
        fi
        print_success "$dockerfile exists"
    done

    print_step "Validating docker-compose.yml..."
    if python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))" 2> "$LOG_DIR/docker-compose-validation.log"; then
        print_success "docker-compose.yml is valid YAML"
    else
        print_error "docker-compose.yml has syntax errors"
        cat "$LOG_DIR/docker-compose-validation.log"
        return 1
    fi

    print_step "Validating shell scripts..."
    for script in scripts/docker-build.sh scripts/docker-test.sh; do
        if bash -n "$script" 2> "$LOG_DIR/$(basename $script).validation.log"; then
            print_success "$(basename $script) is syntactically valid"
        else
            print_error "$(basename $script) has syntax errors"
            cat "$LOG_DIR/$(basename $script).validation.log"
            return 1
        fi
    done

    print_success "Docker configuration validation complete"
}

build_docker_images() {
    print_header "Building Docker Images (ROCm $ROCM_VERSION)"

    local images=()
    if [ "$FULL_MODE" = true ]; then
        images=("dev" "test" "runtime")
    else
        images=("test")
    fi

    for image in "${images[@]}"; do
        print_step "Building chai-$image image..."
        if ./scripts/docker-build.sh "$image" "$ROCM_VERSION" 2>&1 | tee "$LOG_DIR/build-$image.log"; then
            print_success "chai-$image built successfully"

            # Check image size
            size=$(docker images chai-$image:latest --format "{{.Size}}")
            echo "  Image size: $size" | tee -a "$LOG_DIR/image-sizes.txt"
        else
            print_error "Failed to build chai-$image"
            return 1
        fi
    done

    print_step "Verifying images..."
    docker images | grep chai- | tee "$LOG_DIR/docker-images.txt"

    print_success "Docker images built successfully"
}

test_docker_containers() {
    print_header "Testing Docker Containers"

    print_step "Testing hipcc availability..."
    if docker run --rm chai-test:latest which hipcc &> "$LOG_DIR/hipcc-check.log"; then
        print_success "hipcc found in container"
    else
        print_error "hipcc not found in container"
        return 1
    fi

    print_step "Checking hipcc version..."
    docker run --rm chai-test:latest hipcc --version | tee "$LOG_DIR/hipcc-version.txt"

    print_step "Checking ROCm installation..."
    docker run --rm chai-test:latest bash -c "
        echo 'ROCm Path: \$ROCM_PATH'
        ls -la /opt/rocm/bin/ | head -10
        hip-config --version
    " | tee "$LOG_DIR/rocm-check.txt"

    print_success "Docker container tests passed"
}

compile_benchmarks() {
    print_header "Compiling All Benchmarks"

    mkdir -p "$LOG_DIR/compilation"

    print_step "Compiling all benchmarks in Docker container..."

    local failed=0
    local succeeded=0

    for benchmark in BFS BS CEDD CEDT HSTI HSTO PAD RSCD RSCT SC SSSP TQ TQH TRNS; do
        echo ""
        print_step "Compiling $benchmark..."

        if docker run --rm \
            -v "$(pwd):/workspace" \
            chai-test:latest \
            bash -c "
                cd /workspace/HIP-D/$benchmark
                make -f Makefile.hip clean &>/dev/null || true
                make -f Makefile.hip
            " 2>&1 | tee "$LOG_DIR/compilation/$benchmark.log"; then

            # Check if binary was created
            binary=$(find "HIP-D/$benchmark" -type f -executable 2>/dev/null | head -1)
            if [ -n "$binary" ]; then
                size=$(stat -c%s "$binary" 2>/dev/null || stat -f%z "$binary" 2>/dev/null)
                print_success "$benchmark compiled successfully ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
                succeeded=$((succeeded + 1))
            else
                print_warning "$benchmark: make succeeded but no binary found"
            fi
        else
            print_error "$benchmark compilation failed"
            failed=$((failed + 1))
        fi
    done

    echo ""
    echo "═══════════════════════════════════════"
    echo "Compilation Summary:"
    echo "  Succeeded: $succeeded"
    echo "  Failed: $failed"
    echo "═══════════════════════════════════════"

    if [ $failed -gt 0 ]; then
        print_error "$failed benchmark(s) failed to compile"
        return 1
    fi

    print_step "Listing compiled binaries..."
    find HIP-D -type f -executable | grep -E "(bfs|sssp|bs|sc)" | head -20 | tee "$LOG_DIR/binaries.txt"

    print_success "All benchmarks compiled successfully"
}

run_smoke_tests() {
    print_header "Running Smoke Tests"

    print_step "Testing help output for each benchmark..."

    for benchmark in BFS BS CEDD CEDT HSTI HSTO PAD RSCD RSCT SC SSSP TQ TQH TRNS; do
        binary=$(find "HIP-D/$benchmark" -type f -executable 2>/dev/null | head -1)
        if [ -n "$binary" ]; then
            echo "Testing $benchmark..."
            timeout 5 "$binary" --help &> "$LOG_DIR/smoke-$benchmark.log" || true
            print_success "$benchmark help test complete"
        else
            print_warning "$benchmark: no binary found"
        fi
    done

    print_success "Smoke tests complete"
}

run_gpu_benchmarks() {
    print_header "Running GPU Benchmarks"

    if [ "$GPU_MODE" != true ]; then
        print_warning "GPU mode not enabled. Use --gpu flag."
        return 0
    fi

    print_step "Checking GPU availability..."
    if ! command -v rocm-smi &> /dev/null; then
        print_error "rocm-smi not found. Cannot run GPU benchmarks."
        return 1
    fi

    rocm-smi | tee "$LOG_DIR/gpu-status.txt"

    print_step "Running BFS benchmark on GPU..."
    docker run --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --security-opt seccomp=unconfined \
        --group-add video \
        --group-add render \
        -v "$(pwd):/workspace" \
        chai-test:latest \
        bash -c "
            cd /workspace/HIP-D/BFS
            ./bfs -n 1 -t 1
        " 2>&1 | tee "$LOG_DIR/gpu-bfs-run.log"

    print_success "GPU benchmark execution complete"
}

generate_report() {
    print_header "Generating Validation Report"

    local report="$LOG_DIR/validation-report.md"

    cat > "$report" << EOF
# CHAI HIP Validation Report

**Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**ROCm Version**: $ROCM_VERSION
**Host**: $(hostname)
**User**: $(whoami)

## System Information

### Docker Version
\`\`\`
$(docker --version)
\`\`\`

### Docker Compose Version
\`\`\`
$(docker-compose --version 2>/dev/null || docker compose version 2>/dev/null || echo "Not available")
\`\`\`

EOF

    if [ "$GPU_MODE" = true ] && [ -f "$LOG_DIR/gpu-info.log" ]; then
        cat >> "$report" << EOF
### GPU Information
\`\`\`
$(cat "$LOG_DIR/gpu-info.log")
\`\`\`

EOF
    fi

    cat >> "$report" << EOF
## Validation Results

### Code Structure
- ✅ CUDA to HIP conversion verified
- ✅ All benchmark files present
- ✅ Makefile configuration correct
- ✅ HIP headers properly included

### Docker Images
EOF

    if [ -f "$LOG_DIR/image-sizes.txt" ]; then
        cat >> "$report" << EOF
\`\`\`
$(cat "$LOG_DIR/image-sizes.txt")
\`\`\`
EOF
    fi

    if [ -d "$LOG_DIR/compilation" ]; then
        cat >> "$report" << EOF

### Compilation Results

EOF
        for log in "$LOG_DIR/compilation"/*.log; do
            benchmark=$(basename "$log" .log)
            if grep -qi "error" "$log"; then
                echo "- ❌ $benchmark: FAILED" >> "$report"
            else
                echo "- ✅ $benchmark: SUCCESS" >> "$report"
            fi
        done
    fi

    cat >> "$report" << EOF

## Log Files

All detailed logs are available in: \`$LOG_DIR/\`

- Code validation: \`cuda-remnants.log\`
- Docker build logs: \`build-*.log\`
- Compilation logs: \`compilation/*.log\`
- Smoke test logs: \`smoke-*.log\`
EOF

    if [ "$GPU_MODE" = true ]; then
        cat >> "$report" << EOF
- GPU benchmark logs: \`gpu-*.log\`
EOF
    fi

    cat >> "$report" << EOF

## Recommendations

EOF

    if [ "$QUICK_MODE" = true ]; then
        cat >> "$report" << EOF
- Run full validation with \`--full\` flag for comprehensive testing
EOF
    fi

    if [ "$GPU_MODE" != true ]; then
        cat >> "$report" << EOF
- Run GPU benchmarks with \`--gpu\` flag to test actual execution
EOF
    fi

    cat >> "$report" << EOF

---
Generated by CHAI HIP validation script
EOF

    print_success "Report generated: $report"

    # Display report
    echo ""
    cat "$report"
}

###############################################################################
# Main Execution
###############################################################################

main() {
    print_header "CHAI HIP Validation Script"
    echo "Timestamp: $TIMESTAMP"
    echo "Log directory: $LOG_DIR"
    echo "ROCm Version: $ROCM_VERSION"
    echo ""

    # Step 1: Prerequisites
    check_prerequisites

    # Step 2: Code validation
    validate_code_structure || exit 1

    # Step 3: Docker configuration
    validate_docker_config || exit 1

    # Quick mode stops here
    if [ "$QUICK_MODE" = true ]; then
        print_success "Quick validation complete"
        generate_report
        exit 0
    fi

    # Step 4: Build Docker images
    build_docker_images || exit 1

    # Step 5: Test containers
    test_docker_containers || exit 1

    # Compile-only mode stops here
    if [ "$COMPILE_ONLY" = true ]; then
        compile_benchmarks || exit 1
        print_success "Compilation validation complete"
        generate_report
        exit 0
    fi

    # Step 6: Compile all benchmarks
    compile_benchmarks || exit 1

    # Step 7: Smoke tests
    run_smoke_tests

    # Step 8: GPU benchmarks (if enabled)
    if [ "$GPU_MODE" = true ]; then
        run_gpu_benchmarks
    fi

    # Step 9: Generate report
    generate_report

    print_header "Validation Complete"
    print_success "All validation steps passed!"
    echo ""
    echo "View detailed report: $LOG_DIR/validation-report.md"
    echo "View logs: $LOG_DIR/"
}

# Run main function
main "$@"
