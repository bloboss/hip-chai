#!/bin/bash
# Automated CUDA to HIP Porting Script for CHAI Benchmarks
# This script automates the conversion of CUDA-D benchmarks to HIP-D

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="${SCRIPT_DIR}/CUDA-D"
HIP_DIR="${SCRIPT_DIR}/HIP-D"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if hipify-clang is available
check_hipify() {
    if ! command -v hipify-clang &> /dev/null; then
        log_error "hipify-clang not found. Please install ROCm or run inside Docker container."
        exit 1
    fi
    log_info "hipify-clang found: $(which hipify-clang)"
}

# Convert CUDA files to HIP
convert_cuda_files() {
    local benchmark=$1
    log_info "Converting CUDA files in ${benchmark}..."

    cd "${HIP_DIR}/${benchmark}"

    # Convert .cu files
    for cufile in *.cu; do
        if [ -f "$cufile" ]; then
            log_info "  Converting ${cufile}..."
            hipify-clang "$cufile" -o "${cufile%.cu}.hip.cpp" -- -std=c++11 -x cuda
            if [ $? -eq 0 ]; then
                # Rename to replace original
                mv "${cufile%.cu}.hip.cpp" "${cufile%.cu}.cpp"
                # Keep original .cu as backup
                mv "$cufile" "${cufile}.bak"
                log_info "  ✓ Converted ${cufile} -> ${cufile%.cu}.cpp"
            else
                log_error "  ✗ Failed to convert ${cufile}"
            fi
        fi
    done

    cd "${SCRIPT_DIR}"
}

# Update Makefile for HIP
update_makefile() {
    local benchmark=$1
    local makefile="${HIP_DIR}/${benchmark}/Makefile"

    if [ ! -f "$makefile" ]; then
        log_warn "No Makefile found for ${benchmark}"
        return
    fi

    log_info "Updating Makefile for ${benchmark}..."

    # Create backup
    cp "$makefile" "${makefile}.bak"

    # Update compiler
    sed -i 's/CXX=nvcc/CXX=hipcc/' "$makefile"

    # Update environment variable names
    sed -i 's/CHAI_CUDA_LIB/CHAI_HIP_LIB/g' "$makefile"
    sed -i 's/CHAI_CUDA_INC/CHAI_HIP_INC/g' "$makefile"

    # Update library paths (add ROCm libraries)
    sed -i 's|LIB=-L/usr/lib/ -L\$(CHAI_HIP_LIB) -lm|LIB=-L/usr/lib/ -L\$(CHAI_HIP_LIB) -L/opt/rocm/lib -lm -lamdhip64|' "$makefile"

    # Update include paths
    sed -i 's|INC=-I\$(CHAI_HIP_INC)|INC=-I\$(CHAI_HIP_INC) -I/opt/rocm/include|' "$makefile"

    # Update source files (.cu -> .cpp)
    sed -i 's/\.cu/.cpp/g' "$makefile"

    log_info "  ✓ Updated Makefile for ${benchmark}"
}

# Update header files
update_headers() {
    local benchmark=$1
    log_info "Updating headers for ${benchmark}..."

    # Update common.h - change CUDA_COMPILER to HIP_COMPILER
    local common_h="${HIP_DIR}/${benchmark}/support/common.h"
    if [ -f "$common_h" ]; then
        sed -i 's/_CUDA_COMPILER_/_HIP_COMPILER_/g' "$common_h"
        log_info "  ✓ Updated common.h"
    fi

    # Rename cuda-setup.h to hip-setup.h if exists
    local cuda_setup="${HIP_DIR}/${benchmark}/support/cuda-setup.h"
    local hip_setup="${HIP_DIR}/${benchmark}/support/hip-setup.h"
    if [ -f "$cuda_setup" ]; then
        cp "$cuda_setup" "$hip_setup"
        # Update includes in the file
        sed -i 's/#include <cuda\.h>/#include <hip\/hip_runtime.h>/g' "$hip_setup"
        sed -i 's/cudaError_t/hipError_t/g' "$hip_setup"
        sed -i 's/cudaSuccess/hipSuccess/g' "$hip_setup"
        log_info "  ✓ Created hip-setup.h"
    fi

    # Update main.cpp and kernel.cpp to use hip-setup.h
    for file in "${HIP_DIR}/${benchmark}/main.cpp" "${HIP_DIR}/${benchmark}/kernel.cpp"; do
        if [ -f "$file" ]; then
            sed -i 's/"support\/cuda-setup\.h"/"support\/hip-setup.h"/g' "$file"
        fi
    done
}

# Process a single benchmark
process_benchmark() {
    local benchmark=$1
    log_info "Processing benchmark: ${benchmark}"
    log_info "========================================"

    convert_cuda_files "$benchmark"
    update_makefile "$benchmark"
    update_headers "$benchmark"

    log_info "✓ Completed processing ${benchmark}"
    echo ""
}

# Main execution
main() {
    log_info "CHAI CUDA to HIP Porting Script"
    log_info "================================"
    echo ""

    # Check prerequisites
    check_hipify

    # Check if HIP-D directory exists
    if [ ! -d "$HIP_DIR" ]; then
        log_error "HIP-D directory not found at ${HIP_DIR}"
        log_info "Please ensure HIP-D directory exists (copy of CUDA-D)"
        exit 1
    fi

    # Get list of benchmarks
    benchmarks=($(ls -d ${HIP_DIR}/*/ | xargs -n 1 basename))

    log_info "Found ${#benchmarks[@]} benchmarks to process"
    echo ""

    # Process each benchmark
    for benchmark in "${benchmarks[@]}"; do
        # Skip if not a valid benchmark directory
        if [ ! -d "${HIP_DIR}/${benchmark}" ]; then
            continue
        fi

        process_benchmark "$benchmark"
    done

    log_info "========================================"
    log_info "All benchmarks processed successfully!"
    log_info "========================================"
    echo ""
    log_info "Next steps:"
    log_info "1. Set environment variables:"
    log_info "   export CHAI_HIP_LIB=/opt/rocm/lib"
    log_info "   export CHAI_HIP_INC=/opt/rocm/include"
    log_info "   export ROCM_PATH=/opt/rocm"
    echo ""
    log_info "2. Build and test a benchmark:"
    log_info "   cd HIP-D/BFS"
    log_info "   make"
    log_info "   ./bfs -h"
    echo ""
    log_info "3. Review the HIP-PORTING-PLAN.md for detailed information"
}

# Run main function
main "$@"
