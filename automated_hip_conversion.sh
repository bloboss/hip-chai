#!/bin/bash
# Automated HIP Conversion Script for CHAI Benchmarks
# This script converts all CUDA-D benchmarks in HIP-D to HIP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIP_DIR="${SCRIPT_DIR}/HIP-D"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CHAI CUDA to HIP Automated Conversion${NC}"
echo "========================================"
echo ""

# Function to convert a single benchmark
convert_benchmark() {
    local benchmark=$1
    local benchmark_dir="${HIP_DIR}/${benchmark}"

    echo -e "${YELLOW}Converting ${benchmark}...${NC}"

    cd "${benchmark_dir}"

    # 1. Convert main.cpp
    if [ -f "main.cpp" ]; then
        echo "  - Converting main.cpp to HIP"
        ${SCRIPT_DIR}/convert_cuda_to_hip.sh main.cpp > /dev/null 2>&1
    fi

    # 2. Convert kernel.h
    if [ -f "kernel.h" ]; then
        echo "  - Converting kernel.h to HIP"
        ${SCRIPT_DIR}/convert_cuda_to_hip.sh kernel.h > /dev/null 2>&1
    fi

    # 3. Convert kernel.cu to kernel.cu.hip
    if [ -f "kernel.cu" ]; then
        echo "  - Converting kernel.cu to HIP"
        cp kernel.cu kernel.cu.bak
        ${SCRIPT_DIR}/convert_cuda_to_hip.sh kernel.cu > /dev/null 2>&1
        mv kernel.cu kernel.cu.hip

        # Add HIP header to kernel file
        sed -i '1i\#include <hip/hip_runtime.h>' kernel.cu.hip

        # Replace CUDA kernel launch with HIP
        # Change <<<>>> to hipLaunchKernelGGL where needed
        # This is a simplified approach - may need manual review
    fi

    # 4. Create hip-setup.h in support directory
    if [ -f "support/cuda-setup.h" ] && [ ! -f "support/hip-setup.h" ]; then
        echo "  - Creating support/hip-setup.h"
        cp "${HIP_DIR}/BFS/support/hip-setup.h" support/hip-setup.h
    fi

    # 5. Create Makefile.hip
    if [ -f "Makefile" ]; then
        echo "  - Creating Makefile.hip"
        cat > Makefile.hip << 'EOF'
CXX=hipcc
CXX_FLAGS=-std=c++11

# Check for HIP library path
ifndef CHAI_HIP_LIB
    CHAI_HIP_LIB=/opt/rocm/lib
endif

# Check for HIP include path
ifndef CHAI_HIP_INC
    CHAI_HIP_INC=/opt/rocm/include
endif

# Check for ROCm path
ifndef ROCM_PATH
    ROCM_PATH=/opt/rocm
endif

LIB=-L/usr/lib/ -L$(CHAI_HIP_LIB) -L$(ROCM_PATH)/lib -lm -lamdhip64
INC=-I$(CHAI_HIP_INC) -I$(ROCM_PATH)/include

EOF

        # Extract dependencies and sources from original Makefile
        DEP_LINE=$(grep "^DEP=" Makefile | sed 's/kernel\.cu/kernel.cu.hip/g' | sed 's/cuda-setup/hip-setup/g')
        SRC_LINE=$(grep "^SRC=" Makefile | sed 's/kernel\.cu/kernel.cu.hip/g')
        EXE_LINE=$(grep "^EXE=" Makefile)

        echo "$DEP_LINE" >> Makefile.hip
        echo "$SRC_LINE" >> Makefile.hip
        echo "$EXE_LINE" >> Makefile.hip

        cat >> Makefile.hip << 'EOF'

all:
	$(CXX) $(CXX_FLAGS) $(SRC) $(LIB) $(INC) -o $(EXE)

clean:
	rm -f $(EXE)
EOF
    fi

    echo -e "  ${GREEN}✓ ${benchmark} converted${NC}"
    cd "${SCRIPT_DIR}"
}

# Get list of benchmarks
BENCHMARKS=($(ls -d ${HIP_D}/*/ 2>/dev/null | xargs -n 1 basename))

if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    echo "Error: No benchmarks found in HIP-D directory"
    exit 1
fi

echo "Found ${#BENCHMARKS[@]} benchmarks to convert"
echo ""

# Convert each benchmark
for benchmark in BFS BS CEDD CEDT HSTI HSTO PAD RSCD RSCT SC SSSP TQ TQH TRNS; do
    if [ -d "${HIP_DIR}/${benchmark}" ]; then
        convert_benchmark "$benchmark"
    fi
done

echo ""
echo -e "${GREEN}========================================"
echo "Conversion Complete!"
echo -e "========================================${NC}"
echo ""
echo "To build a benchmark with HIP:"
echo "  cd HIP-D/<benchmark>"
echo "  cp Makefile.hip Makefile"
echo "  make"
echo ""
echo "Or set environment variables and build directly:"
echo "  export CHAI_HIP_LIB=/opt/rocm/lib"
echo "  export CHAI_HIP_INC=/opt/rocm/include"
echo "  export ROCM_PATH=/opt/rocm"
echo "  cd HIP-D/BFS"
echo "  make -f Makefile.hip"
