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

DEP=kernel.cpp kernel.h main.cpp kernel.cu.hip support/common.h support/hip-setup.h support/timer.h support/verify.h
SRC=main.cpp kernel.cpp kernel.cu.hip
EXE=rsct

all:
	$(CXX) $(CXX_FLAGS) $(SRC) $(LIB) $(INC) -o $(EXE)

clean:
	rm -f $(EXE)
