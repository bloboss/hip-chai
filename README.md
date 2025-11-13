# Chai

v1.0-alpha

## Overview

Chai is a benchmark suite of Collaborative Heterogeneous Applications for Integrated-architectures. The Chai benchmarks are designed to use the latest features of heterogeneous architectures such as shared virtual memory and system-wide atomics to achieve efficient simultaneous collaboration between host and accelerator devices.

Each benchmark has multiple implementations. This release includes the OpenCL-D, OpenCL-U, CUDA-D, CUDA-U, CUDA-D-Sim, CUDA-U-Sim, and HIP-D implementations. The C++AMP implementations are underway. If you would like early access to premature versions, please contact us.

**NEW**: HIP-D implementation now available for AMD GPUs and alternative computing platforms!

## Instructions

Clone the repository:

  ```
  git clone https://github.com/chai-benchmarks/chai.git
  cd chai
  ```

Export environment variables:

  ```
  export CHAI_OCL_LIB=<path/to/OpenCL/lib>
  export CHAI_OCL_INC=<path/to/OpenCL/include>
  ```

Select desired implementation:

  ```
  cd OpenCL-U
  ```

Select desired benchmark:

  ```
  cd BFS
  ```

Compile:

  ```
  make
  ```

Execute:

  ```
  ./bfs
  ```

For help instructions:

  ```
  ./bfs -h
  ```

## HIP-D Implementation (AMD GPUs)

The HIP-D implementation enables CHAI benchmarks to run on AMD GPUs and other alternative computing platforms using the HIP (Heterogeneous-compute Interface for Portability) framework.

### Quick Start with HIP

Export environment variables:

  ```
  export CHAI_HIP_LIB=/opt/rocm/lib
  export CHAI_HIP_INC=/opt/rocm/include
  export ROCM_PATH=/opt/rocm
  ```

Select HIP implementation:

  ```
  cd HIP-D
  ```

Select desired benchmark:

  ```
  cd BFS
  ```

Compile:

  ```
  make
  ```

Execute:

  ```
  ./bfs -h
  ```

### Porting CUDA to HIP

To port CUDA benchmarks to HIP, use the provided automation tools:

**Using Docker (Recommended)**:

  ```
  docker build -f Dockerfile.hipify -t chai-hipify .
  docker run -it --rm -v $(pwd):/workspace chai-hipify
  ./port_to_hip.sh
  ```

**Using Local ROCm Installation**:

  ```
  ./port_to_hip.sh
  ```

For detailed porting instructions, see [HIP-PORTING-PLAN.md](HIP-PORTING-PLAN.md).

## Citation

Please cite the following paper if you find our benchmark suite useful:

* J. Gómez-Luna, I. El Hajj, L.-W. Chang, V. Garcia-Flores, S. Garcia de Gonzalo, T. Jablin, A. J. Peña, W.-M. Hwu.
  **Chai: Collaborative Heterogeneous Applications for Integrated-architectures.**
  In *Proceedings of IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2017.
  [\[bibtex\]](https://chai-benchmarks.github.io/assets/ispass17.bib)

## Chai Benchmarks for CPU-FPGA Systems
The FPGA synthesizable version of Chai benchmarks can be found in this <a href="https://github.com/chai-benchmarks/chai-fpga" target="_blank">chai-fpga</a> repository. 
