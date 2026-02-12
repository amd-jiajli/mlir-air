# Project Brief: MLIR-AIR

## Overview

**MLIR-AIR** (MLIR for AIE Runtime) is a compiler infrastructure that enables mapping AI/ML workloads onto AMD AI Engines (AIE) in NPU devices.

## Core Purpose

Transform high-level tensor/linalg MLIR representations into executable binaries (xclbin) that run on AMD NPU hardware through a series of transformations:
- Tiling for AIE tile sizes
- Vectorization for AIE vector units
- Memory hierarchy optimization (L1/L2/DDR)
- Lowering to AIE-specific constructs (tiles, cores, DMAs, channels)

## Target Devices

| Device | Architecture | Description |
|--------|-------------|-------------|
| **NPU1 (Phoenix)** | AIE2 | First-generation NPU |
| **NPU2 (Strix)** | AIE2P | Enhanced capabilities |

## Project Goals

1. **High-level Input**: Accept tensor/linalg MLIR from various frontends (Triton, IREE, hand-written)
2. **Automatic Optimization**: Apply transformations for optimal AIE utilization
3. **Hardware Mapping**: Generate efficient core assignments and data movement patterns
4. **Runtime Integration**: Produce XRT-compatible executables

## Success Metrics

- Correctness: Generated code produces correct results
- Performance: Efficient utilization of AIE compute and memory resources
- Usability: Clear compilation flow with debugging capabilities

## Key Stakeholders

- AI/ML developers targeting AMD NPU hardware
- Compiler engineers extending the infrastructure
- Researchers exploring novel optimization techniques

## References

- Wang et al., "[From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR](https://arxiv.org/abs/2510.14871)", arXiv:2510.14871, October 2025
- Repository: [github.com/Xilinx/mlir-air](https://github.com/Xilinx/mlir-air)
