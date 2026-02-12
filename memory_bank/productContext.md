# Product Context: MLIR-AIR Purpose & Vision

## Why This Project Exists

MLIR-AIR bridges the gap between high-level AI/ML workloads and AMD's NPU hardware. It solves the fundamental challenge of efficiently mapping complex tensor computations onto a heterogeneous array of AI Engine (AIE) compute tiles.

## Problems It Solves

### 1. Complexity of NPU Programming
**Problem:** AMD NPUs contain arrays of AI Engine tiles with complex memory hierarchies, DMA engines, and interconnect networks. Direct programming is extremely difficult.

**Solution:** MLIR-AIR provides high-level abstractions (`air.herd`, `air.channel`) that automatically handle:
- Physical tile placement
- Data movement orchestration
- Synchronization and dependencies
- Memory allocation across hierarchy levels

### 2. Performance Optimization Gap
**Problem:** Getting good performance on NPUs requires careful optimization of:
- Data tiling for memory hierarchy
- Vectorization for AIE vector units
- Double-buffering (ping-pong) for latency hiding
- Channel routing to avoid congestion

**Solution:** Transform dialect scripts allow declarative specification of these optimizations, with the compiler handling low-level details.

### 3. Portability Across NPU Generations
**Problem:** Different NPU versions (NPU1/AIE2, NPU2/AIE2P) have different capabilities and constraints.

**Solution:** The AIR dialect abstracts device-specific details, allowing the same high-level code to target multiple devices with device-specific lowering.

## Target Users

### Primary Users
1. **AI/ML Framework Developers**: Integrating NPU support into frameworks like PyTorch, TensorFlow
2. **Compiler Engineers**: Extending the compiler with new optimizations
3. **Performance Engineers**: Optimizing specific workloads for NPU execution

### Secondary Users
1. **Researchers**: Exploring novel compilation techniques for spatial architectures
2. **Hardware Architects**: Evaluating workload mappings and identifying hardware improvements

## User Experience Goals

### For Workload Developers
- Write high-level tensor operations
- Use transform scripts to specify optimization strategy
- Get efficient NPU code without manual low-level programming

### For Compiler Developers
- Clear pass pipeline architecture
- Well-defined dialect boundaries
- Comprehensive debugging tools (mlir-browser, IR printing)

### For Performance Engineers
- Profiling integration
- Ability to inspect all intermediate representations
- Fine-grained control via transform dialect

## How It Should Work

### Ideal Flow
```
1. User writes/generates high-level MLIR (linalg/tensor)
2. User provides transform script specifying optimization strategy
3. Compiler handles:
   - Tiling for memory hierarchy
   - Vectorization for AIE units
   - Placement of computation on tiles
   - Data movement scheduling
   - Synchronization
4. User gets xclbin that runs efficiently on NPU
```

### Key Interaction Points
1. **Input MLIR**: From Triton, IREE, or hand-written
2. **Transform Script**: User-specified optimization strategy
3. **Device Selection**: NPU1 vs NPU2
4. **Output**: xclbin + instruction binary for XRT

## Success Criteria

1. **Correctness**: Generated code produces mathematically correct results
2. **Performance**: Competitive with hand-optimized implementations
3. **Usability**: Clear error messages, good debugging, documented APIs
4. **Maintainability**: Modular passes, clear dialect boundaries

## Context Within AMD Ecosystem

```
┌─────────────────────────────────────────────────────────┐
│  AI Frameworks (PyTorch, TensorFlow, etc.)              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Frontend Compilers (Triton, IREE, etc.)                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ Linalg/Tensor MLIR
┌─────────────────────────────────────────────────────────┐
│  MLIR-AIR ← This project                                │
│  • Transform dialect optimization                        │
│  • AIR dialect (high-level NPU abstraction)             │
│  • AIE dialect (physical mapping)                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ xclbin
┌─────────────────────────────────────────────────────────┐
│  XRT Runtime                                             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  AMD NPU Hardware                                        │
└─────────────────────────────────────────────────────────┘
