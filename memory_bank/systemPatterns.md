# System Patterns: MLIR-AIR Architecture

## Dialect Hierarchy

```
┌─────────────────────────────────────────────┐
│  Input: Linalg/Tensor MLIR                  │
│  (from Triton, IREE, or hand-written)       │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  AIR Dialect (High-Level Abstraction)       │
│  • air.launch, air.segment, air.herd        │
│  • air.channel, air.dma_memcpy_nd           │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  AIE Dialect (Physical Mapping)             │
│  • aie.tile, aie.core, aie.lock             │
│  • aie.objectFifo, aie.buffer               │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Hardware (AMD NPU)                         │
│  • AIE cores, DMAs, switchboxes             │
│  • xclbin + instruction binary              │
└─────────────────────────────────────────────┘
```

## Dialects Overview

| Dialect | Purpose |
|---------|---------|
| **AIR** | High-level abstraction for parallel execution on AIE arrays |
| **AIRRt** | Runtime dialect for host-device interaction |
| **AIE** | Low-level dialect representing physical AIE tiles, cores, locks, buffer descriptors |

## Memory Hierarchy

| Memory Space | Code | Size per Tile | Description |
|-------------|------|---------------|-------------|
| DDR | 0 | Shared | Global memory (host accessible) |
| L2 (MemTile) | 1 | 256KB | Shared memory across cores |
| L1 (Local) | 2 | 64KB | Per-core local memory |

## Key Abstractions

| Abstraction | Description |
|-------------|-------------|
| **air.launch** | Host-side invocation of AIE computation |
| **air.segment** | A partition of the AIE array |
| **air.herd** | A 2D array of compute tiles executing the same code |
| **air.channel** | Data communication abstraction between memory levels |
| **air.dma_memcpy_nd** | N-dimensional DMA transfer |

## Compilation Pipeline

### Stage 1: Transform Dialect Application
- Input: Linalg/Tensor MLIR
- Transform script applies: tiling, memory allocation, vectorization, AIR construct creation
- Output: Tiled/vectorized AIR MLIR

### Stage 2: run.py Pre-processing
- `air-override-memref-memory-space`: Set default memory space
- `air-wrap-func-with-parallel`: Create launch grid
- `air-par-to-launch`: Create air.launch
- `air-copy-to-dma`: Convert memory copies

### Stage 3: aircc.py - AIR Optimization & Lowering
**Optimization passes:**
- `air-dependency`: Build dependency graph
- `air-broadcast-detection`: Detect broadcast patterns
- `air-dma-to-channel`: Convert DMAs to channels
- `air-fuse-channels`: Multiplex channels
- `air-ping-pong-transform`: Double buffering

**Placement passes:**
- `air-collapse-herd`: Reshape herds to fit columns
- `air-place-herds`: Physical tile assignment

**Conversion:**
- `air-to-aie`: Generate AIE dialect
- `airrt-to-npu`: Generate NPU instructions

### Stage 4: aiecc.py - AIE Compilation
- `convert-vector-to-aievec`: Vector → AIE intrinsics
- `aie-objectFifo-stateful-transform`: Lower objectFIFOs
- `aie-create-pathfinder-flows`: Physical routing
- Core compilation: LLVM → ELF per core
- Packaging: CDO → PDI → xclbin

### Stage 5: Runtime Execution
- XRT loads xclbin
- Send instructions, transfer data
- Execute and verify results

## Key Intermediate Files

| Stage | File | Description |
|-------|------|-------------|
| After placement | `placed.air.mlir` | AIR with physical tile assignments |
| After AIR→AIE | `aie.air.mlir` | AIE dialect with cores, locks, buffers |
| After NPU gen | `npu.air.mlir` | With NPU instruction sequence |
| After routing | `input_physical.mlir` | With switchbox configuration |
| Final | `air.xclbin` | Packaged executable |
| Final | `air.insts.bin` | NPU instruction binary |

## Design Patterns

### Transform Script Pattern
Transform scripts use MLIR's transform dialect to apply transformations declaratively:
```mlir
transform.sequence %arg0 : !pdl.operation failures(propagate) {
^bb1(%arg1: !pdl.operation):
    // Match, transform, fuse, tile, vectorize
}
```

### Memory Hierarchy Pattern
1. Tile outer dimensions for L2 (batch/rows)
2. Tile inner dimensions for L1 (compute blocks)
3. Vectorize innermost loops for AIE vector units

### Data Movement Pattern
- Use `air.channel` for L2↔L1 transfers
- Apply ping-pong buffering for overlap
- Fuse channels to reduce routing pressure

## Architecture Constraints

### AIE2/AIE2P Vector Widths
| Data Type | Vector Width |
|-----------|-------------|
| bf16 | 16 lanes |
| int8/int4 | 32 lanes |

### Known Limitations
- AIE only supports 1D vectors (no 2D vector operations)
- Routing congestion limits channel count
- Buffer sizes constrained by tile memory (64KB L1, 256KB L2)
