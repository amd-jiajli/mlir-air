# Active Context: Current Work Focus

## Current Example

**Directory:** `test/xrt/42_triton_softmax_bf16/`

This is the primary example currently being worked on for optimization.

## Current Goal

Optimize the transform script for better performance on softmax operations.

## Key Files

| File | Purpose |
|------|---------|
| `test/xrt/42_triton_softmax_bf16/transform_aie2p.mlir` | Transform script to modify |
| `test/xrt/42_triton_softmax_bf16/run.py` | Python test driver |
| `test/xrt/42_triton_softmax_bf16/Makefile` | Build and test commands |

## Testing Commands

```bash
cd test/xrt/42_triton_softmax_bf16

# Check correctness (compiles and verifies output)
make run

# Check performance (runs profiling)
make profile

# Per-pass IR output for aircc stage
make run DEBUG_AIRCC=1
```

## Debugging Transform Scripts

Transform scripts run as a single MLIR pass, so `DEBUG_AIRCC=1` won't show internal steps.

**To debug transform script internals:**
```mlir
// Add to transform script
transform.print %arg1 {name = "=== After step X ==="} : !pdl.operation
```

## Recent Changes

- Created comprehensive transform pipeline analysis (`memory_bank/42_triton_softmax/transform_pipeline_analysis.md`)
- Identified that `split_reduction` cannot be used (creates 2D vectors, AIE only supports 1D)
- Found the critical step where reductions enter loops: step [45] `tile_using_for [0, 32]`

## Active Decisions

**Optimization Goal:** Move `vector.reduction` operations outside of `scf.for` loops for better performance.

**Current Approach:** Explore existing dialect operations and transform patterns before implementing custom transforms.

**Key Constraint:** AIE only supports 1D vectors (`vector<32xbf16>`), NOT 2D vectors.

## Current Blockers

1. **AIE 1D vector constraint** - Standard `split_reduction` creates 2D vectors which AIE rejects
2. **Reductions inside loops** - Step [45] `tile_using_for [0, 32]` places reductions in loop body

## Next Steps

1. ✅ Understand transform pipeline step-by-step (DONE - see `transform_pipeline_analysis.md`)
2. ✅ Identify intervention point (DONE - step [45] or post-vectorization)
3. ⬜ Explore existing dialect ops to achieve reduction hoisting
4. ⬜ Test modifications with softmax example
5. ⬜ Verify correctness and measure performance improvement

## Quick Links

- Transform script: `test/xrt/42_triton_softmax_bf16/transform_aie2p.mlir`
- Compilation pipeline: `memory_bank/systemPatterns.md`
- Debugging guide: `memory_bank/techContext.md`
