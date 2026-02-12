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

*Track recent modifications here*

## Active Decisions

*Document active technical decisions being made*

## Current Blockers

See `progress.md` for detailed status of optimization attempts.

## Next Steps

1. Review current transform script performance
2. Identify optimization opportunities
3. Test modifications for correctness
4. Profile for performance improvements

## Quick Links

- Transform script: `test/xrt/42_triton_softmax_bf16/transform_aie2p.mlir`
- Compilation pipeline: `memory_bank/systemPatterns.md`
- Debugging guide: `memory_bank/techContext.md`
