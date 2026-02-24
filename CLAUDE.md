# MLIR-AIR

MLIR-AIR is a compiler framework that maps AI/ML workloads onto AMD NPU hardware using MLIR.

- **Input**: Linalg/Tensor MLIR (from Triton, IREE, or hand-written)
- **Output**: `xclbin` + `insts.bin` (executable on AMD NPU via XRT)
- **Targets**: NPU1 (Phoenix, AIE2 architecture), NPU2 (Strix, AIE2P architecture)

## Environment Setup

Run once per shell session before any build/run commands:

```bash
cd /home/jiajli/apps/mlir-air
source ./sandbox/bin/activate
source ./utils/env_setup.sh install/ /home/jiajli/apps/mlir-air/my_install/mlir-aie/install $(python3 -m pip show llvm-aie | grep Location | awk '{print $2}')/llvm-aie my_install/mlir
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `mlir/lib/Conversion/` | Dialect lowering passes (AIR->AIE, AIRRt->NPU, etc.) |
| `mlir/lib/Transform/` | Optimization passes (dependency, placement, tiling, etc.) |
| `mlir/include/air/` | Dialect definitions (TableGen + headers) |
| `mlir/test/` | Lit tests for individual passes |
| `python/air/compiler/aircc/` | `aircc.py` compiler driver (main.py, cl_arguments.py) |
| `python/air/backend/` | Python backends (xrt.py, xrt_runner.py, cpu_backend.py) |
| `tools/air-opt/` | Main MLIR pass driver |
| `tools/aircc/` | Compiler driver entry point (aircc.py) |
| `test/xrt/` | End-to-end XRT tests (each subdir has Makefile + transform scripts) |
| `programming_examples/` | Example programs (softmax, flash_attention, GEMM, etc.) |
| `runtime_lib/` | Host and CPU runtime libraries |

## Compilation Pipeline

```
Linalg MLIR
    |  [1] Transform Script (transform_aie2p.mlir) -- tiling, vectorization, memory hierarchy
    v
AIR MLIR
    |  [2] run.py -- memory space override, launch grid, DMA conversion
    v
    |  [3] aircc.py -- dependency, broadcast, placement, AIR->AIE, NPU instr gen
    v
AIE MLIR
    |  [4] aiecc.py -- vector lowering, routing, ELF gen, CDO/PDI, xclbin packaging
    v
xclbin + insts.bin
    |  [5] XRT Runtime -- load xclbin, send instructions, transfer data, verify
    v
Results
```

Key intermediate files: `placed.air.mlir`, `aie.air.mlir`, `npu.air.mlir`, `input_physical.mlir`, `air.xclbin`, `air.insts.bin`

## Key Concepts

**Dialects**: AIR (high-level parallel execution), AIRRt (host-device runtime), AIE (physical tiles/cores/DMAs)

**Memory hierarchy**:
- Space 0: DDR (host accessible)
- Space 1: L2 / MemTile (shared, 256KB/tile)
- Space 2: L1 / Local (per-core, 64KB/tile)

**Core abstractions**: `air.launch` (host invocation), `air.segment` (array partition), `air.herd` (2D compute tile array), `air.channel` (data communication), `air.dma_memcpy_nd` (N-D DMA transfer)

## Tools & Commands

### air-opt
```bash
air-opt input.mlir -canonicalize -cse -o output.mlir
air-opt input.mlir --air-transform="filename=transform.mlir" -o output.mlir
air-opt input.mlir -pass1 --mlir-print-ir-after-all 2>&1   # debug IR
```

### aircc.py
```bash
aircc.py input.mlir -o output.xclbin --device npu2
aircc.py input.mlir -o output.xclbin --device npu2 -v          # verbose
aircc.py input.mlir -o output.xclbin --device npu2 --debug-aircc  # per-pass IR
```

### Build & run tests
```bash
cd test/xrt/<test_name>
make run                    # build and run on NPU
make run DEBUG_AIRCC=1      # with per-pass IR output (saved to build_peano/air_project/debug_ir/)
make compile-xclbin         # compile only
make profile                # profile performance
make run TARGET=npu1 M=512 N=256  # custom target/dimensions
```

## Debugging

1. **transform.print** (best for transform scripts): Add `transform.print %arg1 {name = "checkpoint"} : !pdl.operation` in the transform script to inspect IR at specific points. Output goes to stderr.

2. **DEBUG_AIRCC=1**: Run `make run DEBUG_AIRCC=1`. Per-pass IR files saved to `build_peano/air_project/debug_ir/` as `pass_XXX_after_<passname>.mlir`.

3. **air-opt flags**: `--mlir-print-ir-after-all`, `--mlir-print-ir-after-change`, `--mlir-print-ir-before=<pass>`, `--mlir-print-debuginfo`, `--mlir-print-stacktrace-on-diagnostic`.

4. **Common issues**: Pass not finding operations (use `transform.debug.emit_remark_at` to inspect), transform script failing silently (check stderr), compilation hanging (check dependency resolution or routing).

## Transform Script Development

Transform scripts define how high-level operations map to AIE hardware. Located at `test/xrt/*/transform_aie2p.mlir` (or `transform_aie2.mlir` for NPU1).

**Key operations**: tile for AIE sizes, allocate buffers to L1/L2, vectorize for AIE vector units (16-lane bf16, 32-lane int8), create herds and DMAs.

**Patterns**:
- Tile outermost dimensions first for better data locality
- Assign L2 (memory_space=1) to intermediate buffers, L1 (memory_space=2) to innermost computation
- AIE2P: 16 lanes for bf16, 32 lanes for int8/int4

**Pitfalls**:
- Ping-pong buffers must have matching tile sizes between producer and consumer
- Too aggressive channel fusion can cause routing congestion
- ObjectFifo sizing must account for double-buffering
- AIE only supports 1D vectors; transforms producing 2D vectors will fail at backend

## Detailed Reference Documentation

For deeper dives, see `docs_ai_generated/`:
- `PROJECT_OVERVIEW.md` — comprehensive project overview
- `compilation_pipeline_detailed.md` — 879-line pass-by-pass pipeline breakdown
- `README_debugging.md` — in-depth debugging techniques
- `air-opt.help.doc` — all air-opt passes and options
- `aie-opt.help.doc` — all aie-opt passes and options
- `memory/project_knowledge.md` — concise project reference
