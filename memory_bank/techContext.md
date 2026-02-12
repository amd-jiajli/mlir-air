# Technical Context: MLIR-AIR Development

> **Note:** Environment setup is in `.clinerules/02-env_setup.md`

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `mlir/lib/Conversion/` | Dialect lowering passes (AIR→AIE, etc.) |
| `mlir/lib/Transform/` | Optimization passes |
| `python/air/compiler/aircc/` | Python compiler driver (aircc.py) |
| `test/xrt/` | End-to-end XRT tests |
| `programming_examples/` | Example programs |
| `tools/air-opt/` | Main optimization driver |
| `tools/mlir-browser/` | IR visualization tool |
| `docs_ai_generated/` | AI-accumulated documentation |

## Key Tools

### 1. air-opt

The main pass driver for AIR/AIE transformations.

```bash
# List all registered passes
air-opt --list-passes

# Run specific passes
air-opt input.mlir -canonicalize -cse -o output.mlir

# Apply transform script
air-opt input.mlir \
  -air-override-memref-memory-space="scope=func memory-space=1" \
  --air-transform="filename=transform.mlir" \
  -o output.mlir

# Debug: print IR after all passes
air-opt input.mlir -pass1 -pass2 --mlir-print-ir-after-all 2>&1 | tee log.txt
```

### 2. aircc.py

The compiler driver that orchestrates the full compilation flow.

```bash
# Basic usage
aircc.py input.mlir -o output.xclbin --device npu2

# With verbose output
aircc.py input.mlir -o output.xclbin --device npu2 -v

# With debug IR output (per-pass)
aircc.py input.mlir -o output.xclbin --device npu2 --debug-aircc
```

### 3. mlir-browser

Interactive HTML visualization of IR transformations.

```bash
# Debug IR mode (recommended for fine-grained debugging)
python3 tools/mlir-browser/mlir_browser.py \
  --debug-ir build_peano/air_project/debug_ir \
  -o browser.html --no-server

# Aggregate mode (browse intermediate files)
python3 tools/mlir-browser/mlir_browser.py \
  --aggregate build_peano/air_project/ \
  -o browser.html --no-server
```

## Debugging Methods

### Method 1: transform.print (Recommended for Transform Scripts)

Add print statements directly to transform script:
```mlir
transform.sequence %arg0 : !pdl.operation failures(propagate) {
^bb1(%arg1: !pdl.operation):
    transform.print %arg1 {name = "=== Before transforms ==="} : !pdl.operation
    // ... transformations ...
    transform.print %arg1 {name = "=== After transforms ==="} : !pdl.operation
}
```

### Method 2: DEBUG_AIRCC Mode

```bash
# Enable per-pass IR output
make run DEBUG_AIRCC=1

# Output files saved to: build_peano/air_project/debug_ir/
```

### Method 3: air-opt IR Printing Flags

| Flag | Description |
|------|-------------|
| `--mlir-print-ir-after-all` | Print IR after every pass |
| `--mlir-print-ir-after-change` | Print only when IR changes |
| `--mlir-print-ir-before=<pass>` | Print IR before specific pass |
| `--mlir-print-debuginfo` | Include debug information |
| `--mlir-print-op-generic` | Print in generic form |
| `--dump-pass-pipeline` | Show the pass pipeline |
| `--mlir-print-stacktrace-on-diagnostic` | Stack trace on errors |

## Tool References

Full documentation available in:
- `docs_ai_generated/air-opt.help.doc` - All air-opt passes and options
- `docs_ai_generated/aie-opt.help.doc` - All aie-opt passes and options

## Build and Test Commands

```bash
cd test/xrt/<test_name>

# Run with default settings (NPU2)
make run

# Run on NPU1 with custom dimensions
make TARGET=npu1 M=512 N=256 run

# Compile only (no execution)
make compile-xclbin

# Profile performance
make profile

# Debug with per-pass IR output
make run DEBUG_AIRCC=1
```

## Dependencies

- **MLIR/LLVM**: Core compiler infrastructure
- **mlir-aie**: AIE dialect and lowering passes
- **Peano**: LLVM-based AIE compiler (for core ELF generation)
- **XRT**: Xilinx Runtime (for execution)
- **bootgen/xclbinutil**: Packaging tools

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Pass not finding operations | Use `transform.debug.emit_remark_at` to inspect available ops |
| Transform script fails silently | Check stderr for `transform.print` output |
| Compilation hangs | Look for infinite loops in dependency resolution or routing |
| Vector type errors | AIE only supports 1D vectors; avoid 2D vector shapes |
