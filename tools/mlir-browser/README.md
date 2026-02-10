# MLIR Browser

An interactive visualization tool for exploring MLIR IR transformations step-by-step.

## Features

- **Pass Timeline**: View the sequence of passes and their effects on the IR
- **Interactive Navigation**: Use slider, buttons, or keyboard to navigate between snapshots
- **Syntax Highlighting**: MLIR-aware syntax highlighting for better readability
- **Diff View**: Three modes for comparing consecutive IR states:
  - **Split+Diff**: Side-by-side with line highlighting (added=green, removed=red)
  - **Unified**: Git-style unified diff with `+`/`-` prefixes
- **Search**: Find specific patterns in the IR
- **Aggregate Mode**: Browse existing intermediate files from a compilation
- **Keyboard Shortcuts**: 
  - `←` `→` Navigate between passes
  - `D` Cycle through view modes (Single → Split+Diff → Unified)
  - `U` Switch to Unified diff view
  - `S` Switch to Single view
  - `Home` / `End` Jump to first/last

## Quick Start

### 1. Debug IR Mode - Browse pass-by-pass IR files from DEBUG_AIRCC=1

The most detailed view: see the IR after **every single pass** in the compilation pipeline.

**Step 1: Run with DEBUG_AIRCC=1:**
```bash
cd test/xrt/42_triton_softmax_bf16
make run DEBUG_AIRCC=1
```

**Step 2: Generate interactive browser:**
```bash
python3 tools/mlir-browser/mlir_browser.py \
  --debug-ir build_peano/air_project/debug_ir \
  -o debug_browser.html \
  --no-server
```

**Step 3: Open in browser:**
```bash
firefox debug_browser.html  # or your preferred browser
```

This loads all 55 pass files with:
- Proper pass names from `pass.log` (including full arguments like `air-place-herds{num-rows=6 num-cols=8}`)
- Checkpoint markers (★) for important milestones (AIR Placement, AIR to AIE, NPU Instructions)
- Side-by-side diff view to see exactly what each pass changes

### 2. Transform Debug Mode - Browse transform.print output

The most common use case for MLIR-AIR development: debug transform scripts by viewing IR after each phase.

**Step 1: Add `transform.print` to your transform script:**
```mlir
// In transform_aie2p.mlir, after each phase:
transform.print %arg1 {name = "After Phase 1"} : !pdl.operation
// ... more transformations ...
transform.print %arg1 {name = "After Phase 2"} : !pdl.operation
```

**Step 2: Run and capture output:**
```bash
cd test/xrt/42_triton_softmax_bf16
make run 2>&1 | tee ir_emitted/output.log
```

**Step 3: Generate interactive browser:**
```bash
python3 tools/mlir-browser/mlir_browser.py \
  --transform-log ir_emitted/output.log \
  -o ir_emitted/transform_browser.html \
  --no-server
```

**Step 4: Open in browser:**
```bash
# Open the generated HTML file
firefox ir_emitted/transform_browser.html  # or your preferred browser
```

### 2. Basic usage with simple passes

```bash
cd tools/mlir-browser

python mlir_browser.py input.mlir --passes "canonicalize;cse"
```

### 2. Usage with transform script (for MLIR-AIR)

```bash
python mlir_browser.py input.mlir \
  --pre-passes "-air-override-memref-memory-space=scope=func memory-space=1" \
  --transform transform.mlir \
  --output browser.html \
  --no-server
```

### 3. Full matmul example

```bash
cd /home/jiajli/apps/mlir-air/test/xrt/47_triton_matmul_ver4_strix_8x4_int8_int32

python3 /home/jiajli/apps/mlir-air/tools/mlir-browser/mlir_browser.py \
  asm_src.mlir \
  --pre-passes "-air-override-memref-memory-space=scope=func memory-space=1" \
  --transform transform_aie2p.mlir \
  --passes "air-wrap-func-with-parallel=loop-bounds=2,4,1;air-par-to-launch=depth=0 has-air-segment=true;canonicalize;cse;air-copy-to-dma" \
  --output matmul_browser.html \
  --no-server
```

This captures 8 IR snapshots:

### 4. Aggregate mode - Browse existing intermediate files

After running `make run` in a test directory, you can browse all the intermediate MLIR files:

```bash
# First, run the test to generate intermediate files
cd /home/jiajli/apps/mlir-air/test/xrt/42_triton_softmax_bf16
make run

# Then browse the intermediate files
python3 /home/jiajli/apps/mlir-air/tools/mlir-browser/mlir_browser.py \
  --aggregate build_peano/air_project/ \
  --output pipeline_browser.html \
  --no-server
```

This captures all intermediate MLIR files from the full compilation pipeline:
- **AIR Placement** (`placed.air.mlir`) - After aircc placement passes
- **AIR to AIE** (`aie.air.mlir`) - After air-to-aie conversion
- **NPU Instructions** (`npu.air.mlir`) - After airrt-to-npu
- **Buffer Addresses** (`input_with_addresses.mlir`) - After aiecc address assignment
- **Physical Routing** (`input_physical.mlir`) - After routing
- **ELF Injection** (`input_physical_with_elfs.mlir`) - After ELF generation
- **NPU Lowering** (`npu_insts.mlir`) - After NPU lowering pipeline
- **LLVM Lowering** (`*_input_opt_with_addresses.mlir`) - After LLVM lowering

This mode is great for understanding the full compilation pipeline without re-running compilation.

### 5. Original pass-by-pass capture example
1. **Initial IR** - Original input
2. **After AIROverrideMemRefMemorySpace** - Memory space override
3. **After AIRTransformInterpreterPass** - Transform script applied
4. **After AIRWrapFuncWithParallelPass** - Parallel wrapping
5. **After ParallelToLaunch** - Parallel to launch conversion
6. **After Canonicalizer** - Canonicalization
7. **After CSE** - Common subexpression elimination
8. **After CopyToDma** - Copy to DMA conversion

## Command Line Options

```
usage: mlir_browser.py [-h] [--passes PASSES] [--transform TRANSFORM]
                       [--pre-passes PRE_PASSES] [--air-opt-path AIR_OPT_PATH]
                       [--port PORT] [--export-only] [--output OUTPUT]
                       [--no-server] [--aggregate BUILD_DIR]
                       [--transform-log LOG_FILE] [--debug-ir DEBUG_IR_DIR]
                       [input]

Arguments:
  input                    Input MLIR file (not required for aggregate/transform-log/debug-ir mode)

Options:
  --passes, -p             Semicolon-separated list of passes
                           Use semicolons to separate passes with arguments
                           Example: "canonicalize;cse;air-copy-to-dma"
  --transform, -t          Transform script file
  --pre-passes             Passes to run before main passes
                           Example: "-air-override-memref-memory-space=scope=func memory-space=1"
  --air-opt-path           Path to air-opt binary (auto-detected if not specified)
  --port                   Port for web server (default: 8080)
  --export-only            Only export JSON, don't generate HTML
  --output, -o             Output HTML file path (default: mlir_browser.html)
  --no-server              Don't start the web server, just generate HTML
  --aggregate, -a          Aggregate mode: read existing intermediate MLIR files
                           from a build directory (e.g., build_peano/air_project/)
  --transform-log, -l      Transform log mode: parse transform.print output from
                           a log file (for debugging transform scripts)
  --debug-ir, -d           Debug IR mode: read pass-by-pass IR files from debug_ir
                           directory generated by DEBUG_AIRCC=1
                           Example: --debug-ir build_peano/air_project/debug_ir
```

## Transform Log Mode Details

Transform log mode (`--transform-log`) is designed to work with `transform.print` output from transform scripts. This is the recommended way to debug transform scripts in MLIR-AIR.

**How it works:**
1. Add `transform.print` operations to your transform script after each phase
2. Run the compilation and capture stdout/stderr to a log file
3. Use `--transform-log` to parse the log and generate an interactive browser

**The log file should contain markers like:**
```
[[[ IR printer: <phase name> ]]]
<IR content>
```

**Example workflow:**
```bash
# 1. Run with transform.print and capture output
cd test/xrt/42_triton_softmax_bf16
make run 2>&1 | tee ir_emitted/output.log

# 2. Generate browser
python3 tools/mlir-browser/mlir_browser.py \
  --transform-log ir_emitted/output.log \
  -o transform_browser.html \
  --no-server

# 3. View in browser
firefox transform_browser.html
```

**Benefits:**
- See exactly how each transform operation affects the IR
- Compare IR states side-by-side with diff highlighting
- Navigate through phases using keyboard shortcuts
- Search for specific patterns in the IR

## Aggregate Mode Details

Aggregate mode is designed to work with the output of `make run` in test directories.
It reads existing intermediate MLIR files and presents them in pipeline order:

| Pipeline Stage | File | Description |
|---------------|------|-------------|
| AIR Placement | `placed.air.mlir` | After air-collapse-herd, air-place-herds, air-renumber-dma |
| AIR to AIE | `aie.air.mlir` | After air-to-aie pass (converts AIR constructs to AIE dialect) |
| NPU Instructions | `npu.air.mlir` | After air-opt-shim-dma-bds, air-to-std, airrt-to-npu |
| Buffer Addresses | `input_with_addresses.mlir` | After INPUT_WITH_ADDRESSES_PIPELINE |
| Physical Routing | `input_physical.mlir` | After aie-create-pathfinder-flows (routing) |
| ELF Injection | `input_physical_with_elfs.mlir` | After ELF generation |
| NPU Lowering | `npu_insts.mlir` | After NPU lowering pipeline |
| LLVM Lowering | `*_input_opt_with_addresses.mlir` | After AIE_LOWER_TO_LLVM pipeline |

## Pass Argument Syntax

When passes have arguments with spaces, use semicolons to separate passes:

```bash
# CORRECT: Use semicolons to separate passes with space-containing arguments
--passes "air-par-to-launch=depth=0 has-air-segment=true;canonicalize;cse"

# The script will automatically quote the arguments properly
```

## How It Works

1. **Capture**: Runs `air-opt` with `--mlir-print-ir-after-all`
2. **Parse**: Extracts individual IR snapshots from the output
3. **Generate**: Creates a self-contained HTML file with embedded snapshots
4. **Browse**: Open the HTML file in any browser to navigate through snapshots

## Output Files

- `mlir_browser.html` - Self-contained HTML file with all IR snapshots embedded
- `ir_snapshots.json` - JSON file with snapshot data (when using `--export-only`)

## Requirements

- Python 3.7+
- `air-opt` binary (from MLIR-AIR installation)
- Modern web browser

## Troubleshooting

### No snapshots captured

If only 1 snapshot (initial IR) appears:
1. Check if passes are being parsed correctly (look at the "Running:" output)
2. Make sure pass arguments are separated by semicolons
3. Verify the transform script exists and is valid

### Pass arguments not recognized

If passes with arguments fail:
- Use semicolons to separate passes: `--passes "pass1;pass2"`
- The script handles quoting automatically for arguments with spaces

### air-opt not found

Specify the path explicitly:
```bash
python mlir_browser.py input.mlir --passes "..." \
  --air-opt-path /path/to/air-opt
```

## Browser Interface

The generated HTML provides:

- **Left sidebar**: List of all passes with navigation
- **Main view**: Current IR content with syntax highlighting
- **Diff view**: Toggle with 'D' key to see side-by-side comparison
- **Search box**: Filter IR content
- **Navigation controls**: Slider and buttons for pass navigation

## Extending

### Adding new syntax highlighting rules

Edit the `highlightSyntax` function in `index.html` to add new patterns:

```javascript
const patterns = [
    // Add your pattern
    [/\b(your_keyword)\b/g, '<span class="your-class">$1</span>'],
    // ...
];
```

### Custom CSS themes

Modify the CSS in `index.html` to change colors. The main classes are:
- `.keyword` - MLIR keywords
- `.type` - Type names
- `.dialect` - Dialect prefixes
- `.function` - Function names
- `.comment` - Comments
- `.string` - String literals
