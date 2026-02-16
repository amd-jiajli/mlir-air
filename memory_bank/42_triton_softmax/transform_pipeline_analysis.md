# Transform Pipeline Analysis for Triton Softmax BF16

This document provides a step-by-step analysis of the 59 transformation steps in the MLIR-AIR transform pipeline for the softmax kernel, with emphasis on key changes and the critical point where reductions enter loops.

## Executive Summary

**Key Finding**: The `vector.reduction` operations end up **inside** `scf.for` loops because of how step [45] `tile_using_for [0, 32]` tiles the linalg.generic operations that contain reductions. The intervention point for moving reductions outside loops would need to occur **before or during** vectorization and the final tiling step.

## Pipeline Phases Overview

| Phase | Steps | Description |
|-------|-------|-------------|
| 1. Initial Cleanup | [00-03] | Canonicalization, fold unit extents, CSE |
| 2. Op Fusion | [04-16] | Fuse elementwise ops and multi-op linalg |
| 3. Tiling & Memory | [17-37] | Tile for parallelism, allocate L1/L2 buffers |
| 4. Bufferization | [38-43] | Convert tensors to memrefs |
| 5. **Vector Tiling** | [44-45] | **CRITICAL: Creates vectorized loops with reductions inside** |
| 6. AIE Lowering | [46-51] | Convert to air.herd, DMA, vectorize |
| 7. Type Conversion | [52-57] | Cast reductions and exp to bf16 |
| 8. Final Cleanup | [58-59] | Final canonicalization, CSE |

---

## Detailed Phase Analysis

### Phase 1: Initial Cleanup [00-03]

#### [00] Initial IR
The starting IR from Triton contains:
- `tensor<4x1024xbf16>` input with reinterpret_cast from raw pointers
- Two `linalg.reduce` operations (maxnumf and addf) 
- Multiple `linalg.generic` for elementwise ops (extf, subf, exp, truncf, mulf)
- Reductions reduce dimension 0 on transposed tensors (1024x4 → 4)

**Key Structure:**
```mlir
// Max reduction
%reduced = linalg.reduce ins(%transposed : tensor<1024x4xf32>) outs(%8 : tensor<4xf32>) dimensions = [0] 
  (%in: f32, %init: f32) { %26 = arith.maxnumf %in, %init : f32; linalg.yield %26 : f32 }

// Sum reduction  
%reduced_3 = linalg.reduce ins(%transposed_2 : tensor<1024x4xf32>) outs(%16 : tensor<4xf32>) dimensions = [0]
  (%in: f32, %init: f32) { %26 = arith.addf %in, %init : f32; linalg.yield %26 : f32 }
```

#### [01] match func.func
No change - matches the function for subsequent transforms.

#### [02] canonicalize + fold_unit_extent
Simplifies tensor operations:
- Removes redundant tensor.empty() 
- Collapses unit dimensions (e.g., `tensor<4x1xf32>` → `tensor<4xf32>`)
- Simplifies tensor.expand_shape/collapse_shape

#### [03] CSE
Common subexpression elimination - removes duplicate operations.

---

### Phase 2: Operation Fusion [04-16]

#### [05] air.fuse_elementwise_linalg ⭐
**MAJOR CHANGE**: Fuses multiple elementwise operations into combined linalg.generic ops.

Before: Separate ops for extf, subf, exp, truncf, mulf
After: Three fused generics:
1. Max reduction input (unchanged)
2. `sub-exp` fused: `extf → subf → exp`
3. `final` fused: `divf → truncf → extf → subf → exp → truncf → mulf`

#### [07] air.transpose_reduce ⭐
**Changes reduction dimension** from 0 to 1 by transposing back:
```mlir
// Before: reduce dimension [0] on 1024x4
%reduced = linalg.reduce ins(%transposed : tensor<1024x4xf32>) outs(...) dimensions = [0]

// After: reduce dimension [1] on 4x1024
%transposed_2 = linalg.transpose ins(%transposed) ... permutation = [1, 0]
%reduced = linalg.reduce ins(%transposed_2 : tensor<4x1024xf32>) outs(...) dimensions = [1]
```

This is important because it allows the reduction to be along the contiguous dimension (last dim) which maps better to vector operations.

#### [08] structured.generalize ⭐
**Converts `linalg.reduce` to `linalg.generic`**:
```mlir
// Before
%reduced = linalg.reduce ins(%transposed_2 : tensor<4x1024xf32>) outs(%8 : tensor<4xf32>) dimensions = [1]

// After  
%10 = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
  iterator_types = ["parallel", "reduction"]
} ins(%transposed_2 : tensor<4x1024xf32>) outs(%8 : tensor<4xf32>) {
^bb0(%in: f32, %out: f32):
  %22 = arith.maxnumf %in, %out : f32
  linalg.yield %22 : f32
}
```

#### [09-10] Canonicalize + CSE
Removes the redundant transpose operations that were added by transpose_reduce.

#### [15-16] air.fuse_multi_op_linalg ⭐
Fuses the extf (bf16→f32) into the reduction generics:
```mlir
// Fused max reduction - now includes extf
%8 = linalg.generic {iterator_types = ["parallel", "reduction"]} 
  ins(%3 : tensor<4x1024xbf16>) outs(%7 : tensor<4xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %18 = arith.extf %in : bf16 to f32  // fused
    %19 = arith.maxnumf %18, %out : f32
    linalg.yield %19 : f32
}

// Fused sum reduction - now includes extf + subf + exp + addf
%13 = linalg.generic {iterator_types = ["parallel", "reduction"]}
  ins(%3, %8 : tensor<4x1024xbf16>, tensor<4xf32>) outs(%12 : tensor<4xf32>) {
  ^bb0(%in: bf16, %in_3: f32, %out: f32):
    %18 = arith.extf %in : bf16 to f32
    %19 = arith.subf %18, %in_3 : f32
    %20 = math.exp %19 : f32
    %21 = arith.addf %20, %out : f32
    linalg.yield %21 : f32
}
```

---

### Phase 3: Tiling & Memory Hierarchy [17-37]

#### [17] bufferize_to_allocation (L2)
Adds L2 (memory space 1) buffer for output:
```mlir
%alloc_2 = memref.alloc() : memref<4x1024xbf16, 1>
%17 = bufferization.to_tensor %alloc_2 restrict writable
```

#### [18] tile_using_forall [1] ⭐
**FIRST TILING**: Creates `scf.forall` to parallelize across the 4 rows:
```mlir
%18 = scf.forall (%arg8) in (4) shared_outs(%arg9 = %17) -> (tensor<4x1024xbf16>) {
  %extracted_slice = tensor.extract_slice %3[%arg8, 0] [1, 1024] [1, 1]
  // Each iteration processes 1x1024 slice
}
```

#### [19-21] fuse_into_containing_op
Fuses the two reduction generics and fill operations into the forall body.
After this, each forall iteration independently computes:
1. Max reduction on 1x1024 slice → 1xf32 max value
2. Sum reduction on 1x1024 slice → 1xf32 sum value  
3. Final output 1x1024 slice

#### [26] bufferize_to_allocation (L1)
Adds L1 (memory space 2) buffers for the scalar reduction results:
```mlir
%alloc_5 = memref.alloc() : memref<1xf32, 2>  // for max
%alloc_6 = memref.alloc() : memref<1xf32, 2>  // for sum
```

#### [30-33] bufferize_to_allocation + promote_tensor (L1)
Creates L0 buffers (memory space 2) for intermediate tensor tiles:
```mlir
%11 = bufferization.alloc_tensor() {memory_space = 2 : i64} : tensor<1x1024xbf16>
%12 = bufferization.materialize_in_destination %extracted_slice in %11
```

---

### Phase 4: Bufferization [38-43]

#### [38] one_shot_bufferize ⭐⭐
**MAJOR CHANGE**: Converts all tensors to memrefs. The IR is now purely buffer-based.

Before: `tensor<1x1024xbf16>`, `linalg.generic` with tensor semantics
After: `memref<1x1024xbf16, 2>`, `linalg.generic` with memref semantics

**Key structure at this point (still pre-vectorization):**
```mlir
scf.forall (%arg8) in (4) {
  // L0 buffers
  %alloc_4 = memref.alloc() : memref<1x1024xbf16, 2>
  memref.copy %subview, %alloc_4  // copy from L1 to L0
  
  // Max reduction (1x1024 → 1)
  linalg.generic {iterator_types = ["parallel", "reduction"]}
    ins(%alloc_4 : memref<1x1024xbf16, 2>) outs(%alloc_6 : memref<1xf32, 2>)
  
  // Sum reduction (1x1024 → 1)
  linalg.generic {iterator_types = ["parallel", "reduction"]}
    ins(%alloc_4, %alloc_6 : ...) outs(%alloc_8 : memref<1xf32, 2>)
  
  // Output computation (1x1024)
  linalg.generic {iterator_types = ["parallel", "parallel"]}
    ins(%alloc_4, %alloc_6, %alloc_8 : ...) outs(%alloc_10 : memref<1x1024xbf16, 2>)
}
```

**Important**: At this stage, the reductions are on full 1x1024 tiles. There are NO scf.for loops for vectorization yet.

#### [43] air.remove_uninitialized_copy
Removes redundant copies where destination is overwritten immediately.

---

### Phase 5: Vector Tiling [44-45] ⭐⭐⭐ CRITICAL PHASE

#### [44] match linalg.generic
Matches the three generic operations for the next transform.

#### [45] tile_using_for [0, 32] ⭐⭐⭐ THE CRITICAL STEP
**THIS IS WHERE REDUCTIONS ENTER LOOPS**

Tiles the 1x1024 operations into scf.for loops with vector width 32:

```mlir
// BEFORE [44]: Single linalg.generic over 1x1024
linalg.generic {iterator_types = ["parallel", "reduction"]}
  ins(%alloc_4 : memref<1x1024xbf16, 2>) outs(%alloc_6 : memref<1xf32, 2>) {
    // operates on full 1024 elements
}

// AFTER [45]: scf.for loop with 32-element tiles
scf.for %arg9 = %c0 to %c1024_7 step %c32 {
  %subview_18 = memref.subview %alloc_4[0, %arg9] [1, 32] [1, 1]
  %subview_19 = memref.subview %alloc_6[0] [1] [1]
  linalg.generic {iterator_types = ["parallel", "reduction"]}
    ins(%subview_18 : memref<1x32xbf16, ...>) outs(%subview_19 : memref<1xf32, ...>) {
      // operates on 32 elements per iteration
      // REDUCTION IS NOW INSIDE THE LOOP
}
```

**The three loops created:**
1. Loop 1 (max reduction): Iterates 32 times, each doing 32-element maxnumf reduction
2. Loop 2 (sum reduction): Iterates 32 times, each doing 32-element add reduction
3. Loop 3 (output): Iterates 32 times, writing 32 elements

---

### Phase 6: AIE Lowering [46-51]

#### [48] air.par_to_herd ⭐
Converts `scf.parallel` (from forall) to `air.herd`:
```mlir
air.herd @herd_0 tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c1) 
  args(%arg12=%alloc, %arg13=%alloc_0) : memref<4x1024xbf16, 1 : i32>, memref<4x1024xbf16, 1> {
    // Each herd tile processes one row
}
```

#### [50] air.copy_to_dma
Converts `memref.copy` to `air.dma_memcpy_nd`:
```mlir
air.dma_memcpy_nd (%alloc_5[] [] [], %arg12[%arg8, %c0] [%c1_7, %c1024_8] [...])
```

#### [51] air.herd_vectorize ⭐⭐
**VECTORIZES THE LOOP BODY**: Converts linalg.generic to vector operations.

```mlir
// BEFORE: linalg.generic inside scf.for
scf.for %arg14 = %c0 to %c1024_2 step %c32 {
  linalg.generic {...} ins(%subview_18 : memref<1x32xbf16, ...>) {
    %3 = arith.extf %in : bf16 to f32
    %4 = arith.maxnumf %3, %out : f32
  }
}

// AFTER: vector operations inside scf.for
scf.for %arg14 = %c0 to %c1024_5 step %c32 {
  %5 = vector.transfer_read %subview[%c0, %c0], %4 : ..., vector<1x32xbf16>
  %6 = vector.transfer_read %subview_12[%c0], %3 : ..., vector<1xf32>
  %7 = arith.extf %5 : vector<1x32xbf16> to vector<1x32xf32>
  %8 = vector.multi_reduction <maxnumf>, %7, %6 [1] : vector<1x32xf32> to vector<1xf32>
  vector.transfer_write %8, %subview_12[%c0] : vector<1xf32>, ...
}
```

**RESULT**: `vector.multi_reduction` is now INSIDE the `scf.for` loop.

---

### Phase 7: Type Conversion [52-57]

#### [53] air.vector_type_cast reductions to bf16
Inserts truncf/extf around multi_reduction to compute in bf16:
```mlir
%7 = arith.extf %5 : vector<1x32xbf16> to vector<1x32xf32>
%8 = arith.truncf %7 : vector<1x32xf32> to vector<1x32xbf16>  // NEW
%9 = arith.truncf %6 : vector<1xf32> to vector<1xbf16>        // NEW
%10 = vector.multi_reduction <maxnumf>, %8, %9 [1] : vector<1x32xbf16> to vector<1xbf16>
%11 = arith.extf %10 : vector<1xbf16> to vector<1xf32>        // NEW
```

#### [55] air.vector_type_cast exp to bf16
Similarly converts exp to operate on bf16:
```mlir
%12 = arith.truncf %11 : vector<1x32xf32> to vector<1x32xbf16>
%13 = math.exp %12 : vector<1x32xbf16>  // bf16 exp
```

---

### Phase 8: Final Cleanup [58-59]

#### [58] canonicalize + vector patterns
- Converts 2D vectors with unit dimension to 1D: `vector<1x32xbf16>` → `vector<32xbf16>`
- Converts `vector.multi_reduction` to `vector.reduction`

**FINAL IR:**
```mlir
air.herd @herd_0 tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c1) {
  // Loop 1: Max reduction - REDUCTION INSIDE LOOP
  scf.for %arg14 = %c0 to %c1024_5 step %c32 {
    %4 = vector.transfer_read ... : vector<32xbf16>
    %5 = memref.load %alloc_7[%c0]
    %6 = arith.truncf %5 : f32 to bf16
    %7 = vector.reduction <maxnumf>, %4, %6 : vector<32xbf16> into bf16  // ⚠️ INSIDE LOOP
    %8 = arith.extf %7 : bf16 to f32
    memref.store %8, %alloc_7[%c0]
  }
  
  // Loop 2: Sum reduction - REDUCTION INSIDE LOOP
  scf.for %arg14 = %c0 to %c1024_5 step %c32 {
    %11 = math.exp %10 : vector<32xbf16>
    %13 = vector.reduction <add>, %11, %12 : vector<32xbf16> into bf16  // ⚠️ INSIDE LOOP
  }
  
  // Loop 3: Final normalization
  scf.for %arg14 = %c0 to %c1024_5 step %c32 {
    %13 = math.exp %12 : vector<32xbf16>
    %15 = arith.mulf %13, %14 : vector<32xbf16>
    vector.transfer_write %15, ...
  }
}
```

---

## Problem Analysis

### Why Reductions End Up Inside Loops

1. **Step [45]**: `tile_using_for [0, 32]` tiles the 1x1024 reduction into 32 iterations of 32-element reductions
2. **Step [51]**: Vectorization converts each 32-element reduction to `vector.multi_reduction<op>, vector<1x32xT>, vector<1xT>`
3. **Step [58]**: Simplifies to `vector.reduction<op>, vector<32xT>, T`

The accumulator update happens **every iteration**:
```mlir
scf.for %arg14 = %c0 to %c1024 step %c32 {
  // Load current accumulator
  %acc = memref.load %alloc[%c0]
  
  // Read 32 elements
  %vec = vector.transfer_read ...
  
  // Reduce 32 elements with accumulator
  %new_acc = vector.reduction <add>, %vec, %acc
  
  // Store updated accumulator
  memref.store %new_acc, %alloc[%c0]
}
```

### The Ideal Structure (for performance)

```mlir
// Initialize partial sums as vector
%partials = vector.splat %init : vector<32xbf16>

// Accumulate partial sums (no inter-iteration dependency)
scf.for %arg14 = %c0 to %c1024 step %c32 iter_args(%acc = %partials) {
  %vec = vector.transfer_read ...
  %new_acc = arith.addf %acc, %vec : vector<32xbf16>
  scf.yield %new_acc
}

// Final reduction OUTSIDE loop
%result = vector.reduction <add>, %partials : vector<32xbf16> into bf16
```

---

## Intervention Points

### Option 1: Before step [45]
**Goal**: Restructure the linalg.generic before tiling to use a different reduction strategy.

### Option 2: After step [51], before step [52]  
**Goal**: Apply a custom transform to hoist the scalar reduction out of the loop.

### Option 3: New custom pass
**Goal**: Recognize the pattern after vectorization and restructure to:
1. Keep vector additions inside the loop
2. Move final scalar reduction outside

### Constraint
AIE2 only supports 1D vectors (`vector<32xbf16>`), NOT 2D vectors (`vector<Nx32xbf16>`).
The standard `transform.structured.split_reduction` creates 2D vectors and thus cannot be used.

---

## Files Reference

- Transform script: `transform_aie2p.mlir`
- Input IR: `input_ir/4x1024_mul_inv_bf16.mlir`
- Transform output log: `local_saved/transform_print/transform_output.log`
