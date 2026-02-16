# Reduction Hoisting Design Proposal

## Problem Statement

The current transform pipeline generates `vector.reduction` operations **inside** `scf.for` loops:

```mlir
// Current pattern (expensive - 32 horizontal reductions per loop)
scf.for %i = %c0 to %c1024 step %c32 {
  %vec = vector.transfer_read ... : vector<32xbf16>
  %acc = memref.load %alloc[%c0] : bf16
  %result = vector.reduction <add>, %vec, %acc : vector<32xbf16> into bf16  // 32x expensive!
  memref.store %result, %alloc[%c0] : bf16
}
```

This results in 32 horizontal reductions (one per loop iteration), which is inefficient.

## Desired Pattern

```mlir
// Target pattern (efficient - only 1 horizontal reduction total)
%init_vec = vector.splat %init : vector<32xbf16>
%final_vec = scf.for %i = %c0 to %c1024 step %c32 iter_args(%acc = %init_vec) -> vector<32xbf16> {
  %vec = vector.transfer_read ... : vector<32xbf16>
  %new_acc = arith.addf %acc, %vec : vector<32xbf16>  // Fast vector add
  scf.yield %new_acc : vector<32xbf16>
}
%result = vector.reduction <add>, %final_vec : vector<32xbf16> into bf16  // Only 1 reduction!
memref.store %result, %alloc[%c0] : bf16
```

## Constraint

**AIE2 only supports 1D vectors** (`vector<32xbf16>`), NOT 2D vectors (`vector<Nx32xbf16>`).

The standard MLIR `transform.structured.split_reduction` creates 2D tensors/vectors for partial sums, which fails on AIE backend.

## Available Related Transforms

From `AIRTransformOps.td`, these existing transforms show similar patterns:

| Transform | Description | Relevance |
|-----------|-------------|-----------|
| `air.hoist_cast_pair` | Hoists extf/truncf pairs out by changing iter_arg type | **High** - similar mechanism |
| `air.flatten_for_iter_args` | Flattens vector iter_args using shape_cast | **Medium** - iter_arg manipulation |
| `air.hoist_vector_transfer_pointers` | Adds pointer iter_args to loops | **Medium** - adds iter_args |
| `air.hoist_loop_invariant_transfers` | Hoists read/write pairs | **Low** - different pattern |

## Proposed Solution: `air.hoist_vector_reduction`

### Transform Definition

```tablegen
def HoistVectorReductionOp : Op<Transform_Dialect, "air.hoist_vector_reduction",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Hoist vector.reduction operations out of scf.for loops";
  let description = [{
    This transform identifies `scf.for` loops containing `vector.reduction` operations
    where the reduction accumulator is loaded before and stored after each iteration.
    It restructures the loop to:
    1. Initialize a vector accumulator (using vector.splat or vector.broadcast)
    2. Replace scalar reduction with vector addition inside the loop
    3. Move the final scalar reduction outside the loop
    
    Pattern matched:
    ```mlir
    scf.for %i = ... {
      %vec = vector.transfer_read ...
      %acc = memref.load %alloc[%idx] : scalar_type
      %new_acc = vector.reduction <add|mul|maxnumf|...>, %vec, %acc
      memref.store %new_acc, %alloc[%idx]
    }
    ```
    
    Transformed to:
    ```mlir
    %init = memref.load %alloc[%idx] : scalar_type
    %init_vec = vector.splat %init : vector<NxT>
    %final_vec = scf.for %i = ... iter_args(%acc_vec = %init_vec) -> vector<NxT> {
      %vec = vector.transfer_read ...
      %new_acc_vec = arith.addf %acc_vec, %vec : vector<NxT>  // or corresponding op
      scf.yield %new_acc_vec : vector<NxT>
    }
    %result = vector.reduction <add|...>, %final_vec : vector<NxT> into scalar_type
    memref.store %result, %alloc[%idx]
    ```
    
    Supported reduction kinds: add, mul, maxnumf, maxsi, maxui, minnumf, minsi, minui
    
    Requirements:
    - The loop must contain exactly one vector.reduction with a scalar accumulator
    - The accumulator must be loaded/stored to memory at each iteration
    - The vector type must remain 1D (for AIE compatibility)
  }];
  
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";
}
```

### Implementation Strategy

The implementation in `mlir/lib/Transform/AIRLinalgCodegen.cpp` would:

1. **Pattern Detection**:
   ```cpp
   // Match scf.for containing vector.reduction
   for (auto forOp : target.getOps<scf::ForOp>()) {
     auto reductionOps = findVectorReductions(forOp);
     for (auto reductionOp : reductionOps) {
       if (isHoistable(reductionOp, forOp)) {
         hoistReduction(reductionOp, forOp);
       }
     }
   }
   ```

2. **Hoistability Check**:
   - Accumulator is a scalar (not vector)
   - Accumulator comes from memref.load before reduction
   - Reduction result goes to memref.store after
   - No other uses of the scalar accumulator in the loop body

3. **Transformation Steps**:
   ```cpp
   // 1. Hoist the initial load before the loop
   auto initLoad = getAccumulatorLoad(reductionOp);
   initLoad->moveBefore(forOp);
   
   // 2. Create vector.splat to initialize vector accumulator
   auto initVec = rewriter.create<vector::SplatOp>(loc, vecType, initLoad.getResult());
   
   // 3. Create new scf.for with vector iter_arg
   auto newForOp = scf::ForOp::create(rewriter, loc, lb, ub, step, {initVec});
   
   // 4. Replace vector.reduction with element-wise op
   // For add: arith.addf %iter_arg, %input_vec
   // For mul: arith.mulf %iter_arg, %input_vec
   // For max: arith.maxnumf %iter_arg, %input_vec
   
   // 5. Create vector.reduction after the loop
   auto finalReduction = rewriter.create<vector::ReductionOp>(
       loc, reductionKind, newForOp.getResult(0));
   
   // 6. Create store after the loop
   rewriter.create<memref::StoreOp>(loc, finalReduction, alloc, indices);
   ```

### Reduction Kind Mapping

| Reduction Kind | Vector Operation |
|----------------|------------------|
| `add` | `arith.addf` / `arith.addi` |
| `mul` | `arith.mulf` / `arith.muli` |
| `maxnumf` | `arith.maxnumf` |
| `maxsi` | `arith.maxsi` |
| `maxui` | `arith.maxui` |
| `minnumf` | `arith.minnumf` |
| `minsi` | `arith.minsi` |
| `minui` | `arith.minui` |

## Alternative: Pattern-Based Rewrite

Instead of a dedicated transform op, we could add patterns to existing canonicalization:

```cpp
// In mlir/lib/Transform/AIRMiscPasses.cpp
struct HoistVectorReductionPattern : public OpRewritePattern<scf::ForOp> {
  LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const {
    // Match pattern and transform
    // ...
  }
};
```

And expose it via `transform.apply_patterns`:
```mlir
transform.apply_patterns to %func {
  transform.apply_patterns.air.hoist_vector_reduction
}
```

## Testing

Test cases should verify:
1. Basic add reduction hoisting
2. Max/min reduction hoisting  
3. Multiplication reduction hoisting
4. Multiple reductions in different loops
5. Nested loops (only hoist from innermost)
6. Maintain 1D vector throughout (no 2D vectors created)

## Integration with Transform Script

```mlir
// In transform_aie2p.mlir, add after vectorization:
%loops_with_reduction = transform.structured.match ops{["scf.for"]} in %vectorized_herd
transform.air.hoist_vector_reduction %loops_with_reduction
```

## Files to Modify

1. `mlir/include/air/Dialect/AIR/AIRTransformOps.td` - Add op definition
2. `mlir/lib/Transform/AIRLinalgCodegen.cpp` - Add implementation
3. `mlir/test/Transform/AIRLinalgCodegen/` - Add tests
4. `test/xrt/42_triton_softmax_bf16/transform_aie2p_reduction_hoisted.mlir` - Use new transform

## Current Status

**Not yet implemented**. The transform script `transform_aie2p_reduction_hoisted.mlir` 
documents the optimization goal but uses the standard pipeline without reduction hoisting.

## References

- `HoistCastPairOp` implementation: Shows how to manipulate iter_args
- `HoistVectorTransferPointersOp`: Shows how to add iter_args to loops
- `FlattenForIterArgsOp`: Shows vector shape manipulation in iter_args
