// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton Softmax Tiling Recipe Transform Script - Reduction Hoisting Experiment
//===----------------------------------------------------------------------===//
// 
// OPTIMIZATION GOAL: Move reduction operations outside of the vectorized loop
//
// CURRENT PATTERN (after vectorization):
//   for i in 0..32:
//       load vector<32xf32>
//       vector.reduction<add> → scalar  // Expensive! 32 horizontal reductions
//       add scalar to accumulator
//
// DESIRED PATTERN:
//   accumulator = vector<32xf32> zeros
//   for i in 0..32:
//       load vector<32xf32>
//       accumulator += vector           // Fast vector add
//   vector.reduction<add> accumulator → scalar  // Only 1 reduction!
//
// CHALLENGE:
//   MLIR's split_reduction with inner_parallel achieves this mathematically,
//   but creates intermediate 2D tensor shapes (e.g., tensor<batch x 32 x 32>)
//   that become 2D vectors (vector<32x32xf32>) after vectorization.
//   AIE hardware only supports 1D vectors, causing backend failure.
//
// THIS SCRIPT: Explores alternative optimizations while documenting the
//   reduction hoisting goal for future custom implementation.
//
//===----------------------------------------------------------------------===//

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
        //===================================================================
        %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        
        transform.apply_patterns to %func0 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !pdl.operation
        transform.apply_cse to %func0 : !pdl.operation

        //===================================================================
        // PHASE 2: Operation Fusion and Reduction Preparation
        //===================================================================
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fused_func = transform.air.fuse_elementwise_linalg %func1
        
        // Transpose reductions to innermost dimension
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %fused_func : (!pdl.operation) -> !pdl.operation
        %transformed_reduces = transform.air.transpose_reduce %reduces
        %generalized_reduces = transform.structured.generalize %transformed_reduces : (!pdl.operation) -> !pdl.operation
        
        transform.apply_patterns to %fused_func {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %fused_func : !pdl.operation

        // Split handles for manipulation
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fill1, %fill2 = transform.split_handle %fill : (!pdl.operation<"linalg.fill">) -> (!pdl.operation<"linalg.fill">, !pdl.operation<"linalg.fill">)
        %generic = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %generic1, %generic2, %generic3, %generic4, %generic5 = transform.split_handle %generic : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)
        
        %fused_generic1 = transform.air.fuse_multi_op_linalg %generic1, %generic2
        %fused_generic2 = transform.air.fuse_multi_op_linalg %generic3, %generic4

        //===================================================================
        // PHASE 3: Tiling and Producer-Consumer Fusion
        //===================================================================
        %generic5_output_buf, %new_generic5 = transform.structured.bufferize_to_allocation %generic5
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !pdl.operation

        // Tile for batch dimension (creates parallel iterations for each row)
        %tiled_generic_5, %forall_5 =
        transform.structured.tile_using_forall %generic5 tile_sizes [1] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        // Fuse producers into the tiled loop
        %tiled_fused_generic_2, %4 = transform.structured.fuse_into_containing_op %fused_generic2 into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %tiled_fused_generic_1, %5 = transform.structured.fuse_into_containing_op %fused_generic1 into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
        %fused_fill, %7 = transform.structured.fuse_into_containing_op %fill into %forall_5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)

        //===================================================================
        // PHASE 4: Post-Fusion Canonicalization
        //===================================================================
        %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func2 : !pdl.operation
        
        //===================================================================
        // PHASE 5: L1 Memory Allocation
        //===================================================================
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %tiled_generic1, %tiled_generic2, %tiled_generic3 = transform.split_handle %generics2 : (!pdl.operation<"linalg.generic">) -> (!pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">, !pdl.operation<"linalg.generic">)

        %op0 = transform.get_operand %tiled_generic1[0] : (!pdl.operation) -> !transform.any_value
        transform.structured.promote_tensor to 2 %op0 : !transform.any_value

        %gen1_in_buffer, %gen1_in_new = transform.structured.bufferize_to_allocation %tiled_generic1
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        
        %gen2_in_buffer, %gen2_in_new = transform.structured.bufferize_to_allocation %tiled_generic2
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation
        
        %gen3_in_buffer, %gen3_in_new = transform.structured.bufferize_to_allocation %tiled_generic3
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !pdl.operation

        //===================================================================
        // PHASE 6: Pre-Bufferization Canonicalization
        //===================================================================
        %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func5 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func5 : !pdl.operation
        
        //===================================================================
        // PHASE 7: Complete Bufferization
        //===================================================================
        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!pdl.operation) -> !pdl.operation

        //===================================================================
        // PHASE 8: Post-Bufferization Cleanup
        //===================================================================
        %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        transform.apply_patterns to %func6 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        transform.apply_cse to %func6 : !pdl.operation
        transform.apply_patterns to %func6 {
            transform.apply_patterns.canonicalization
        } : !pdl.operation
        
        %func_op_updated = transform.air.remove_uninitialized_copy %func6

        //===================================================================
        // PHASE 9: Vectorization Tiling
        //===================================================================
        // CURRENT: Tile by [0, 32] creates 32 iterations, each doing vector<32> reduction
        // 
        // OPTIMIZATION NOTE: The ideal transformation would be:
        //   1. Keep iteration count at 32
        //   2. Change accumulator from scalar to vector<32xf32>
        //   3. Inside loop: vector<32> += loaded_vector (fast vector add)
        //   4. After loop: single vector.reduction<add> (only 1 reduction!)
        //
        // This requires a custom transform that:
        //   a) Creates vector iter_arg for the loop
        //   b) Converts scalar reduction to vector accumulation
        //   c) Inserts final vector reduction after loop
        //
        // For now, using standard approach with vector width 32
        
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %inner_most_generics, %vec_loops:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [0, 32]
          : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!pdl.operation) -> !pdl.operation
        %parallel = transform.loop.forall_to_parallel %forall_as_herd : (!pdl.operation) -> !pdl.operation
        %herd = transform.air.par_to_herd %parallel

        // Convert memory copies to DMA operations
        %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!pdl.operation) -> !pdl.operation
        %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd
        
        // Vectorize herd operations
        %vectorized_herd = transform.air.herd_vectorize %herd

        // Cast vector reduce to bf16 for AIE vectorized reduction intrinsic
        %vector_reductions_in_herd = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result10 = transform.air.vector_type_cast %vector_reductions_in_herd {target_element_type = bf16}

        // Cast vector exp to bf16 for AIE vectorized exp intrinsic
        %vector_exps_in_herd = transform.structured.match ops{["math.exp"]} in %vectorized_herd : (!pdl.operation) -> !pdl.operation
        %result11 = transform.air.vector_type_cast %vector_exps_in_herd {target_element_type = bf16}

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation

        // Convert size-1 vectors to scalars
        %func7_transformed = transform.air.convert_size1_vector_to_scalar %func7
        
        //===================================================================
        // PHASE 11: Final Cleanup and Lowering
        //===================================================================
        transform.apply_patterns to %func7_transformed {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        } : !pdl.operation
        transform.apply_cse to %func7_transformed : !pdl.operation
    }
}
