//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test: Hoist maxnumf vector.reduction out of scf.for loop
// This pattern appears in softmax max-finding loops.
//
// Before: scalar accumulator loaded/stored each iteration with vector.reduction inside loop
// After: vector iter_args accumulation with single post-loop vector.reduction

// CHECK-LABEL: @test_hoist_maxnumf_reduction
// CHECK: arith.constant dense<0xFF80> : vector<32xbf16>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<32xbf16>)
// CHECK:   vector.transfer_read
// CHECK:   arith.maxnumf {{.*}} : vector<32xbf16>
// CHECK:   scf.yield {{.*}} : vector<32xbf16>
// CHECK: vector.reduction <maxnumf>
// CHECK-NOT: vector.reduction {{.*}} inside scf.for
func.func @test_hoist_maxnumf_reduction() {
  %cst_neg_inf = arith.constant 0xFF800000 : f32
  %poison = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c32 = arith.constant 32 : index

  %input = memref.alloc() {alignment = 64 : i64} : memref<1x1024xbf16, 2>
  %acc_mem = memref.alloc() : memref<1xf32, 2>
  memref.store %cst_neg_inf, %acc_mem[%c0] : memref<1xf32, 2>

  scf.for %i = %c0 to %c1024 step %c32 {
    %subview = memref.subview %input[0, %i] [1, 32] [1, 1]
      : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
    %vec = vector.transfer_read %subview[%c0, %c0], %poison {in_bounds = [true]}
      : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
    %acc = memref.load %acc_mem[%c0] : memref<1xf32, 2>
    %acc_bf16 = arith.truncf %acc : f32 to bf16
    %result = vector.reduction <maxnumf>, %vec, %acc_bf16 : vector<32xbf16> into bf16
    %result_f32 = arith.extf %result : bf16 to f32
    memref.store %result_f32, %acc_mem[%c0] : memref<1xf32, 2>
  }
  return
}

// -----

// Test: Hoist add vector.reduction out of scf.for loop
// This pattern appears in softmax sum-of-exp loops.

// CHECK-LABEL: @test_hoist_add_reduction
// CHECK: arith.constant dense<0.000000e+00> : vector<32xbf16>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<32xbf16>)
// CHECK:   arith.addf {{.*}} : vector<32xbf16>
// CHECK:   scf.yield {{.*}} : vector<32xbf16>
// CHECK: vector.reduction <add>
func.func @test_hoist_add_reduction() {
  %cst_zero = arith.constant 0.000000e+00 : f32
  %poison = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c32 = arith.constant 32 : index

  %input = memref.alloc() {alignment = 64 : i64} : memref<1x1024xbf16, 2>
  %acc_mem = memref.alloc() : memref<1xf32, 2>
  memref.store %cst_zero, %acc_mem[%c0] : memref<1xf32, 2>

  scf.for %i = %c0 to %c1024 step %c32 {
    %subview = memref.subview %input[0, %i] [1, 32] [1, 1]
      : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
    %vec = vector.transfer_read %subview[%c0, %c0], %poison {in_bounds = [true]}
      : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
    %acc = memref.load %acc_mem[%c0] : memref<1xf32, 2>
    %acc_bf16 = arith.truncf %acc : f32 to bf16
    %result = vector.reduction <add>, %vec, %acc_bf16 : vector<32xbf16> into bf16
    %result_f32 = arith.extf %result : bf16 to f32
    memref.store %result_f32, %acc_mem[%c0] : memref<1xf32, 2>
  }
  return
}
