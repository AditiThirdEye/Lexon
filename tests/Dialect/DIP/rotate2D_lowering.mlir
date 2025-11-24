// RUN: lexon-opt -verify-diagnostics %s | lexon-opt | FileCheck %s

func.func @lexon_rotate2d_f32(%input : memref<?x?xf32>, %angle : f32, %output : memref<?x?xf32>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xf32>, f32, memref<?x?xf32>
  dip.rotate_2d %input, %angle, %output : memref<?x?xf32>, f32, memref<?x?xf32>
  return
}
