# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=unused-variable

import gc
import logging
import numpy as np
import os
import sys
import tempfile
import traceback
import unittest

import iree.compiler
import iree.runtime

COMPILED_ADD_SCALAR = None


def compile_add_scalar():
    global COMPILED_ADD_SCALAR
    if not COMPILED_ADD_SCALAR:
        COMPILED_ADD_SCALAR = iree.compiler.compile_str(
            """
            func.func @add_scalar(%arg0: i32, %arg1: i32) -> i32 {
              %0 = arith.addi %arg0, %arg1 : i32
              return %0 : i32
            }
            """,
            target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
        )
    return COMPILED_ADD_SCALAR


def create_add_scalar_module(instance):
    binary = compile_add_scalar()
    m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
    return m


def create_simple_static_mul_module(instance):
    binary = iree.compiler.compile_str(
        """
        func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
          %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
          return %0 : tensor<4xf32>
        }
        """,
        target_backends=iree.compiler.core.DEFAULT_TESTING_BACKENDS,
    )
    m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
    return m


def create_simple_dynamic_abs_module(instance):
    binary = iree.compiler.compile_str(
        """
        func.func @dynamic_abs(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
          %0 = math.absf %arg0 : tensor<?x?xf32>
          return %0 : tensor<?x?xf32>
        }
        """,
        target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS,
    )
    m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
    return m


def create_stablehlo_module(instance):
    binary = iree.compiler.compile_str(
        """
func.func @main(
  %image: tensor<28x28xf32>,
  %weights: tensor<784x10xf32>,
  %bias: tensor<1x10xf32>
) -> tensor<1x10xf32> {
  %0 = "stablehlo.reshape"(%image) : (tensor<28x28xf32>) -> tensor<1x784xf32>
  %1 = "stablehlo.dot"(%0, %weights) : (tensor<1x784xf32>, tensor<784x10xf32>) -> tensor<1x10xf32>
  %2 = "stablehlo.add"(%1, %bias) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %3 = "stablehlo.constant"() { value = dense<0.0> : tensor<1x10xf32> } : () -> tensor<1x10xf32>
  %4 = "stablehlo.maximum"(%2, %3) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  "func.return"(%4): (tensor<1x10xf32>) -> ()
}
        """
        
# '''
# func.func @main (%x0: tensor<2xui64>) -> (tensor<2xui64>)
# {
#     %y_, %y = mhlo.rng_bit_generator %x, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
#     "func.return"(%y0): (tensor<2xui64>) -> ()
# }
# '''
        ,
        target_backends=iree.compiler.DEFAULT_TESTING_BACKENDS,
    )
    m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
    return m


instance = iree.runtime.VmInstance()
device = iree.runtime.get_device(iree.compiler.core.DEFAULT_TESTING_DRIVER)
# device = iree.runtime.get_device("local-task")
allocator = device.allocator
hal_module = iree.runtime.create_hal_module(instance, device)


m = create_stablehlo_module(instance)
context = iree.runtime.VmContext(instance, modules=[hal_module, m])
f = m.lookup_function("main")
finv = iree.runtime.FunctionInvoker(context, device, f, tracer=None)
image =  iree.runtime.asdevicearray(device, np.ones((28,28), dtype=np.float32))
weights =  iree.runtime.asdevicearray(device,np.ones((784,10), dtype=np.float32))
bias =  iree.runtime.asdevicearray(device,np.ones((1,10), dtype=np.float32))
result = finv(image, weights, bias)
print("result:", result.to_host())

breakpoint()
