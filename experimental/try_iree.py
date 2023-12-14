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
        """,
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

# m = create_add_scalar_module(instance)
# context = iree.runtime.VmContext(instance, modules=[hal_module, m])
# f = m.lookup_function("add_scalar")
# finv = iree.runtime.FunctionInvoker(context, device, f, tracer=None)
# result = finv(5, 6)
# logging.info("result: %s", result)

# m = create_simple_dynamic_abs_module(instance)
# context = iree.runtime.VmContext(instance, modules=[hal_module, m])
# f = m.lookup_function("dynamic_abs")
# finv = iree.runtime.FunctionInvoker(context, device, f, tracer=None)
# arg0 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
# result = finv(arg0)
# logging.info("result: %s", result)
# np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])

# m = create_simple_static_mul_module(instance)
# context = iree.runtime.VmContext(instance, modules=[hal_module, m])
# f = m.lookup_function("simple_mul")
# finv = iree.runtime.FunctionInvoker(context, device, f, tracer=None)
# arg0 =  iree.runtime.asdevicearray(device, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
# arg1 =  iree.runtime.asdevicearray(device,np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
# result = finv(arg0, arg1)
# logging.info("result: %s", result)
# np.testing.assert_allclose(result, [4.0, 10.0, 18.0, 28.0])

# class DeviceHalTest(unittest.TestCase):
#     def setUp(self):
#         super().setUp()
#         self.device = iree.runtime.get_device("local-task")
#         self.allocator = self.device.allocator
#         # Make sure device setup maintains proper references.
#         gc.collect()

#     def testGcShutdownFiasco(self):
#         init_ary = np.zeros([3, 4], dtype=np.int32) + 2
#         ary = iree.runtime.asdevicearray(self.device, init_ary)

#         # Drop all references to backing objects in reverse order to try to
#         # trigger heap use-after-free on bad shutdown order.
#         self.allocator = None
#         gc.collect()
#         self.device = None
#         gc.collect()

#         # Now drop the ary and make sure nothing crashes (which would indicate
#         # a reference counting problem of some kind): The array should retain
#         # everything that it needs to stay live.
#         ary = None
#         gc.collect()

#     def testMetadataAttributes(self):
#         init_ary = np.zeros([3, 4], dtype=np.int32) + 2
#         ary = iree.runtime.asdevicearray(self.device, init_ary)
#         self.assertEqual([3, 4], ary.shape)
#         self.assertEqual(np.int32, ary.dtype)

#     def testExplicitHostTransfer(self):
#         init_ary = np.zeros([3, 4], dtype=np.int32) + 2
#         ary = iree.runtime.asdevicearray(self.device, init_ary)
#         self.assertEqual(repr(ary), "<IREE DeviceArray: shape=[3, 4], dtype=int32>")
#         self.assertFalse(ary.is_host_accessible)

#         # Explicit transfer.
#         cp = ary.to_host()
#         np.testing.assert_array_equal(cp, init_ary)
#         self.assertTrue(ary.is_host_accessible)

#     def testOverrideDtype(self):
#         init_ary = np.zeros([3, 4], dtype=np.int32) + 2
#         buffer_view = self.allocator.allocate_buffer_copy(
#             memory_type=iree.runtime.MemoryType.DEVICE_LOCAL,
#             allowed_usage=iree.runtime.BufferUsage.DEFAULT,
#             device=self.device,
#             buffer=init_ary,
#             element_type=iree.runtime.HalElementType.SINT_32,
#         )

#         ary = iree.runtime.DeviceArray(
#             self.device, buffer_view, override_dtype=np.float32
#         )

#         # Explicit transfer.
#         cp = ary.to_host()
#         self.assertEqual(cp.dtype, np.float32)
#         np.testing.assert_array_equal(cp, init_ary.astype(np.float32))
#         self.assertTrue(ary.is_host_accessible)

#     def testIllegalImplicitHostTransfer(self):
#         init_ary = np.zeros([3, 4], dtype=np.int32) + 2
#         ary = iree.runtime.asdevicearray(self.device, init_ary)
#         # Implicit transfer.
#         with self.assertRaises(ValueError):
#             _ = np.asarray(ary)

#     def testImplicitHostArithmetic(self):
#         init_ary = np.zeros([3, 4], dtype=np.int32) + 2
#         ary = iree.runtime.asdevicearray(
#             self.device, init_ary, implicit_host_transfer=True
#         )
#         sum = ary + init_ary
#         np.testing.assert_array_equal(sum, init_ary + 2)
#         self.assertTrue(ary.is_host_accessible)

#     def testArrayFunctions(self):
#         init_ary = np.zeros([3, 4], dtype=np.float32) + 2
#         ary = iree.runtime.asdevicearray(
#             self.device, init_ary, implicit_host_transfer=True
#         )
#         f = np.isfinite(ary)
#         self.assertTrue(f.all())

#     def testIteration(self):
#         init_ary = np.array([0, 1, 2, 3, 4, 5])
#         ary = iree.runtime.asdevicearray(
#             self.device, init_ary, implicit_host_transfer=True
#         )

#         for index, value in enumerate(ary):
#             self.assertEqual(index, value)

#     def testSubscriptable(self):
#         init_ary = np.array([0, 1, 2, 3, 4, 5])
#         ary = iree.runtime.asdevicearray(
#             self.device, init_ary, implicit_host_transfer=True
#         )

#         for index in range(0, 6):
#             value = ary[index]
#             self.assertEqual(index, value)

#     def testReshape(self):
#         init_ary = np.zeros([3, 4], dtype=np.float32) + 2
#         ary = iree.runtime.asdevicearray(
#             self.device, init_ary, implicit_host_transfer=True
#         )
#         reshaped = ary.reshape((4, 3))
#         self.assertEqual((4, 3), reshaped.shape)

#         np_reshaped = np.reshape(ary, (2, 2, 3))
#         self.assertEqual((2, 2, 3), np_reshaped.shape)

#     def testDeepcopy(self):
#         init_ary = np.zeros([3, 4], dtype=np.float32) + 2
#         orig_ary = iree.runtime.asdevicearray(
#             self.device, init_ary, implicit_host_transfer=True
#         )
#         copy_ary = copy.deepcopy(orig_ary)
#         self.assertIsNot(orig_ary, copy_ary)
#         np.testing.assert_array_equal(orig_ary, copy_ary)

#     def testAsType(self):
#         init_ary = np.zeros([3, 4], dtype=np.int32) + 2
#         orig_ary = iree.runtime.asdevicearray(
#             self.device, init_ary, implicit_host_transfer=True
#         )
#         # Same dtype, no copy.
#         i32_nocopy = orig_ary.astype(np.int32, copy=False)
#         self.assertIs(orig_ary, i32_nocopy)

#         # Same dtype, copy.
#         i32_nocopy = orig_ary.astype(np.int32)
#         self.assertIsNot(orig_ary, i32_nocopy)
#         np.testing.assert_array_equal(orig_ary, i32_nocopy)

#         # Different dtype, copy.
#         f32_copy = orig_ary.astype(np.float32)
#         self.assertIsNot(orig_ary, f32_copy)
#         self.assertEqual(f32_copy.dtype, np.float32)
#         np.testing.assert_array_equal(orig_ary.astype(np.float32), f32_copy)

#     def testBool(self):
#         init_ary = np.zeros([3, 4], dtype=np.bool_)
#         init_ary[1] = True  # Set some non-zero value.
#         ary = iree.runtime.asdevicearray(self.device, init_ary)
#         self.assertEqual(repr(ary), "<IREE DeviceArray: shape=[3, 4], dtype=bool>")
#         np.testing.assert_array_equal(ary.to_host(), init_ary)


# if __name__ == "__main__":
#     unittest.main()
