/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/conv.h"

#include <stddef.h>
#include <iostream>
#include <cstdint>
#include <vector>

// Only use multi-threaded Eigen if ruy is disabled.
#if !defined(TFLITE_WITH_RUY)
#define TFLITE_WITH_MULTITHREADED_EIGEN
#endif

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
#include "tensorflow/lite/kernels/eigen_support.h"
#endif
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"
// b/131835803 forces us to include multithreaded_conv.h before optimized_ops.h
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
#include "tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h"
#endif
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace snsidn {

// This file has 4 implementation of Conv.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  // kMultithreadOptimized is a mixture of an Eigen-based kernel when threads
  // are available and kGenericOptimized when we must use only one thread.
  kMultithreadOptimized,
  // The kernel uses use CBLAS interface for matrix multiplication.
  // It's fast when an optimized CBLAS implementation is available (e.g. Apple
  // Accelerate Framework), and it's slow when falling back to naive
  // implementation.
  kCblasOptimized,
};

const int kTensorNotAllocated = -1;

static constexpr size_t kMaxIm2colBufferSizeMobile = 1024 * 1024 * 1024;  // 1GB

int8_t filter_r   aw[5760]={-5, 1, -5, -17, 14, -8, 32, -9, 38, -10, -9, 0, 4, -7, 0, -19, -12, 3, 17, -9, -1, -5, 5, -3, 1, -7, 5, -31, -4, 12, 0, 8, 9, -19, -4, -8, -10, -2, 0, 8, 7, -17, -12, 15, -3, 20, 9, -20, 7, 3, 3, -2, -4, -127, 4, 2, -7, 1, -24, -6, -3, 6, -4, -4, 2, -1, -7, 17, -1, 0, 14, -11, -24, -1, 0, 17, 4, 3, -4, -13, -30, 4, 18, 2, 3, -22, 4, -1, 0, -6, 11, -2, 3, -16, -29, -17, 8, 30, 9, 3, -4, 103, 17, -8, 1, -6, -12, 5, -1, 26, 13, -6, 3, -6, 24, -21, -7, -4, -46, -1, -15, -8, 6, -1, 13, -1, 15, 3, 2, -19, 1, -1, -5, 10, 11, 24, 3, 4, -1, 16, 19, 1, 19, -1, -59, -32, -57, -39, 52, -7, -48, -21, -58, 18, 8, -27, -32, -28, -32, 25, 5, -37, 71, 28, -56, 22, 84, -68, -35, 20, -10, 12, 16, 12, 26, 35, -54, 1, 34, 18, 0, 8, 6, -19, -22, 5, 6, 18, 9, -34, 2, -4, -8, 23, -26, -41, 8, 16, -6, -17, -21, -8, -2, 2, 26, -27, 23, 29, -5, 5, -44, 45, -127, 27, -3, 21, 8, -30, -10, -8, -8, -36, -3, -8, 18, -15, 16, 13, -32, 11, -8, -1, -20, 11, -40, -16, 4, -32, 16, -97, 7, 41, -22, -77, 32, 29, 29, 32, 23, 35, -7, 39, -74, -63, 54, 4, -10, 5, 59, -34, 7, 37, -33, 23, 6, -19, 0, -13, 64, -12, 4, 14, -9, 1, -34, 9, 23, 31, -25, 39, -3, 8, -16, 2, 52, -6, 41, 3, 3, -35, -1, 19, -14, -14, -6, 46, 46, 35, 19, 7, -13, 97, -12, 6, 19, -26, -4, -14, -13, -20, 13, 0, -36, 11, -49, 29, -53, -17, 10, -31, -3, 9, -19, 25, 0, 3, -51, 13, 2, 46, 15, -105, 95, -71, -5, 27, 8, -17, 10, -15, 15, -82, -11, -19, -5, -3, 5, -8, 2, -29, 0, -10, -27, 5, 8, -24, 28, -48, 12, -5, 4, -36, 29, -10, -30, -30, 127, 64, 38, 19, -83, 5, 55, 66, -23, 26, 24, -39, -36, -20, -5, 10, -3, -28, -6, 24, 7, -18, -1, 75, 10, -26, -2, 12, 14, -7, -4, -122, -43, -1, -29, -23, -24, 27, 5, 40, 19, -5, 21, 2, 6, -86, 48, -51, -56, 10, -15, 14, -6, -10, 4, -61, -11, -8, -29, 7, -16, 5, -94, -24, -2, -4, 2, 35, 3, 45, 56, -6, -27, 8, 17, -7, -38, -47, 17, 12, -2, 2, 117, -105, -3, 4, -79, -34, -11, 22, -1, -5, -75, -22, 62, -2, -48, 1, 6, 7, -58, 59, -64, -50, -85, 120, -35, 58, -37, 25, -15, 0, 21, 29, -8, -13, -27, -66, -12, 47, 15, -44, -15, -52, 59, -46, -18, 19, -5, 0, -39, 16, 8, -61, -80, -20, 4, -30, 65, 12, 21, 6, -19, 0, 38, -45, 50, 12, 25, -29, -4, -75, 18, -48, -39, -87, 42, 43, -14, -64, 67, -18, 94, -21, -34, 28, -9, 35, 36, 62, 28, -16, 4, -79, 8, -11, -41, -68, 6, -49, -44, -4, -24, -28, 28, -23, -51, -65, -71, 47, 41, 127, 13, -25, -43, -76, 28, -22, -15, -100, -6, -4, -101, -37, -82, 9, -2, 11, 49, 41, 25, -59, 20, -7, -127, 10, -24, 22, 21, -51, -7, 13, 2, -40, 9, -7, -70, 26, 16, -15, 44, 37, 5, 20, 68, -13, 29, -17, -27, 77, 38, -33, 36, -23, 49, -113, -40, 27, 46, 54, 12, 6, -21, 55, -11, 6, -25, -31, -11, -45, 22, -116, -12, 20, 5, 20, 30, -7, -33, -13, -14, 12, -9, -22, 23, -1, -25, 11, 73, 14, -19, 9, 37, 53, 0, 5, 12, 43, 89, -7, 52, -93, 33, -8, 11, 50, 7, 17, 7, 46, 22, 13, 6, 59, 28, -89, 49, 4, 41, -1, -30, 31, 56, -99, 2, -21, 18, -31, 27, 39, -54, -4, 6, 28, -117, -9, 28, 71, -13, 35, 38, -34, 31, -7, 71, 60, -57, -6, 27, 16, 15, -9, -10, 44, -59, -25, -6, -5, -7, -5, 48, 11, -1, 0, -53, -7, -54, 15, -33, 11, 25, 8, 22, 16, -20, -5, -11, 54, -6, -9, -6, -15, 39, 9, -5, -2, 14, -45, -8, 14, -12, -10, 22, 20, -5, 30, -38, -4, -4, 44, 56, 40, -1, 11, -5, 32, 8, -57, 6, 6, 40, -26, -29, -14, -14, -36, 15, 13, -18, -14, -22, 2, 2, 11, 6, -10, 18, 12, -13, -17, 77, -127, -44, 26, -9, -10, 6, -5, 18, 3, 33, -8, -30, -24, 20, -53, -3, -23, -17, 50, -7, -35, -36, -6, -9, -9, -9, 27, 6, 43, 54, -16, 11, -1, 10, 14, 14, -4, 8, 4, -17, 55, -47, 31, 0, -3, -6, 0, 33, 23, -50, 50, 25, -4, 41, 17, 38, -18, -35, 77, 9, 11, -6, -11, -39, 7, -2, 5, -9, 31, -13, 7, -15, -12, 0, -2, -67, -8, -7, 37, -4, -7, -35, 108, 20, 4, 86, -3, 5, 4, 6, 13, -3, -35, -19, -7, 29, -18, -21, -8, 15, -2, -9, -4, -6, -3, -22, 18, -5, 35, 0, 19, -8, -22, -56, -11, -8, 22, -7, -10, -6, -2, 62, 13, 8, 72, -8, -115, -7, 4, -1, 88, -10, -1, -1, 7, -16, -7, 2, 1, -7, -2, -9, 7, 9, 7, 7, -23, -25, -16, -19, -8, 10, 35, 0, -8, 50, -22, 20, 30, 12, -17, 7, -28, -11, 19, -7, -16, -4, 0, 4, -24, -22, 2, -2, -31, 127, -30, 1, -51, 31, -25, 7, 3, -26, 6, 3, 40, 65, -17, -2, -12, 91, -25, 3, -48, -53, -4, -20, 0, 14, -53, 10, 2, -55, 8, -6, -26, -5, 0, -4, -16, 65, 40, 9, -1, -21, -11, 11, -1, 44, -13, 27, -63, 24, -61, 52, -114, 1, 68, -101, -34, -23, -10, 63, 98, 50, -22, 14, -25, -1, 19, -69, -2, -28, 16, 50, 47, 23, 38, 6, -26, 38, 60, 17, 32, 6, -62, -15, 1, -17, 16, -10, -14, 18, 15, 13, 29, 55, -2, 0, -48, -30, 39, 52, 53, 30, 70, -21, -22, -103, -20, 6, 43, 48, 0, 45, 9, -13, -70, 73, -117, 67, -39, -29, 37, 64, -55, -60, 32, -50, -84, 53, 23, -87, 0, -51, -85, 27, -63, -65, -31, 10, -56, -47, -16, -52, 7, 24, 114, -25, -9, -19, -20, -82, -127, -2, 1, 58, -16, 11, -45, 60, -4, 29, -6, 82, -16, -10, 80, -56, 24, -71, -41, 15, -31, -28, 35, 48, 22, 76, -9, 10, -16, -36, -4, -13, 5, 32, 12, -27, 33, 43, 8, -5, -48, -2, -16, 1, 3, -1, 90, 32, -9, 27, 1, -1, 0, -3, 17, -2, -11, -16, 3, -7, 0, 10, -2, 5, -6, -7, 4, -14, -9, -18, -10, -9, 28, 4, 2, -6, 9, 13, -1, -12, -4, 8, 9, -6, 53, -13, 4, 16, 30, -15, 62, 1, 27, 4, -12, 3, -10, 0, 1, -11, -12, -9, -3, 2, 2, -4, 16, 9, 3, -23, -6, -28, -7, -3, -5, -6, -4, -4, -3, -12, 6, 84, 18, -1, 3, 17, 38, -1, 28, -8, -8, 4, -7, 6, 5, -3, 4, 15, -42, -23, -2, -8, -13, -127, -6, -9, -11, -3, 0, 8, -54, -6, -13, -2, -3, -3, -7, 6, 34, -27, -3, 33, -14, -14, 7, 18, 14, 23, -10, -5, -14, 6, -11, -5, -3, -7, -4, 13, -11, -9, -9, 52, -3, 21, -4, -4, 2, -5, 12, 6, -2, 77, 57, 5, -34, -33, 7, 17, -32, 0, 0, -28, -20, 18, -19, 22, 25, 30, 1, 2, -32, 4, 5, 3, 12, 36, 2, 8, 15, 33, -34, -10, -64, 14, -62, -20, -23, -4, -6, -8, -32, -8, 31, 2, -52, -23, -31, -22, 18, -47, 10, -55, 5, 13, 1, 12, 16, -37, -18, 4, 15, 47, -127, -24, -2, 12, 64, -14, 13, 42, -7, 38, 10, -3, 56, 12, -19, -2, -35, -39, 30, -15, 13, -26, 20, -10, -49, 12, -39, -15, 1, -33, -37, -10, -44, 2, 21, 4, 70, 53, 14, 11, 2, 4, -58, 7, -2, -2, 8, 4, -32, 0, -3, -17, 30, -9, -48, -70, 28, -24, -25, -6, 1, -32, 7, -12, 5, 14, 15, -25, -38, 45, -1, -45, 25, -13, 3, 28, -9, 10, -31, 10, 9, 101, 127, -80, -26, -101, 67, -15, 15, 25, -22, 13, -12, -34, 7, 51, 26, 28, -10, 42, 1, 22, 29, 29, -12, 22, 12, 1, -81, 46, 17, -13, 30, 16, -19, 7, -45, 24, -80, -3, -90, 4, 58, 17, 70, 12, -6, 13, -55, 22, 21, 30, 13, -13, 17, 7, -8, -8, 33, 31, 30, 26, 39, -26, -4, 17, 10, -22, -74, -9, 7, 3, -30, 26, 23, 37, -27, -25, -23, 53, 4, 105, -22, 31, -35, 15, -7, 14, 6, 22, -53, -5, -37, 27, 25, -13, -32, 0, 1, 33, 11, 14, -12, 9, -17, 17, 17, -31, 6, 21, -3, -48, 6, 30, 51, -54, -5, 27, -30, 35, -21, -19, 35, -1, 1, 15, -3, 22, -1, 6, 33, -53, 63, 29, 10, 15, 5, -53, 6, -55, -2, -7, 38, -5, -31, -69, 33, 43, 13, 87, -12, -4, -20, -2, -22, -6, 13, -6, -18, -60, 47, -95, 5, -14, -1, 23, 42, 20, 26, -56, -1, -7, 10, -11, 10, 2, 15, -16, 36, 20, -120, -22, -86, 3, -79, -67, -9, -68, -48, -26, 12, -12, 11, 27, -49, 14, 1, -16, -14, -22, 7, -4, 0, 46, 26, 21, 2, 22, 1, 25, 11, -50, -115, 5, 24, -17, 38, 0, 10, -75, -1, -17, -5, -1, -8, 60, -11, 10, -40, -10, 9, -18, -14, -36, 45, 18, 4, 30, -52, 9, -28, 4, -6, 27, 1, 18, -8, 28, -59, -14, -77, 4, -5, 8, 7, -20, -9, 18, -56, 26, 12, -31, -29, -12, 67, 7, 13, -95, -10, 11, 0, -4, 3, -20, 35, 19, -13, 18, -10, 8, -31, -127, 6, 63, 8, -17, -89, -8, 7, -39, 21, 37, -81, -90, 85, 5, -21, 67, 21, 64, 16, 78, 54, -50, -68, 11, 39, -31, -39, -32, -60, -26, -29, 9, 88, 25, -21, -27, -35, 50, -77, 3, 0, -18, 39, -39, -54, 22, -55, 55, 61, 106, -51, -8, -33, -40, 81, -58, -56, -23, -1, 11, -9, 11, 11, 52, -16, -90, -40, -30, -56, 64, 16, -30, -56, -18, -29, 0, -32, 74, -36, -28, -52, -43, -58, -127, -2, 102, 4, 97, -37, -18, 28, -16, 23, 2, 44, 9, -1, -10, -27, -7, -62, -31, 6, 31, -82, -80, -22, 4, 52, -84, 20, -20, -31, 26, -10, 8, 39, 1, -52, 62, 94, 20, 56, 28, -6, -36, -37, -80, -4, 66, 20, -3, -7, -39, -2, 19, 29, -35, 15, -94, 4, -41, -11, -46, 16, 29, -43, 52, -23, 51, -30, -6, -25, -6, 41, -7, -5, -24, -78, -56, -25, -52, 49, -42, 14, -7, 10, 5, 25, 15, 49, -43, -16, -40, 15, 2, 57, 46, -49, 3, 2, 44, -6, -4, 11, -24, -59, -28, -33, -23, -1, 19, 9, -33, -19, -21, -21, -22, 50, -21, 27, -7, -9, -37, -59, 13, -33, 11, -11, -10, 14, -27, 10, 11, 24, -127, -21, -6, -8, 44, -38, -11, -6, -31, -39, 31, -26, 16, 52, 15, 61, -51, -15, -14, 44, -6, -21, 34, -1, -28, 5, 44, 20, -36, 26, 28, -1, 19, -5, 4, -36, -43, 3, 7, 25, 41, 60, -19, -40, 7, -26, -14, -14, -28, 31, 7, 37, -24, -40, 35, 10, 13, -30, 10, -42, 20, -62, 58, -38, 14, 41, -21, 48, 27, -50, 3, -12, 8, 14, -20, 47, -16, 52, -12, 4, 26, 9, -37, -120, -22, 3, -26, -5, 37, -11, 46, -28, -84, 0, -35, -23, 41, 10, -88, 2, -86, 8, 53, 10, -46, 23, 67, 13, -44, 14, 21, -42, -1, 60, 17, -4, -14, 8, 3, 19, -18, 44, -11, 64, 22, -8, 0, -72, 2, -2, -51, 83, -15, -1, -19, -17, -3, -45, -15, -12, 74, 31, -70, -16, 68, -96, 27, 2, 76, -73, 74, 26, -46, -77, 16, 113, 9, -71, 127, 21, 56, -109, -56, -34, -8, 27, 4, -5, -6, -15, 71, 30, 13, -15, -24, 39, -11, 6, 11, 6, -35, -3, 3, -37, -5, -23, -39, -34, -14, 50, -68, 66, -51, -74, -35, 45, -61, -1, -5, 4, -16, -99, -10, 6, 88, -57, -31, -9, -26, 1, 19, 45, -5, -8, 45, 51, -4, -18, -33, -14, 78, -33, 83, 8, 36, 4, 4, 29, -62, -26, 110, 20, -33, 0, 4, 4, -13, -29, 41, -25, -110, 24, 1, 70, -37, -58, 31, 5, -24, -6, -27, 35, 24, -1, 7, 43, -24, -8, 38, -31, 24, -36, -13, 15, 2, 73, 28, -13, -1, -23, 21, -66, -55, 22, -58, -16, 127, 17, 4, 13, -14, 18, -64, -10, 6, 20, -49, -68, -23, 69, -15, -50, 11, -31, -22, -18, 19, -25, 29, -15, 25, -51, -15, 0, 7, 9, -60, -96, -12, 43, 14, 97, 20, -58, 11, -2, -48, -2, -17, -29, -2, 29, 19, 16, -29, -6, 18, 74, 74, -49, 35, 10, 30, 28, 31, 13, 17, 42, 4, -6, -15, 49, 42, 28, 26, -55, -19, -3, -12, 5, 49, 17, -14, 1, 17, -39, -25, 4, 19, 9, 40, 30, -2, -18, -51, -27, -56, -29, 122, -52, -79, 51, 57, 37, -14, -26, 14, -22, -2, 16, -43, -20, -78, -21, 7, -24, -32, -47, -35, -68, 28, 29, -31, -25, -59, 65, -27, 14, -47, 48, 0, -11, -2, 2, 0, -73, 11, 73, 63, 108, -77, -17, -30, 36, 15, 8, -71, 13, 47, -14, -46, -18, 7, 39, 2, 93, -61, 1, -1, -84, -25, -51, 42, -33, -30, -60, -72, -10, -68, 45, 46, -23, 25, 74, 8, -82, 54, -12, 107, -84, 64, -47, 5, -31, 0, -18, -44, -60, -69, 35, 14, -127, 42, -16, 29, 10, 52, -29, 3, 12, -66, -76, -60, 66, -72, -2, 58, -24, -57, 71, 26, 47, 68, -43, -6, 0, -23, -58, 34, -1, -53, -17, -62, 27, -45, 26, -29, -26, -6, 59, 55, -14, 14, -88, 76, 25, 35, 22, -8, -29, -7, -9, -32, 2, -10, -74, -55, -4, -17, -17, -15, -98, -7, -13, 10, -33, -6, 38, 38, 28, -20, -2, 7, 12, 16, -4, -10, -4, -50, 3, -10, 14, -6, 22, -7, 41, 31, 38, -31, 31, -30, -35, 33, 68, -93, 37, -3, -31, -32, -1, -3, -20, 15, -24, -47, -15, 29, -3, -35, 67, 29, 42, 12, 16, 2, -43, -13, 50, 48, 28, -15, 17, -127, -22, 20, 1, 11, 3, -25, 14, 64, -25, -1, -5, 15, -15, -35, 12, -15, -51, -24, 22, 5, -7, -7, 5, -15, 12, 2, -11, -38, -80, -25, -33, 0, -31, -3, -8, 38, -20, 70, 8, 37, -5, -5, 26, 2, -29, 2, -24, 8, -42, 9, -24, 26, -9, -14, -31, 0, 2, -7, -50, 17, -32, -47, 0, 10, 19, -26, 56, -19, 4, -29, -26, -33, 20, -9, 11, 31, 76, 82, 10, 54, -16, 16, 26, 24, 34, 61, 76, -21, 39, 5, 7, 38, -20, 23, 14, 42, -36, -9, 44, 46, -13, 20, -17, -10, 25, 23, 23, -31, 20, -39, 63, -61, 28, -10, 94, -48, -18, -16, -7, -57, 21, 7, -28, 18, 95, 29, -49, 40, -49, -29, 89, 40, 22, 49, 24, -51, 96, -60, -44, -25, 47, 18, 5, 87, 45, 105, -2, -1, 54, -26, -17, 93, 48, 95, -19, 17, -116, 3, -43, 15, -26, 27, 5, 27, 2, 0, 12, 67, 30, -18, -19, -7, 21, 94, -23, 15, 22, -23, -35, 8, 19, 69, -6, -13, 47, -27, 18, 25, -21, 4, 127, -26, 0, 1, 54, -16, 15, 92, -4, 4, -33, 51, -31, 17, -84, -6, 25, 11, -7, -16, 18, -19, 22, -19, -51, -17, -2, -31, -45, 34, -21, 32, -5, -2, 10, 0, -30, -13, -8, 28, 4, -14, -15, -26, -22, 12, -16, 2, -24, 31, -35, -6, -43, -8, 5, -14, -38, -18, 40, -16, 2, 16, -14, -9, -35, -19, -48, -36, -19, -48, 34, 32, -52, -2, 18, -28, 33, 0, 7, -11, 9, 11, 6, -4, -3, -26, -6, 12, -4, -15, 14, 11, -3, 57, -20, -24, -1, -9, -15, 16, -4, -31, 9, 127, 2, -1, -2, -15, -54, 7, -32, 7, -8, -9, -1, 21, 15, -22, 4, -19, 83, 48, -13, -4, -15, -77, 52, 3, -8, -28, 30, -5, 52, -58, 4, -14, -17, -76, 24, -6, 83, -24, -12, -53, 16, 10, 21, -4, -25, -45, 11, -5, -7, -17, 7, 9, 10, -35, 37, -16, 18, 17, 48, -46, -12, 16, -13, 37, -6, 41, -42, 40, 14, -13, 35, 24, 0, -90, -10, -46, -18, -36, 21, -5, 127, 17, 9, 35, 50, -48, 38, 40, -19, -10, -18, -9, -19, 26, -7, -8, -11, 61, 26, -12, -7, 52, 40, -46, -31, 5, 5, 0, -106, -83, 2, 21, -7, -31, -21, -12, -18, -73, -33, -31, 20, -25, 22, 30, 9, -17, 20, -22, 18, -39, 38, -42, 50, 4, -40, -26, 45, 6, 1, 10, 32, -11, 3, -23, -38, -99, -38, 24, 27, -63, -26, -19, 49, 67, 6, 27, -40, -33, 25, 112, 0, -39, 28, 45, -15, -15, -52, -37, 41, 2, -11, 47, -73, 9, -3, 8, 12, -14, 43, -64, -5, -68, 42, 10, 13, -26, -19, -55, -34, 6, 8, -24, -12, 13, 27, 22, 28, -3, 10, -30, 0, -43, 22, -8, 19, 56, 3, 5, 6, 7, -37, 35, -3, -11, 80, -10, 40, 3, -26, -80, 9, -79, -47, -28, -53, -8, -2, 34, -53, -5, -5, -5, 18, 80, 34, 28, 54, -5, 62, -44, -6, -48, -19, -14, -70, 43, 15, -79, 48, -50, 40, 10, 34, 8, -53, -5, -45, -38, -20, -2, 2, -120, 62, 13, -41, -47, -24, -52, -35, 24, -19, 72, -10, 17, -5, 41, 120, -32, 71, 9, -17, -48, -19, 84, -25, 24, -55, -11, 69, -14, -79, 63, 22, -110, 88, -5, -36, 1, 4, -72, -82, 12, -42, -36, 27, 16, -36, -63, -7, -2, -35, 84, -75, 35, 127, -11, -69, -33, -17, -39, -62, -63, -56, -27, -14, -3, -36, 31, -79, -95, 19, -103, -4, 16, -23, 56, -16, -71, -21, -78, -13, -65, 48, -35, -12, 4, -4, 32, -19, 67, -36, 4, 6, 4, 3, 25, -66, -41, -1, -11, -21, 7, 10, 10, -38, 14, -10, -28, 0, 22, 11, 37, 15, -23, 3, 8, 20, 26, -40, 42, 29, 20, -16, 10, -48, 11, -51, 37, -41, 12, -8, -2, 17, 1, -61, 82, 45, 16, -54, -1, 127, -3, -30, 14, 18, -11, 22, 17, 12, -1, 0, 3, 17, 15, 17, 15, 5, 26, -1, 2, 5, -19, 27, 10, 4, 12, 0, 16, 5, 57, -28, 123, 5, 45, -16, -39, -36, -11, -14, -14, 18, 18, 20, 14, 14, 10, -3, -20, -51, -92, 24, 11, 4, -34, -4, 11, 4, -6, -1, -8, 83, 41, -10, 21, 15, 59, 24, -7, -19, -9, 8, -12, 10, -27, -71, -40, 44, -111, 3, 17, -10, 4, 5, 2, 27, 49, 6, 19, 0, 0, -8, -101, 13, 34, 0, 13, 51, 18, 48, 10, 6, -28, 127, -4, 45, 16, 19, -36, 63, -20, -1, -30, -40, -7, -3, 8, 9, -8, 21, -61, -37, -28, 9, 13, 74, -52, -11, -38, 5, -1, -29, 26, 47, -17, 52, -16, 49, 29, -38, -90, 81, 36, 68, 9, 30, 6, 14, -28, 8, -2, -47, -35, 31, 16, 33, 18, 22, -42, 15, -40, 0, 30, 17, 18, 24, 45, 33, 24, -61, -31, 13, 52, -71, -107, 2, 93, -44, -1, -10, 58, 20, -4, 37, 20, -14, -50, 1, 11, 13, 24, -17, -13, 120, 0, 14, 46, 31, 14, -76, 26, 26, -12, -51, 17, -20, 31, -7, 25, -5, 12, -39, 25, -87, 33, -15, 49, -2, 4, 56, 19, -12, -74, -17, -45, 13, 3, 12, 6, -2, 6, 20, 4, -22, -14, 23, -44, 20, -71, 9, -32, -17, -31, 0, 6, -15, 56, -9, 2, -43, 32, 7, -17, -24, -14, 8, -38, 127, -13, 9, -37, -13, -11, -39, -21, 13, -17, -8, -34, -12, -21, -95, -19, -4, -3, -30, -39, 3, -5, 17, 3, -27, 14, 9, -10, -3, 15, 4, -25, -24, 10, -4, 4, -13, -9, -24, 14, 6, 5, -4, -19, -23, 10, 5, -22, -12, -4, -42, 9, -4, -12, -23, 18, -25, -14, 2, -1, -31, -17, -22, -7, -36, -34, 16, -4, -30, -7, 38, 7, -18, -22, -1, 2, -12, 20, -37, -31, 9, -14, -25, -4, -15, -10, -16, -14, -24, -17, -14, -35, -14, 18, -10, -13, -13, 6, -4, 39, -2, 13, -10, -33, 21, -46, -12, 6, -21, -6, -40, -1, 7, -33, -26, 13, 1, 7, 20, 22, -37, 19, -24, -14, -6, -10, -11, -73, 39, -74, -30, 38, 39, -33, 16, -18, -40, 116, -26, 37, -24, 26, -22, -29, -27, -41, 20, 11, 45, -107, -35, 40, 35, -73, -25, 6, 69, 28, -42, 0, -1, 46, -94, -14, -13, -21, -29, -33, -67, 12, 1, -35, -23, -16, 11, 44, -26, -27, -56, 29, -6, -15, -52, 127, -6, -35, -3, 29, 35, 24, 35, -62, -2, -24, -38, -8, -9, 39, -16, -108, 47, 43, 1, -36, 41, 7, 33, -13, -56, -27, -25, -29, -18, 5, -21, -3, 9, 14, 11, 5, -42, -103, 28, -98, 0, -3, 57, 32, -5, -9, -72, -22, 27, -41, -77, -45, 86, -86, -41, 39, 21, -127, 13, 77, -68, -33, 28, 17, -50, 18, 23, -78, -22, -30, 8, -45, -54, -43, 0, 0, 10, -20, -5, -32, -7, -40, 32, 2, 19, -37, 30, -9, 66, -4, 34, -60, -48, -23, -17, -56, 22, -12, -1, 19, -30, -1, 13, 40, 12, -3, -26, -35, -1, -40, 9, 55, -27, 8, 21, -10, -23, 3, -48, -4, 28, 9, 36, 39, -7, 20, -77, -31, -101, -19, 110, 76, -24, -31, 42, 28, -3, -15, -6, -26, -9, -28, -1, 7, -20, 64, -12, -13, 7, -25, -17, -6, -20, 33, 41, -5, 5, 45, -80, 21, 36, 3, 16, 1, 18, 96, -18, 108, -18, -10, -23, 80, 5, -7, -25, 33, 5, 12, 8, -8, 62, 35, 34, 18, 59, -20, 32, -12, 81, -37, 40, -11, -19, -22, 16, -14, -29, 33, 0, 3, 0, 37, -8, 39, -127, 76, 7, 15, -17, -16, -42, -64, 31, 62, -21, 7, 6, -19, -19, 23, 58, 52, -17, -33, -16, -20, 11, 83, 1, -76, 15, 26, 20, 29, -63, 70, 12, -19, -44, 42, -9, 109, -45, -39, -30, -73, 2, 27, 5, -58, -8, -1, -64, -15, 34, 47, 7, 39, 2, -17, -22, 32, -58, 0, 10, -22, -8, 2, -10, -17, 17, -39, 49, -45, -3, -5, -98, -119, 21, -35, 26, -33, 40, -7, -25, 36, 53, 77, -36, -23, 6, 44, -41, 9, -6, 29, 18, 21, -94, -26, -34, 15, 15, 22, 37, -30, 8, -16, 23, -37, -40, -13, 20, 34, 25, 48, 60, 33, 127, -34, -59, -46, -2, -3, -26, 27, 9, -36, -24, -9, -10, -7, -17, -15, 51, 52, 46, 7, -41, 33, 17, -14, 12, -67, -78, -14, -9, 77, -44, -27, 3, 49, 4, 5, -57, 59, 5, -60, 91, 37, -42, 12, 43, 11, 13, 45, -10, 2, 80, 13, 30, -2, -52, -112, 23, 17, -41, 9, -22, 89, 5, -45, -57, 99, -24, -21, -6, 9, 35, 12, -19, -54, 23, -52, -3, -117, 9, -56, 31, -14, 3, -35, -26, 50, -49, -70, -33, -48, -41, 0, 83, -42, 68, 21, -3, -13, 40, 20, 2, 19, 13, 51, -50, -7, -88, -19, 52, 6, -29, -28, -55, -17, 23, 2, 15, -4, -24, -3, 4, -39, -18, 51, 59, 42, 8, 0, 3, 27, -28, -4, 5, -3, -50, 32, -71, -13, -3, -24, -51, -127, 7, -31, -19, -19, 24, -12, -25, -38, -67, 21, -46, -11, 13, 14, 45, 4, -21, -16, 27, 2, -10, -12, -97, 12, -26, -18, -36, 0, 31, -64, -74, -7, -23, 9, -10, -34, -56, 12, -62, -23, -9, 18, -22, -32, -37, -1, 1, -25, 40, -39, -55, 30, -26, -51, -10, 2, 5, -45, 5, 4, -4, 4, 0, 37, 127, 72, -7, 21, -14, -1, -15, 15, -28, 21, 43, 25, -16, -30, 24, -6, 24, 2, 0, 12, 20, 24, 101, 28, 10, 26, -18, 1, 102, 28, -30, -23, -6, 23, 23, -11, 20, 45, 19, 37, 6, 7, 87, -17, -2, 4, -13, 19, -77, -4, 22, 8, 0, 26, 37, 34, 8, 22, 15, 12, -19, -2, 28, 8, 14, -53, 40, 13, 6, 0, 3, -10, 9, 6, -29, 45, -17, 1, 3, -40, -37, -4, -14, 11, 30, 7, 13, -19, -36, 16, 6, -17, 81, -5, -15, 28, 44, -65, -11, 2, 40, 20, 2, 5, 65, -16, 34, 22, 29, -38, 34, -19, -8, 80, 17, -32, 40, -18, -19, -40, -17, 52, 22, 18, 54, -6, 29, 9, -10, -33, -27, -2, 15, 15, -76, -2, 25, 21, 12, -14, -36, -10, 61, 30, -2, -21, 81, -31, 24, -21, 11, -22, -30, -32, -37, 32, -45, -38, -20, -22, -16, 20, -18, 15, 4, -35, -12, 32, 127, -21, 55, -17, -3, 1, -111, 26, 28, 5, -42, 14, -53, -7, 2, 32, -52, -17, 29, -17, 6, 10, 48, -3, -3, 46, -61, -6, 3, 26, -4, -7, -42, -4, -21, -41, -1, -27, -1, 29, 106, 16, -26, 40, 16, 11, 6, -3, 18, -2, 19, -11, 31, 19, 16, 22, 0, 33, -1, -13, 3, -6, 13, -14, -5, 45, 5, 27, 13, 30, 7, -23, -11, -9, 17, 27, -18, -7, -12, -8, 35, 19, -24, -42, -16, 15, 10, -19, 40, 37, 5, 17, -3, -24, -32, 40, -13, 23, -9, -40, -18, -7, 4, -25, 48, 15, -31, 19, -14, 17, 2, -7, -30, 0, -24, 15, 2, 17, 35, -7, -50, -66, 13, -14, 29, 11, 21, -2, 5, 6, 16, 1, -14, -14, 27, 14, 12, 31, 16, -9, 30, 0, -18, 6, 9, 0, 1, 2, -26, 7, -8, -28, -5, -1, -6, 3, 18, 19, 4, -21, 3, 11, -31, -1, -98, -1, 15, 22, -78, 5, -1, 4, 25, 27, 12, 14, -3, 23, 25, -2, 26, 33, -17, -11, 9, 7, 14, 39, -4, 0, 17, 1, 0, -48, -3, 45, 23, 0, -3, 17, 8, -4, 19, -12, 12, -18, 15, 22, -15, 44, 54, -5, -127, 85, -46, 23, 68, -63, -6, 2, 17, 22, -22, 9, 3, -9, 9, 8, 13, -7, 33, 15, 12, 18, 1, 21, 12, -4, 13, -26, 6, 6, 7, -5, 10, 13, 10, 18, -26, 8, 0, 34, 7, 1, 20, 40, 7, -6, 1, 8, -1, 16, 86, 10, 93, -45, 101, 39, -30, -5, 38, -32, 46, -31, -22, 29, 0, -8, 4, -77, -3, 8, 2, -4, 8, 28, -53, 13, -33, 10, -33, -32, -35, 0, 7, 31, 24, 7, -26, -12, 1, -22, 58, -27, -15, -22, 54, 5, 81, -4, 0, -6, -13, -17, 25, -41, 27, 49, -4, -4, 2, -2, 20, 29, 32, 47, 5, 33, -23, -9, -3, 45, -16, -52, 13, -25, -29, 19, -45, -66, -20, -127, -8, -60, -46, -17, -16, 14, -23, -38, -34, 48, 25, 51, -27, -50, -120, -84, -11, 0, 14, 71, -83, -19, -8, -19, 54, 26, 19, -10, 3, 8, -25, -30, 5, -24, 8, 26, 5, -97, 13, 10, -16, 7, -15, 38, 19, 25, -5, -2, -1, 34, 27, 12, 33, -31, 21, -3, 29, 62, 4, 1, -9, 0, 46, 0, -12, 1, -21, 18, 40, -4, -2, 5, 3, -18, 13, -15, -3, -18, 15, 27, 3, -21, -7, 15, 7, -18, -2, -3, -5, -4, 31, 3, -38, 3, -14, -10, 4, -5, 18, -6, 8, 9, 1, 1, -26, -5, 4, -19, -72, 3, -23, 17, -55, 0, -22, 6, -4, -6, 4, -13, 7, 1, 2, 0, 4, -1, -2, -25, 6, 26, 12, -2, -11, -9, -17, -4, -11, 21, -10, 43, 21, -20, -42, 3, 5, 13, -9, 5, -29, 22, -8, -7, 2, 1, 3, -14, -60, -2, -22, 20, 6, 6, -71, -127, -5, 22, 7, 6, -18, -2, 28, -7, 3, -1, -12, -13, -25, -12, -72, -15, -5, -22, -2, 33, 7, 46, -21, 65, 9, -6, 6, 7, 10, 0, 5, 2, 30, -7, 9, 16, 9, -6, -2, 2, 23, -33, 5, -32, 57, -58, -23, -8, -37, -70, -68, -51, -10, 18, 58, -10, 5, -49, 17, -4, -29, -56, -51, -20, -27, 18, -14, -43, -19, 24, 16, -44, -61, 0, 5, -39, 6, -6, 34, -60, -67, -56, -16, -38, 57, 6, 82, 7, 95, -3, -17, -36, -42, -18, -11, -23, -19, -78, -44, 59, -16, 11, -19, -17, -30, -10, -31, -29, 55, 20, -70, 28, -66, -58, 40, 27, 6, -15, 30, -33, -93, 4, 18, 91, 23, -114, 45, 2, 86, 5, -75, -21, 38, 36, 35, -49, -103, 70, 29, -45, -51, -16, -4, -3, -46, 78, 11, -23, 46, -15, 45, -14, 55, -23, -51, -53, -33, -117, -36, 82, -31, -5, -13, -127, -4, 25, -37, 13, 59, 3, 8, -31, 0, 38, -61, -18, -98, 72, 74, -26, 13, -37, 1, 102, 4, -115, -8, -123, 22, 4, -11, 52, 116, 10, 21, 15, -3, -95, -48, -83, -24, -53, -44, -9, -10, -34, 7, -14, -99, -6, -51, 86, -6, 3, -90, -12, -35, -61, -5, -3, 32, -24, -20, 42, -99, -6, -3, -51, -66, 53, 127, -19, 33, 35, -49, 8, 48, -71, -1, -19, 71, 70, -26, -1, 8, -8, 39, -32, -9, 72, -50, 25, 0, 9, -23, 22, -30, 52, -8, -122, -21, -20, -13, 3, 19, -27, 100, -75, 29, -46, 26, -7, 22, 42, 17, 107, 88, 31, -21, 44, 65, 96, 52, -12, 14, -10, -5, -27, -107, -77, -2, -79, -57, -109, -100, -24, 67, 16, -32, 3, -12, 44, -7, -52, 11, -4, -9, 2, -19, -48, -18, 69, 43, 28, 39, -73, 28, 3, 64, 0, -118, 91, 68, 15, -6, -75, 4, -26, -4, 7, -12, 6, 40, 23, -116, 127, -14, 104, 12, 15, 28, 7, 1, 7, 2, -14, -18, -12, 23, 8, 7, 41, 10, -7, 35, -3, -45, -22, 0, 1, -14, 19, -56, 4, 4, -5, 3, -2, -8, 4, 5, 13, 58, 25, 7, 9, -34, 30, 57, 9, -6, 32, 78, -3, -7, -1, 31, 30, 16, 15, 3, 30, 30, -3, 26, 31, 31, -10, 7, 61, -12, 12, -12, -13, 22, -5, -6, 11, 1, -26, 6, 12, -6, 63, 26, -7, 32, 14, 4, -17, 3, 25, -19, 18, 8, 13, 66, 4, -49, 25, -65, 27, -15, -6, 14, 6, -25, 9, -36, -35, 10, 13, 12, -49, 26, 16, -10, 0, -2, 39, 18, -34, -38, 1, 1, 30, 18, -5, -5, 6, 11, 21, -23, 9, -7, 19, 19, -2, 52, 44, 11, 32, 9, 17, -40, 13, -59, -82, 21, -4, 55, -32, -48, -1, 21, -21, 26, -3, 13, 44, 59, -83, 44, -53, 11, -123, 11, -45, 39, 12, 16, -28, 21, -16, -56, -14, -8, -3, 34, 39, -39, 6, -34, 11, 63, 59, 66, 46, 45, 58, 50, 25, 30, 19, 127, 42, 16, 30, 10, -1, 55, -66, 49, 34, -13, 34, 53, 2, 24, 122, -50, -30, 70, 15, 24, -59, -45, -7, -4, 34, -41, -6, -49, -10, 123, 16, 37, 5, 20, 10, -44, 23, 16, 12, 34, 22, 112, -74, 31, -1, -18, 1, 78, 56, 11, 32, 61, 61, 51, 23, 6, 17, 5, -16, 32, 11, 8, -10, -60, -19, 40, -107, -1, -6, -11, -39, -49, 25, -35, -7, 31, 6, 40, 25, 45, 8, 38, -27, -23, -17, 30, 40, 13, 51, 22, -16, 14, -36, 6, -41, -31, 113, -19, -102, -76, 89, 39, -68, -3, 19, -8, 26, -3, 64, 7, 5, -12, -23, 34, -82, -38, 30, 5, 18, -49, 4, 75, 17, -102, -24, -32, 47, -28, -35, 40, 38, 25, 28, -47, -13, 9, -54, 24, 32, -71, 0, -65, 21, 26, 7, 81, 25, 40, -38, -9, -37, -2, 15, -5, 8, -18, 22, -35, 44, 10, 34, 54, -40, -31, -64, 23, -36, -4, -70, 8, -41, -64, 110, 72, 19, 64, 51, 12, 15, 33, 42, 31, 38, -24, 4, -127, -66, -37, 44, -40, 24, 45, -21, 17, 7, -101, -50, 5, -29, 55, -37, -41, 24, 38, 13, 25, 25, -78, -56, -32, 1, 36, -32, -24, 16, 17, 14, -118, -48, 35, 48, 20, -26, 6, 6, 41, -8, -35, -101, 38, 8, -37, 11, 17, -61, -4, -11, -21, -8, 18, 23, 47, -14, -30, 1, -14, -12, 26, 15, 19, 6, 44, -68, -127, -3, -21, -31, 3, -4, -21, 52, -11, -7, 18, 4, -20, 28, 46, -38, 19, 9, 46, 24, 7, -7, -12, -44, 35, -39, -39, -19, 7, -6, -48, 54, -16, -42, -13, -17, -8, -30, 3, 20, -35, 26, -57, -29, 4, -16, -20, 9, -18, 79, -103, -77, -54, 11, -11, -45, 36, 33, -11, 13, 21, 21, 10, -13, 13, 56, 40, -75, 93, -24, 24, 53, -18, 17, 14, 74, -20, -44, 38, 59, -11, 21, 3, -18, -2, -13, 42, 40, -30, -30, 13, -24, 45, 25, 19, -10, -18, -1, -12, -41, 21, -5, 16, 49, -17, 73, 8, 54, -13, -12, -29, 35, 76, -36, -58, 77, 74, 19, -30, 49, -19, 3, 1, 26, -13, -36, -28};

float bias_raw[40]={-10.505208015441895, 5.482931613922119, 17.3843936920166, -2.008500099182129, -6.656646728515625, 17.835657119750977, -17.754623413085938, 8.876652717590332, -24.791744232177734, 23.415584564208984, 8.950373649597168, 18.230918884277344, 10.496207237243652, 17.18072509765625, 9.01466178894043, -10.21682357788086, 3.838258743286133, 20.229129791259766, -14.631470680236816, 23.287424087524414, 7.560101509094238, -3.4721343517303467, 8.761000633239746, -1.8002983331680298, 17.803869247436523, 14.890010833740234, -19.106534957885742, 4.776075839996338, 28.550609588623047, -20.861431121826172, -10.456238746643066, 8.524324417114258, -4.04214334487915, 6.687891483306885, 8.717767715454102, 0.5993421673774719, -22.225116729736328, -12.809319496154785, 17.071937561035156, 0.7941679358482361};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const int stride_width=1;
const int stride_height=1;
const TfLiteFusedActivation activation=kTfLiteActNone;
const int dilation_width_factor=1;
const int dilation_height_factor=1;
const int filter_dims_size=4;
const int32_t filter_dims_raw[4]={40,1,1,144};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={40};
const TfLitePadding paddings=kTfLitePaddingSame;
const TfLiteType filter_type=kTfLiteInt8;
const TfLiteType bias_type=kTfLiteFloat32;
const float scale_filter=0.0;
const int32_t zero_point_filter=0;
const float scale_bias=0.0;
const int32_t zero_point_bias=0;
// const float scales_filter=;
// const int32_t zero_points_filter=;
// const float scales_bias=;
// const int32_t zero_points_bias=;

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int im2col_id = kTensorNotAllocated;
  int hwcn_weights_id = kTensorNotAllocated;
  int input_quantized_id = kTensorNotAllocated;
  int scaling_factors_id = kTensorNotAllocated;
  int input_offset_id = kTensorNotAllocated;
  int accum_scratch_id = kTensorNotAllocated;
  // Row sums are used to cache filter sums for hybrid zero-point calculations.
  int row_sums_id = kTensorNotAllocated;

  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int> per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // Indexes are the offset to the memory buffer in the array used to keep track
  // of the allocated temporaries.
  int32_t im2col_index;
  int32_t hwcn_weights_index;
  int32_t input_quantized_index;
  int32_t scaling_factors_index;
  int32_t accum_scratch_index;
  int32_t input_offset_index;
  int32_t row_sums_index;

  bool need_hwcn_weights = false;
  bool have_weights_been_transposed = false;
  bool need_im2col = false;
  // If it's true, it means im2col is needed but gets disabled because the
  // temporary im2col tensor requires too much memory (i.e.
  // >= kMaxIm2colBufferSize);
  bool im2col_oversized = false;

  bool supports_multithreaded_kernel = false;
  bool is_hybrid_per_channel = false;
  bool compute_hybrid_row_sums = true;

  // Number of convolution groups.
  int32_t groups = 1;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

void ExtractConvParams(TfLitePadding padding, int stride_width, int stride_height, 
                               int dilation_width_factor, int dilation_height_factor,
                               TfLiteFusedActivation activation,
                               TfLiteConvParams* data_params) {
  // TfLiteConvParams data_params;
  data_params->padding = padding;
  data_params->stride_width = stride_width;
  data_params->stride_height = stride_height;
  data_params->dilation_width_factor = dilation_width_factor;
  data_params->dilation_height_factor = dilation_height_factor;
  data_params->activation = activation;
  // return data_params;
}

void GetConvTensor(TfLiteType type, const char* name, TfLiteIntArray* tensor_dims_data, 
                       TfLiteQuantizationParams quant_params,
                       char* tensor_data, TfLiteAffineQuantization* quant_struct,
                       size_t bytes_size, TfLiteTensor* tensor) {
  tensor->type = type;
  tensor->name = name;
  tensor->dims = tensor_dims_data;
  tensor->params = quant_params;
  // tensor->data.raw = reinterpret_cast<char*>(tensor_data);
  tensor->data.raw = tensor_data;
  tensor->bytes = bytes_size;
  tensor->allocation_type = kTfLiteMemNone;
  // data_0.allocation = allocation;
  tensor->is_variable = false;
  if (type != kTfLiteFloat32) {
    tensor->quantization.type = kTfLiteAffineQuantization;
    tensor->quantization.params = quant_struct;
  } else {
    tensor->quantization.type = kTfLiteNoQuantization;
  }
  tensor->sparsity = nullptr;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to use as scratch space for im2col, and
  // to carry information from Prepare() to Eval().
  auto* data = new OpData;
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
  eigen_support::IncrementUsageCounter(context);
#endif
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
  eigen_support::DecrementUsageCounter(context);
#endif
  delete reinterpret_cast<OpData*>(buffer);
}

// Naive implementation of transpose for floats. Could be optimized to be more
// cache friendly, but for now it's a one-time cost on first run, and we would
// prefer to remove the need to do this at all eventually.
void TransposeFloatTensor(const TfLiteTensor* input, TfLiteTensor* output) {
  const int rows = output->dims->data[1];
  const int cols = output->dims->data[0];
  const float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input_data[i * cols + j];
      output_data[j * rows + i] = in_value;
    }
  }
}

// Check if im2col needs to be allocated, as some version of optimized Conv dont
// use it. If any change is supporting im2col in any of the Conv versions, then
// it should be updated here as well
bool IsIm2ColRequired(const TfLiteTensor* input, TfLiteConvParams* params,
                      const TfLiteTensor* filter, OpData* data, bool is_hybrid,
                      KernelType kernel_type) {
  // If HWCN weights are required, Im2Col not required
  if (data->need_hwcn_weights) return false;

  // segregate based on dilated conv & non-dialated conv
  const bool need_dilated_im2col =
      params->dilation_width_factor != 1 || params->dilation_height_factor != 1;
  const bool need_non_dilated_im2col =
      params->stride_width != 1 || params->stride_height != 1 ||
      filter->dims->data[2] != 1 || filter->dims->data[1] != 1;

  const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;

  // Return early as basic requirement is not met
  if (!need_im2col) return false;

  // Special case for Hybrid, as it supports only non-dilated im2col currently
  const bool is_hybrid_non_dilated = is_hybrid && need_non_dilated_im2col;
  const bool is_quantized = input->type == kTfLiteUInt8 ||
                            input->type == kTfLiteInt8 ||
                            input->type == kTfLiteInt16;

  switch (kernel_type) {
    case kReference:
      if (is_hybrid) {
        return true;
      } else {
        return false;
      }
    case kGenericOptimized:
    case kCblasOptimized:
      if (is_hybrid && !need_non_dilated_im2col) {
        return false;
      } else {
        return true;
      }
    case kMultithreadOptimized:
      if (is_hybrid_non_dilated || is_quantized ||
          !data->supports_multithreaded_kernel) {
        return true;
      } else {
        return false;
      }
    default:
      return false;
  }
}

// Allocate temporary tensors (`im2col`, `hwcn_weights` if necessary).
// Note: `context->AddTensors` might invalidate pointers to existing tensors.
// Therefore the logic to add tensors are isolated into this function.
static TfLiteStatus AllocateTemporaryTensorsIfRequired(
    TfLiteContext* context, TfLiteNode* node, bool is_hybrid,
    bool is_per_channel, KernelType kernel_type, size_t im2col_bytes) {
  // auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  TfLiteConvParams data_params;
  ExtractConvParams(paddings, stride_width, stride_height, dilation_width_factor, dilation_height_factor, activation, &data_params);
  TfLiteConvParams* params = &data_params;

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // TF_LITE_ENSURE(context, node->inputs->size >= 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor filter_tensor;
  TfLiteIntArray* filter_dims_data = TfLiteIntArrayCreate(filter_dims_size);
  int size_filter = 1;
  for (int i = 0; i < filter_dims_size; i++) {
    // std::cout << "dims_raw: " << dims_raw[i] << std::endl;
    filter_dims_data->data[i] = filter_dims_raw[i];
    size_filter *= filter_dims_raw[i];
  }
  size_t bytes_size_filter = sizeof(float) * size_filter;
  TfLiteQuantizationParams filter_params;
  filter_params.scale=scale_filter;
  filter_params.zero_point=zero_point_filter;

  TfLiteFloatArray* scale_array_filter = TfLiteFloatArrayCreate(1);
  scale_array_filter->data[0] = scale_filter;
  TfLiteIntArray* zero_point_array_filter = TfLiteIntArrayCreate(1);
  zero_point_array_filter->data[0] = zero_point_filter;

  TfLiteAffineQuantization quant_struct_filter;
  quant_struct_filter.scale = scale_array_filter;
  quant_struct_filter.zero_point = zero_point_array_filter;
  quant_struct_filter.quantized_dimension = 0;
  // float* filter_data;
  // filter_tensor_data = filter_raw;
  GetConvTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data),
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;
  // TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

  // If we're using the optimized multithreaded EigenTensor implementation of
  // convolution, it expects the filter weights to be transposed compared to
  // the normal TF Lite buffer format. Typical TF Lite weights are
  // [filter_count, filter_height, filter_width, input_depth], but for the float
  // implementation we need them as [filter_height, filter_width, input_depth,
  // filter_count]. We get to that format by transposing, and create a temporary
  // buffer to store the results.
  // This path is only used for float processing, so only create the buffer if
  // we're running with that data type.
  data->need_hwcn_weights =
      input->type == kTfLiteFloat32 && data->supports_multithreaded_kernel;

  // We don't always need to allocate im2col. It is only used in some versions
  // of the optimized Conv. This test just mimics something that happens inside
  // optimized_ops.h, in order to avoid a DCHECK(!im2col_data).
  data->need_im2col =
      IsIm2ColRequired(input, params, filter, data, is_hybrid, kernel_type);

  // If im2col_oversized is found to be true, we have to fallback to an
  // execution path (like kReference in float/quantized cases) that doesn't
  // require im2col operation. Therefore, we have to skip checking the hybrid
  // case (but not the hybrid-per-channel one) where there's no such a fallback
  // execution path.
  // TODO(b/178743262): Consider making this check conditioned on the available
  // memory of the system, rather than coupling to the mobile platform check.
  if (IsMobilePlatform() && !(is_hybrid && !is_per_channel) &&
      data->need_im2col && im2col_bytes >= kMaxIm2colBufferSizeMobile) {
    data->need_im2col = false;
    data->im2col_oversized = true;
  }
  int temporaries_count = 0;
  if (data->need_im2col) {
    data->im2col_index = temporaries_count;
    if (data->im2col_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->im2col_id);
    }
    ++temporaries_count;
  }
  if (data->need_hwcn_weights) {
    data->hwcn_weights_index = temporaries_count;
    if (data->hwcn_weights_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->hwcn_weights_id);
    }
    ++temporaries_count;
  }

  if (is_hybrid) {
    // Allocate tensor to store the on-the-fly quantized inputs.
    data->input_quantized_index = temporaries_count;
    if (data->input_quantized_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->input_quantized_id));
    }
    ++temporaries_count;

    // Allocate tensor to store the quantization params computed during
    // on-the-fly input quantization.
    data->scaling_factors_index = temporaries_count;
    if (data->scaling_factors_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->scaling_factors_id));
    }
    ++temporaries_count;

    // Allocate tensor to store the accumulators for the matrix multiply.
    data->accum_scratch_index = temporaries_count;
    if (data->accum_scratch_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->accum_scratch_id));
    }
    ++temporaries_count;
    if (is_per_channel) {
      data->input_offset_index = temporaries_count;
      if (data->input_offset_id == kTensorNotAllocated) {
        TF_LITE_ENSURE_OK(
            context, context->AddTensors(context, 1, &data->input_offset_id));
      }
      ++temporaries_count;

      data->row_sums_index = temporaries_count;
      if (data->row_sums_id == kTensorNotAllocated) {
        TF_LITE_ENSURE_OK(context,
                          context->AddTensors(context, 1, &data->row_sums_id));
      }
      ++temporaries_count;
    }
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);

  return kTfLiteOk;
}

TfLiteStatus Prepare(KernelType kernel_type, TfLiteContext* context,
                     TfLiteNode* node) {
  // std::cout << "codes runs here #-1" << std::endl;
  // auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  TfLiteConvParams data_params;
  ExtractConvParams(paddings, stride_width, stride_height, dilation_width_factor, dilation_height_factor, activation, &data_params);
  TfLiteConvParams* params = &data_params;

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  // std::cout << "codes runs here #-2" << std::endl;
  bool has_bias = false;
  // Check number of inputs/outputs
  // TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  // const TfLiteTensor* filter;
  // TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  // TfLiteTensor* filter;
  TfLiteTensor filter_tensor;
  TfLiteIntArray* filter_dims_data = TfLiteIntArrayCreate(filter_dims_size);
  int size_filter = 1;
  for (int i = 0; i < filter_dims_size; i++) {
    // std::cout << "dims_raw: " << dims_raw[i] << std::endl;
    filter_dims_data->data[i] = filter_dims_raw[i];
    size_filter *= filter_dims_raw[i];
  }
  size_t bytes_size_filter = sizeof(float) * size_filter;
  TfLiteQuantizationParams filter_params;
  filter_params.scale=scale_filter;
  filter_params.zero_point=zero_point_filter;

  TfLiteFloatArray* scale_array_filter = TfLiteFloatArrayCreate(1);
  scale_array_filter->data[0] = scale_filter;
  TfLiteIntArray* zero_point_array_filter = TfLiteIntArrayCreate(1);
  zero_point_array_filter->data[0] = zero_point_filter;

  TfLiteAffineQuantization quant_struct_filter;
  quant_struct_filter.scale = scale_array_filter;
  quant_struct_filter.zero_point = zero_point_array_filter;
  quant_struct_filter.quantized_dimension = 0;
  // float* filter_data;
  // filter_tensor_data = filter_raw;
  GetConvTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data),
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;

  // Check dimensionality of input, filter
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
  // Check input channels matching filter
  // Filter input channel can be a factor of channels of input (grouped conv)
  // or equals (normal conv).
  auto input_channel = input->dims->data[3];
  auto filter_input_channel = filter->dims->data[3];
  TF_LITE_ENSURE_EQ(context, input_channel % filter_input_channel, 0);
  data->groups = input_channel / filter_input_channel;
  // std::cout << "codes runs here #-3" << std::endl;
  // Check types. (We assume that UINT8 refers to quantized tensors)
  TfLiteType input_type = input->type;
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt8 || input_type == kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

  if (input_type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }
  // Filter must have zero zero-points in per-channel quantization.
  if (input_type == kTfLiteInt16 || input_type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    for (int i = 0; i < affine_quantization->zero_point->size; ++i) {
      TF_LITE_ENSURE_EQ(context, affine_quantization->zero_point->data[i], 0);
    }
  }
  // std::cout << "codes runs here #-4" << std::endl;
  const TfLiteTensor* bias = nullptr;

  // TODO(ahentz): At this point the optimized versions require 'bias'. We can
  // either change that or document that convolution requires it.
  // TF_LITE_ENSURE(context, has_bias);

  if (has_bias) {
    // TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &bias));
    if (input_type == kTfLiteUInt8 || input_type == kTfLiteInt8) {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else if (input_type == kTfLiteInt16) {
      TF_LITE_ENSURE(context, (bias->type == kTfLiteInt32) ||
                                  (bias->type == kTfLiteInt64));
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input_type);
    }
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }
  // std::cout << "codes runs here #-5" << std::endl;
  const bool is_hybrid =
      (input->type == kTfLiteFloat32 &&
       (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8));

  if (is_hybrid && filter->type == kTfLiteInt8 &&
      filter->quantization.type == kTfLiteAffineQuantization &&
      filter->quantization.params &&
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)
          ->scale &&
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params)
              ->scale->size > 1) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    const float scale = affine_quantization->scale->data[0];
    for (int i = 1; i < affine_quantization->scale->size; i++) {
      if (affine_quantization->scale->data[i] != scale) {
        data->is_hybrid_per_channel = true;
        break;
      }
    }
  }
  // std::cout << "codes runs here #-6" << std::endl;
  // The multi-threaded kernel supports neither dilation nor hybrid kernels, and
  // is incompatible with mutable input filters that might change between evals.
  data->supports_multithreaded_kernel =
      (kernel_type == kMultithreadOptimized) &&
      (context->recommended_num_threads != 1) && !is_hybrid &&
      (params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1) &&
      (filter->allocation_type != kTfLiteArenaRw) && !IsDynamicTensor(filter);

  int channels_in = filter->dims->data[3];
  int channels_out = filter->dims->data[0];
  int width = input->dims->data[2];
  int height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int batches = input->dims->data[0];
  // std::cout << "codes runs here #-7" << std::endl;
  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  size_t im2col_type_size;
  TF_LITE_ENSURE_STATUS(GetSizeOfType(context, input->type, &im2col_type_size));
  // Note that we intentionally promote the first multiplicand (i.e. 'batches')
  // to 'size_t' to avoid integer overflow here.
  const size_t im2col_bytes = static_cast<size_t>(batches) * out_height *
                              out_width * channels_in * filter_height *
                              filter_width * im2col_type_size;
  TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
      context, node, is_hybrid, data->is_hybrid_per_channel, kernel_type,
      im2col_bytes));
  // std::cout << "codes runs here #-8" << std::endl;
  // TF_LITE_ENSURE(context, has_bias);

  // Note that full fixed-point inference requires that all tensors have their
  // parameters set. This is usually done during quantized training or
  // calibration.
  if (input_type != kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    // std::cout << "affine_quantization->scale->size: " << affine_quantization->scale->size << std::endl;
    TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 ||
                             affine_quantization->scale->size == channels_out));

    data->per_channel_output_multiplier.resize(channels_out);
    data->per_channel_output_shift.resize(channels_out);
    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), channels_out));
  }
  // std::cout << "codes runs here #-9" << std::endl;
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  auto output_status = context->ResizeTensor(context, output, output_size);

  if (output_status != kTfLiteOk) return output_status;

  if (data->need_im2col) {
    node->temporaries->data[data->im2col_index] = data->im2col_id;

    TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(4);

    auto filter_input_channel = filter->dims->data[3];
    im2col_size->data[0] = output_size->data[0];
    im2col_size->data[1] = output_size->data[1];
    im2col_size->data[2] = output_size->data[2];
    im2col_size->data[3] = filter_input_channel * filter_height * filter_width;

    TfLiteTensor* im2col =
        &context->tensors[node->temporaries->data[data->im2col_index]];
    im2col->type = input->type;
    if (is_hybrid) {
      im2col->type = filter->type;
    }
    im2col->allocation_type = kTfLiteArenaRw;
    auto im2col_status = context->ResizeTensor(context, im2col, im2col_size);
    if (im2col_status != kTfLiteOk) return im2col_status;
  }

  if (data->need_hwcn_weights) {
    node->temporaries->data[data->hwcn_weights_index] = data->hwcn_weights_id;
    TfLiteIntArray* hwcn_weights_size = TfLiteIntArrayCreate(2);

    // Because we're treating the filter weights as a matrix when we do the
    // transpose, we allocate the buffer with a two-dimensional shape, where one
    // dimension is the number of elements in each filter, and the second is the
    // total number of filters.
    auto filter_input_channel = filter->dims->data[3];
    hwcn_weights_size->data[0] =
        (filter_height * filter_width * filter_input_channel);
    hwcn_weights_size->data[1] = channels_out;

    TfLiteTensor* hwcn_weights =
        &context->tensors[node->temporaries->data[data->hwcn_weights_index]];
    hwcn_weights->type = input_type;
    hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;

    auto hwcn_weights_status =
        context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
    if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

    // TODO(petewarden): If Resize() is called when the size hasn't actually
    // changed, this will do extra redundant work.
    data->have_weights_been_transposed = false;
  }

  if (is_hybrid) {
    node->temporaries->data[data->input_quantized_index] =
        data->input_quantized_id;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->input_quantized_index,
                                  &input_quantized));
    input_quantized->type = kTfLiteInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
    // std::cout << "codes runs here #-10" << std::endl;
    node->temporaries->data[data->scaling_factors_index] =
        data->scaling_factors_id;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->scaling_factors_index,
                                  &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    // Only one scale factor per batch is typically necessary. See optimized
    // implementation for why we need to allocate for the height of the inputs
    // flattened to 2D.
    TF_LITE_ENSURE(context, channels_in != 0);
    const int height = NumElements(input) / channels_in;
    int scaling_dims[1] = {height};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = height;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    node->temporaries->data[data->accum_scratch_index] = data->accum_scratch_id;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, data->accum_scratch_index,
                                       &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    const int scratch_width = batches * out_height * out_width;
    int accum_scratch_dims[2] = {channels_out, scratch_width};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_scratch_size = TfLiteIntArrayCreate(2);
      accum_scratch_size->data[0] = channels_out;
      accum_scratch_size->data[1] = scratch_width;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, accum_scratch,
                                                       accum_scratch_size));
    }

    if (data->is_hybrid_per_channel) {
      const auto* affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization*>(
              filter->quantization.params);
      TF_LITE_ENSURE_EQ(
          context, affine_quantization->scale->size,
          filter->dims->data[affine_quantization->quantized_dimension]);
      node->temporaries->data[data->input_offset_index] = data->input_offset_id;
      TfLiteTensor* input_offsets;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, data->input_offset_index,
                                    &input_offsets));
      input_offsets->type = kTfLiteInt32;
      input_offsets->allocation_type = kTfLiteArenaRw;
      // See above comment for the need to allocate for height of inputs.
      TF_LITE_ENSURE(context, channels_in != 0);
      const int height = NumElements(input) / channels_in;
      const int input_offset_dims[1] = {height};
      if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1,
                                     input_offset_dims)) {
        TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
        input_offsets_size->data[0] = input_offset_dims[0];
        TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                         input_offsets_size));
      }
      node->temporaries->data[data->row_sums_index] = data->row_sums_id;
      TfLiteTensor* row_sums;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, data->row_sums_index, &row_sums));
      row_sums->type = kTfLiteInt32;
      row_sums->allocation_type = kTfLiteArenaRwPersistent;
      // See above comment for the need to allocate for height of inputs.
      const int row_sums_dims[1] = {channels_out};
      if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
        TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
        row_sums_size->data[0] = row_sums_dims[0];
        TF_LITE_ENSURE_OK(
            context, context->ResizeTensor(context, row_sums, row_sums_size));
      }
    }
  }
  // std::cout << "codes runs here #-11" << std::endl;
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(kernel_type, context, node);
}

template <KernelType kernel_type>
void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* im2col,
                   TfLiteTensor* output) {
  auto input_offset = -input->params.zero_point;
  auto filter_offset = -filter->params.zero_point;
  auto output_offset = output->params.zero_point;

  KernelType effective_kernel_type;
  if ((kernel_type == kMultithreadOptimized ||
       kernel_type == kCblasOptimized) &&
      (params->dilation_width_factor != 1 ||
       params->dilation_height_factor != 1)) {
    // kMultithreadOptimized and kCblasOptimized do not support dilation.
    // Therefore, fallback to optimized.
    effective_kernel_type = kGenericOptimized;
  } else {
    effective_kernel_type = kernel_type;
  }

  // We have to fallback to reference execution path when im2col is needed but
  // disabled because to-be-allocated temporary im2col tensor is too large.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
  }

  // Grouped convolution is right now only supported on reference kernel.
  if (data->groups != 1) {
    effective_kernel_type = kReference;
  }

  ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  switch (effective_kernel_type) {
    case kReference: {
      reference_ops::Conv(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(filter), GetTensorData<uint8_t>(filter),
          GetTensorShape(bias), GetTensorData<int32_t>(bias),
          GetTensorShape(output), GetTensorData<uint8_t>(output),
          GetTensorShape(im2col), GetTensorData<uint8_t>(im2col),
          /* cpu_backend_context = */ nullptr);
      break;
    }
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized: {
      // There is only one optimized implementation for Quantized Conv.
      optimized_ops::Conv(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(filter), GetTensorData<uint8_t>(filter),
          GetTensorShape(bias), GetTensorData<int32_t>(bias),
          GetTensorShape(output), GetTensorData<uint8_t>(output),
          GetTensorShape(im2col), GetTensorData<uint8_t>(im2col),
          CpuBackendContext::GetFromContext(context));
      break;
    }
  }
}

template <KernelType kernel_type>
void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, OpData* data,
                             const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             TfLiteTensor* im2col) {
  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  KernelType effective_kernel_type = kernel_type;
  // We have to fallback to reference execution path when im2col is needed but
  // disabled because to-be-allocated temporary im2col tensor is too large.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
  }

  // Grouped convolution is right now only supported on reference kernel.
  if (data->groups != 1) {
    effective_kernel_type = kReference;
  }

  switch (effective_kernel_type) {
    case kReference: {
      reference_integer_ops::ConvPerChannel(
          op_params, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<int8>(input), GetTensorShape(filter),
          GetTensorData<int8>(filter), GetTensorShape(bias),
          GetTensorData<int32>(bias), GetTensorShape(output),
          GetTensorData<int8>(output));
      break;
    }
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized: {
      optimized_integer_ops::ConvPerChannel(
          op_params, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<int8>(input), GetTensorShape(filter),
          GetTensorData<int8>(filter), GetTensorShape(bias),
          GetTensorData<int32>(bias), GetTensorShape(output),
          GetTensorData<int8>(output), GetTensorShape(im2col),
          GetTensorData<int8>(im2col),
          CpuBackendContext::GetFromContext(context));
      break;
    }
  }
}

template <KernelType kernel_type>
void EvalQuantizedPerChannel16x8(TfLiteContext* context, TfLiteNode* node,
                                 TfLiteConvParams* params, OpData* data,
                                 const TfLiteTensor* input,
                                 const TfLiteTensor* filter,
                                 const TfLiteTensor* bias, TfLiteTensor* output,
                                 TfLiteTensor* im2col) {
  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  KernelType effective_kernel_type = kernel_type;
  // We have to fallback to reference execution path when im2col is needed but
  // disabled because to-be-allocated temporary im2col tensor is too large.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
  }

  // Grouped convolution is right now only supported on reference kernel.
  if (data->groups != 1) {
    effective_kernel_type = kReference;
  }

  // To prevent 32bit accum overflow for 16x8 quantization, it enables the
  // optimized path only when zero_point is 0.
  bool has_non_zero_point = input->params.zero_point ||
                            filter->params.zero_point ||
                            output->params.zero_point;

  // Fallback to reference kernel when bias_type is int64 as
  // there is no optimized kernel for int64 bias yet.
  if (bias && bias->type == kTfLiteInt64) {
    reference_integer_ops::ConvPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), GetTensorShape(input),
        GetTensorData<int16>(input), GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<std::int64_t>(bias), GetTensorShape(output),
        GetTensorData<int16>(output));
  } else if (effective_kernel_type == kReference || has_non_zero_point) {
    reference_integer_ops::ConvPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), GetTensorShape(input),
        GetTensorData<int16>(input), GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<std::int32_t>(bias), GetTensorShape(output),
        GetTensorData<int16>(output));
  } else {
    optimized_integer_ops::ConvPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), GetTensorShape(input),
        GetTensorData<int16_t>(input), GetTensorShape(filter),
        GetTensorData<int8_t>(filter), GetTensorShape(bias),
        GetTensorData<std::int32_t>(bias), GetTensorShape(output),
        GetTensorData<int16_t>(output), GetTensorShape(im2col),
        GetTensorData<int16_t>(im2col),
        CpuBackendContext::GetFromContext(context));
  }
}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteConvParams* params, OpData* data,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* im2col,
               TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
  // std::cout << "codes runs here #4" << std::endl;
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  KernelType effective_kernel_type = kernel_type;
  // Fall back to the optimized path if multi-threaded conv is unsupported.
  if ((kernel_type == kMultithreadOptimized) &&
      !data->supports_multithreaded_kernel) {
    effective_kernel_type = kGenericOptimized;
  }
  // std::cout << "codes runs here #5" << std::endl;
  // When im2col is needed (which is implied when 'im2col_oversized' is true),
  // the GEMMM-based optimized path requires im2col data be allocated to ensure
  // the correctness. Therefore, when im2col is disabled because of the
  // oversized temporary im2col tensor, fallback to a non-optimized path is
  // needed.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
    // As detailed by tflite::multithreaded_ops::Conv implementation in
    // multithreaded_conv.h, the Eigen-based execution doesn't need im2col data.
    // Therefore, we could rely on it as a better-optimized fallback than the
    // reference one.
    if (data->supports_multithreaded_kernel) {
      effective_kernel_type = kMultithreadOptimized;
    }
#endif
  }
  // std::cout << "codes runs here #6" << std::endl;
  // Grouped convolution is right now only supported on reference kernel.
  if (data->groups != 1) {
    effective_kernel_type = kReference;
  }

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  switch (effective_kernel_type) {
    case kReference: {
      reference_ops::Conv(op_params, GetTensorShape(input),
                          GetTensorData<float>(input), GetTensorShape(filter),
                          GetTensorData<float>(filter), GetTensorShape(bias),
                          GetTensorData<float>(bias), GetTensorShape(output),
                          GetTensorData<float>(output), GetTensorShape(im2col),
                          GetTensorData<float>(im2col));
      break;
    }
    case kCblasOptimized:
    case kGenericOptimized: {
      optimized_ops::Conv(op_params, GetTensorShape(input),
                          GetTensorData<float>(input), GetTensorShape(filter),
                          GetTensorData<float>(filter), GetTensorShape(bias),
                          GetTensorData<float>(bias), GetTensorShape(output),
                          GetTensorData<float>(output), GetTensorShape(im2col),
                          GetTensorData<float>(im2col),
                          CpuBackendContext::GetFromContext(context));
      break;
    }
    case kMultithreadOptimized: {
#if defined(TFLITE_WITH_MULTITHREADED_EIGEN)
      // std::cout << "codes runs here #7" << std::endl;
      const float* filter_data;
      if (data->need_hwcn_weights) {
        filter_data = GetTensorData<float>(hwcn_weights);
      } else {
        filter_data = GetTensorData<float>(filter);
      }
      // int index;
      // for (index = 0; index < 432; index++){
      //   // std::cout << "filter_data[" << index << "] = " << filter_data[index] << std::endl;
      //   std::cout << filter_data[index] << ", ";
      // }
      multithreaded_ops::Conv(
          *eigen_support::GetThreadPoolDevice(context), op_params,
          GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), filter_data, GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorShape(output),
          GetTensorData<float>(output), GetTensorShape(im2col),
          GetTensorData<float>(im2col));
      break;
#else   // !defined(TFLITE_WITH_MULTITHREADED_EIGEN)
      // See Register_CONV_2D: we should never be here when TFLITE_WITH_RUY
      // was enabled. We #if out this code in order to get the corresponding
      // binary size benefits.
      TFLITE_DCHECK(false);
#endif  // defined(TFLITE_WITH_MULTITHREADED_EIGEN)
    }
  }
}

template <KernelType kernel_type>
TfLiteStatus EvalHybridPerChannel(TfLiteContext* context, TfLiteNode* node,
                                  TfLiteConvParams* params, OpData* data,
                                  const TfLiteTensor* input,
                                  const TfLiteTensor* filter,
                                  const TfLiteTensor* bias,
                                  TfLiteTensor* im2col, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  const int batch_size = SizeOfDimension(input, 0);
  TF_LITE_ENSURE(context, batch_size != 0);
  const int input_size = NumElements(input) / batch_size;
  TfLiteTensor* quantized_input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                                     &quantized_input_tensor));
  int8_t* quantized_input_ptr_batch =
      GetTensorData<int8_t>(quantized_input_tensor);
  TfLiteTensor* scaling_factors_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->scaling_factors_index,
                                     &scaling_factors_tensor));
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors_tensor);
  TfLiteTensor* input_offset_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_offset_index,
                                     &input_offset_tensor));
  int32_t* input_offset_ptr = GetTensorData<int32_t>(input_offset_tensor);

  for (int b = 0; b < batch_size; ++b) {
    const int offset = b * input_size;
    tensor_utils::AsymmetricQuantizeFloats(
        GetTensorData<float>(input) + offset, input_size,
        quantized_input_ptr_batch + offset, &scaling_factors_ptr[b],
        &input_offset_ptr[b]);
  }

  int8_t* im2col_ptr = nullptr;
  int8_t* filter_ptr = nullptr;
  if (im2col != nullptr) {
    im2col_ptr = im2col->data.int8;
  }
  filter_ptr = filter->data.int8;
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);

  KernelType effective_kernel_type = kernel_type;
  // We have to fallback to reference execution path when im2col is needed but
  // disabled because to-be-allocated temporary im2col tensor is too large.
  // See b/178743262 for the detailed motivation.
  if (data->im2col_oversized) {
    effective_kernel_type = kReference;
  }

  // Grouped convolution is right now only supported on reference kernel.
  if (data->groups != 1) {
    effective_kernel_type = kReference;
  }

  ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  switch (effective_kernel_type) {
    case kReference:
      reference_ops::HybridConvPerChannel(
          op_params, scaling_factors_ptr, GetTensorShape(input),
          quantized_input_ptr_batch, GetTensorShape(filter), filter_ptr,
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(im2col), im2col_ptr, affine_quantization->scale->data,
          input_offset_ptr);
      break;
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized: {
      TfLiteTensor* row_sums;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, data->row_sums_index, &row_sums));
      TfLiteTensor* scratch;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, data->accum_scratch_index, &scratch));
      optimized_ops::HybridConvPerChannel(
          op_params, scaling_factors_ptr, GetTensorShape(input),
          quantized_input_ptr_batch, GetTensorShape(filter), filter_ptr,
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(im2col), im2col_ptr, affine_quantization->scale->data,
          input_offset_ptr, GetTensorShape(scratch),
          GetTensorData<int32>(scratch), GetTensorData<int32_t>(row_sums),
          &data->compute_hybrid_row_sums,
          CpuBackendContext::GetFromContext(context));
      data->compute_hybrid_row_sums = false;
      break;
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        TfLiteConvParams* params, OpData* data,
                        const TfLiteTensor* input, const TfLiteTensor* filter,
                        const TfLiteTensor* bias, TfLiteTensor* im2col,
                        TfLiteTensor* accum_scratch, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  const int batch_size = SizeOfDimension(input, 0);
  TF_LITE_ENSURE(context, batch_size != 0);
  const int input_size = NumElements(input) / batch_size;

  const float* input_ptr = GetTensorData<float>(input);
  TfLiteTensor* quantized_input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                                     &quantized_input_tensor));
  int8_t* quantized_input_ptr_batch =
      GetTensorData<int8_t>(quantized_input_tensor);
  TfLiteTensor* scaling_factors_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->scaling_factors_index,
                                     &scaling_factors_tensor));
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors_tensor);

  // Per-batch input quantization for higher accuracy.
  {
    ruy::profiler::ScopeLabel label("ConvHybridQuantizeInputs");
    for (int b = 0; b < batch_size; ++b) {
      float unused_min, unused_max;
      const int offset = b * input_size;
      tensor_utils::SymmetricQuantizeFloats(
          input_ptr + offset, input_size, quantized_input_ptr_batch + offset,
          &unused_min, &unused_max, &scaling_factors_ptr[b]);
      scaling_factors_ptr[b] *= filter->params.scale;
    }
  }

  switch (kernel_type) {
    case kReference:
    case kGenericOptimized:
    case kMultithreadOptimized:
    case kCblasOptimized: {
      // There is only one implementation for hybrid kernel.
      ConvParams op_params;
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width = data->padding.width;
      op_params.padding_values.height = data->padding.height;
      op_params.stride_width = params->stride_width;
      op_params.stride_height = params->stride_height;
      op_params.dilation_width_factor = params->dilation_width_factor;
      op_params.dilation_height_factor = params->dilation_height_factor;
      op_params.float_activation_min = output_activation_min;
      op_params.float_activation_max = output_activation_max;
      if (data->groups == 1) {
        optimized_ops::HybridConv(
            op_params, scaling_factors_ptr, GetTensorShape(input),
            quantized_input_ptr_batch, GetTensorShape(filter),
            GetTensorData<int8_t>(filter), GetTensorShape(bias),
            GetTensorData<float>(bias), GetTensorShape(accum_scratch),
            GetTensorData<int32_t>(accum_scratch), GetTensorShape(output),
            GetTensorData<float>(output), GetTensorShape(im2col),
            GetTensorData<int8_t>(im2col),
            CpuBackendContext::GetFromContext(context));
      } else {
        // This case is handled by (fallbacked to) per channel hybrid group conv
        // and shouldn't hit this branch.
        TF_LITE_KERNEL_LOG(
            context,
            "Group convolution currently not supported for hybrid kernel.");
        return kTfLiteError;
      }
      break;
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type, TfLiteType input_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  // std::cout << "codes runs here #0" << std::endl;
  // auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  TfLiteConvParams data_params;
  ExtractConvParams(paddings, stride_width, stride_height, dilation_width_factor, dilation_height_factor, activation, &data_params);
  TfLiteConvParams* params = &data_params;

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  // std::cout << "codes runs here #1" << std::endl;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor filter_tensor;
  TfLiteIntArray* filter_dims_data = TfLiteIntArrayCreate(filter_dims_size);
  int size_filter = 1;
  for (int i = 0; i < filter_dims_size; i++) {
    // std::cout << "dims_raw: " << dims_raw[i] << std::endl;
    filter_dims_data->data[i] = filter_dims_raw[i];
    size_filter *= filter_dims_raw[i];
  }
  size_t bytes_size_filter = sizeof(float) * size_filter;
  TfLiteQuantizationParams filter_params;
  filter_params.scale=scale_filter;
  filter_params.zero_point=zero_point_filter;

  TfLiteFloatArray* scale_array_filter = TfLiteFloatArrayCreate(1);
  scale_array_filter->data[0] = scale_filter;
  TfLiteIntArray* zero_point_array_filter = TfLiteIntArrayCreate(1);
  zero_point_array_filter->data[0] = zero_point_filter;

  TfLiteAffineQuantization quant_struct_filter;
  quant_struct_filter.scale = scale_array_filter;
  quant_struct_filter.zero_point = zero_point_array_filter;
  quant_struct_filter.quantized_dimension = 0;

  // float* filter_data;
  // filter_tensor_data = filter_raw;
  GetConvTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data), 
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;

  TfLiteTensor bias_tensor;
  const TfLiteTensor* bias;
  if (has_conv_bias) {
    TfLiteIntArray* bias_dims_data = TfLiteIntArrayCreate(bias_dims_size);
    int size_bias = 1;
    for (int i = 0; i < bias_dims_size; i++) {
      // std::cout << "dims_raw: " << dims_raw[i] << std::endl;
      bias_dims_data->data[i] = bias_dims_raw[i];
      size_bias *= bias_dims_raw[i];
    }
    size_t bytes_size_bias = sizeof(float) * size_bias;
    TfLiteQuantizationParams bias_params;
    bias_params.scale=scale_bias;
    bias_params.zero_point=zero_point_bias;

    TfLiteFloatArray* scale_array_bias = TfLiteFloatArrayCreate(1);
    scale_array_bias->data[0] = scale_bias;
    TfLiteIntArray* zero_point_array_bias = TfLiteIntArrayCreate(1);
    zero_point_array_bias->data[0] = zero_point_bias;

    TfLiteAffineQuantization quant_struct_bias;
    quant_struct_bias.scale = scale_array_bias;
    quant_struct_bias.zero_point = zero_point_array_bias;
    quant_struct_bias.quantized_dimension = 0;
    
    // float* bias_data;
    // bias_tensor_data = bias_raw;
    GetConvTensor(bias_type, "bias", bias_dims_data, bias_params,
                        reinterpret_cast<char*>(bias_tensor_data), 
                        &quant_struct_bias, bytes_size_bias, &bias_tensor);
    bias = &bias_tensor;
  } else {
    bias = nullptr;
  }

  TfLiteTensor* im2col =
      data->need_im2col
          ? &context->tensors[node->temporaries->data[data->im2col_index]]
          : nullptr;
  TfLiteTensor* hwcn_weights =
      data->need_hwcn_weights
          ? &context->tensors[node->temporaries->data[data->hwcn_weights_index]]
          : nullptr;

  if (data->need_hwcn_weights && !data->have_weights_been_transposed) {
    TransposeFloatTensor(filter, hwcn_weights);
    data->have_weights_been_transposed = true;
  }
  // std::cout << "codes runs here #3" << std::endl;
  TFLITE_DCHECK_EQ(input_type, input->type);
  switch (input_type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      if (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8) {
        if (data->is_hybrid_per_channel ||
            // TODO(b/162870360): Fallback to PerChannel implementation
            // before we have grouped hybrid convolution.
            data->groups != 1) {
          TF_LITE_ENSURE_OK(context, EvalHybridPerChannel<kernel_type>(
                                         context, node, params, data, input,
                                         filter, bias, im2col, output));
        } else {
          TfLiteTensor* accum_scratch =
              &context->tensors[node->temporaries
                                    ->data[data->accum_scratch_index]];
          TF_LITE_ENSURE_OK(context,
                            EvalHybrid<kernel_type>(context, node, params, data,
                                                    input, filter, bias, im2col,
                                                    accum_scratch, output));
        }
      } else {
        EvalFloat<kernel_type>(context, node, params, data, input, filter, bias,
                               im2col, hwcn_weights, output);
      }
      break;
    case kTfLiteUInt8:
      EvalQuantized<kernel_type>(context, node, params, data, input, filter,
                                 bias, im2col, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel<kernel_type>(context, node, params, data, input,
                                           filter, bias, output, im2col);
      break;
    case kTfLiteInt16:
      EvalQuantizedPerChannel16x8<kernel_type>(
          context, node, params, data, input, filter, bias, output, im2col);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  // std::cout << "codes runs here #10" << std::endl;
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));

  switch (input->type) {
    case kTfLiteFloat32:
      return EvalImpl<kernel_type, kTfLiteFloat32>(context, node);
    case kTfLiteUInt8:
      return EvalImpl<kernel_type, kTfLiteUInt8>(context, node);
    case kTfLiteInt8:
      return EvalImpl<kernel_type, kTfLiteInt8>(context, node);
    case kTfLiteInt16:
      return EvalImpl<kernel_type, kTfLiteInt16>(context, node);
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace conv

TfLiteRegistration* Register_snsidn_REF() {
  static TfLiteRegistration r = {snsidn::Init, snsidn::Free,
                                 snsidn::Prepare<snsidn::kReference>,
                                 snsidn::Eval<snsidn::kReference>};
  return &r;
}

TfLiteRegistration* Register_snsidn_GENERIC_OPT() {
  static TfLiteRegistration r = {snsidn::Init, snsidn::Free,
                                 snsidn::Prepare<snsidn::kGenericOptimized>,
                                 snsidn::Eval<snsidn::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_snsidn_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {snsidn::Init, snsidn::Free,
                                 snsidn::Prepare<snsidn::kMultithreadOptimized>,
                                 snsidn::Eval<snsidn::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_snsidn_CBLAS_OPT() {
//   static TfLiteRegistration r = {snsidn::Init, snsidn::Free,
//                                  snsidn::Prepare<snsidn::kCblasOptimized>,
//                                  snsidn::Eval<snsidn::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_snsidn() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_snsidn_MULTITHREADED_OPT();
#else
  return Register_snsidn_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
