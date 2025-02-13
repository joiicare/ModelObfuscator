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
namespace sgpyea {

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

int8_t filter_r   aw[3456]={33, -37, 58, 1, 34, -24, 59, -29, -19, 97, -9, -64, -21, -59, 2, -32, 81, 19, 49, 66, -48, 9, 12, -33, 19, 23, 73, -45, -10, 42, -5, 75, 2, -26, -9, -11, -62, -57, 6, 65, 12, -17, -11, 83, 13, -21, -43, 43, -3, 39, 112, 69, 32, 0, -5, 67, -21, 2, -9, 23, -63, -36, -71, 28, -25, 6, 98, -19, -79, 45, 4, -8, 80, 46, 65, 38, -2, -2, 29, -43, 40, 62, -29, 19, -68, 20, -4, 7, 14, -98, 3, 3, 56, -1, 8, 56, 12, -55, 22, -40, -62, -29, -45, 62, 31, 21, 54, 19, -37, -58, 26, 11, 45, -73, 0, -23, -53, 24, -14, 43, -20, 36, 5, 84, 30, -38, -19, 13, 43, -39, -47, -24, 16, 7, 74, -10, 28, -127, 52, 6, -4, -2, -37, -44, 41, -11, 24, 62, 47, 8, -27, -36, -80, -4, -46, -127, -7, 2, 16, -15, 31, 11, -100, -64, 41, 24, 64, 86, 0, -29, -6, -109, -38, 56, 10, -22, 25, -23, 0, -26, 27, 6, 24, -34, 6, -65, 14, 45, -16, 6, 38, 20, 23, 13, 24, -20, 71, -9, -12, 25, -9, 30, -15, 21, -24, 53, 96, 26, 43, -68, 14, -13, -13, 76, -3, -22, -30, 25, -12, -12, 3, 0, 3, 21, -14, -24, -88, 5, -13, 17, -21, -18, -51, 97, -36, 11, -29, 11, 20, -7, 7, 32, 12, 39, -1, 51, 31, -33, 59, -11, 12, 61, 34, 34, -18, -41, 72, 13, -13, 17, 49, 2, -12, 61, -89, -42, -18, -96, -33, -24, 25, -30, 30, -20, -22, 6, 2, -12, -23, 68, -10, 54, 74, -2, 7, 73, 39, -6, -18, 16, -17, 21, -6, -24, 63, -46, -39, 3, 70, 3, 9, 21, -1, -30, 72, -101, 16, -27, 11, -42, 68, 59, 69, -65, -8, 11, 5, -45, 55, 50, -21, -6, 8, 35, 25, -68, 12, 30, 5, 31, -9, -12, 16, -59, -12, -1, -8, 19, -16, -2, 8, -1, -15, 0, 9, 51, 15, -39, 9, 12, 63, -5, 34, -35, -26, 25, 31, 20, 26, -7, 87, 19, -6, -37, 8, -15, -20, 57, -25, -43, -59, -46, 14, -4, 1, -7, -20, -38, 92, -15, -44, -9, -74, 28, -59, 70, -63, 68, -7, 53, 74, 25, -5, -41, -18, 127, 4, -57, -12, 7, -33, 51, -34, 10, 96, -14, 54, -41, -18, -16, -17, -75, -82, -5, 81, 35, 1, 3, -11, 77, -14, -16, 37, 10, -61, 106, 3, 19, 11, -60, 48, -9, 63, 6, -21, 15, 0, -2, 17, 2, 21, -19, -30, -7, -15, -23, 85, 77, 13, 23, -1, 5, 0, 9, 21, -8, -1, 14, 1, -16, 11, -11, 10, 15, 12, 22, -127, -6, 8, -3, 44, -18, 10, 0, 6, 37, -19, -14, 9, 11, -53, 10, 6, -35, -7, -9, -93, -17, -66, 26, -5, 14, -11, -13, 1, 15, 6, -11, 5, -81, 75, 7, 9, -5, -1, 31, 22, 11, -36, 32, -7, 11, 0, 10, 8, 13, 25, -75, 17, -35, -5, -2, 19, 3, -8, -1, -4, 1, -8, -26, 6, -6, -6, -9, 6, 4, 21, 6, -36, 8, -4, 1, -42, -2, 20, 15, -3, 17, -5, -6, -12, 8, -31, 1, -6, 35, -5, 21, 13, -2, -43, -39, -23, 18, 12, 4, -3, 3, 6, 7, 17, 3, -105, 11, 13, 9, 17, -7, 22, 21, 6, 25, 31, -1, -24, 8, 18, 88, -1, -10, 79, -9, -2, 26, 15, -9, 19, -4, 49, 58, 13, -57, -12, 74, 5, 17, 41, -23, -61, 2, -62, 15, 32, 29, 72, -42, 16, -7, -4, 44, 10, -32, 23, -1, 32, 0, 23, 61, 35, 35, -24, 35, -40, -26, -57, 6, -31, -35, 5, 121, -4, 38, -15, -14, -41, -57, -14, 19, -15, -15, 2, -5, 24, 93, -1, 21, 11, -40, -39, 6, -9, 111, 52, -12, -20, 20, -37, -7, 3, -43, -33, -9, -61, 2, -4, 33, 9, 57, 12, -55, 7, 0, 19, -3, 46, 44, -1, -28, 30, 43, -8, -3, 28, 83, 24, 14, 127, 19, -19, -79, -30, -37, 9, 36, 31, -87, 16, 12, -37, 1, -18, 26, 14, 67, -90, 0, 1, 6, -10, 16, -54, -21, -7, -17, -4, -5, -21, 1, -7, -5, 28, 3, 13, -20, 122, -127, -14, -29, -7, -9, -26, -8, -7, -22, -12, 27, 14, -3, -22, -26, -25, 32, -18, 17, -96, 11, -8, 1, 38, 21, 0, -14, 4, -83, 16, 23, -13, -11, 15, -1, 4, 0, -8, 0, -104, -23, 15, -16, 31, -10, -66, 25, -7, -40, 4, -6, 1, 57, -59, -13, 4, -7, 8, -14, -21, 7, 95, -5, 5, -16, 6, 2, 28, -4, -27, -92, -7, -68, 5, -2, 4, 1, 13, 14, 16, 2, 15, -28, 2, -1, 8, -27, 17, 11, -10, -2, 36, -8, -24, -3, 66, 1, -15, -25, 4, -11, -9, 37, 2, 5, -6, -2, 6, 7, 24, -94, -11, -16, 35, 16, 27, -11, -2, 19, -5, -13, -6, -15, 6, -21, -12, -7, 7, -5, 0, -36, 8, 24, 36, -34, -8, -17, 6, 16, -27, -3, -45, -8, 20, -14, -54, 17, 65, -64, -73, 55, 48, 13, 8, 18, -10, -5, 19, 25, -2, 3, -14, -21, 18, -18, 15, 10, -2, -4, -18, 60, -37, 21, 6, -28, -46, 11, 9, -9, -7, -15, 27, -20, 29, -22, -7, -16, 23, -13, -9, -39, 22, 2, -27, 21, 38, 0, 15, 6, -20, -18, -11, 6, -28, 8, -11, -1, -22, -68, -17, -52, 5, -10, 1, -2, 8, 23, 35, -2, -8, 35, -3, -81, 24, -59, -48, -27, 27, -34, -30, -15, -24, -54, 44, -27, 19, 54, -60, 23, -12, 29, 26, 32, 1, -12, -12, -7, 9, 26, -1, -17, -17, -44, 106, 17, -9, 11, -4, -3, 20, 46, -28, -21, -2, 7, -24, 127, -12, -58, 2, 11, 10, -61, -20, 14, -39, 49, 14, -67, -15, 39, 16, 78, 12, 75, -2, 6, -1, 14, -23, -5, -60, 34, -22, -45, -32, -12, -12, 49, 45, 56, 52, 20, -27, 70, -57, -12, 31, 24, -17, -25, -38, 73, -55, -36, -5, -27, 8, 1, 77, 49, 9, -11, 12, 21, 1, 11, -8, -14, 5, -23, 56, 22, 9, -29, 81, -36, -35, -37, 82, -29, 5, -19, 20, 4, 29, 15, -1, 38, -18, -18, -26, -18, -3, -1, -3, -18, -18, -19, 11, 33, -18, 45, 9, -19, -3, 13, -5, 50, 38, -51, -36, -64, 14, -25, -48, 13, 20, -24, 43, 105, -63, 30, 36, -47, 5, -77, 10, -9, 27, -27, -27, 15, 56, 14, 12, 18, 78, 9, -24, -48, -19, 5, -24, -83, -3, -57, 77, -41, 32, -127, -45, -37, -26, -43, -56, 4, -7, 14, -53, 7, -15, -1, -67, -1, 9, 11, 27, -34, 21, 8, 45, -55, -21, -80, 9, 36, -22, -8, -12, -84, -47, 67, 8, -44, -22, -47, -35, 43, -9, 21, -47, -4, 8, -19, -14, -14, -99, 14, 30, -90, 3, 24, 3, -11, -13, -31, 29, -18, -27, 35, 36, -26, -104, 19, 77, 25, 3, 34, -23, 87, 47, -28, -8, 12, 0, 24, -5, 36, 45, -13, -40, 9, -115, 112, -30, -27, -45, 14, 86, -27, -39, 85, -9, 58, 38, 75, 77, -12, 22, 65, 28, -15, 56, 41, -33, 53, 14, -78, 30, 30, -5, -33, 31, -18, -82, -6, 59, 41, -20, -54, 20, 27, -14, 56, 23, -57, -27, -46, 36, 64, -6, 34, 37, -16, 30, -127, -54, -60, -20, -84, 38, -24, 19, -61, 47, -8, -100, -5, 61, 15, 7, 25, -3, 24, -41, 89, 4, 54, 65, 20, -23, 14, -44, 19, 4, -22, 74, 71, -5, -32, 7, 78, 89, 59, 32, -42, 2, 56, 84, -41, 31, 40, 89, -26, -3, -64, 4, 2, 12, 26, 6, 28, -36, -10, -64, -52, -8, 16, 23, -3, -49, -46, -35, -30, -12, 26, -23, 5, -26, 13, 5, -50, -82, 13, -25, 8, -15, -2, -24, -9, -30, -43, 72, 72, -11, -13, -9, 9, -54, 40, -48, -22, -47, -7, 86, 9, 27, 4, -50, 15, 47, 29, -51, -15, -33, -23, -55, -20, -42, -11, 3, 61, -16, -55, -43, -21, -1, -113, -32, 22, 30, -108, -44, 12, -16, -38, -37, -14, -4, -42, -48, -35, -25, -108, -35, -4, 20, -1, -127, -10, -57, 22, -36, -13, 33, -7, -6, 115, -58, -45, -1, -24, -28, -56, 22, -27, 50, -76, 27, 5, -121, 31, 6, 35, -2, 16, 26, -35, -3, 38, 50, -41, -72, -46, -13, -54, -52, -57, 33, -24, 29, -6, 30, 5, 4, 19, 16, -61, -26, -62, 4, -16, 28, 28, 43, 107, 35, 58, 17, -25, -61, -14, -1, 35, -25, 22, 15, -21, 5, 49, 13, 0, 58, 38, -46, 27, 24, 5, -16, -58, 31, 20, -39, 31, -62, -16, 14, -36, -114, 12, 52, -4, -31, -103, -127, -59, -16, -77, -24, 17, -58, -64, 20, 47, -10, 35, 24, 58, 48, 17, 59, -28, -38, -18, -39, -23, -24, 0, -4, -55, -4, 3, -3, 52, -47, -5, 2, -30, 50, 1, 62, 43, 51, 12, 25, -127, 57, 80, -12, -29, -7, -64, 27, -13, 3, -19, -95, 49, 13, 4, 41, -127, 28, -97, 24, -11, -9, 8, 36, -49, 63, -19, 9, -15, -32, -37, 6, 20, -31, 8, 22, -46, 31, 7, -82, 17, -53, 25, 0, 55, -27, -30, -5, 17, -12, 19, -21, -11, -3, 46, -12, -36, 16, -4, 16, -1, -1, 24, -21, 66, 19, 87, -23, -10, 127, 29, 12, 19, 11, -5, 39, 2, 0, 48, 16, -38, 68, -5, -57, -4, -43, 7, 45, -63, 0, 10, -62, 13, 12, 3, 56, 3, -3, 9, 40, -9, -47, 37, -25, 70, -24, -2, -3, 10, 40, -2, 61, -70, 52, -63, -24, 31, 21, -108, 44, 50, -21, 59, 59, -40, 73, -4, 12, -24, -23, 6, -84, -92, 10, 68, 0, -56, -12, 13, -94, 27, -53, -3, -6, -7, -35, 83, 12, -22, 49, -50, -23, -11, -5, -14, -4, -26, 40, 19, -45, -111, 22, 57, 22, -14, 53, -7, 17, 2, 19, 66, -13, 22, -49, 36, 6, 36, 69, 10, -1, 2, 0, 20, -81, -37, 3, -19, 7, 67, 42, 71, -38, -54, 43, 23, -2, -15, 0, 42, 15, 55, 36, 23, 1, 16, 18, 57, 37, -65, 15, 48, -2, -29, -50, 28, 8, 31, -9, 6, -18, 25, -10, 8, 39, 60, 0, 42, 4, 16, -48, -17, 41, 9, -1, -5, 1, -19, 9, 34, 38, 23, -9, 10, 39, -41, -3, 7, -2, -59, -66, 13, 24, -21, 31, -19, 69, 127, 32, 44, 0, -13, -24, 43, -14, 24, -60, 29, 17, 53, 35, -34, -51, 6, 10, -4, -26, 67, -27, 86, -38, 79, 36, 31, 1, -33, 4, -18, 0, -23, -4, -91, -26, -18, 6, 35, -7, 45, 4, 6, -28, 17, 38, -18, 36, 59, -25, 50, 41, 10, -15, 23, 4, 0, -27, 4, -13, 19, -2, 5, -4, 7, -16, 21, 32, 20, -97, -22, 1, 25, -4, 0, -14, -76, -39, -1, 61, -25, 34, 19, -49, 16, 4, 4, -10, 25, -51, -14, 25, 12, 6, -5, 127, -24, 34, -48, -1, -59, 120, 3, -57, 96, 4, 7, 4, 0, 4, -14, -28, -31, 2, 8, 32, 13, 20, 2, 7, 7, 6, 18, -3, -7, 12, 5, -5, 19, 13, 5, -61, -12, -9, 40, 2, 10, -46, 4, 5, -36, 16, -53, 17, 32, -14, -64, -3, 38, -1, 19, -1, -11, -24, -12, 34, 10, -20, -26, 11, 1, 10, -26, 12, 32, 37, 55, 29, 1, 5, 17, 3, -31, 37, 12, 13, -13, 22, 61, -26, 30, 13, 33, -1, -9, -28, 37, 2, 89, -4, 2, 1, -92, 11, -5, -14, -1, -22, 25, 84, 12, 12, -24, -74, -8, -13, -16, -46, -33, -19, -74, -40, 24, 12, 17, -48, -2, 0, -87, 17, -23, -1, 3, -17, -47, -63, -46, 64, -91, -2, -17, -26, -6, -7, 16, -42, -98, 24, -6, 23, 24, 9, 1, -66, -20, 25, -25, 29, -33, 95, -2, -98, 0, -7, 14, 11, 27, 9, 22, 14, -31, 55, 18, -68, 37, -82, -76, 34, 15, 10, 3, -24, 14, 39, -18, 19, 6, -39, 86, 21, -27, 65, 13, 5, -2, 13, 15, 34, 12, -27, -18, -10, -28, -41, 26, 0, -43, -50, -39, -16, 28, -15, -17, -115, -27, -8, 24, 4, 24, 36, 27, -111, 97, -34, -32, 35, 2, 13, -106, -17, 5, -6, -60, -4, -1, -34, 25, 9, 24, -18, 28, 16, 30, -8, -78, 32, 127, 4, 7, 7, -51, -94, 6, 7, -71, 24, -7, -6, -15, 14, -54, -30, 50, 33, 7, -4, 18, 2, 15, 70, -64, -108, -67, -36, 93, -10, 81, 18, 4, -23, -4, -10, -37, -64, 60, 2, 21, -55, 30, -47, -34, 33, 51, 43, -30, -2, 8, 17, 57, -94, 11, -12, 47, -7, 4, -71, -7, 23, -32, -54, 8, 47, -49, -16, 23, -42, 62, -27, -92, 20, -37, 51, -36, 3, -1, 39, 8, 6, -12, -34, 57, 8, -10, 35, -9, -43, 18, -16, 50, 9, -23, 1, -43, 54, 47, -39, -9, 29, 10, 8, -25, 2, 8, 10, -12, -12, -4, -70, 11, -28, -76, -42, -52, -37, -4, -66, 66, -38, 16, -70, 53, -21, -67, -13, 26, -28, -36, 58, 34, -25, -121, 17, -66, 72, 69, -8, 65, 45, -42, -83, 127, -25, -28, 11, -36, -46, 42, -17, 38, -18, 21, 3, 16, -1, -30, -29, -1, -32, -9, 1, 12, 1, -7, 31, 26, -17, -47, 13, 2, 91, -19, 40, -45, -7, -13, -3, 12, -36, -28, 21, 14, -2, -20, -82, 6, 6, -22, 1, 2, -4, -23, 33, -59, 23, 28, 7, -14, -3, 20, -1, 5, -8, -12, 5, -48, -33, -39, 10, 15, 56, 2, -40, -46, 1, 10, 16, -12, -24, 1, -24, -4, -1, -45, 9, 0, 0, 27, -24, 40, 12, -22, 0, 6, -22, 11, 63, 15, -48, 6, -21, 35, -10, 1, 32, -44, 5, -51, -1, 29, -51, 32, 5, 5, -34, -40, 53, -76, -28, 87, 10, 76, -31, -1, 20, -11, 19, 13, 24, -6, -45, -127, -66, -13, -36, -10, 11, 0, -2, 6, -16, 3, -19, -13, -50, 122, -4, 65, -9, -15, -45, 14, 9, -29, 37, -32, -8, -7, -7, 39, -2, 6, 20, -4, 15, -25, 3, 17, -61, -6, -70, -127, 65, -28, 78, 0, 5, 12, 3, 0, -5, 9, 3, 3, -13, -124, 19, 44, -35, -5, -17, 20, -36, 27, 19, 36, -7, -98, -27, -8, -7, -28, -8, -12, -6, 20, 4, 28, -2, 7, -24, 21, -23, 4, -14, -15, -8, 15, 1, -17, -45, 1, 35, 5, 16, -9, 26, 25, -6, 0, 36, -5, 37, -16, 3, 41, 10, -9, -9, 25, 7, -44, 24, 6, 7, -16, -22, 9, -54, 17, 23, 12, 0, 47, 11, -20, 10, 30, -6, -21, -16, 13, -22, -18, -6, -12, 16, 8, -9, -21, -42, -4, -5, 31, -48, -12, 5, -4, -50, -28, 15, -4, 18, -45, 26, -28, 16, -33, 43, 8, -20, 12, 2, -1, 25, 25, -22, -3, 0, -27, -1, -39, -54, 18, -7, -20, 33, 19, -26, 3, -60, -26, -127, 67, -59, 31, 32, 75, -13, 41, -87, -24, -7, 25, 92, -17, 27, -23, -35, 56, -51, 19, -26, 14, 19, -23, 87, 33, -43, -22, 10, 17, -52, 20, -42, 47, 12, 5, 12, -23, 2, -11, -37, 13, -1, 22, 14, -27, 1, 24, 20, -4, -19, 11, 29, -1, -1, -14, -35, -13, 1, -13, -33, 9, -6, 19, -4, -8, -20, -10, -2, -4, 10, -53, 50, -35, 6, 51, 23, -60, -25, -36, 0, -14, -2, -14, -9, 24, -24, 18, 17, -24, 34, 122, 6, 3, 15, -33, -3, 42, 25, -38, 20, -2, -95, -15, 19, 8, -62, -25, 8, 15, -14, -12, 22, -24, -7, 83, -15, 79, -37, 12, 36, 109, -42, 56, -1, 11, 19, -55, 70, -43, 48, -4, 2, -80, -17, -34, 85, -44, -33, 51, 30, -46, -8, -37, 104, -84, -78, 23, -15, -12, -70, -78, -43, 56, -45, 92, -104, 44, -14, -58, -19, -17, 11, -40, -17, 16, -33, -70, -6, 6, 76, 60, -127, 66, -82, -6, 5, -13, 56, -4, 65, 6, -9, 48, -13, 26, 12, 14, -32, -7, 32, -8, 43, -64, -16, -96, -77, -2, -22, 21, -41, 82, 34, 24, -23, -17, 14, -69, -19, -12, 41, 7, 22, -29, 112, 2, 35, -10, -36, 44, -45, 2, 56, 50, 40, 28, 42, 27, -53, -57, 24, 51, 79, 18, -23, -1, 75, 2, 102, 64, 62, -58, -20, 35, 17, -20, -36, 51, 40, -3, 27, 99, -18, 8, -7, -64, -121, -2, -47, -38, -13, -27, -53, 35, -47, -97, -29, 13, -9, -18, 57, 93, -3, 36, -104, 15, 3, 13, 9, -6, -12, 22, -21, 11, -8, 100, 36, -5, 3, 43, -43, -30, 43, 93, -25, 36, 38, -74, -4, -7, 38, 36, 41, 1, 19, 67, -5, -14, 32, -14, 19, -6, -30, 42, -6, -127, -7, -100, -35, 43, -13, 1, -3, 5, 17, 5, -21, -67, 13, -24, -43, 11, 51, -14, -24, 5, -2, 11, 14, 10, 110, -3, 0, -6, -18, 10, 7, -10, -11, 35, 19, 16, 1, -21, -72, 14, 10, 14, 4, 27, 11, 17, -23, 41, 0, -46, -6, 4, -9, 8, 2, 17, -26, 51, 16, -40, -5, 20, 8, 22, 57, 34, 19, 19, 37, 101, 3, -7, -14, 8, -18, 36, -2, -21, 3, -35, 6, 32, -9, -10, 9, 5, 0, -30, -4, -49, 13, 22, 2, -44, -1, 39, -15, -5, -14, -20, -31, 46, -2, 46, -22, -4, 26, -25, -7, -55, -3, 3, -6, -43, -33, 28, -43, -12, -69, 106, 63, 66, -42, -27, 9, 4, -14, -11, 2, -46, 61, -21, 36, -48, -1, -27, 88, -44, -2, -44, 37, 45, -40, 18, 14, -49, -64, 38, 25, 19, -14, 20, -22, -30, -6, 7, 7, -11, -5, -33, 7, -13, -24, -8, 10, 65, -57, 68, -84, 30, -25, -24, 16, -78, 5, 6, -5, -22, -8, 22, 9, -48, 22, -14, -40, -19, -34, 10, -10, 0, 23, 27, 13, -68, -2, 55, -21, 12, -18, 5, -23, -22, 8, -62, -24, -7, -38, -14, 69, 105, 8, 40, -41, -127, -24, -30, -34, -29, -71, -45, 15, -23, 24, 7, 45, -31, -12, -90, 38, 36, -27, -70, 20, -3, -23, -43, -2, -60, -4, -3, -26, -43, 5, -28, 3, -20, 22, -22, 4, 26, 34, 4, -8, 8, 12, -16, 14, -25, -1, -6, 10, 0, 6, 23, 73, -12, -55, 19, 17, 4, 17, -4, -1, -10, 127, -2, 2, -11, 2, 4, 77, 21, 6, 10, -12, -11, 11, 19, 9, 77, -16, 108, 4, -8, 5, 0, 0, -20, 5, 3, -3, 16, 16, 6, -6, 20, 3, -1, 5, -15, 9, 2, -24, -5, -4, -7, 13, 4, -10, 6, -1, 8, 4, -1, -8, -67, 19, -8, 5, -8, -5, 5, 8, -8, 1, 3, 0, -17, 8, -17, -126, -9, 4, -9, 9, -6, -18, -98, 4, -5, -3, -10, -10, -20, -2, -4, -14, -30, 9, 3, 30, 9, 10, -7, 11, -5, -109, 50, -38, 2, -21, 1, -6, -3, 1, -25, -4, -21, -7, -13, -1, -29, 10, 29, 7, 23, -7, 10, 65, -6, -5, -3, 4, 69, -4, -68, -84, -51, 9, 6, 2, 4, -44, -10, 110, 4, -27, -28, -16, 70, 61, 11, -15, -10, -38, 8, 9, -13, 3, -17, -83, 38, 91, -19, 28, 0, -53, 0, -37, 22, 23, 2, 18, 8, -15, -12, -11, -28, 121, 103, 4, 5, 9, -33, 63, -51, -30, 26, 40, -11, -93, -118, 25, -57, -7, -49, 72, -43, -56, 20, 35, -24, -23, -12, -12, -97, -18, 7, -23, 24, 9, 3, -92, -115, 72, 17, -37, 17, 45, 9, 32, 22, 16, -94, -9, 54, 1, 19, 7, -6, 37, -28, 30, -31, 68, -113, -8, -38, 49, -86, -8, 46, -26, 27, -55, -114, 26, 3, 119, -21, -21, -52, -69, 23, 66, -23, 127, 114, -27, 12, -97, -53, -15, 46, 5, 122, -50, -22, 6, 25, 67, 29};

float bias_raw[24]={1.2200630903244019, -0.8905134797096252, -1.753718376159668, -1.2123632431030273, -1.2555344104766846, 1.6292816400527954, -8.312779426574707, -0.6835771799087524, 0.6196351051330566, 3.9515304565429688, 3.5523531436920166, 0.1180729866027832, -2.3841421604156494, 0.25951457023620605, 0.622147262096405, -3.4385826587677, -0.42334163188934326, 4.370458602905273, -1.3193798065185547, 0.11378109455108643, 3.1534852981567383, 6.672092437744141, -0.3466372489929199, -1.5588912963867188};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const int stride_width=1;
const int stride_height=1;
const TfLiteFusedActivation activation=kTfLiteActNone;
const int dilation_width_factor=1;
const int dilation_height_factor=1;
const int filter_dims_size=4;
const int32_t filter_dims_raw[4]={24,1,1,144};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={24};
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

TfLiteRegistration* Register_sgpyea_REF() {
  static TfLiteRegistration r = {sgpyea::Init, sgpyea::Free,
                                 sgpyea::Prepare<sgpyea::kReference>,
                                 sgpyea::Eval<sgpyea::kReference>};
  return &r;
}

TfLiteRegistration* Register_sgpyea_GENERIC_OPT() {
  static TfLiteRegistration r = {sgpyea::Init, sgpyea::Free,
                                 sgpyea::Prepare<sgpyea::kGenericOptimized>,
                                 sgpyea::Eval<sgpyea::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_sgpyea_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {sgpyea::Init, sgpyea::Free,
                                 sgpyea::Prepare<sgpyea::kMultithreadOptimized>,
                                 sgpyea::Eval<sgpyea::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_sgpyea_CBLAS_OPT() {
//   static TfLiteRegistration r = {sgpyea::Init, sgpyea::Free,
//                                  sgpyea::Prepare<sgpyea::kCblasOptimized>,
//                                  sgpyea::Eval<sgpyea::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_sgpyea() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_sgpyea_MULTITHREADED_OPT();
#else
  return Register_sgpyea_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
