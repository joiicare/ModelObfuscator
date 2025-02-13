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
namespace ethpnd {

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

int8_t filter_r   aw[3456]={-3, -4, -2, -35, 45, 29, -8, -12, 127, 15, -32, -37, -57, 39, -14, 39, -1, -28, 45, -7, 57, -20, 4, -51, 88, 28, -65, 76, 8, -55, -26, 0, 18, 7, -6, 49, 38, 16, 89, -30, -68, 48, -9, 127, -112, -6, 57, 23, -28, -18, 15, -82, -127, 75, -3, 13, -110, -11, -25, -6, 48, 15, -4, -59, -1, -23, -3, 10, 7, 5, 3, 81, -10, 92, -6, 17, -127, -9, 26, -102, -3, 47, 14, -1, -2, -46, 31, 22, -17, 26, -6, -45, 22, -15, 14, -88, 52, -127, 49, 18, -29, -17, 56, -65, 40, 0, 104, 59, 5, 65, 23, -80, 43, 37, -2, -5, 59, -9, -2, -8, -29, -25, -7, 127, 8, -127, -8, 1, -39, -48, 43, 12, 16, -48, -12, 17, -2, -5, -23, -37, -12, 2, -1, 11, 13, -13, -16, 4, -10, 14, -27, 11, -7, -22, -2, 12, -4, 15, -14, 7, -38, -127, -1, 11, 20, 0, 13, -14, -18, -9, 127, 3, -2, 3, 15, 94, -2, -91, -33, 14, -6, 3, 77, -54, -39, 39, 92, -14, -59, -51, 9, -85, -3, -3, -4, 0, -3, 14, 24, -24, -4, -40, -4, 12, 44, 12, -17, 38, -34, 127, -2, -7, -7, -66, 12, -1, 17, -70, -23, -6, -24, 8, -3, -21, -17, 19, 112, -56, -59, 75, 66, -6, 61, 90, -55, -40, -29, -127, 56, 2, -21, -39, 127, 0, -29, 5, 1, 84, 28, -32, -53, 41, -23, -19, 92, 27, -26, 0, 41, -45, -90, -25, -40, -37, -30, -114, -43, -52, 127, 37, -3, 16, 7, -48, -7, -51, -13, 5, 0, 35, 67, -26, 62, -3, 0, 40, 20, 123, -58, 41, -25, -24, 97, -19, -50, -127, -70, 70, -14, -4, -62, 62, 3, -43, 25, 29, -78, -47, 55, -8, -5, 34, -42, 36, 55, 10, 24, -6, -8, 89, -14, -93, -68, 107, -95, 101, 24, -93, 40, -64, 21, -41, -94, -127, 49, -36, 45, -27, 51, 18, -88, 25, 64, 127, 64, -58, -5, 38, 36, -36, -12, 60, -51, 23, 57, 27, -83, -7, -24, -39, -111, -127, 36, 25, 58, -29, 37, 47, -74, 110, -31, 27, 39, -10, 33, 21, 4, -19, 34, 105, -45, -54, 25, -67, -61, 65, 48, -1, -32, 2, -39, -23, 24, -88, 21, 87, 12, 20, -8, 117, 127, -37, -35, 126, 36, 21, 47, 10, -3, -13, -49, -11, 9, 3, -90, 42, 15, 1, 13, -28, 21, 3, 19, -45, -10, 127, -56, -19, 31, 7, 24, -29, -49, 19, 84, -2, 28, 0, 40, -47, -42, 10, -23, -34, 73, 17, -21, 43, 52, -127, 20, -60, 90, -34, -27, 9, 70, 57, 6, -11, 21, 9, 7, -25, 40, -115, 35, -12, -36, -5, 107, 23, -2, -29, 11, -81, 1, 48, -2, 127, 74, -46, 32, 5, -4, -6, 3, -15, -65, -12, -127, 34, -68, -53, -11, 35, 9, -3, 33, -8, -4, 17, 23, -20, 64, 2, -72, 15, -18, 5, 21, -57, -13, -27, -27, -4, -21, -19, -127, 11, -20, 1, -17, 60, -3, -18, -4, 27, -127, -15, 24, -50, 27, 15, 72, -81, -62, 25, 69, -27, -82, -52, 92, 85, 61, 20, 2, -38, 10, -10, 40, 26, 85, 76, 20, 72, 58, -23, 39, 3, 29, 1, -36, 37, 98, 5, -89, -22, -11, 105, -66, 22, -127, 8, -11, 3, 88, -9, -96, 40, -104, 0, 23, 127, 122, 1, 80, 19, 80, -24, -94, 0, -9, 20, 89, 8, -73, -61, -5, 28, 55, 4, -11, -29, 69, 29, -3, 10, 30, -102, 13, -17, -18, 5, 39, 17, 5, -8, 25, -36, 11, 26, 1, 127, 51, -32, -27, -19, -11, -6, -16, -3, -75, 5, -3, -33, 42, -98, 18, -10, 83, 57, 127, -19, 82, -19, 23, -56, 27, 6, -14, -114, 40, 108, 7, -34, 43, -127, 0, -41, -40, 48, 54, 26, -14, -24, 37, -60, 5, 35, -9, 118, -67, -76, -31, 1, -10, 6, 74, -127, 28, -63, 45, 40, 49, 14, 72, -31, -17, 68, 76, 25, 19, 8, 53, -12, 58, -16, -61, 4, -14, -1, -26, -127, -31, 24, -4, -59, -37, -77, -81, 56, 53, -25, -9, 70, 10, -9, 4, 66, -23, -36, 5, -29, 14, -3, -13, 77, -35, 43, -29, 48, 83, 2, 85, -7, 50, -44, 89, -127, 108, -10, -2, -41, -39, -83, -79, -15, -35, 14, -47, -22, 26, -2, 4, -49, 3, -16, 127, 33, 14, 33, -23, -22, -61, 23, 29, -28, -127, 12, 68, -44, 71, -38, -47, -125, -42, 0, -28, -73, -54, 57, 55, -34, 1, -118, -110, -16, 80, -14, 7, 20, -62, -118, -13, -127, 49, 126, -31, 27, 1, 17, -16, -35, 2, 45, -38, 14, 25, -9, 56, 23, 19, -23, 5, -4, 41, 14, -52, 16, -21, -1, 11, -70, 0, -49, -22, -27, -40, -4, -127, 2, -27, -7, -43, 78, -13, -19, -12, 43, 94, -77, -19, 7, -50, 11, -44, -127, 5, 42, 40, -81, 39, -65, -73, 110, -3, 4, -62, 68, 33, 21, 25, 49, -12, 127, 68, -19, -18, 7, -102, 72, -3, 86, 78, 47, -122, -102, -94, 78, 0, -66, 39, 35, -47, 19, 26, 86, 1, 21, 26, 7, 45, -21, 16, -10, -36, 36, -17, -9, 12, 8, 26, -28, -10, -21, 10, 2, 37, 54, 127, -1, -26, -15, 85, 8, -27, 13, 9, 40, 90, -27, -38, 83, -27, 61, -17, 23, -123, -8, -87, -10, -127, -24, -29, 44, -44, 8, -9, -2, 18, 5, 10, -31, 41, 70, 127, -27, 3, -55, -2, -99, -60, -48, 108, -39, -14, -61, -15, 58, 27, -27, -25, 13, -39, 15, -2, -7, -22, 47, 72, 26, -110, -13, 14, 56, -20, 127, -6, 25, -55, 65, 47, 20, -17, 7, -76, 5, -26, 7, 48, 39, 46, 14, 67, 2, 90, -38, 5, -17, -34, -32, 127, 2, 13, 16, -24, -2, -47, 41, -21, 50, -124, -39, 31, -66, -16, 82, -6, -15, -32, -22, 21, -8, 7, -20, -23, 29, -18, -21, 5, -127, -7, 9, 1, -45, 9, -9, 6, 40, 57, 6, -23, 19, -14, -127, 8, -50, 11, -33, -85, 11, 39, -4, 11, 16, 58, 16, -46, 42, -23, -30, -1, -15, 18, 12, -29, 34, -14, 3, 127, 12, -33, -5, -7, 108, -22, -69, -58, 45, 21, -8, -127, 6, 14, -7, 40, -31, -42, 56, -32, 25, -36, 5, -29, -4, 3, -36, 14, -35, 36, 16, -7, 33, 44, -116, 98, 0, -55, 19, 2, 102, 41, 52, 28, -13, 79, -19, 127, 109, -76, -3, -4, -83, -40, 53, -18, -43, -20, -14, -8, 127, 21, -124, -6, 8, -40, -33, 43, 9, 16, -48, -11, 4, 13, -4, -29, -36, -31, 26, -9, -4, 34, -32, -48, 90, 51, -127, -8, 27, -55, -44, 19, 70, 56, -61, 26, -16, -26, 38, -24, 88, -117, -5, -16, 67, -17, 41, 18, -17, -3, -3, -32, 33, -64, 46, 66, 20, -127, -33, 2, 66, -27, 82, 30, -3, 29, 5, 2, 26, 26, -16, -51, -10, -16, 23, 20, -12, 12, 22, 53, 34, -97, 1, -28, 66, -50, 127, -11, 35, -41, 66, 39, 4, -86, 61, 126, -7, 78, -16, 127, -65, -59, 60, -42, -54, 118, -5, 11, 52, 87, -45, 28, -89, 124, -52, 27, 40, -62, 8, 2, 12, -65, -8, 29, -41, -31, 110, -15, 13, -21, 8, 1, -21, -9, -49, -29, 54, -36, 7, -9, -127, 2, -2, 1, -127, -66, -100, 1, 0, -83, -6, 3, 6, 41, 39, -2, -10, -11, -8, 15, 11, -42, 11, -1, 36, 21, -20, -26, -2, -35, 8, 79, 40, -9, 23, -27, 6, 33, -100, 2, 68, 39, -45, 127, -49, -6, -29, -110, 0, 35, -61, -12, 8, 2, -9, -16, 23, -52, 30, -22, -6, 2, -82, 48, -2, 56, 33, 127, -73, 53, -13, 21, -48, 71, -9, -127, 9, -14, -10, 16, -111, -54, -17, -39, -45, -100, -48, -103, 36, 39, -35, -10, 77, -6, -13, -57, 50, -36, 65, -37, 7, -115, -8, 38, -90, -27, 69, -13, -16, -34, -19, 27, -22, 32, -24, -16, 10, -10, -15, 1, -127, 52, 127, 13, 2, -70, 1, 25, -62, 16, -62, 24, -3, -41, -31, 77, -10, 4, -11, 31, -89, 7, -5, 21, 2, -8, -84, -34, -23, 127, 25, 4, 28, -11, -42, -15, -27, 0, 2, 6, 10, 42, -30, 56, 24, -4, -2, -8, 105, -102, -127, 0, 12, 46, -7, 1, 51, -37, 90, 2, 10, 15, -5, -3, 25, 19, -55, 7, 68, -26, -6, -7, -37, -47, 0, 127, -35, 44, -6, -57, -41, -18, -22, 28, -66, -13, 8, 66, -25, -15, 30, -53, -2, 19, -5, -4, -17, 58, 26, 9, -27, 65, 27, 9, -13, 9, -103, 27, -9, -20, 9, 58, 17, -2, -33, 17, -54, -2, 50, -10, 127, -127, -96, 19, 16, -8, -14, 21, 8, -43, 125, -14, 10, -9, 3, 13, 18, 5, -30, -14, 53, -32, -3, -10, -87, -57, 34, 26, 31, 9, 11, 6, -7, 86, -2, -5, 34, -57, 62, -56, 17, -58, -30, -127, 48, -119, 19, -43, 55, 31, 46, 19, -52, 75, 29, 32, -47, 18, -125, 32, -29, -13, -9, 85, 25, 29, -21, 24, -84, -2, 40, 5, 127, 53, 41, 36, 20, -23, -8, -65, 20, 26, -32, 5, 53, 20, 34, -127, -24, -76, -41, 1, 31, 8, -25, -27, 5, -21, -29, -16, -7, -15, -9, -21, -5, -27, -3, 10, -13, -29, -4, 40, 14, -7, -127, 20, -9, 37, -1, 12, -7, -24, 35, -127, 0, 31, -13, 45, -32, -65, -79, 3, -58, -99, 18, -27, -94, 55, 10, 33, 50, -3, -40, 3, -28, -56, -42, 89, -3, -30, 10, -69, -5, 61, 12, -43, 127, 2, 6, 79, 49, -6, -13, 20, -58, -81, -19, 2, 27, 49, 50, -37, 122, -43, -127, -1, -28, -21, -40, 46, 33, 17, -73, 70, 6, -5, 9, -34, -14, -30, -23, 7, -2, -7, -7, -100, 4, -19, 11, 45, 34, 37, -1, 50, -5, 112, -19, 9, -40, -80, 26, 127, -11, 19, 5, -5, -22, 48, 34, 15, -28, 58, 19, 13, -2, 17, -93, 30, -18, -15, 4, 51, 40, 12, -2, 14, -48, 17, 19, 14, 127, 88, -21, -112, 41, -127, 18, 24, 126, 125, 0, 71, 26, 106, -48, -101, -5, 26, 15, 117, 0, -80, -82, 0, 26, 3, 14, 65, -4, -14, 2, 11, -17, 4, -43, 86, -127, 14, -69, -3, 10, -7, -32, 17, 90, -9, 6, 7, -19, -56, -38, 6, -8, -26, -81, -6, 34, 65, -31, -21, -7, -52, -127, -6, -4, 0, -1, -63, -32, 107, -48, 19, 21, 44, -16, 81, -3, -18, 5, -9, -8, -5, -50, 47, -127, 22, -64, -7, 25, 0, 38, 0, 89, 2, 9, -9, -41, 77, -6, -99, 24, -85, 17, 35, 127, 118, 7, 78, 12, 80, -64, -62, 16, 19, 10, 92, 13, -110, -32, 11, 7, 68, 63, -5, 55, 20, -37, 16, 2, -30, 58, -29, 26, 8, 25, 127, 1, -68, 2, 26, 106, 49, -31, 111, 54, 114, -23, -71, 7, -17, 9, 73, 127, -38, 33, 28, -25, 39, 20, -25, 14, -20, -15, 26, -58, 6, -20, 2, -49, 32, 15, -12, -87, 80, -127, -8, -30, 72, 9, 15, 3, -27, 43, -5, 63, -11, -22, 43, -2, -15, -6, -2, -78, 11, -64, -110, -4, -93, 27, 10, 34, 35, -60, 50, 58, -127, -5, 22, 55, -58, -28, 8, 52, -80, 28, -106, -15, -21, 22, 127, 0, 11, -2, 42, 107, -21, -91, -17, -13, -45, -23, 49, -69, -23, -40, 109, 14, -60, -46, 8, -79, 46, -43, -29, 12, -42, -4, 15, 110, 36, -16, 33, 8, 21, -11, 44, -36, -69, 90, 84, -18, -26, 52, 127, -67, 1, -1, -14, 0, -20, 0, -8, -10, -4, 1, -10, 41, 43, -66, -1, 12, 24, 17, 127, 5, 28, 12, 22, -6, -18, 21, 19, 1, -6, 12, -39, -22, 39, 0, 57, -34, 5, 73, -12, 32, 32, -71, -127, 22, -73, -79, 26, 43, -36, 13, 45, 6, 37, -6, -26, -42, -11, -10, 5, -26, -47, 84, 3, -40, -67, 46, -127, 16, -2, 18, 26, 18, -99, 90, -33, -8, 10, 13, 42, 37, 57, 67, 127, 13, 79, 50, 16, 35, 33, -112, -34, -12, -22, -22, 41, 60, 74, 28, -20, -4, 16, 23, 23, 14, 21, -14, 11, 60, 41, 1, -68, 2, -47, 77, -12, 38, -127, 30, 28, -18, 14, -28, 94, 6, -44, 20, 0, 61, 62, -44, -51, 108, -17, 19, -5, 29, -116, -4, -22, -28, -127, -53, -30, -4, -4, 4, -104, 7, 23, -25, 28, -127, -72, 82, -19, -57, -32, -56, -38, 37, 81, 10, 19, 3, 49, 11, 20, 50, -31, -12, 127, -1, -16, 9, 3, 73, 28, -37, 14, 15, 22, -3, 74, 13, -34, 12, 31, -23, -52, -17, 7, -27, 42, 112, 4, -20, -127, 11, 37, -93, 10, -38, 31, -29, -21, -44, 121, 11, 42, -6, 14, -121, 17, -4, 28, -55, 89, -127, 58, 11, -31, -15, 118, -52, 37, 41, 99, 87, -34, 21, 13, -71, 32, -87, 23, 13, 10, 26, 19, 9, -96, -14, 23, 20, -41, -23, 37, -5, -56, 127, -12, 22, -10, 7, -11, 18, 8, -8, -38, 72, 3, -45, 16, -121, 27, -47, -103, -26, -74, 41, 12, 32, 30, -83, 54, 72, -65, -10, -58, 42, -126, 20, -16, 59, -117, 25, 127, -30, 41, -18, -63, -2, -51, 3, -24, 54, 24, 9, 36, 17, 63, -54, -4, -8, 47, 13, 127, -14, 9, -35, -34, -22, 50, 61, -1, -78, 81, 26, 5, 7, -28, 52, 22, 43, 18, -54, -4, -51, 33, 56, -49, 19, -127, 92, -6, -34, 19, -20, -5, 9, 13, 6, 33, -53, 33, -34, -72, -19, 127, 24, 5, -34, 17, -103, -13, 0, -38, -11, 13, -24, -112, 2, 39, -23, 17, -6, 71, -63, -65, 31, 91, 10, -127, -31, 100, 59, 31, -43, -8, -1, 64, 34, 20, 32, 68, 86, 19, -16, -55, 19, 15, -57, 45, -93, 60, -9, -43, -25, 125, 40, -2, -58, 16, -127, -5, 51, 6, 57, 27, 17, -5, 118, 96, 127, -12, -33, 2, 1, 36, 29, -3, -15, 7, 25, -2, -22, 5, -9, -12, 0, -4, -69, -3, -2, 7, 51, -48, 120, 1, 3, -127, -10, 17, 25, 56, -17, 7, -43, 0, -12, -23, 4, -31, 16, -3, 42, 47, -23, -127, 6, -2, -2, -14, -120, -32, 46, -32, -11, -7, -15, -57, 31, 25, 34, -21, 16, 50, 7, 6, 55, 61, -52, 31, 7, -1, -12, -14, -29, -61, -33, -127, 23, -60, -30, -11, 9, -19, 34, 24, -9, 7, 0, -15, -28, 92, 127, 3, 1, -123, 9, 4, -79, 44, -83, 31, -27, -35, -41, 120, 14, 15, -37, 22, -110, -11, 33, 15, 19, -104, 9, 2, 29, -57, -22, 47, -56, -56, 120, -7, 7, -4, -14, 20, 1, 6, -7, -37, 62, -9, -32, 2, -127, -127, -1, 34, 1, 19, -8, 13, -124, 17, -33, 3, 40, -54, 1, 42, -6, -61, 1, 14, 59, 1, 80, -35, 40, 11, 16, 17, -90, -29, 127, 22, 8, 33, 29, -30, -15, -7, 55, -53, -4, -1, 73, 23, 7, -23, 4, 10, -10, -10, 15, 75, -2, 42, -10, 15, -35, -49, 10, -29, -41, -79, 48, 7, 20, -12, -57, -127, -16, -18, -18, 14, 23, -47, -67, -49, -76, -20, 84, -69, -28, 44, -54, 35, -26, -44, 18, -98, -43, -19, 13, 10, -45, -127, 109, 57, -87, 49, 102, -24, 31, -127, -22, 5, -67, 12, 10, 20, -1, -23, -42, 67, -14, 19, -10, -16, -49, -15, 4, 25, -64, 66, -2, -127, 23, -39, -4, 60, 15, 1, -2, -18, 37, 1, -10, -94, 34, 9, 2, 27, 19, -8, -18, -11, 23, 27, -41, 79, 3, -74, 27, 31, 97, 57, -75, -47, 99, -2, 13, -16, 56, -111, 0, 32, -21, -127, -60, -34, -31, -54, -81, -50, -59, -38, 63, -57, -11, 44, -69, 40, -24, -61, 0, -109, -38, -2, -4, 10, -57, -127, 106, 35, -71, -66, -31, 35, -21, 0, 5, 68, -46, -43, 56, 23, -45, -52, -92, 83, 68, 127, -73, 86, -70, -7, -37, -59, 22, -49, 23, 49, 7, 25, -2, 12, -57, 7, -21, 1, -20, -42, 82, -9, 13, -52, -35, -127, 14, -21, 5, -65, 29, 75, 21, -62, 50, -13, -28, -29, 4, 35, 4, -13, 43, -2, 17, 80, -32, -68, 19, -5, 127, -70, -9, 42, 25, -45, -94, 0, -24, 127, 5, 21, 29, -50, 16, -24, 1, 44, -16, 0, 44, 8, 29, 71, 51, 7, -40, 27, 94, 23, 13, -27, 25, -15, -15, 7, -84, 4, -51, -35, -23, -49, 30, -127, 5, -38, -15, -102, 83, -53, 2, -25, 69, -44, 14, 27, -1, -21, 21, -90, -68, 63, -53, 61, -6, 28, 92, -13, 28, 49, -26, -127, 50, -85, -121, 60, 38, -10, -42, -36, -9, -8, 19, 127, 23, -18, 54, -3, -60, -8, -67, 67, 46, 73, -46, 62, -55, -42, -6, -33, 1, -18, 34, 12, 6, 8, 0, 7, -1, -6, 7, -9, -27, 1, 12, 9, 31, 20, -127, -15, 8, -36, 3, 4, 13, -97, -49, 30, -34, 15, 0, -1, -34, -45, -23, 24, -55, -19, -8, -70, 15, 27, -12, 44, -118, 9, 87, 127, -45, -18, -31, -43, 11, -27, 10, 21, -13, 13, -30, 4, -1, -34, 5, 30, 10, -38, 4, 6, 22, -39, 35, 127, -8, -90, 41, 98, -7, 29, 4, -47, -127, 75, -75, -98, 85, 17, -29, -69, -36, -96, 6, -26, 70, -27, 60, -58, 50, -15, -40, -34, -127, 67, 2, -18, -83, -127, -65, 109, 81, 27, -86, 72, 59, -28, -30, -22, -34, 66, -110, 1, -84, 62, 115, -1, -2, -127, 12, 8, -80, 10, -18, 25, 0, -21, -37, 75, -5, 26, -22, 29, -96, 10, 3, 23, -52, -34, 58, -9, 29, -20, 2, 12, -34, 64, -59, 18, 16, -57, 59, -86, 18, -60, -50, -127, 83, -56, 6, -57, 50, -30, -127, -26, -17, 125, -3, 6, 54, -28, -2, -1, -13, 10, -5, 27, 27, 53, -16, 44, 39, -3, -20, 9, 95, -14, -1, -24, 3, 1, 4, 0, 9, -17, 22, 99, 7, -127, 33, 35, 15, -18, 120, -26, 19, -70, 72, 40, 11, -9, 70, -12, 28, -127, -22, 6, -62, -10, 26, 17, -4, -22, -30, 16, -15, 21, -9, -1, -43, 24, -13, 13, -95, -93, -127, 21, -25, 56, 17, 13, 30, -18, 50, -25, -22, 11, -14, -21, 30, 37, -17, 35, 53, -40, -14, 9, -22, 16, -11, -52, -4, -15, 0, 61, 77, 2, 113, 127, -53, -1, -63, 44, -25, 25, -105, 123, -70, -12, -28, 15, 12, 3, 35, 36, 53, 6, -43, 25, -19, -42, 55, -22, -17, -6, 14, 127, 18, 13, -50, 43, 41, 116, -70, -24, 63, -27, 22, -36, 2, 9, -3, 120, 46, -21, -12, -16, 0, 127, -1, 28, 55, -13, -6, 73, -14, 113, 87, -14, -46, 62, -53, 0, -7, 23, -1, -10, -36, -31, 74, -8, -99, -31, -34, -11, -41, -87, -66, -10, -75, -97, -127, 55, 33, 46, 89, -11, 18, -127, 0, 9, -54, 19, -3, 19, -2, -13, -31, 53, -6, 19, -15, 18, -58, 12, -9, 16, -48, -17, -19, 127, 12, -23, 16, -7, -48, 12, -33, -2, 20, -107, -17, 0, 61, 9, 2, -74, 20, -127, -104, -89, 63, -102, -35, -28, 97, -39, -103, 33, -35, -64, 100, 2, 36, 8, -22, 32, 1, -47, -23, -79, 98, -41, -9, -7, -127, 14, -1, 6, 55, -16, 0, 0, -40, -58, 4, 22, -19, 0, 127, -7, 58, -11, -19, 102, -2, -51, 10, -2, -1, 94, 15, -93, 5, -40, 0, 13, 2, 22, -29, 5, -2, 22, -47, -127, 0, 4, 16, 36, 52, -14, -24, -2, 17, 9, 20, 22, 8, -2, 0, 7, 0, 9, 13, -35, -2, -2, -43, -27, 5, 6, -127, 35, 3, -42, -1, -15, 17, 31, 26, 11, 103, -23, -28, 11, 41, 90, -34, -10, 43, 25, 44, -38, 7, -55, 109, -35, 36, -127, 53, 40, -2};

float bias_raw[144]={-0.003744776826351881, -2.178795576095581, -0.015865430235862732, -1.2558845281600952, 1.899176001548767, -0.08944626897573471, -0.026240192353725433, 1.271543025970459, 0.821892261505127, 2.740539789199829, 0.7193674445152283, -0.02447323501110077, 2.5299181938171387, -0.43215999007225037, 2.2495670318603516, -0.04668344557285309, 0.31018662452697754, 6.158798694610596, -2.0388433933258057, -0.0993841290473938, 0.14931903779506683, 1.9518873691558838, 0.7589998245239258, 2.3379383087158203, 2.351820468902588, -0.005989527329802513, 0.834870457649231, 0.442639023065567, 1.691422700881958, -0.4828334152698517, -0.029921256005764008, 0.492510586977005, -0.025309361517429352, 0.5111663937568665, 2.053165912628174, -0.1411597728729248, -0.11957435309886932, -2.5114052295684814, 2.1109235286712646, 0.7784035801887512, 0.08673777431249619, -0.7529356479644775, 0.07278534024953842, 0.47279584407806396, -0.9667394161224365, -0.8629062175750732, -0.21183758974075317, 5.312616348266602, -0.23246745765209198, 1.9220936298370361, 0.03217902034521103, 0.17372484505176544, 0.07411758601665497, -0.02886977419257164, 0.048165589570999146, -0.07841357588768005, 0.5436301231384277, -0.021465227007865906, -1.4602937698364258, 1.7587025165557861, 0.04122534766793251, 1.6352633237838745, -0.020047727972269058, -0.009766757488250732, 3.089247941970825, -0.20433637499809265, -0.9389521479606628, 0.18214598298072815, 1.3416709899902344, -0.0224456787109375, -0.3119644522666931, 0.4780005216598511, -0.44900694489479065, 2.90354585647583, 1.0091460943222046, -3.3890881538391113, 2.9661126136779785, 3.325798988342285, -1.1706441640853882, -0.7456977367401123, 0.21005763113498688, 0.027385741472244263, -0.754352867603302, 0.004525600932538509, 0.5962454080581665, -0.4182508587837219, 3.00999116897583, 0.5296602249145508, -0.04348980635404587, 1.903550386428833, -0.497244656085968, 0.011629310436546803, -0.11936188489198685, 1.1416528224945068, -0.5099949836730957, 0.37318694591522217, 0.30635759234428406, 4.464869499206543, -0.2336445152759552, -0.7622694969177246, -0.03190133720636368, 0.15084029734134674, 0.46702951192855835, 1.2937688827514648, 0.3472011685371399, 0.029456809163093567, -0.06942301243543625, 0.4812924563884735, 0.029672974720597267, 0.3469706177711487, -2.611452579498291, -0.05362378805875778, 1.9991025924682617, 2.0127174854278564, -1.0282502174377441, -0.7864431142807007, 0.7158107161521912, -0.8479968905448914, -0.8424317836761475, 1.677750825881958, 1.887345314025879, 2.2865679264068604, -3.6258082389831543, -1.4643498659133911, -1.01318359375, 3.063880681991577, 1.111679196357727, 0.00415844889357686, 2.3136701583862305, -0.03036254271864891, -0.044392094016075134, 1.375218152999878, 0.0897371917963028, -1.8136179447174072, -1.4621896743774414, -1.1229677200317383, 0.26634156703948975, -0.008337192237377167, 0.01439414918422699, 0.2838136553764343, 0.36373960971832275, 1.860623836517334, -0.04646998643875122, -0.0756862461566925};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const int stride_width=1;
const int stride_height=1;
const TfLiteFusedActivation activation=kTfLiteActNone;
const int dilation_width_factor=1;
const int dilation_height_factor=1;
const int filter_dims_size=4;
const int32_t filter_dims_raw[4]={144,1,1,24};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={144};
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

TfLiteRegistration* Register_ethpnd_REF() {
  static TfLiteRegistration r = {ethpnd::Init, ethpnd::Free,
                                 ethpnd::Prepare<ethpnd::kReference>,
                                 ethpnd::Eval<ethpnd::kReference>};
  return &r;
}

TfLiteRegistration* Register_ethpnd_GENERIC_OPT() {
  static TfLiteRegistration r = {ethpnd::Init, ethpnd::Free,
                                 ethpnd::Prepare<ethpnd::kGenericOptimized>,
                                 ethpnd::Eval<ethpnd::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_ethpnd_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {ethpnd::Init, ethpnd::Free,
                                 ethpnd::Prepare<ethpnd::kMultithreadOptimized>,
                                 ethpnd::Eval<ethpnd::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_ethpnd_CBLAS_OPT() {
//   static TfLiteRegistration r = {ethpnd::Init, ethpnd::Free,
//                                  ethpnd::Prepare<ethpnd::kCblasOptimized>,
//                                  ethpnd::Eval<ethpnd::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_ethpnd() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_ethpnd_MULTITHREADED_OPT();
#else
  return Register_ethpnd_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
