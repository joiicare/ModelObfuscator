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
namespace wunffw {

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

int8_t filter_r   aw[3456]={-41, -127, -3, -12, 92, -10, -10, 73, -17, 34, 1, -53, 15, -25, 20, 1, 78, 25, 20, -11, 40, -11, 24, 25, 0, -25, -55, -12, -11, 5, 6, 47, 31, -15, -13, 2, -33, -10, 11, -13, -18, -127, -45, 10, 12, 59, -38, -37, -8, -5, 8, -127, -43, 114, 21, 8, 67, 29, -61, -49, -33, 80, -3, 3, 8, -25, 44, 7, 64, -10, -41, -2, -56, 96, 19, -6, 23, -3, 5, 16, -11, 33, -83, -49, 49, -107, -1, 54, -5, -127, 62, -26, -32, -16, 67, 34, 45, 29, -3, 21, 67, -93, -14, -22, -118, -5, 87, 87, 56, -78, 38, 9, -22, 68, -45, 14, -127, 9, 23, -10, 0, 8, 61, 8, -11, -8, 24, 100, -11, -86, 11, -7, -31, -23, 19, -68, 1, -119, 127, 10, -23, -28, -17, -77, 33, -12, -62, -16, -23, 3, 12, 70, -3, 22, 95, 27, 38, -14, -8, -29, -20, 20, 127, -5, 96, 61, -36, -29, 35, -46, -89, -5, -15, 3, -15, 44, 112, -127, 50, 11, -94, -63, 50, 9, 0, 44, 52, -29, 13, 81, 45, 110, 25, 127, 105, -16, 68, -10, 49, 38, 10, -55, -26, 28, -20, 12, 18, 25, 80, -59, -41, -88, 21, 26, 71, 105, -20, -32, -13, -15, -127, 81, 19, 24, -116, -7, -6, 6, 41, -19, 15, -52, -14, -101, -17, 23, -54, 38, 18, 79, 25, -59, -9, -20, 127, 17, -12, 37, -20, -34, -14, -15, 12, 18, 13, 8, 32, -18, 41, 26, 6, 2, -7, 116, -36, -127, -27, 37, 80, -47, 60, 77, -64, 64, 22, -25, 4, -39, 88, 29, 23, -30, 31, 17, -43, 26, 4, 5, 98, 68, 7, 35, -20, -29, -14, 41, -32, 79, -3, 3, 56, 1, 66, -12, -9, -76, 72, 11, 82, -11, 127, -7, 10, 5, 7, 126, 47, -127, -66, 14, -48, -29, 49, 31, 46, -47, 9, -18, -2, 35, -42, -17, 34, -21, -22, 7, 7, 6, -9, 127, 49, 91, 2, 7, 48, 3, 3, 3, -36, -26, -1, 7, 2, -4, -10, 3, 14, -1, 8, -29, -21, -54, -15, -46, -19, 109, 9, 29, 66, -61, 5, -19, -12, 13, -127, 34, 1, 5, 19, -57, -31, 4, 28, -22, 30, 44, 127, -1, 23, -14, 63, -42, -55, 52, 11, -34, -52, -32, -6, 119, 127, -120, -69, 32, 9, -7, 77, 74, -14, -82, 98, -26, -84, 36, -99, -16, 127, -70, 17, 30, 47, -49, 17, 81, -82, -117, -52, 56, -125, 20, -67, -9, 46, -20, 27, -4, 27, -11, 59, -45, -23, 6, -121, -98, 16, -16, 20, -11, 54, -127, -93, -50, -93, -89, 102, -4, 6, 20, 14, 5, 26, 25, 87, -11, 19, 32, 9, 2, 18, 6, 1, 21, 30, 127, -8, 6, -84, 18, 42, 19, 16, -8, 32, 7, 27, 0, -29, -61, -4, -34, -8, -13, 4, 127, 13, -38, -35, -49, -122, 38, 23, -19, -50, 1, -2, 16, 10, 6, 32, -23, 70, -73, -23, 95, -64, -1, 13, -127, -1, 8, 2, -59, 108, -24, 32, 40, 31, 69, 19, 37, 14, 3, 19, 7, 10, 6, 7, 14, -7, 9, 17, 7, -31, -14, 18, 127, -9, 11, -53, 20, -33, 4, -15, -4, -20, 17, -7, 7, -17, 11, -31, 71, 33, -30, -127, 3, 65, 14, 17, 106, 27, 2, -39, 75, 44, 11, 30, -32, 28, 12, -18, -12, 35, -11, -6, -66, -78, 66, 70, 45, 24, 55, 127, 7, -21, 38, 49, 32, 21, -36, 28, 13, 58, 60, 2, -21, -88, -41, -6, -16, -47, -38, 127, 77, -37, -37, 50, 1, 53, -18, 25, -80, -48, 17, -104, -37, -5, 0, 3, -34, 49, -69, -19, -11, 29, 3, -108, -67, 127, 79, 5, -100, -5, 14, 34, -11, -46, 29, -75, -127, 38, -71, 49, 50, 69, 51, -14, 75, -20, -39, -17, -27, 47, 10, 31, -1, 84, 14, -24, -13, 4, -17, -8, 36, -48, 1, -4, -1, 8, -68, 25, -86, -89, 66, -45, 72, -127, -17, -61, -76, -123, 71, -36, 20, -66, 13, -4, -8, -5, 113, 34, 127, 0, -6, 8, 1, 15, 12, -22, -17, 21, 13, -3, -12, -1, 3, 1, -2, 16, -18, 73, -1, -117, 3, -1, 2, -16, -127, -34, 43, 5, -63, -18, -34, -97, 36, 16, -8, -48, 54, 39, 30, -31, 88, -29, -32, -4, -66, -123, -127, 4, 26, -14, 1, -36, -44, -12, 40, -2, -26, 5, -6, 15, 8, -20, 26, 5, 78, 36, 57, 17, 38, -76, -38, 44, -106, -44, 127, -22, 3, -8, -65, -71, 48, 54, -87, -20, -14, 30, 22, 17, -32, -17, -20, -74, 26, 4, 6, 112, 7, -31, 9, -17, -70, -35, 21, 127, 52, 17, -116, -7, 10, -47, 59, 26, -32, 32, 26, -1, 126, 127, 96, 15, -51, 67, 17, 20, 16, -19, 6, 10, 59, -4, -29, 31, -2, -13, -9, 13, -84, -38, 90, -7, 31, -103, -39, 76, -126, -56, 115, -37, -25, -31, -67, 6, 2, 45, -16, -17, 1, 20, -1, 36, -127, -68, 15, 94, 19, 54, -34, 22, -76, -77, 2, -21, -28, -42, 56, 20, -3, -35, -127, -126, -30, -2, 40, 127, 30, -99, -97, 65, 8, 75, -17, 6, 36, -53, 127, -57, 17, 56, 24, -45, -4, 15, 39, 2, 100, 15, -27, 11, -44, 2, -3, 23, -96, -127, -72, 1, -3, -89, -23, -2, 23, 46, 33, -33, 2, -28, 41, 23, 22, -24, -48, -1, 74, -18, -5, 10, -56, -85, 10, -5, 14, -127, -28, 7, 21, 61, -17, 0, -50, -1, 35, -31, -1, 2, -8, -20, 65, -11, -23, -40, -39, -38, 16, 124, 52, 76, -48, -15, 55, -33, -127, -42, 19, -89, 48, -98, 20, -10, -36, 99, 23, -25, -92, -43, 5, 33, -11, 41, -119, 33, -30, -50, -87, 77, 14, 62, -127, -60, -25, -10, -31, -22, -15, 31, -26, -16, -2, 20, -21, -16, 15, 127, -7, -7, 15, 19, -30, 8, -70, -40, 47, 75, 26, 45, -108, -37, -12, -113, -14, -12, 0, 22, -101, -20, 127, 11, -7, 57, 17, -44, -31, -16, 50, -7, 1, 5, -5, 22, -7, 96, -38, -26, -13, 82, 17, -28, 1, 43, 4, 37, -10, 13, -76, 19, 1, -6, -29, 98, 36, 31, 5, 15, -5, -67, -30, -127, 109, -51, 118, 90, 10, 26, -9, 25, 11, -8, 89, -60, -46, -12, -127, 0, 61, 58, -88, 59, -20, -4, -20, 43, 78, -65, 22, 63, 2, 21, -8, -1, -16, -2, 55, -67, 127, -61, 37, 42, 7, 55, -105, 2, -77, -64, -52, 49, 52, 46, -22, -34, -16, -38, 13, 27, 17, -23, 17, 5, 25, 77, -52, 10, 18, 32, -51, 127, 11, 37, 11, 73, -15, 25, 24, -6, -127, 110, 47, -35, -10, -78, -11, 58, 81, 63, -119, 26, -51, 0, 14, -105, -11, 27, -1, -33, -70, 32, 27, 25, 45, 13, 20, -15, -46, -68, 11, 18, -1, 45, 127, -25, 40, -1, -12, 100, 6, -58, 3, -29, -3, -39, 0, -19, 67, -30, -41, 46, -24, -7, 58, -27, 15, -10, -22, 20, 18, -5, 3, -24, 86, -34, -52, -127, -28, 36, -33, 17, -77, 75, 43, 21, -21, -26, 1, 6, -35, -42, 13, -43, 46, 41, -72, 8, -45, 17, 113, 127, 86, -94, -78, -1, 126, -127, -25, 25, 13, 79, 0, -76, -120, -104, 80, 28, 28, 10, -57, 51, -17, 103, -4, 29, 92, 26, 0, -26, 0, -26, 0, 39, 26, 0, 0, 52, 0, 46, 0, 0, -7, 0, 0, 13, 0, 0, 0, 0, 0, 5, 7, 2, -93, -17, -127, 0, 11, 37, -1, -16, -4, 8, 19, -19, -6, -1, 30, 1, 8, -10, -2, -14, 8, 58, 24, -35, 30, 54, -75, 37, -13, -85, 0, 67, 70, 24, -77, 41, 11, 2, -22, -46, 19, -127, 35, 17, -25, 13, 4, -10, 61, 62, -29, 14, -7, 127, 6, -13, -14, -55, 27, -17, 53, -16, -11, 32, 2, 21, -14, 20, -48, 3, 122, 61, 38, -122, -24, 25, -68, -17, 76, -11, 12, 30, -10, -27, -18, 20, 52, -35, -49, 12, -29, -30, -127, 70, 17, -31, -5, 53, -18, -13, -127, -60, 95, -48, -72, -35, -83, -71, 1, 78, -69, -3, 35, 46, 16, -1, 86, -34, -5, 35, -11, 14, -3, -52, -21, -18, -1, 15, -5, -10, 21, -31, -32, -1, -23, -5, -25, 37, 50, 127, -26, 28, 37, -23, -32, 22, 6, 108, -3, 42, 65, -10, -23, -11, 33, 43, -3, 24, -127, 18, 20, -107, 124, 56, -21, -58, -127, -18, -32, 54, 38, 83, 39, 9, -1, 1, -25, -24, -56, -23, 50, 63, 65, -15, -39, -71, -23, -108, 38, 28, -18, 0, 9, 20, -5, 7, -76, 18, 127, -73, 35, 83, -33, -2, 67, -10, -19, -14, -67, 15, 47, 57, 91, -38, -12, 15, -127, -19, 33, -64, 16, 81, 9, -42, -38, -44, -32, -1, -54, 40, 44, -23, -22, 103, -32, -37, 0, -47, -70, -65, -6, -9, -6, 15, -10, -48, 19, 57, -15, -127, -67, 121, 47, -10, -98, 9, -3, 8, -1, 53, -3, -8, 21, -11, -2, -42, -2, 20, 40, 26, -11, 127, 105, 22, 26, 9, 35, 93, 13, 73, 56, 76, 31, 25, 6, -10, 1, 16, 104, -48, -18, -2, 23, 63, 16, -34, -45, -63, 53, -34, 3, 9, 127, 28, 9, -19, 20, 18, 45, 65, 76, -38, 42, -127, -7, 8, -31, 34, 7, 13, -17, -35, -34, 110, -7, 48, -46, 19, -39, 19, -27, 20, -37, 47, 54, 8, -127, 65, 83, -31, -9, -67, 34, 39, 50, 33, -15, -10, -45, -4, 26, -23, 3, -29, 36, -45, -58, -2, -38, 18, -25, -11, 22, 127, -7, -6, 12, 38, -32, -19, -52, 1, 23, 48, -8, 35, -72, -42, 6, -63, -19, 10, -77, -25, -25, 127, -6, -9, 11, -47, -27, -58, -40, -26, -4, 52, -3, 36, -100, 55, 12, -18, 70, 99, 84, 26, 0, 26, 13, 0, 0, 0, 0, -7, 0, 0, -52, 0, 26, 0, -26, 0, -26, -105, 46, -26, 0, -39, 52, -70, -65, -7, 29, -67, 9, 31, -3, 21, -17, -34, -57, -52, 11, 55, 34, 9, -127, 47, -29, 101, -28, 8, 27, -65, -31, 12, -28, -10, 8, -42, -14, -21, 61, -3, -30, 6, 11, -53, -53, 13, 59, -10, -24, 19, 40, 127, -91, -109, -52, 53, -44, 51, 42, 34, -51, -45, -11, 42, -39, -21, -34, -36, 13, 56, -35, 16, -127, 5, 86, 99, -47, -64, 42, 108, -17, 39, -21, 11, -80, -46, 17, 27, -60, -12, 8, -33, 9, 57, -75, -8, -33, 37, 11, 127, 21, -27, -16, 18, 89, -109, 127, 1, 21, -116, -9, -12, -6, 47, 1, 2, -62, 6, 17, -15, 4, -7, 13, 2, 93, 31, 14, -28, 39, 127, 9, 46, -56, 125, 38, 15, -33, -75, 65, 26, 121, 20, -48, 91, -3, -72, 14, 36, -102, -79, -127, 16, 5, 47, -25, -9, -46, -66, 85, -46, -53, -19, -51, 43, 14, 24, -78, 47, 108, -40, -2, 42, -32, 56, 41, -127, -3, 23, -10, 123, 13, -48, -42, 16, -108, 13, -41, -96, -42, 9, -22, 10, 60, 44, -28, 40, -25, 69, 29, -15, -51, 83, 47, -2, -18, 2, -127, 13, -49, -2, 13, 98, 41, -10, -84, -1, -77, 34, 42, 19, 104, 97, -3, -88, 26, 48, -20, 43, -97, -127, 105, -95, 17, -29, -11, -73, 39, 34, -14, 33, -19, 63, 63, 71, 36, -16, 0, -4, 26, -22, 6, -46, -7, -68, 43, 48, -39, -108, 87, 35, 39, 3, 127, 67, -1, 24, 33, -16, 10, -127, -94, 74, 16, 61, -16, 30, 50, -49, 119, -43, 1, 23, 14, 4, 0, 25, -13, -5, 51, -27, -3, 60, -64, 62, 34, -15, -31, 52, 27, 9, -2, 41, -107, 6, -3, -10, -9, 11, -4, -47, -26, 53, -40, 1, 33, 17, 127, 16, 7, -13, -56, 71, -126, 10, -11, 127, 3, -4, -12, -51, 22, -17, 65, -10, 3, 29, -6, 15, -22, 5, -64, -101, -127, -4, -44, 60, 57, 69, -6, -9, -17, -13, -21, 7, -49, -48, 70, 77, 82, 12, -16, -92, -30, -91, 29, -19, -12, 2, -74, 8, -127, -35, 11, 10, -4, 3, -16, -2, -20, 10, -22, 25, 33, -21, -19, 33, 7, -21, 1, -22, -66, 25, 3, -53, -7, 81, -84, 61, -86, -9, 107, 42, 0, 49, 30, 127, 21, -1, 51, 36, 37, 59, -14, -85, 127, 1, 0, 39, 6, 44, -1, 55, 59, 34, -62, 125, -41, 6, 18, 35, -109, 21, -5, -44, -73, 54, 82, -5, -20, -21, -70, 27, 17, -6, -23, -56, -32, 30, 33, 28, -39, 36, 23, -28, -127, -21, -9, 54, -39, -17, -41, -117, 39, 67, 10, 25, -2, 11, 22, 4, -46, -99, 120, -62, 127, 30, -62, -51, -80, -4, -77, -21, -63, 13, -4, -64, 50, 109, 21, 12, -21, 61, 1, -8, 127, -65, 65, 9, -13, -25, 19, 52, -46, -56, 63, -46, -46, 15, -23, 19, -20, -93, -1, -47, 3, -26, 48, -9, 4, 59, 15, 46, -41, 12, 33, 90, 4, 127, -30, 65, -52, 42, -26, -1, -7, 127, -8, 25, -1, -19, 57, -17, 15, -21, 7, -17, -30, 70, -4, 12, -58, 45, -44, -24, 1, -6, -23, -46, 44, -39, -1, 32, -15, 20, -127, -34, -21, -29, 14, -88, 17, -69, 5, 23, -27, -65, 61, 4, -15, 10, 79, 7, 5, -60, -3, -20, -1, 31, 45, 6, 37, 41, -8, 59, -60, 11, -11, 30, -15, 127, -20, 19, 32, 35, -13, -5, -7, 4, -127, -64, -127, -8, 12, -77, -12, 7, 5, 45, 7, -5, -24, -6, 30, -15, 2, -6, -6, -18, 43, -32, -6, 0, 24, -24, 13, -28, 43, -6, 27, 36, -1, -127, -28, 35, 3, 0, 122, 10, 0, -14, 31, 14, 24, 35, 20, -8, 31, 127, -39, -4, -22, 106, 2, 22, 12, -35, -4, -18, 62, -18, 34, 12, -12, 38, -39, -14, -88, 1, 54, -12, 38, -114, -35, 54, -25, -32, 95, 12, -2, 1, -13, -3, 12, 18, 2, -40, -18, -13, -36, -19, -127, -30, -20, 2, 19, 35, -10, 60, -51, -51, 43, -1, -55, -10, 16, 109, 30, 55, -127, 51, -20, -12, 67, 65, -14, -1, -6, 4, 68, -35, 127, 2, 9, -84, -12, 12, 17, 35, -24, -4, -33, -6, 15, -24, 3, 2, -10, -3, 32, 23, 6, 127, 10, 26, -6, 44, 71, -45, -21, -41, -11, -11, -18, 49, -9, -24, -9, 83, -16, -36, 26, 34, -57, -21, -35, -20, -36, 9, 9, 3, -23, -41, -28, 24, 18, 4, -45, 56, 34, -16, -127, 3, -8, 49, -39, 4, -20, 43, 58, 21, -18, -33, 4, 27, -97, -12, -46, -8, -87, -27, -68, 127, 44, 20, -46, 18, -116, -1, 82, 98, 6, -102, -8, 66, 34, -18, -21, -19, -85, 20, 127, -36, -22, 6, -36, 30, -54, 104, 86, -17, 33, -26, -9, -56, -85, 0, 42, 127, 12, 25, -27, 69, 26, -38, 16, -40, -19, -9, -82, 50, -27, 7, -71, 127, -12, 2, 9, 47, -8, 26, -49, 35, -7, -28, 15, 54, 100, 84, 54, -39, 74, 127, -9, 40, 58, -65, -14, -44, -97, -57, -30, 6, -14, 62, -19, -23, -4, -53, -1, -32, 30, 18, 11, -8, 25, 52, -80, -4, 2, 91, -12, 127, -12, -6, -35, 33, -18, -49, -117, -9, -68, 109, 42, -51, 52, -8, 16, -61, -6, 43, 58, 40, -7, -10, -62, 50, 123, 127, -73, -79, 66, -65, 39, -59, 2, 21, -4, 63, -21, 38, -51, -126, -28, 127, -30, 55, -9, 16, -68, 9, 26, -20, 96, 41, -36, -30, -29, 3, -30, -127, -36, -1, 28, -87, -11, -8, -16, 20, 16, 13, -38, 3, -3, 0, 7, -24, 20, 4, 86, -18, -31, -11, 30, -18, -13, 43, -31, -16, -32, -15, -6, 13, 8, 27, 42, -14, -127, 25, -7, 42, -31, 34, 3, -19, -10, 23, -11, 12, -5, -38, -9, -10, -16, 18, -11, -2, 21, -23, -24, -4, -33, 4, -33, 24, 47, 127, -12, -61, -10, 16, -19, 1, 22, 127, -23, 5, -22, -24, -17, 27, -83, -46, 61, 59, -36, 37, -70, -49, -9, -124, -4, 73, -70, -127, 16, -57, 0, -74, -105, 7, 94, -39, 44, 22, -54, -62, 51, 25, -74, 47, -33, 42, 1, 1, 52, 26, 27, -12, -2, 127, 21, -29, -29, 19, 2, 41, 40, 12, -58, 26, 18, 0, -2, -33, -8, 14, -18, -12, -88, -57, 49, -68, -10, 15, 0, 98, 17, -41, 14, -65, 39, 3, -97, 68, 77, 40, 20, 127, 7, 24, 74, 56, 3, 3, -2, -10, -34, 36, -127, 1, -8, 84, 10, 0, -17, -50, 25, 12, 47, -2, -5, 31, 1, -8, 0, 5, -32, -11, 69, -21, 27, -127, -18, -8, -68, -3, 28, 13, -5, -37, -27, 22, -12, -2, -14, -19, -43, 41, -13, -5, -102, -29, -102, -36, -33, 94, 12, -33, 13, -40, 4, -41, 10, 54, 10, 15, -3, -2, -127, 32, 90, -4, 3, 31, 42, -78, 60, 69, 0, 82, -18, 45, -79, -52, 14, 34, -65, -40, 84, 7, 7, -23, -110, -127, 43, -16, 74, 32, 47, 68, -3, -24, 19, -11, 7, -44, 61, 31, -28, -28, 47, -33, 1, -77, -29, -53, 127, -51, 35, -64, 18, -19, -31, -127, 61, 111, 4, 91, -18, 16, -125, -20, 7, 2, -38, -41, 112, 35, -18, -58, -82, -125, 17, 27, 61, -73, 57, 36, 31, 13, -123, 30, 127, 12, -34, -20, 18, 2, 36, -3, -2, 7, 16, -15, 28, 20, 20, -3, -49, 13, -18, 59, -42, -123, 13, -35, -9, -23, 53, 9, 32, 44, -16, -2, -69, 7, -16, 63, 127, 79, -23, 36, -39, -3, -25, -38, -20, -9, 15, -69, -6, -65, 13, -1, -13, 87, 76, -31, 5, -18, 72, -113, 5, 74, 20, 85, 82, -127, -9, -56, -36, 19, 9, -48, -13, -30, 32, -42, 100, 1, 13, 11, 11, -83, -10, 15, -11, -7, 34, 8, -13, 83, -127, 17, 2, -16, 60, 12, 0, 127, -11, 93, 1, -17, 3, -56, 32, -32, 89, -46, -17, 44, 23, -75, -2, 92, -5, 49, 30, -21, -127, -15, 55, -4, -9, -85, 0, 11, 71, 45, -43, 71, -11, -54, -63, -50, 64, 16, -65, -5, -3, 0, -25, -127, -29, 26, -44, -71, 60, -37, -58, 108, 9, -122, -115, 31, -26, 23, 28, -83, -24, 88, 1, 21, -84, -19, 11, -4, -42, -76, -15, -18, -57, -71, 0, -15, 2, 91, 2, 17, -8, -22, -127, -54, -9, 83, -14, -34, -64, -15, -19, -30, 62, -72, 12, 127, 50, 90, -14, -39, -2, -61, -19, -20, 34, -68, 5, -22, 36, -10, -16, 97, 58, -26, -24, 4, 120, -32, 127, -13, 16, -30, -6, -10, -7, -6, -23, 5, -34, 9, 7, -18, -5, 27, 7, 10, 36, -17, -54, -12, 37, -72, 9, -4, 61, 60, 20, 21, -30, -80, 26, 53, 17, 31, 127, 109, -18, -102, 44, 21, -5, 72, 33, 16, -37, 117, 24, -18, -37, -14, -89, -7, -32, 23, 40, 66, -2, -16, -127, -14, -56, -15, 36, -15, 122, -74, -17, 44, -18, 29, 8, 127, -92, -88, 31, -7, -25, -3, -40, 93, 127, 88, -123, 81, -50, -1, -49, 32, 44, -20, -53, 60, 2, 127, -27, 25, 75, -44, 115, -21, 11, 15, -53, -36, -17, 10, -90, 105, 57, 11, 29, 85, 82, 8, -39, 66, -18, -9, 27, -46, 15, -7, 31, 109, -39, -127, 89, 10, 17, 22, -11, -96, -8, -25, -25, -39, 27, 5, 12, 5, -127, 17, -57, -52, -23, -76, -19, 43, 23, 52, -15, 16, -5, 1, 11, -18, -19, 29, -31, -47, -28, 101, 116, 40, -6, -50, 7, 1, -58, 36, -96, 20, -21, -19, -24, 55, 1, 14, -14, 20, -127, 21, 32, -56, 48, 24, 3, -1, 64, -39, -5, -32, 58, 35, -2, -9, 2, -33, 20, -49, -22, 1, 127, -5, 16, 4, -5, -74, 26, -52, -126, -13, -68, 118, 61, -37, 41, -31, 55, -80, 21, 98, 1, -1, 39, 70, -115, 104, 127, -34, -17, 22, 23};

float bias_raw[144]={2.077878475189209, -2.036112070083618, 2.0429298877716064, -3.588334083557129, 0.12766925990581512, -0.19702613353729248, -1.5596727132797241, 2.3351712226867676, -1.9998446702957153, -0.3236321210861206, 0.22677451372146606, -2.145900249481201, -0.6594845056533813, -1.8041497468948364, 0.9514527320861816, -0.8583985567092896, -1.2123680114746094, -0.6867433786392212, -0.6176581978797913, -0.0721416249871254, 1.928826093673706, -1.2948973178863525, 3.97312331199646, 2.5209531784057617, 0.409987211227417, 0.7186415791511536, 0.23027637600898743, -1.2931451797485352, -0.27594083547592163, 1.6967713832855225, 2.486889123916626, 0.07082229852676392, -1.6393111944198608, -0.4454052746295929, 0.11787744611501694, -1.0601567029953003, 1.9270333051681519, 0.14376023411750793, 1.6951195001602173, 2.330913782119751, 6.354482650756836, 1.0982626676559448, -0.4574394226074219, -0.41833558678627014, 2.0946433544158936, -0.705554187297821, -0.8743371963500977, 2.568950653076172, 2.1508312225341797, 0.3264169991016388, 1.9175015687942505, -1.4219170808792114, 0.22635942697525024, 0.3581380546092987, 0.28683286905288696, -1.0116568803787231, 1.5709447860717773, 0.6588903069496155, -0.6205974817276001, -0.04247633367776871, 0.5235955119132996, 2.0075623989105225, -0.9979832172393799, -1.336488127708435, 0.22954760491847992, 1.3214001655578613, 2.931629180908203, 1.8636921644210815, -0.21934255957603455, -2.5671849250793457, -2.1045360565185547, 0.06148732826113701, 0.2492719441652298, 1.72442626953125, -0.3829437494277954, -3.9083151817321777, 0.4696941077709198, -1.038119912147522, -1.1960115432739258, -1.3331420421600342, 0.1171911433339119, 0.49635064601898193, 1.233227014541626, -0.31950610876083374, 2.692960500717163, 1.0366765260696411, 1.2684324979782104, 0.12268885225057602, 1.898874044418335, -1.0009610652923584, -0.04530705139040947, -2.352463483810425, -1.3462998867034912, 0.03028322383761406, -0.4483920931816101, -0.1389746367931366, 1.277774691581726, 2.915132761001587, 2.4007129669189453, 2.6231493949890137, 1.7449527978897095, -0.3948637843132019, 1.561423420906067, 0.26770827174186707, -0.19415315985679626, -2.6167805194854736, -0.6770682334899902, -1.0287457704544067, 2.29410982131958, 0.04112013429403305, 3.066256046295166, -0.3484511375427246, 0.09926827996969223, 0.17822802066802979, -1.2926311492919922, -1.7432727813720703, 0.6543087363243103, 0.9500787258148193, -1.4776935577392578, 0.058686524629592896, -0.9141879081726074, 0.037879496812820435, 1.7926411628723145, 3.88832688331604, 0.003838556120172143, -2.3623666763305664, 2.602466344833374, -0.9628481268882751, 0.54879230260849, -1.761441946029663, 1.867981195449829, 2.1632866859436035, 0.4830590784549713, -0.29748809337615967, 2.1321756839752197, 2.9047043323516846, -0.15794529020786285, 5.163308620452881, -2.0917932987213135, -1.2927449941635132, -0.07834170013666153, 0.20365102589130402, 3.3098623752593994, -0.19127868115901947};

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

TfLiteRegistration* Register_wunffw_REF() {
  static TfLiteRegistration r = {wunffw::Init, wunffw::Free,
                                 wunffw::Prepare<wunffw::kReference>,
                                 wunffw::Eval<wunffw::kReference>};
  return &r;
}

TfLiteRegistration* Register_wunffw_GENERIC_OPT() {
  static TfLiteRegistration r = {wunffw::Init, wunffw::Free,
                                 wunffw::Prepare<wunffw::kGenericOptimized>,
                                 wunffw::Eval<wunffw::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_wunffw_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {wunffw::Init, wunffw::Free,
                                 wunffw::Prepare<wunffw::kMultithreadOptimized>,
                                 wunffw::Eval<wunffw::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_wunffw_CBLAS_OPT() {
//   static TfLiteRegistration r = {wunffw::Init, wunffw::Free,
//                                  wunffw::Prepare<wunffw::kCblasOptimized>,
//                                  wunffw::Eval<wunffw::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_wunffw() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_wunffw_MULTITHREADED_OPT();
#else
  return Register_wunffw_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
