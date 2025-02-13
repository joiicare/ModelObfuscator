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

#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv.h"

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace oiiuii {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// This file has three implementation of DepthwiseConv.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

const int kTensorNotAllocated = -1;

int8_t filter_r   aw[3600]={24, -6, 20, -14, 3, -2, 13, -3, -6, -9, -46, -12, 7, -2, -6, -21, -1, -34, -13, -33, -5, 2, -2, 8, -20, 2, -14, 10, -15, -6, 31, -3, -36, 19, 1, -29, -3, -16, -9, -8, 8, -2, -17, 8, -1, 12, -27, 10, 2, -7, -32, 0, 0, -2, -5, 3, 70, 18, 11, 6, 25, -8, 9, -30, 3, 22, -10, 7, -1, -17, 5, 4, -12, -8, -13, -30, 16, 7, 6, -9, -2, 5, -13, 17, -10, -1, 12, 7, -24, -16, -66, 35, -31, -8, -6, -7, -13, -3, -2, -7, -18, -9, -3, 16, 6, 22, 40, -8, -26, -7, 5, -39, -34, 0, -4, 0, 14, 0, -6, 23, 1, -4, 4, 5, -5, 11, 5, 6, -36, -39, 32, 2, 2, -15, 14, 0, -11, 4, 14, 1, -5, 0, 5, -16, 24, -9, 17, -15, -1, 9, 23, -4, -10, -5, -65, 20, 35, -19, -13, -51, 7, -74, -4, 23, -9, -43, -10, 13, -9, 24, -15, 2, -53, 0, 11, 10, -18, -30, -27, -64, 19, 7, -3, -18, -26, -21, -31, 21, 8, 21, -25, 12, 28, -16, -56, -11, 17, -19, 39, 33, 101, -18, 20, 9, 23, -67, -20, -20, -4, 15, 2, 25, -4, -24, -42, -19, -5, 40, -13, -14, -8, -47, -6, -7, -19, 2, -16, 33, -1, -24, 32, 4, -33, -2, -69, 27, -78, 6, 5, -13, -79, -18, -19, -6, -15, -19, -20, 7, -17, 15, 47, -6, -27, -18, -2, -88, 37, -30, -16, -21, -14, -15, -8, 28, -48, -9, -30, -20, 11, -14, 22, 10, -18, -70, 23, -3, 35, -4, 14, 15, -7, 35, 28, -7, -21, -46, 21, 11, -6, -1, -2, 4, -8, 27, 18, -22, 1, -8, 10, 58, -73, -63, -96, -89, 4, -69, 8, 59, -7, -22, -15, -8, 47, 29, -6, -9, -60, -4, -10, 23, 2, -104, -41, -55, 38, 13, -70, -25, -44, -40, 71, 22, 18, -3, 18, 15, 48, -17, -9, -19, -15, -34, 25, 36, 11, -66, 4, -36, -14, 73, -36, 66, -4, -15, 18, 45, -12, 8, -89, -38, 4, -37, 15, 11, -24, -21, -24, -6, -34, -14, 3, 23, 7, -39, -47, 7, -48, -14, 11, 16, -96, 23, 8, -19, -114, -27, -24, -9, 38, -30, -32, -22, -58, -33, 6, -3, -5, -25, -15, -91, 21, -13, -32, -35, -37, -29, -2, -10, -75, -11, -41, -34, 30, -36, 24, 4, -1, -111, -11, -33, 52, 1, -1, 37, 0, 44, 55, -12, -27, 24, 29, 41, -22, -4, -18, 15, 0, 15, 17, 15, -7, -3, 106, 21, 2, -66, 67, -66, -5, -75, -5, -21, -12, 0, 3, -5, -23, -34, 17, -10, -25, -12, -4, -2, 34, -38, 0, -11, 16, -14, -50, -17, 6, -18, 56, 13, 10, -26, 0, 9, 24, -12, 13, -6, -38, -18, -25, 32, -60, -22, -17, -37, -17, 4, 18, 28, 9, -60, -6, 26, 9, 28, -38, -18, 5, -13, 7, -12, 2, 41, -11, -7, -19, 3, 5, -4, 0, -21, -34, 12, -38, -44, 74, 10, -76, 7, -2, -7, -72, -14, -16, -10, 30, -20, -18, -7, -24, -32, -53, -11, 23, -16, -2, -68, -29, 46, -15, -25, -8, -14, 2, -15, -32, 2, -21, -10, 12, -13, 14, -4, 10, -66, -31, -18, 0, -5, 10, 27, -5, 26, 45, -9, -16, -5, 14, 7, -14, -7, -19, 8, 3, -1, 11, 15, -6, -2, 68, -20, 31, -13, -6, -18, -6, -96, -13, -4, -6, -12, 6, -26, -18, -5, 31, 0, -1, -7, -2, -7, 25, 21, -4, 7, -4, -17, 0, -6, 2, -3, -16, 5, -4, -13, 3, 9, -8, -3, 5, 0, 4, -3, -22, 14, -30, -11, -12, 4, 10, -13, 4, 0, -2, -46, -14, 2, 19, 7, 12, 2, -2, 5, -16, -25, 8, -8, 5, -1, -5, 1, -1, -4, -7, 1, 19, 7, -28, -4, 23, 11, -31, -8, -6, -5, -6, -2, -2, -11, 8, -8, -6, 5, 8, -8, -44, -9, 16, -5, 1, -24, -10, 10, -3, 1, 14, 1, -1, 8, 12, 4, 5, 6, -2, 13, 4, 4, -5, -31, -20, 7, -28, -13, 15, 1, -10, 4, 34, 4, -4, 5, 4, -19, 64, -4, 67, -16, -8, 14, 22, -21, -8, -12, -71, 36, 3, -19, 6, 32, 8, -93, -3, -53, -9, 31, -10, 45, -7, -36, 46, -29, -18, 4, 15, 8, -58, 13, 12, -56, 34, -1, -46, -6, -29, -24, -35, 19, 7, 17, -25, 13, -11, -20, -53, -8, 18, -19, -11, -53, 106, -52, 14, -33, -40, 43, 52, -17, -7, 73, -9, 36, -7, -20, -2, -38, -2, -26, 13, -19, 8, 23, -1, -19, -19, -4, -13, 15, -20, -23, 20, -8, -34, -3, -59, 45, -5, 6, -2, -10, 10, -16, -16, -6, -9, -20, -19, 8, -5, -4, 90, 2, -67, -1, -4, 27, -44, -26, -17, -24, 85, -17, 7, -67, -25, -11, -2, -24, 10, 18, 19, -9, 6, -45, 60, -24, 13, -17, 13, -9, -6, -40, 41, -36, -23, 33, 23, -43, 70, 59, 63, 16, -51, -14, -22, -85, 33, -65, -127, 94, 26, -90, -57, -5, 52, -114, 47, -35, 45, 17, -50, 76, -27, 78, 62, -49, -97, 60, 81, 70, -99, -27, 104, -96, 82, 81, -70, -54, -127, -93, 16, -24, 62, 94, 40, -15, 82, 20, 19, -56, 58, 11, 93, -98, 107, -127, 74, -93, 22, -127, -23, -102, 5, 78, 59, -27, -58, 31, -17, -102, 39, 78, 62, 47, -40, -53, -56, -63, 9, -62, 24, 66, 37, -88, 91, -43, -90, -43, -88, 98, -21, 53, 64, -85, 30, 13, 26, 46, -84, 13, 9, -44, -71, 70, 98, 52, -75, 30, -73, 64, -37, -30, -78, -92, -34, -79, 33, -16, -107, -68, 42, -89, 71, -52, -19, 19, -14, -100, 77, -49, 78, 41, -37, 46, 39, -91, 98, -54, 15, -88, -18, -40, 1, 87, 5, 73, -81, -46, -64, -89, 73, -91, -28, 124, -122, -124, -13, -82, 79, -70, 91, 127, 83, -118, -84, -6, 86, 30, 71, 30, -121, 78, 82, 105, 24, -77, 39, -122, 105, 103, 75, -87, -52, -127, 125, -61, 96, 69, 92, -50, 127, 62, 125, -90, -69, 47, 62, -127, -29, -73, 89, -28, 102, 51, -87, 58, -9, -20, 101, -63, -97, 97, -27, -113, 78, -55, 90, 89, -87, -25, -105, -82, 46, -92, 74, 101, 82, -110, -98, -67, -127, -109, 28, 111, -78, 83, 86, -105, 50, 46, 66, 71, -2, 48, 47, -91, -108, 95, 11, 77, 2, 59, -98, 103, 117, 127, -107, -117, -100, -105, 57, 99, -127, -91, 67, -110, 97, -108, -59, 65, -49, -110, -14, 43, 127, 84, -82, 106, 75, -127, 127, 33, 53, -5, -57, 81, -65, 48, -60, 57, -42, -9, -16, -33, 30, -42, 110, 45, 67, -100, 127, -127, 39, -90, 44, 6, 31, 9, -34, -68, -36, -105, 35, 64, -77, 17, 23, 55, 82, -28, -63, -81, 73, 58, 127, -54, 46, -88, -14, -13, 59, 4, 69, -11, 51, 14, 75, -48, -37, 15, -77, -86, -112, 35, 8, 68, 43, 31, 66, 79, -19, -104, 53, -16, -45, 62, -13, -97, 59, 2, 40, 40, -57, 85, -60, -57, 15, -43, 56, 70, 23, -77, -76, -25, -74, 30, 127, 70, -101, 45, 40, -53, 36, 12, 19, 29, 70, 15, 16, -62, -69, -17, -93, 39, 73, 28, -58, 74, -33, 19, -72, -66, -18, -69, 43, 46, -87, -54, 23, -79, 63, -33, -15, 48, -49, -50, -105, 75, 81, 30, -26, 65, 34, -80, 50, 59, 9, 55, -10, 83, -72, -8, -66, 11, 0, 16, 23, 1, -10, 2, 86, -14, -3, -35, -42, -78, -1, -109, -6, -39, -17, 1, 14, -43, 3, 26, -12, 6, 4, -20, 4, 0, 34, 8, -16, -2, 23, -20, 40, -3, 5, -20, -42, 20, 3, -23, 15, 16, -24, -22, 9, 0, 46, -17, -23, -45, -58, 36, -19, 63, 1, -13, 15, -9, -30, -98, -13, 31, 11, -6, 1, -35, 12, 12, -18, -17, -13, -22, -2, -14, -17, -2, 0, 23, -25, -15, 69, 11, -22, 45, 70, 32, -56, 0, 0, 1, 8, -18, -20, -15, 23, -17, -17, -7, -19, -25, -88, -6, 60, -2, -2, 48, -7, -44, -13, -11, 83, -11, 11, -6, -16, -4, -19, -18, 5, 20, 21, 8, -28, -9, -51, 21, 17, -20, 17, 2, -12, -28, 5, 14, -22, -3, 23, -16, 118, 2, 127, 5, -18, 33, 21, -35, -5, -21, 9, 63, 52, -63, 8, 76, 18, -106, 13, -19, 3, -33, -6, 82, -3, -32, -5, -73, -12, -7, -70, 21, -90, -27, -54, -48, 52, 27, 59, -14, -57, -39, 16, 18, 6, -4, 21, 19, -11, -22, 15, -20, 40, -35, -13, 23, -1, -101, -9, -25, -29, 47, 57, 23, -44, 101, 21, 50, -16, 5, -18, -68, 6, -35, 8, 10, -21, 21, -24, -35, -38, -21, 4, -56, -23, -39, 20, -12, -19, -76, 28, -9, 40, 19, 12, -18, 31, -28, -19, -5, 39, -29, -31, -25, -65, -61, 107, 16, -107, 9, -19, 46, 16, -48, -30, -29, 127, -30, 10, -1, -41, -26, -39, -55, 26, 14, 22, -26, 49, -1, 67, 27, -18, -15, 4, 10, -3, 8, 47, -78, -28, 24, 28, -23, 95, 92, 84, 79, -88, -40, -63, -111, 70, -100, -13, 127, -77, -112, -123, 88, 83, -127, 97, -126, 80, 122, -80, 127, -20, 127, -52, -103, -89, 74, -93, 95, -127, 31, -55, -108, 95, 97, 127, -80, -63, -117, 127, -66, 88, 82, 112, -49, 51, 57, 125, -97, 34, 46, 127, 31, -40, -50, 71, 41, -127, -80, -24, -99, -66, 127, 101, -55, -94, 90, 55, -127, 69, 127, 91, 94, -85, -68, -94, -101, 43, -104, 83, -54, 62, -110, 122, -83, -2, -66, 23, -16, 73, 81, 93, -113, 94, 51, 65, 77, -28, 49, 48, -99, -127, -96, 106, 86, -125, 77, -103, 101, -127, 117, -103, -106, -46, -109, 76, -66, -110, -97, 70, -127, 100, -65, -65, -86, 91, 8, 110, 106, 20, 64, -87, 83, 79, 37, 30, -94, 53, 37, -53, -127, 6, 127, 2, 127, -127, -127, -127, -127, 127, -127, -42, 93, -68, -118, 125, 47, 127, -43, 127, 107, 127, 77, -127, -20, 127, -35, -127, 61, -107, 127, 1, 127, 45, 120, 127, -127, 127, 127, 75, -127, 83, -127, 92, -127, 127, 127, 127, -127, 76, 127, 99, -127, -127, 127, 61, 17, -60, 106, 127, 127, -118, -77, -127, -62, -68, 30, 127, -127, -127, 127, 127, -115, 127, -59, 127, 127, -127, -20, -127, -127, 127, -127, 127, 5, 127, -127, -123, -127, 23, 127, 20, -18, 61, 127, 127, -127, 127, 127, 127, 127, -127, 127, 127, -127, -121, 74, 5, 127, 0, 127, -127, 127, 109, 59, -127, -127, -124, -127, 127, -127, -125, -127, 127, -113, 127, -127, -127, -127, 127, 36, 13, 63, 124, 127, -127, 127, 127, 70, -26, 72, 127, -127, -127, 11, -96, 77, -94, 56, -75, -37, -52, -104, 59, -71, -36, -15, 73, -101, 42, -26, 71, -121, 73, 95, 60, -117, -62, -124, -44, -126, -107, 127, -84, 62, 84, 85, 113, 24, 94, -105, 116, 99, -73, -73, 95, -109, -125, -42, 86, 68, 98, -50, 22, 39, -54, -82, 14, 46, -109, 3, -7, 114, 74, 26, 42, 87, 78, 101, -19, -94, 98, -38, -85, 71, 40, -122, 80, 2, 79, 70, -78, 127, -81, -84, 49, -99, 63, 47, 41, -112, -127, -50, 31, 98, -9, -10, -4, 70, 68, -93, 82, 46, 49, 65, 3, 50, 51, -80, -102, 127, -127, 75, 127, 58, -90, 79, 74, -101, -100, -102, -22, -96, 73, 26, -109, -81, 46, -113, 78, -50, -48, -70, 82, 56, -127, -67, 106, 48, -61, 71, 55, 53, -60, 127, 36, 64, -35, 98, -127, -8, -120, -4, -14, 35, 24, -20, -11, -3, -1, -20, -21, -47, -33, -42, 8, -88, 7, -26, -17, -28, -1, -59, 6, 53, -50, 30, 0, -13, 73, 10, 38, -47, -9, -55, 54, 12, -67, 2, -4, -35, -44, 24, 7, 3, 2, 18, -19, -30, -55, -8, 51, -30, -21, 1, 13, 52, 2, -36, 56, -19, 26, 25, -2, -119, 8, 50, -9, -4, -32, -61, 13, 13, 5, -4, -10, -26, -15, -22, -29, -13, -9, 45, -36, -22, 70, 8, 13, -11, 3, -7, -23, 5, 8, -10, 20, -28, -25, -9, 21, -25, -25, -12, -38, 47, -110, -1, 104, -4, -3, 34, -29, -15, -23, -16, 107, -22, 2, 33, -40, -21, -26, -43, 7, 16, 29, -17, 38, 45, -81, -62, 45, -23, 9, -7, -12, 16, -42, 25, -30, 12, 32, 9, 79, -8, 81, 16, -8, 18, 21, 2, -10, -12, 100, 8, 63, -33, -49, 42, 7, -110, -5, 9, -7, -12, 3, 42, -1, 7, -58, -29, -16, -15, -12, -2, -77, -31, -1, 1, 22, -2, 16, -14, -11, -17, 41, 15, 3, -23, 31, 8, -5, -14, 27, -13, 23, -20, -6, 21, -76, 0, -21, 37, 42, 12, 31, 49, 31, 61, -6, 29, 4, 18, -13, -34, -1, -24, -13, -10, -14, 19, -13, -16, -21, -1, 11, -48, -16, -11, 14, 3, 20, 19, 73, -47, 16, 1, -1, 0, 6, -19, -12, -6, 38, -18, -20, -12, -25, -3, 73, 3, -65, 5, 6, 3, 12, 53, -11, -15, 74, -9, 2, 3, -11, -5, -11, -23, 9, 19, 18, 8, -39, 31, 39, 30, -91, -17, 13, 15, -10, 15, -28, -34, -17, -8, 13, 16, 71, 48, 56, 57, -40, -11, -15, -36, 21, -52, 115, 16, -127, -96, -9, 94, 38, -115, 44, -108, 28, -38, -44, 70, -23, 71, -44, -35, -88, 4, -45, 40, -57, 45, -17, -66, 62, 55, -65, -63, 47, -71, 54, -16, 53, 1, 69, -18, -31, 14, 39, -62, -22, 14, 87, 25, -127, 88, 1, 41, 74, 15, -9, -28, 85, 75, 56, -11, -39, 74, 35, -78, 40, 82, 38, 37, -45, -49, -54, -56, 13, -46, 54, -100, 22, -77, 94, -23, 80, 114, 108, -103, 70, 40, 31, -56, 16, 14, 16, 27, 63, 14, 16, -50, -91, -47, 69, 40, -68, 25, -52, -12, -31, 61, -65, -66, -30, -66, 53, 68, -83, -36, 26, -94, 68, -32, -18, 54, -50, 86, 77, -58, -97, 29, -35, 75, 31, 37, -77, -37, 8, 62, -8, -81, 0, 78, -8, 71, -66, -41, -56, -84, 54, -69, -34, -45, 44, -127, 104, 100, 60, -74, 73, 4, 56, 127, -72, -7, 83, -55, 2, 12, -127, 47, -127, 68, 31, 127, -95, -106, 88, 94, -89, -77, 103, -100, -76, -48, 81, 73, 73, -47, -90, 42, -74, -86, -72, 48, 33, 36, -25, 118, 55, -23, -32, -125, -75, -127, 127, 27, 118, -38, -68, 77, 93, -88, 68, -43, 69, 75, -71, -15, -71, -69, 51, -82, 58, -127, 56, -97, -76, -48, 126, 61, -14, -127, 127, 70, 58, -99, 15, 54, 44, 57, -38, 50, 52, -69, -121, -117, -4, 59, 22, 37, -89, -45, -43, -92, -100, -94, -74, -98, 95, 32, -120, -56, 44, -112, 96, -83, -55, 69, -35, 127, 16, -127, -30, 66, -71, 97, 57, 36, -124, 26, 39, -25, -34, -74, -78, 30, -69, 13, -35, -8, -16, -53, 13, -23, -121, -111, 19, -61, -51, 42, 34, -65, 31, 87, 19, 54, -40, -64, -36, -55, 34, 25, -62, 40, -105, 36, 65, 31, 17, -78, 59, 71, 1, -40, 34, -58, -94, -10, 44, 57, 17, -21, -43, 7, -127, -47, 36, 12, -74, 29, 113, 26, 52, -68, -101, 62, 24, 15, 73, -51, 56, -8, -37, 8, 32, -68, 37, 3, 32, 22, -50, 79, -43, -42, 17, -42, 11, -74, 17, -57, -72, -23, 79, -61, -121, -101, 82, 39, 34, -56, 4, 16, 8, 36, -43, 18, 16, -19, -62, -33, -97, 32, 81, 13, -38, -58, 113, -13, -57, -58, -11, -57, 37, -82, -71, -36, 20, -72, 56, -34, -15, 17, -14, 87, -66, -23, 4, 19, -22, 49, 20, 8, -90, 42, 7, -43, -5, 46, -69, -10, -60, -14, -7, 19, 20, 0, -16, -1, -57, -93, -10, 0, 5, -4, 0, -55, -9, 3, -16, -35, -5, -44, -2, 33, 24, 10, 3, -4, -21, -4, 19, -43, 33, -27, 17, -1, 1, -7, -18, -14, 6, 15, 0, 13, -27, 5, -13, -15, -33, -7, 25, -16, -19, 29, 92, -17, 6, -14, -48, 8, 17, 41, 23, -60, -11, 25, 0, -22, -17, -25, 1, 8, -1, -15, 8, -27, -7, -8, -16, 2, -13, -18, -20, -3, 49, -1, 16, -9, -72, -42, 18, -3, 3, -1, 5, -18, -16, -6, -2, -15, -16, 12, -11, 7, -71, -3, 38, -5, 10, -25, -26, -14, -8, -6, 59, -6, -11, -50, -6, -10, -2, -11, 3, 16, 19, -13, 6, 37, -57, 20, 4, -24, 18, 0, -14, 5, -43, 6, -15, 30, 16, 21, 30, -8, 28, 7, 5, 0, 10, 17, -7, -10, 63, -31, 3, 1, -30, 7, -8, -51, -16, -3, -8, 2, 2, 9, -20, 12, 28, 7, -21, -8, 12, -9, -36, 3, -9, 0, -12, -12, -29, -5, -9, 1, 10, 7, -7, -11, 5, 16, -16, -5, -7, -6, 0, -3, 0, 0, -32, -1, -10, 0, 1, 11, 3, 25, 16, 22, -12, 4, 14, 4, 10, 4, -6, -13, -10, -25, 12, 1, 5, -5, -4, 4, 0, 0, -5, 2, 2, 8, 19, 19, 15, -14, -9, -11, -10, -1, -11, -5, -3, -5, 8, -10, -6, 2, 4, 5, 21, -9, -36, -7, 13, -20, -7, 24, 0, -2, 3, 3, 0, 4, 3, 5, -2, 10, -4, 5, 2, 6, -2, 17, 18, -26, -77, -11, 16, 2, -10, -1, -27, 5, -5, 8, -1, 7, 17, -12, 16, 10, 6, 12, 23, 8, -16, -7, 80, -42, -74, -37, 40, 41, -6, -65, -12, -34, -18, -11, 4, 17, -3, 4, 40, 5, -54, -10, 14, -7, -2, 6, 3, -14, 9, -9, 8, -20, 14, -12, -30, 23, 0, -26, -2, 16, -43, -19, -18, -7, -19, -17, 29, -7, -48, 26, -19, -11, 11, 23, -5, 5, -13, 14, -14, 26, 11, 8, -2, -10, 1, 39, -7, -20, 10, -37, -3, -4, -16, 4, -4, 2, -11, -14, 20, 18, 26, -5, 55, -11, 22, -6, -13, 1, -73, -18, -24, -20, 28, -18, -17, 1, -32, 22, 19, -13, -19, -22, 9, -47, -15, -31, -9, -25, -8, -8, 9, -7, -41, 8, -19, -19, -1, -9, 22, 3, 12, 42, 21, -8, -100, -7, 17, 25, -16, -9, -34, 2, -21, -6, 19, -19, -8, -7, -10, 7, -2, 31, 26, -11, -11, -6, 0, -91, 69, -44, 23, 65, -9, -72, 3, -15, -18, -23, -5, -12, 48, -13, 17, -20, -45, -14, 23, 4, 8, 6, 8, -27, 27, 3, 24, -14, 6, -34, -100, 25, 9, -1, -3, 16, -86, -26, -44, -7, -15, -28, 9, -8, 17, 23, -12, -22, 54, -68, -21, -65, -33, -4, 5, 49, 3, -3, -12, -22, -3, -32, 5, 2, -13, -10, -10, -1, -27, -5, -11, -14, -13, -19, -35, 21, 16, -39, 0, -13, 81, 0, -10, -13, -97, -23, -28, -17, 10, -27, -28, -8, -45, -4, -8, -16, 16, -34, -3, -76, -69, -44, -25, -28, -26, -20, 16, 14, -56, -5, -49, -34, 12, -15, 27, 8, -3, 55, 7, 52, -82, -7, 3, 32, -14, -16, -26, -16, -30, 11, 34, -53, -26, -12, -24, -8, 0, 16, 19, 7, -17, 1, -53, -116, 1, -17, -29, 23, -8, -97, -6, 23, -13, 20, -9, -7, -20, -19, 0, -24, -19, -13, 7, -10, 28, 2, -29, -20, 7, -11, 3, -9, -24, -11, -23, 16, 0, 8, -26, 15, -49, -14, -23, -3, 9, -15, -19, 0, 84, -29, -2, -5, 33, -7, -4, -30, -28, -28, -3, 25, 11, -23, 2, -5, -6, -16, -15, -12, -18, 38, -3, 0, -16, 9, -13, -27, -6, -4, -17, 14, 7, -4, -65, -6, 55, -3, -5, -9, -57, -13, -16, -5, -22, -17, -15, 18, -4, -15, -50, -14, 22, -20, 7, -44, 38, -16, -9, -5, -9, -6, 1, 34, -19, -5, -19, -5, 5, -4, 15, 12, -17, 29, -19, 42, -50, -7, 13, 17, -10, -11, -7, -14, -15, -42, 14, -8, -27, -5, -24, -7, 5, 0, 9, 2, -8, -7, -29, -79, -10, -4, -12, -8, -1, -88, -12, 9, -2, -2, -5, -24, -20, -3, 4, -1, -10, -5, -27, -7, 16, -1, -13, 0, -22, -12, 7, -3, -15, 1, 17, 5, -8, -3, -17, 18, -16, -1, 5, -2, 8, -2, -14, 8, 48, -40, 1, 5, -6, 33, 6, 20, -10, -24, -4, 2, 4, -11, 13, 8, -3, 3, -12, -20, 9, -6, 3, -1, -5, 1, -7, -19, -2, 4, 8, 7, 13, -5, -62, -11, 12, -10, -4, -5, -4, -5, -3, -11, -4, -9, -6, 15, 7, -6, -41, -11, 13, -6, 7, -31, -2, -26, 2, 2, 1, 3, -4, 4, 8, 2, -5, 15, -1, 2, 2, 11, -39, 8, -27, 2, -37, -11, 14, 7, -10, -4, 1, 5, -3, 11, -1, 9};

float bias_raw[144]={-0.4290972650051117, 4.646633625030518, -0.534782886505127, 4.040938377380371, 4.735962867736816, 6.667056083679199, 4.492580890655518, 2.58380389213562, 1.89860200881958, 2.449148178100586, 7.721585273742676, -0.01989167556166649, -0.2736625671386719, 2.6426138877868652, -0.8335663080215454, 0.05405261367559433, 1.9066351652145386, 10.2957763671875, 3.5655136108398438, -0.14372846484184265, 1.393507957458496, -0.5969048738479614, 2.4642395973205566, -0.20682692527770996, -1.0823670625686646, -0.3350822627544403, -0.0940103530883789, 2.959568738937378, 2.1921780109405518, 1.2086827754974365, -0.716355562210083, 2.038545608520508, -0.46695971488952637, 3.1333539485931396, -0.6478105783462524, 2.2616682052612305, 1.8605842590332031, 4.64581298828125, -0.7786507606506348, 6.044607162475586, -0.4684275984764099, 2.3268790245056152, 0.12387119233608246, 8.677603721618652, 5.282002925872803, 1.674087643623352, 1.483878254890442, 3.8956363201141357, 1.4761075973510742, 4.84371280670166, -0.4422740936279297, 2.391632080078125, -0.06844794750213623, 1.6950857639312744, -0.5894615650177002, -1.133453607559204, 4.969878673553467, -0.02697470225393772, 2.723238706588745, -0.2952796220779419, -0.5946349501609802, -0.20293951034545898, -0.14852160215377808, -0.20651496946811676, -0.9795097708702087, 0.08308138698339462, 2.6059603691101074, 3.0316357612609863, 8.24004077911377, 1.675663948059082, 2.419255018234253, 2.048346996307373, 0.6123512387275696, -1.5980825424194336, -0.6644223928451538, 2.204357862472534, 5.332076072692871, -0.6256841421127319, -1.5231311321258545, 1.7178566455841064, 1.6113628149032593, 1.7630293369293213, 1.2301846742630005, -0.6831815242767334, 1.8278332948684692, 2.13608980178833, 0.28179842233657837, 5.672379493713379, -0.9799845218658447, -0.7349372506141663, 8.276274681091309, -0.9334125518798828, 0.10713168978691101, -1.0373278856277466, 1.3326075077056885, 1.5709350109100342, 5.471736431121826, 1.4520771503448486, 2.176743507385254, 1.6613136529922485, -0.36365413665771484, 1.0597964525222778, 1.6506099700927734, 2.6404237747192383, 2.3470518589019775, -0.2425617277622223, -0.17400017380714417, 0.5993150472640991, -0.4299948215484619, 1.8042205572128296, 4.007811546325684, -0.25026005506515503, -0.5905665159225464, -0.673663318157196, 0.7153687477111816, 0.5416463613510132, 4.522292613983154, 0.6364307999610901, 3.171191692352295, -0.022210389375686646, 2.5450246334075928, 3.3643405437469482, -0.0025062859058380127, 1.814612865447998, 7.520122528076172, 4.272960186004639, 8.419816017150879, -0.6000932455062866, -1.1019433736801147, 0.247402623295784, -0.6573084592819214, -0.47144225239753723, -0.40981656312942505, 1.3864680528640747, 3.487204074859619, 6.025825500488281, 1.307425856590271, -0.4699759781360626, -0.6982331871986389, 2.7655882835388184, 3.796726942062378, 0.03283584117889404, 6.894745826721191, -0.9930184483528137};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
int stride_width=2;
int stride_height=2;
TfLiteFusedActivation activation=kTfLiteActNone;
int dilation_width_factor=1;
int dilation_height_factor=1;
const int filter_dims_size=4;
const int filter_dims_raw[4]={1,5,5,144};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={144};
TfLitePadding paddings=kTfLitePaddingSame;
TfLiteType filter_type=kTfLiteInt8;
TfLiteType bias_type=kTfLiteFloat32;
const float scale_filter=0.0;
const int32_t zero_point_filter=0;
const float scale_bias=0.0;
const int32_t zero_point_bias=0;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int> per_channel_output_shift;

  // Hybrid per channel temporary tensors.
  int input_quantized_id = kTensorNotAllocated;
  int scaling_factors_id = kTensorNotAllocated;
  int input_offset_id = kTensorNotAllocated;
  int32_t input_quantized_index;
  int32_t scaling_factors_index;
  int32_t input_offset_index;
};

void ExtractDepthConvParams(TfLitePadding padding, int stride_width, int stride_height,
                               int dilation_width_factor, int dilation_height_factor,
                               TfLiteFusedActivation activation,
                               TfLiteDepthwiseConvParams* data_params) {
  // TfLiteDepthwiseConvParams data_params;
  data_params->padding = padding;
  data_params->stride_width = stride_width;
  data_params->stride_height = stride_height;
  data_params->dilation_width_factor = dilation_width_factor;
  data_params->dilation_height_factor = dilation_height_factor;
  data_params->activation = activation;
  // return data_params;
}

void GetDepthConvTensor(TfLiteType type, const char* name, TfLiteIntArray* tensor_dims_data, 
                       TfLiteQuantizationParams quant_params, char* tensor_data,
                       TfLiteAffineQuantization* quant_struct, size_t bytes_size,
                       TfLiteTensor* tensor) {
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
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // auto* params =
  //     reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  TfLiteDepthwiseConvParams data_params;
  ExtractDepthConvParams(paddings, stride_width, stride_height, dilation_width_factor, dilation_height_factor, activation, &data_params);
  TfLiteDepthwiseConvParams* params = &data_params;

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  bool has_bias = false;

  // TF_LITE_ENSURE(context, has_bias || NumInputs(node) == 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
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
  GetDepthConvTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data),
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;
  // TF_LITE_ENSURE_OK(context,
  //                   GetInputSafe(context, node, kFilterTensor, &filter));
  const TfLiteTensor* bias = nullptr;

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 4);
  TF_LITE_ENSURE(context, params->dilation_height_factor > 0);
  TF_LITE_ENSURE(context, params->dilation_width_factor > 0);

  const TfLiteType data_type = input->type;

  const TfLiteType filter_type = filter->type;
  const bool is_hybrid =
      data_type == kTfLiteFloat32 && filter_type == kTfLiteInt8;
  TF_LITE_ENSURE(context,
                 data_type == kTfLiteFloat32 || data_type == kTfLiteUInt8 ||
                     data_type == kTfLiteInt8 || data_type == kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, data_type);
  if (!is_hybrid) {
    TF_LITE_ENSURE(context,
                   filter->type == data_type || data_type == kTfLiteInt16);
  }

  if (data_type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  // Filter in DepthwiseConv is expected to be [1, H, W, O].
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(filter, 0), 1);

  if (has_bias) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBiasTensor, &bias));
    if (data_type == kTfLiteUInt8 || data_type == kTfLiteInt8) {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else if (data_type == kTfLiteInt16) {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt64);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, data_type);
    }
    TF_LITE_ENSURE_EQ(context, NumDimensions(bias), 1);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(filter, 3),
                      SizeOfDimension(bias, 0));
  }

  int channels_out = SizeOfDimension(filter, 3);
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);
  int batches = SizeOfDimension(input, 0);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training or
  // calibration.
  if (data_type != kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE(context, filter->quantization.type != kTfLiteNoQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
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

  if (is_hybrid) {
    TF_LITE_ENSURE(context, filter->quantization.type != kTfLiteNoQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE_EQ(
        context, affine_quantization->scale->size,
        filter->dims->data[affine_quantization->quantized_dimension]);

    int temporaries_count = 0;
    data->input_quantized_index = temporaries_count;
    if (data->input_quantized_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->input_quantized_id));
    }
    ++temporaries_count;
    data->scaling_factors_index = temporaries_count;
    if (data->scaling_factors_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->scaling_factors_id));
    }
    ++temporaries_count;
    data->input_offset_index = temporaries_count;
    if (data->input_offset_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->input_offset_id));
    }
    ++temporaries_count;

    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(temporaries_count);

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
    node->temporaries->data[data->scaling_factors_index] =
        data->scaling_factors_id;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->scaling_factors_index,
                                  &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    const int batch_size = SizeOfDimension(input, 0);
    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }
    node->temporaries->data[data->input_offset_index] = data->input_offset_id;
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, data->input_offset_index,
                                       &input_offsets));
    input_offsets->type = kTfLiteInt32;
    input_offsets->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
      input_offsets_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                       input_offsets_size));
    }
  }

  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(4);
  outputSize->data[0] = batches;
  outputSize->data[1] = out_height;
  outputSize->data[2] = out_width;
  outputSize->data[3] = channels_out;
  return context->ResizeTensor(context, output, outputSize);
}

TfLiteStatus ComputeDepthMultiplier(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    const TfLiteTensor* filter,
                                    int16* depth_multiplier) {
  int num_filter_channels = SizeOfDimension(filter, 3);
  int num_input_channels = SizeOfDimension(input, 3);
  TF_LITE_ENSURE(context, num_input_channels != 0);
  TF_LITE_ENSURE_EQ(context, num_filter_channels % num_input_channels, 0);
  *depth_multiplier = num_filter_channels / num_input_channels;
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteDepthwiseConvParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  TF_LITE_ENSURE_STATUS(ComputeDepthMultiplier(context, input, filter,
                                               &op_params.depth_multiplier));
  if (kernel_type == kReference) {
    reference_ops::DepthwiseConv(
        op_params, GetTensorShape(input), GetTensorData<float>(input),
        GetTensorShape(filter), GetTensorData<float>(filter),
        GetTensorShape(bias), GetTensorData<float>(bias),
        GetTensorShape(output), GetTensorData<float>(output));
  } else {
    optimized_ops::DepthwiseConv<float, float>(
        op_params, GetTensorShape(input), GetTensorData<float>(input),
        GetTensorShape(filter), GetTensorData<float>(filter),
        GetTensorShape(bias), GetTensorData<float>(bias),
        GetTensorShape(output), GetTensorData<float>(output),
        CpuBackendContext::GetFromContext(context));
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteDepthwiseConvParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  auto input_offset = -input->params.zero_point;
  auto filter_offset = -filter->params.zero_point;
  auto output_offset = output->params.zero_point;

  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  TF_LITE_ENSURE_STATUS(ComputeDepthMultiplier(context, input, filter,
                                               &op_params.depth_multiplier));
  if (kernel_type == kReference) {
    reference_ops::DepthwiseConv(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  } else {
    optimized_ops::DepthwiseConv<uint8, int32>(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<uint8_t>(output),
        CpuBackendContext::GetFromContext(context));
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                                     TfLiteDepthwiseConvParams* params,
                                     OpData* data, const TfLiteTensor* input,
                                     const TfLiteTensor* filter,
                                     const TfLiteTensor* bias,
                                     TfLiteTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output->params.zero_point;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  TF_LITE_ENSURE_STATUS(ComputeDepthMultiplier(context, input, filter,
                                               &op_params.depth_multiplier));

  if (kernel_type == kReference) {
    reference_integer_ops::DepthwiseConvPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), GetTensorShape(input),
        GetTensorData<int8>(input), GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<int32>(bias), GetTensorShape(output),
        GetTensorData<int8>(output));
  } else {
    optimized_integer_ops::DepthwiseConvPerChannel(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), GetTensorShape(input),
        GetTensorData<int8>(input), GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<int32>(bias), GetTensorShape(output),
        GetTensorData<int8>(output),
        CpuBackendContext::GetFromContext(context));
  }
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel16x8(
    const TfLiteDepthwiseConvParams* params, const OpData* data,
    const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.weights_offset = 0;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  reference_integer_ops::DepthwiseConvPerChannel(
      op_params, data->per_channel_output_multiplier.data(),
      data->per_channel_output_shift.data(), GetTensorShape(input),
      GetTensorData<int16>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<std::int64_t>(bias), GetTensorShape(output),
      GetTensorData<int16>(output));

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalHybridPerChannel(TfLiteContext* context, TfLiteNode* node,
                                  TfLiteDepthwiseConvParams* params,
                                  OpData* data, const TfLiteTensor* input,
                                  const TfLiteTensor* filter,
                                  const TfLiteTensor* bias,
                                  TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  const int batch_size = SizeOfDimension(input, 0);
  TF_LITE_ENSURE(context, batch_size != 0);
  const int input_size = NumElements(input) / batch_size;
  TfLiteTensor* input_quantized;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                                     &input_quantized));
  int8_t* quantized_input_ptr_batch = input_quantized->data.int8;
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

  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;

  op_params.weights_offset = 0;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  TF_LITE_ENSURE(context, filter->quantization.type != kTfLiteNoQuantization);
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  if (kernel_type == kReference) {
    reference_integer_ops::DepthwiseConvHybridPerChannel(
        op_params, scaling_factors_ptr, GetTensorShape(input),
        quantized_input_ptr_batch, GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<float>(bias), GetTensorShape(output),
        GetTensorData<float>(output), affine_quantization->scale->data,
        input_offset_ptr);
  } else {
    optimized_integer_ops::DepthwiseConvHybridPerChannel(
        op_params, scaling_factors_ptr, GetTensorShape(input),
        quantized_input_ptr_batch, GetTensorShape(filter),
        GetTensorData<int8>(filter), GetTensorShape(bias),
        GetTensorData<float>(bias), GetTensorShape(output),
        GetTensorData<float>(output), affine_quantization->scale->data,
        input_offset_ptr, CpuBackendContext::GetFromContext(context));
  }

  return kTfLiteOk;
}

template <KernelType kernel_type, TfLiteType input_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  // auto* params =
  //     reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  TfLiteDepthwiseConvParams data_params;
  ExtractDepthConvParams(paddings, stride_width, stride_height, dilation_width_factor, dilation_height_factor, activation, &data_params);
  TfLiteDepthwiseConvParams* params = &data_params;

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  // const TfLiteTensor* filter;
  // TF_LITE_ENSURE_OK(context,
  //                   GetInputSafe(context, node, kFilterTensor, &filter));
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
  GetDepthConvTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data),
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;
  // const TfLiteTensor* bias =
  //     (NumInputs(node) == 3) ? GetInput(context, node, kBiasTensor) : nullptr;
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
    GetDepthConvTensor(bias_type, "bias", bias_dims_data, bias_params,
                        reinterpret_cast<char*>(bias_tensor_data), 
                        &quant_struct_bias, bytes_size_bias, &bias_tensor);
    bias = &bias_tensor;
  } else {
    bias = nullptr;
  }

  TFLITE_DCHECK_EQ(input_type, input->type);

  switch (input_type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      if (filter->type == kTfLiteFloat32) {
        return EvalFloat<kernel_type>(context, node, params, data, input,
                                      filter, bias, output);
      } else if (filter->type == kTfLiteInt8) {
        return EvalHybridPerChannel<kernel_type>(context, node, params, data,
                                                 input, filter, bias, output);
      } else {
        TF_LITE_KERNEL_LOG(
            context, "Type %s with filter type %s not currently supported.",
            TfLiteTypeGetName(input->type), TfLiteTypeGetName(filter->type));
        return kTfLiteError;
      }
      break;
    case kTfLiteUInt8:
      return EvalQuantized<kernel_type>(context, node, params, data, input,
                                        filter, bias, output);
      break;
    case kTfLiteInt8:
      return EvalQuantizedPerChannel<kernel_type>(context, node, params, data,
                                                  input, filter, bias, output);
      break;
    case kTfLiteInt16:
      return EvalQuantizedPerChannel16x8(params, data, input, filter, bias,
                                         output);
      break;
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalImpl<kernel_type, kTfLiteFloat32>(context, node);
    case kTfLiteUInt8:
      return EvalImpl<kernel_type, kTfLiteUInt8>(context, node);
    case kTfLiteInt8:
      return EvalImpl<kernel_type, kTfLiteInt8>(context, node);
    case kTfLiteInt16:
      return EvalImpl<kernel_type, kTfLiteInt16>(context, node);
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
}

}  // namespace oiiuii

TfLiteRegistration* Register_oiiuii_REF() {
  static TfLiteRegistration r = {
      oiiuii::Init, oiiuii::Free, oiiuii::Prepare,
      oiiuii::Eval<oiiuii::kReference>};
  return &r;
}

TfLiteRegistration* Register_oiiuii_GENERIC_OPT() {
  static TfLiteRegistration r = {
      oiiuii::Init, oiiuii::Free, oiiuii::Prepare,
      oiiuii::Eval<oiiuii::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_oiiuii_NEON_OPT() {
  static TfLiteRegistration r = {
      oiiuii::Init, oiiuii::Free, oiiuii::Prepare,
      oiiuii::Eval<oiiuii::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_oiiuii_NEON_OPT_UINT8() {
  static TfLiteRegistration r = {
      oiiuii::Init, oiiuii::Free, oiiuii::Prepare,
      oiiuii::EvalImpl<oiiuii::kNeonOptimized, kTfLiteUInt8>};
  return &r;
}

TfLiteRegistration* Register_oiiuii() {
#ifdef USE_NEON
  return Register_oiiuii_NEON_OPT();
#else
  return Register_oiiuii_GENERIC_OPT();
#endif
}

// Warning: Clients using this variant are responsible for ensuring that their
// models only need the UINT8 type. TFLite's op registration mechanism doesn't
// yet allow for more nuanced registration mechanisms.
TfLiteRegistration* Register_oiiuii_UINT8() {
#ifdef USE_NEON
  return Register_oiiuii_NEON_OPT_UINT8();
#else
  return Register_oiiuii();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
