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

#include "tensorflow/lite/kernels/internal/optimized/integer_ops/fully_connected.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/sparse_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace mvlxab {

int8_t filter_raw[4096]={-65, 5, -47, -23, -69, -35, -24, 43, -1, -1, -62, 3, -64, 35, 49, 40, -5, -56, -55, -89, 49, -1, 33, -101, -111, 67, -24, 66, -79, -64, 38, -106, 51, 95, -42, -87, -94, 21, -49, -103, 29, -127, 88, -12, 94, -34, 50, -15, -23, 48, 3, -1, 8, 46, -15, -25, -24, 38, -81, 30, 8, -8, -85, -42, 89, 25, 78, -76, -119, 67, 77, -89, -58, 11, 64, -60, -15, 59, -16, -64, -102, -16, 39, 70, -29, -19, -7, 50, -26, 69, -5, 11, -102, -18, -81, -91, -80, -116, -91, 48, -64, -33, 47, -50, 70, 20, 3, -79, -25, 65, 44, -19, 76, -3, 0, -19, -69, 60, -91, -46, -127, -123, -68, 73, 14, -46, -62, 24, 80, -34, -80, -2, -30, -13, -97, 68, -24, -28, -36, 78, 12, 23, 70, 35, -46, 80, 0, 4, 100, 85, -39, 60, 114, 38, -127, 19, 17, 81, -76, 56, -85, 37, 39, -27, 1, -7, 39, 39, 18, -26, 30, 75, -61, -23, 52, -11, -38, -20, 93, -75, 57, 53, -35, 39, -37, 82, 9, 39, 53, 45, -49, 0, 13, 48, -95, -71, 89, 27, -68, 76, 22, 70, 17, 23, 70, 54, -42, 37, -17, -61, 1, -94, -76, -7, 76, -80, -3, 19, 75, 21, -86, -56, -75, -9, 77, -37, 21, -26, -127, -95, -87, 89, 46, -101, -87, -23, 42, -13, -62, -101, -85, -79, -10, -59, 55, 48, -68, 59, -30, 10, 30, -83, -84, 4, -94, -123, 22, 7, -49, 7, 92, -61, 85, -4, 65, 28, 56, 26, -4, -49, 75, -53, 76, -24, 89, -57, 1, 44, -43, -14, 109, 39, -127, -16, -87, 70, 104, 59, -25, -106, 123, -34, 49, 52, -111, 19, 91, -54, 7, 98, 29, -62, -22, -72, 28, -25, 52, -30, -31, 112, -80, -87, -97, 89, 89, 11, -34, -12, -24, 29, 25, 14, 66, 88, -62, -48, 90, -56, -44, 26, -44, -68, -38, -95, 84, 29, -67, 24, 42, 70, -46, -68, 27, 22, -103, -9, 47, -69, 36, -14, -57, 103, 11, 78, -119, 57, 9, -57, -54, 52, 30, -72, 18, 79, -53, -30, 80, -6, 79, -9, 53, -66, 47, -127, -37, -88, 2, 49, -70, 10, -2, -20, -95, 38, -8, -1, 55, -45, 82, 19, 18, 29, -48, -96, 38, 32, -77, -9, -9, -8, -110, -82, 33, -9, -68, -98, 8, -31, 38, -71, 27, -85, -100, -19, -102, -10, -80, 33, 28, 1, -127, -69, -92, -21, 43, -61, -1, -67, 76, 78, 8, -35, -100, -69, 7, -31, 13, -8, -27, 31, -87, -12, -8, -40, -4, -23, 9, -70, 6, -23, -5, -12, 2, -1, 12, -5, -113, 6, 9, -6, -15, -13, -12, -8, -18, 11, -36, 3, 15, 7, 15, 14, -9, -9, 13, 11, 18, -8, -2, -7, -17, -5, -5, -7, 18, 2, 16, -8, -45, 14, -35, 2, -11, 10, 2, 13, -64, 8, 6, -20, 10, -1, -21, -127, 4, 11, -54, -14, 6, -12, -17, -22, 1, -8, -3, -9, 10, -8, 6, -1, -71, -3, -3, 9, -6, -2, -3, 0, 3, 4, -30, 4, 7, -3, -4, -4, -6, 3, -3, -3, -4, 3, 9, -6, 5, 1, 1, -1, -3, 2, 4, -1, -22, 0, -34, -3, -13, -3, -3, 1, -35, 0, -6, 0, 3, -6, 0, -127, 0, -9, -66, -13, -1, -7, 6, -37, 51, 48, -15, -62, 39, -55, -29, -4, 7, -16, -22, -81, -38, -50, -29, -71, 28, 4, -35, 75, -55, 3, -47, -4, 8, -22, -38, -19, -31, 20, -92, -79, -69, -55, -37, -40, 14, 43, 34, 15, -28, -49, -82, -41, -31, -47, -18, 39, -39, 14, -1, -37, -2, -67, -11, -127, -43, -65, -80, 15, -19, -78, 50, 14, -105, 43, 107, -26, -116, -77, -4, -27, -33, 73, 26, 84, -103, -88, -31, 53, -127, -66, -98, -80, -68, -90, -46, -22, 24, 53, 106, -61, 80, -87, -14, 21, 58, 91, 23, 62, 33, 93, -46, -98, 37, -42, -111, -41, 22, -25, -30, -114, -25, -73, 104, 99, 16, 55, -99, -101, -66, 48, 19, 4, -29, -71, 13, -56, 7, 85, -4, 15, 30, -52, -28, 41, 66, 38, -40, -33, -35, -83, -97, -12, 1, -23, -49, -30, -73, -75, -50, 42, 37, -67, 8, -45, -103, 71, -124, 22, 76, 22, 38, 34, -78, 1, 21, 55, -107, 26, 46, -65, -49, 79, -30, -21, -97, -41, 40, 81, -29, -22, -48, -127, -32, 93, -51, 37, -74, -13, 21, -96, 67, -13, 56, 68, -36, -101, 22, 14, -125, -37, -2, -51, -43, 67, 43, 31, -6, -102, -103, 74, -23, -7, 28, 25, -10, 30, -21, -85, 72, -85, -90, -100, 76, -1, 60, 20, -107, 44, -122, 14, 34, 4, -47, 72, 78, -45, 29, 68, -31, -127, -88, 33, 60, -94, -47, 51, -21, 83, 35, 91, 35, -71, -50, -42, 1, -16, -92, -25, -31, 53, 0, -82, -66, -53, 4, -39, -34, -65, 69, -60, 17, -113, 54, 48, -43, -47, -30, -65, -97, 48, -54, -48, 89, -31, 39, 9, 8, 0, -61, -27, 7, -28, 31, -46, 7, -65, -27, 23, -31, -60, 15, -79, 20, 3, -62, -67, -77, -100, -8, 40, -94, -31, -98, -26, -127, -12, 43, -51, 22, 71, 73, -7, 97, 53, 36, 6, 47, -106, -26, -99, 87, -89, -117, 11, -98, -112, -21, 40, 48, 74, 83, 12, -95, -7, -31, -114, -73, 59, -16, 94, -103, 7, -116, -115, -36, 78, -46, -91, 57, 5, -66, -111, 66, -71, 58, 85, -25, -127, 40, 66, -29, -24, -5, -86, -96, -68, 72, 111, 62, 51, 54, 35, 66, -1, -46, 75, 4, 78, -47, 91, 29, 107, -113, -96, 64, 89, -34, 53, -48, 41, 113, 88, 3, 106, 42, 25, 84, 32, 38, -74, -52, 102, 45, 84, -62, 5, -116, 64, 28, 3, -48, 26, 11, 99, -7, 55, 9, 48, -68, -25, -38, -39, -75, -67, 32, 29, 25, -74, 87, 56, 62, 43, 16, 127, 39, -36, -8, -4, -5, -9, 13, 5, -10, -1, -93, 6, 10, 2, 0, 12, -2, 10, -4, 8, -26, -3, 3, -3, -1, -16, 4, 3, -19, 4, -15, 8, 11, -3, -6, -12, 1, -4, -9, 6, 8, -10, -37, -13, -40, -11, -18, 3, -6, 11, -23, 1, -11, -14, 4, 10, 11, -127, -8, 5, -69, -9, 4, 9, -10, -46, -55, 13, 8, -45, -33, 24, -15, 29, -66, -51, 31, -12, 28, 7, -60, -79, 62, 62, 5, 67, -50, -45, 19, 31, -40, -78, -127, 28, 115, -76, -73, 48, 0, 22, -40, -8, 14, 87, -42, -3, -80, -35, 89, -43, 4, 67, 60, 89, -71, -21, 37, 63, 39, 97, 29, 53, 47, -57, -58, 79, 37, -30, -50, -76, -9, 4, 3, -10, 4, -9, -10, -2, -93, -17, -9, 17, -1, -10, -13, -4, 6, -2, 1, 6, -3, 18, -14, 10, 13, -15, 5, -11, -23, 2, 2, 6, 10, -127, 0, -4, 7, -11, 6, 6, -12, 15, -34, 3, -18, 0, 6, 11, -2, -13, 1, -14, -4, -11, -20, -6, -14, 7, -4, 10, 9, 6, 14, 3, -14, 3, -5, 1, -3, 5, -15, 2, -8, -7, 18, 6, 9, -9, 0, -3, -1, 4, -7, 0, -8, 3, 11, -7, 4, -1, -8, -3, 4, -4, -14, -4, 5, -5, 9, 2, -2, 2, -14, 5, -21, -16, 5, 6, -14, -1, 6, -1, 8, -34, 5, -14, 7, -10, 3, 1, 2, -12, 4, -4, -127, -5, -10, -2, 51, 14, 63, 9, -64, -79, -52, 22, -77, -36, 7, -87, -63, -50, -29, -46, -65, -12, 68, -80, -19, 35, 61, -94, -27, -30, -94, -40, -76, 57, 41, 24, -57, -52, -38, -77, -26, 31, -37, 35, -93, -108, -77, -65, 30, 10, 53, -65, -54, -40, -127, 109, -6, -27, 59, 55, 90, -34, -80, -13, -119, 78, -51, 18, 93, -22, -97, 88, 91, 37, -121, 80, 85, -94, -108, -22, -82, -36, -85, -108, 90, -83, -102, 96, -52, -98, 34, -40, -8, 73, 5, -97, -127, 88, 46, 74, -66, -13, 15, 98, -45, -43, -55, -58, 76, -100, -17, -11, 56, -31, 27, 46, 55, -41, 78, 52, -94, 4, -34, 34, -103, -23, -19, -73, -81, 46, -91, -67, -63, -111, 68, -16, 15, -1, -67, 76, 37, 50, 62, -40, 60, -82, 14, 43, -61, -121, -62, -10, -77, 75, 55, -103, -81, -62, 12, -91, -19, -8, -93, 43, -87, -59, -94, -104, 49, 95, -6, -52, 17, 71, -91, -53, -84, 12, 22, 5, 35, -127, 43, -71, -34, 70, 49, 34, 43, -37, -11, -68, 40, -113, 12, -69, 55, 1, 62, 7, 71, 55, -6, 69, -10, 103, -72, -120, 24, -65, -84, -127, 21, 30, -62, -42, -92, -81, 22, -107, -75, -6, -22, -115, 24, -20, -81, -75, 51, 76, -35, -36, -108, 49, -84, -13, 48, -9, 49, 78, 81, 85, -59, 0, 61, -15, -57, -86, -43, 5, 31, -53, -31, -60, 27, -33, -73, 57, 64, -4, -28, 8, 25, 17, 15, 16, -12, 24, -82, 28, 6, -26, -22, -21, 20, 8, 5, 2, 4, 25, -22, -28, 19, -9, 16, -11, -20, 16, -46, 5, -3, 30, 28, -127, -26, 24, -24, -19, 29, 7, -57, 17, -48, 28, -12, 22, -3, -37, 10, 28, -21, -5, -20, -22, 31, 27, -20, -26, 8, 8, -40, 7, 30, -11, -6, -8, -7, -10, -2, 16, 1, -13, -21, 4, 1, 5, -2, -2, 4, -23, -9, -7, -12, -1, -11, 8, 4, -7, 4, -2, 14, -19, -17, -16, -5, -4, 5, -3, 0, 14, 0, 10, -18, 10, -22, -17, -2, 4, -9, -12, 3, -1, 10, -4, 4, -7, -10, -5, 1, 6, -14, -16, -10, 9, -127, 18, -11, 5, -48, 29, -10, 90, -30, 71, -9, 88, 40, -22, 23, -124, 71, 127, -121, -39, 56, -67, 52, -18, -106, 99, 83, 19, 82, -46, -107, -11, -93, 26, 70, 35, 1, 7, -69, -15, 21, 105, -44, 107, -78, -61, -1, -90, 91, 74, 15, 22, 83, 12, -19, -15, 50, -98, -96, 48, 4, 0, -100, 21, -30, 49, -52, -55, -56, -11, -115, -111, -80, -12, -47, 59, -57, -2, -7, -97, -113, 98, 13, -127, 38, 22, 11, -27, -106, 111, -21, 37, -41, 54, -50, 43, -120, -119, -9, -16, 111, 48, -115, 116, -11, -79, -127, 107, -105, 65, 91, -123, -41, 32, 6, -50, 38, 43, -2, 37, 58, -102, -92, -7, -56, -73, -76, -68, -88, 68, -17, -81, 19, -4, 24, -36, -27, -17, -127, -30, -12, -95, 1, -15, -53, 22, -79, 3, -105, 44, -70, 47, -24, -16, 91, -57, 35, -13, 13, -49, -45, -58, -90, -108, 86, 57, 0, -98, -95, -26, -119, 56, 7, -81, 25, -71, 12, -36, 13, -108, -54, -23, 38, -8, 7, -84, -21, -99, -41, -78, 75, -4, 42, 41, 36, -54, 0, -75, -44, -61, 29, 17, 13, -60, -77, -16, -79, -9, 40, 58, -54, -61, -9, -9, -21, 43, -11, 0, 17, -26, -52, 48, 4, -116, -77, -25, -5, -46, -9, -66, -127, -47, -34, -35, -55, 37, 12, -51, -3, 0, 37, 39, 42, -18, -16, -123, 31, -13, 5, -60, -51, -35, 34, -65, 23, -65, -43, -25, -20, 38, 65, 83, 37, 54, 31, 25, 30, 7, 37, -46, -1, -16, -3, -75, 20, -119, -11, -99, 11, -83, -88, -59, -29, 32, -71, -31, -43, -59, 47, 67, -101, -76, 53, -28, 57, -68, -127, -8, -60, -89, -24, -61, -17, 18, -81, 82, -52, -106, -108, 66, 62, 30, -37, 81, -27, 37, -56, 6, 33, 18, 8, -51, 29, -28, -5, -3, 8, -2, 3, 3, -4, 3, 3, 2, -8, 2, -6, 8, 7, 5, -8, 6, -1, 8, -9, -13, 3, -4, 9, 6, 8, -10, 2, 8, 6, 7, -4, 18, 0, -7, -3, -1, 0, 0, -6, 1, -3, -7, -4, 8, 3, -7, -34, 1, 4, -3, -7, 4, 6, -127, 6, -13, -36, 4, -5, 7, 0, 6, -70, -19, 20, 22, -72, -71, 26, 47, -65, 64, -93, -103, 49, 27, -38, -51, 36, 14, 19, -6, -84, -76, 95, 3, 36, -70, 50, 68, 88, -42, -62, -47, -113, 7, 8, -69, -72, -92, 22, 49, -64, 13, -83, 13, 47, 0, -12, -45, 36, -127, 15, 15, 45, -1, 26, -64, 26, 105, -24, -31, 48, 3, -45, 28, 13, -60, -16, 0, 33, 43, -36, 16, 5, 14, -19, 6, 54, -58, 31, -75, -127, -18, 24, -14, -30, 53, 18, 28, -82, -77, 62, -54, 24, 61, -29, -5, 66, -25, 27, -22, -15, 2, -79, 29, -48, 68, -41, -37, -10, 67, -67, 33, -75, -11, 63, -33, 36, -55, 6, 7, -4, 54, -44, -61, 5, -78, 6, 52, -73, -84, -83, -26, -30, -89, 74, -24, -42, -60, -14, 4, -12, 81, 80, -76, -127, 24, -89, 40, -86, -33, 21, -33, -62, 55, 0, -29, 1, -18, -95, 4, 104, 21, 23, -87, -40, 63, -15, 63, 4, -81, -104, 13, 72, -100, 12, -22, 42, -112, -66, -78, -3, 45, 50, 71, 80, 55, -85, 68, 29, -124, 33, -76, -5, -6, 0, 1, 1, 1, 5, -1, -2, -3, 0, 6, -6, 6, 4, -1, -3, 7, 1, 4, 6, 4, 3, -4, 3, 6, 2, -1, 5, -6, 7, -4, 3, -127, -1, -2, 1, -1, 7, -3, -1, -4, 3, 4, -3, 7, -4, 1, -5, 9, -2, -7, -1, -3, -7, 0, -1, -8, 0, -8, -4, -3, -4, -1, 19, 12, -6, -3, 9, 9, -8, 13, -13, -5, -5, -1, 10, -8, -14, -11, 5, 0, -12, -14, -15, 11, 4, -9, 25, -11, 13, -3, -12, 2, -15, -1, -9, -11, 17, -4, 2, 4, -9, -17, -18, -24, 6, 10, 5, -12, -12, -5, 17, -14, 11, -13, -5, -17, 8, 8, 3, 6, -13, -3, -127, 4, -5, 7, 8, -59, -74, -48, 62, -12, -14, 13, -2, -31, -25, 58, -54, 38, -65, 34, -53, -51, 66, 85, -37, 51, 7, 44, 11, 9, 127, 45, 15, 37, 74, -12, 17, -3, -16, 35, 40, -55, 11, -58, 27, -37, 62, 15, 12, 70, -34, -2, -12, 24, -18, -1, 68, 26, -67, 35, -79, 45, 25, 38, 61, 31, 31, -7, -81, 31, -75, -64, -75, -92, -79, -10, 28, -101, -75, 56, -21, -115, 37, -121, -103, -62, 8, 37, 48, -115, -6, 69, 20, -6, 25, -109, -105, -116, -42, -85, 39, 14, 9, 43, -70, -15, -87, -25, 39, -34, -127, -18, -28, 30, 87, -65, -61, -66, 92, -114, -74, -25, -7, -70, 4, 4, -102, -29, -65, -28, 47, 76, -43, 35, -16, -75, -30, -40, -19, 6, 24, -56, -27, -61, -29, 21, 54, -64, -127, 16, -14, -42, 64, -30, 40, -28, -54, 53, -7, 0, -37, -34, 4, 0, 52, 13, -67, -36, 0, 37, -89, 65, 18, -76, 37, -51, -9, -60, 33, -7, -32, 25, 56, -27, -105, 16, 75, 53, -51, -68, -67, 46, -8, 37, -63, -116, -76, -77, -37, -76, -2, 40, -11, 50, 47, -41, -90, -39, 83, 80, 32, -65, -35, 8, 46, 10, 96, 55, 15, 58, -41, 23, -36, 2, -69, -21, -5, 89, 51, -52, 43, 21, 46, 50, -127, -90, 52, 69, -119, -47, 87, -36, 39, 65, 18, -56, -115, 92, -47, -107, -86, -60, -30, 30, -18, -4, -81, 16, -27, -63, -109, -73, -41, -95, -65, 59, 14, -61, 63, 27, -65, 23, 89, -77, 10, -127, -72, -65, 52, -67, 58, -59, 11, 53, -99, 99, -82, -71, 22, -56, -77, 85, -58, 45, 81, -71, -43, -62, -3, -101, 89, -120, -55, 33, -109, 8, -42, 15, -38, -85, 84, -19, -90, -34, -78, -59, -71, -43, 43, -84, 12, -124, 109, -10, -1, 4, 6, -10, 7, -14, 7, -10, -127, 15, 23, -22, 2, 20, -16, -8, -39, 10, 10, 21, -12, 0, -16, -41, 17, -16, 17, -8, 11, 14, 17, -2, 14, 13, -4, 12, -13, -14, -8, 4, -40, -20, -66, -15, -36, -1, -10, 11, 6, -12, 9, -25, -3, 20, -4, -27, -3, 8, -26, 20, -3, -2, 16, -12, -33, -6, 1, -25, 0, -7, -11, 19, -28, -9, -11, 11, -26, 19, -5, -21, -5, 6, 21, 9, 22, 12, -12, -9, -5, -23, 3, -10, -48, 2, -7, 5, 18, -127, 13, 17, -17, 10, 10, 0, -63, 18, -38, 0, 2, 14, -12, -16, -28, 4, 17, 3, -21, -11, -38, 5, 3, 18, 2, -26, 10, 31, 18, -29, 16, -50, -77, -97, -7, -66, 17, 45, 7, -49, -11, -70, 38, 95, -70, -103, 4, 14, 60, -60, 4, 53, -11, -20, -84, 71, 70, -55, -83, -17, 8, -40, -44, -45, -52, 35, -127, -61, -119, -96, -47, -91, -70, -90, -102, -8, 10, -101, 36, -23, -66, -48, -10, 48, -14, 29, -58, -48, -117, 8, -3, 63, 36, 22, 51, -31, -41, -2, -1, -45, -30, -12, -53, -14, -1, 29, -26, -50, 52, 35, -127, 12, 21, -40, -60, -111, 7, -5, 29, -39, -14, -28, -23, 16, -59, -62, 55, 49, -12, 29, -17, -12, 19, 58, 40, 29, 36, 39, 4, -56, -14, -74, -36, 5, 28, -28, -25, 5, -26, -63, 51, 50, -8, 17, -15, 36, 42, -31, -16, -12, 0, 19, 4, -10, 5, -10, 1, 14, -1, 18, -10, 2, 0, 2, -36, 1, -6, -15, -16, -12, -15, 3, 3, 1, 17, 5, 8, -7, -2, -5, 13, 29, -10, -16, -1, -13, -2, 3, -11, -14, 14, 11, 8, 13, 10, -8, -67, 7, -6, 2, 13, -4, 0, -127, 1, -6, -10, 10, 15, -4, -1, -9, -13, 13, -20, -46, 22, -17, -31, -5, -120, -19, 1, 25, -25, -20, 26, -4, 18, -11, 29, -13, 9, 21, 18, 5, -16, -27, -2, 18, -49, -6, 9, 13, 6, -127, 17, -26, -5, -4, -1, -17, -124, 22, -110, 1, -48, 23, 7, 23, -41, 5, -27, -6, 11, -4, -35, 1, -8, -23, 0, -3, -1, 33, -1, -24, -15, -8, -11, -33, 46, 26, 39, -27, -75, -8, 18, -24, -20, 27, -5, 10, 28, -13, -67, 29, -14, -31, 24, 16, 43, -17, -10, 22, -18, 3, -9, -20, -5, -8, -41, 23, 4, -1, 9, 5, -107, 0, -103, 17, -62, 18, 15, -3, 5, -13, -5, -7, -13, 20, -4, -127, -4, -12, -96, -3, 24, -8, -6, -70, 41, -87, -72, -101, -63, -86, -110, -84, -30, 20, 3, -20, 28, 32, -29, -87, -117, -52, -55, 23, -49, -93, 94, -6, -127, 37, -109, 23, 54, 52, 11, -9, 33, -30, -43, -61, -60, -24, -20, 13, -71, -85, 55, -23, -5, -36, 40, 28, 28, -87, -99, 55, -77, -85, -10, 27, -19, -24, -102, -50, 21, -5, 66, 11, -64, -8, -57, 32, -33, 26, -127, 69, -1, 13, -61, -62, -34, 2, 26, -61, 27, -16, -6, -17, -22, -12, 51, 47, 26, 18, 8, -4, -65, -2, -6, 19, -27, 12, 22, -31, -80, 14, 7, 62, 15, -66, -27, -43, -70, -22, -75, 48, -59, -117, -47, -54, -22, 17, 20, -18, -29, 7, -80, -8, -8, 42, 29, 31, -97, 30, 34, -57, -74, 19, -50, -57, 48, -50, 20, -2, -20, -70, 52, -1, -33, 5, 13, -15, 8, 19, -66, -5, 74, 58, 32, 19, -104, 41, -127, -16, 67, -37, 82, -32, -67, -21, -126, 36, -38, -49, -64, -58, -103, 23, 28, -21, -54, 79, 51, -85, 16, -75, -99, 9, -59, -95, -61, 83, -111, 76, 47, 7, -15, -3, -7, -6, 3, -9, -9, 0, -108, 7, 9, 3, -15, 10, 14, 3, -16, 9, 2, -10, 5, -2, 3, -1, 1, 7, 7, 13, -10, -6, 3, -3, -14, -127, 0, -7, -5, -13, -5, -7, -43, 2, -41, -2, -15, 1, 15, 7, 13, -3, -6, -1, 2, 3, -4, -4, 16, 4, 13, -5, 9, 10, -6, -3, -113, 5, 26, 21, -25, -17, -22, 70, -38, -61, 90, -80, -17, 28, -48, -45, -28, -104, -110, 33, 86, 92, -60, -9, -115, 62, -98, 87, -15, -127, 65, 15, -26, 16, -14, 25, -87, 15, -77, 25, 74, -48, -15, 87, -11, -56, -40, -32, 27, -82, 59, 62, -88, 28, 50, -97, -63, 11, 9, 63, 16, 0, -91, -41, 18, -98, 8, 76, 53, -45, -15, -73, -110, -43, 57, -109, 0, 3, -5, 4, 25, 40, -116, 27, 58, 65, -90, 21, -58, 66, 7, -94, 74, -81, -127, -47, 9, -92, -94, 7, 21, -80, -100, 65, 29, -112, -77, 54, -25, 93, -15, 70, -122, -92, 49, 25, 106, -5, 44, 100, -100, 36, -101, -37, -70, 9, -60, -5, -26, 34, -16, 19, 86, 98, 55, -24, -74, 60, 4, -100, -68, 59, -35, -120, 0, -122, -88, -19, 6, 23, 62, -63, -89, -30, -6, -87, -53, 81, 6, 80, -53, 41, -25, -76, -28, 53, -82, -38, 0, 4, 5, 35, 66, -95, -78, 4, 45, -127, -69, 63, -83, -54, -46, 34, 76, -34, 49, -106, -57, -87, -48, -109, -11, -36, -15, -26, 4, 9, -4, 5, -67, -9, -10, -18, -20, -28, -9, -14, 9, 8, -11, -15, 15, -36, -3, 9, 14, -12, -6, 9, -6, -15, -3, 2, 8, 16, 7, -4, -6, 2, 10, -23, -78, -3, -78, -19, -49, -5, 14, 10, -24, 13, 7, -20, 3, 0, -8, -87, -15, 5, -127, -30, 10, 11, 1, -1, -1, 27, -7, 14, -127, 10, -54, 21, 3, 7, 20, -27, -1, -19, -15, 0, 34, -19, 18, 38, -3, -17, 12, -71, 33, -27, -22, -85, -31, -12, 10, 8, 5, -9, -26, -9, -92, -8, 20, 4, -9, 15, -8, -19, -16, -1, -14, 15, 18, -78, 14, 2, 30, -37, 10, 1, -11, -31, 5, 19, -24, 17, -24, -110, -118, 31, -82, 25, -119, -79, 3, 12, -52, -17, 97, 36, -15, -49, 40, -51, 86, -63, 17, 110, 52, -45, -28, -95, -33, -14, 26, -117, -1, -108, -127, -29, -47, -43, -112, 1, 4, 52, -79, 60, -76, -50, 1, -64, -50, 9, 18, 41, 3, 17, -92, 43, -82, 73, -31, 6, -58, -47, -92, -94, 71, 13, 10, -118, 7, -12, -8, -1, 9, 4, 4, -5, 5, 2, 9, 4, -3, -7, -12, -7, -15, -1, 7, -7, -4, -11, 8, -5, 1, -1, 6, -9, -6, 3, 10, -3, 9, 21, -1, -9, 4, 8, 5, 5, 5, 9, -6, -4, 1, -9, 6, -10, -35, -6, -10, 10, -11, -10, 1, -127, -3, 4, -40, 3, 1, 11, 3, -6, -43, 63, 75, -52, 70, 11, 30, -109, -87, -69, 13, 40, 77, 20, 94, -72, -32, -46, 5, -6, -32, -8, -80, 34, -4, -106, 64, 67, -63, 11, -116, 7, -71, -114, -10, -44, -14, 16, -105, 68, -88, -26, -77, 50, -20, 18, -21, -121, -20, 44, -79, 20, -35, -106, -103, -81, -95, 96, 56, -40, -127, 25, 19, -16, -2, 9, 2, -7, -4, 15, -3, -2, -29, 3, -1, -11, -20, 21, 9, -6, -37, 4, -3, -4, 7, -6, 8, -127, 11, -8, -8, -88, -37, 5, 1, -31, -24, 4, 17, 18, 14, 17, 9, 1, -3, 14, -5, -18, -20, 10, -13, 9, -7, 3, -8, 10, -3, 6, -24, -4, 14, -17, -16, 1, -7, 2, -3, -45, 38, 36, 66, -27, -4, 58, 25, -35, -14, -7, -49, -13, 55, -43, -6, -8, -115, 32, -70, -36, -26, -59, -17, -11, 44, 13, -21, 18, -39, 53, 19, -127, 24, -79, -31, -49, -56, -40, -52, -42, 49, -38, -74, -8, 28, 52, 21, 12, 8, 6, -46, 5, 7, 11, -51, 50, -6, -51, 2, -40, -28, 25, 11, 11, 4, -18, -1, -5, -106, 0, -25, -9, -22, -6, 7, 0, -7, -3, 9, 3, -6, 11, -51, 4, -9, 1, -10, -50, 1, 4, 8, -45, -1, -7, -2, 0, -13, -9, -1, 7, -3, 5, 14, -10, -2, 6, -16, 3, -9, -7, -28, 4, -35, 1, -7, -9, 0, -6, 1, -94, 1, 9, -127, -25, 7, -4, -4, -61};

float bias_raw[64]={-0.053603511303663254, -0.11109606176614761, 0.5324845314025879, -0.08629196137189865, 0.5072014331817627, -0.21167325973510742, -0.0944192185997963, 0.6009714603424072, 0.5266983509063721, -0.17699222266674042, -0.0228977520018816, -0.0840853825211525, -0.07664579153060913, -0.14907322824001312, -0.0551234669983387, -0.4767582416534424, 0.5164396166801453, 0.3935258686542511, 0.47810426354408264, -0.17350900173187256, 0.0036588311195373535, -0.01698262058198452, -0.038251131772994995, -0.05822377651929855, 0.4502699375152588, -0.15555177628993988, 0.34607991576194763, -0.02774963155388832, -0.0950811430811882, 0.04386231675744057, -0.12371549755334854, 0.6218987107276917, -0.037674400955438614, -0.04082002118229866, -0.056818101555109024, 0.5814356803894043, -0.1061311736702919, -0.5207534432411194, -0.031750209629535675, -0.47798511385917664, 0.025801707059144974, -0.04542273283004761, 0.4385741651058197, -0.5716867446899414, -0.30788877606391907, 0.34430092573165894, 0.7130745649337769, -0.46059367060661316, -0.4872778654098511, -0.06250537186861038, -0.3743509352207184, -0.07701530307531357, 0.48352399468421936, -0.05678774043917656, -0.04612978175282478, -0.07726281881332397, -0.504025936126709, -0.19473561644554138, -0.03772176802158356, 0.6282128095626831, -0.05840734764933586, 0.26608797907829285, 0.38402634859085083, -0.2556970417499542};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const TfLiteFusedActivation activation=kTfLiteActRelu;
const TfLiteFullyConnectedWeightsFormat weights_format=kTfLiteFullyConnectedWeightsFormatDefault;
const bool keep_num_dims=false;
const bool asymmetric_quantize_inputs=true;
const TfLiteType filter_type=kTfLiteInt8;
const TfLiteType bias_type=kTfLiteFloat32;
const int32_t filter_dims_raw[2]={64,64};
const int filter_dims_size=2;
const int32_t bias_dims_raw[1]={64};
const int bias_dims_size=1;

const float scale_filter=0.0;
const int32_t zero_point_filter=0;
const float scale_bias=0.0;
const int32_t zero_point_bias=0;

void ExtractFullyConnectedParams(
                               TfLiteFullyConnectedWeightsFormat weights_format, 
                               bool keep_num_dims, bool asymmetric_quantize_inputs,
                               TfLiteFusedActivation activation,
                               TfLiteFullyConnectedParams* data_params) {
  // TfLiteFullyConnectedParams data_params;
  data_params->weights_format = weights_format;
  data_params->keep_num_dims = keep_num_dims;
  data_params->asymmetric_quantize_inputs = asymmetric_quantize_inputs;
  data_params->activation = activation;
  // return data_params;
}


void GetFullyConnectedTensor(TfLiteType type, const char* name, 
                       TfLiteIntArray* tensor_dims_data, 
                       TfLiteQuantizationParams quant_params, char* tensor_data,
                       TfLiteAffineQuantization* quant_struct, size_t bytes_size, TfLiteTensor* tensor) {
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

bool SupportedSparsityFormat(const TfLiteSparsity& sparsity) {
  if (sparsity.dim_metadata[0].format == kTfLiteDimDense &&
      sparsity.dim_metadata[1].format == kTfLiteDimSparseCSR) {
    return true;
  }

  return false;
}

static const int kDimMetadataSizeRandomSparse = 2;
static const int kDimMetadataSizeBlockSparse = 3;

TfLiteStatus CreateLedgerTensor(const TfLiteSparsity* sparsity,
                                TfLiteContext* context, TfLiteTensor* ledger) {
  TF_LITE_ENSURE(context, sparsity != nullptr);
  ledger->type = kTfLiteUInt8;
  ledger->allocation_type = kTfLiteArenaRwPersistent;
  TfLiteIntArray* ledger_size = TfLiteIntArrayCreate(1);
  ledger_size->data[0] = sparsity->dim_metadata[1].array_indices->size +
                         sparsity->dim_metadata[1].array_segments->size - 1;
  return context->ResizeTensor(context, ledger, ledger_size);
}

TfLiteStatus PopulateLedgerData(const TfLiteSparsity* sparsity,
                                TfLiteContext* context, uint8_t* ledger_data) {
  TF_LITE_ENSURE(context, sparsity != nullptr);
  const auto* array_segments = sparsity->dim_metadata[1].array_segments;
  const auto* array_indices = sparsity->dim_metadata[1].array_indices;
  int output_data_ptr = 0;

  for (int i = 0; i < array_segments->size - 1; i++) {
    int row_start = array_segments->data[i];
    int row_end = array_segments->data[i + 1];
    if (row_end - row_start > UINT8_MAX) {
      return kTfLiteError;
    }
    // Copy num of non-zero blocks in row i.
    ledger_data[output_data_ptr] = static_cast<uint8_t>(row_end - row_start);
    output_data_ptr++;

    for (int j = row_start; j < row_end; j++) {
      if (array_indices->data[j] > UINT8_MAX) {
        return kTfLiteError;
      }
      // Copy indices of non-zero blocks in row i.
      ledger_data[output_data_ptr] =
          static_cast<uint8_t>(array_indices->data[j]);
      output_data_ptr++;
    }
  }
  return kTfLiteOk;
}

// This file has four implementations of FullyConnected
enum KernelType {
  kReference,
  kGenericOptimized,
  kLegacyPie,  // Legacy path used by the PIE team and related clients.
};

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index;
  bool compute_row_sums = false;
  // Only used for sparse hybrid fully connected kernels.
  bool ledger_initialized;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;

inline TfLiteStatus CheckTypes(TfLiteContext* context,
                               const TfLiteTensor* input,
                               const TfLiteTensor* filter,
                               const TfLiteTensor* bias, TfLiteTensor* output,
                               TfLiteFullyConnectedParams* params) {
  const bool is_quantized =
      ((filter->type == kTfLiteUInt8) || (filter->type == kTfLiteInt8));
  const bool is_hybrid = is_quantized && (input->type == kTfLiteFloat32);
  const bool is_shuffled =
      is_quantized && (params->weights_format ==
                       kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8);

  // optional bias tensor.
  const bool is_optional_bias_float = !bias || (bias->type == kTfLiteFloat32);
  const bool is_optional_bias_int =
      !bias || (bias->type == kTfLiteInt32) || (bias->type == kTfLiteInt64);

  if (is_quantized) {
    if (is_shuffled) {
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt8);
      TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteUInt8);
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);
      TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
    } else if (is_hybrid) {
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
      TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
      TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
    } else {
      TF_LITE_ENSURE(context, input->type == kTfLiteUInt8 ||
                                  input->type == kTfLiteInt8 ||
                                  input->type == kTfLiteInt16);
      TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                                  output->type == kTfLiteInt8 ||
                                  output->type == kTfLiteInt16);
      TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
    }
  } else {
    // Only float32 is supported currently
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
    TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
  }

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  auto* op_data = new OpData();
  context->AddTensors(context, /*tensors_to_add=*/6,
                      &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus PrepareImpl(TfLiteContext* context, TfLiteNode* node) {
  // auto* params =
  //     reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  TfLiteFullyConnectedParams data_params;
  ExtractFullyConnectedParams(weights_format, keep_num_dims,
                              asymmetric_quantize_inputs, activation, &data_params);
  TfLiteFullyConnectedParams* params = &data_params;
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  // Check we have all the inputs and outputs we need.
  // TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  // Shuffled formats need a workspace to store the shuffled input activations.
  const int expected_outputs_count =
      params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault ? 1
                                                                          : 2;
  TF_LITE_ENSURE_EQ(context, node->outputs->size, expected_outputs_count);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  // const TfLiteTensor* filter;
  // TF_LITE_ENSURE_OK(context,
  //                   GetInputSafe(context, node, kWeightsTensor, &filter));
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
  GetFullyConnectedTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data),
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;

  // const TfLiteTensor* bias =
  //     (node->inputs->size == 3)
  //         ? GetOptionalInputTensor(context, node, kBiasTensor)
  //         : nullptr;
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
    GetFullyConnectedTensor(bias_type, "bias", bias_dims_data, bias_params,
                        reinterpret_cast<char*>(bias_tensor_data), 
                        &quant_struct_bias, bytes_size_bias, &bias_tensor);
    bias = &bias_tensor;
  } else {
    bias = nullptr;
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Check proper datatype match among all Input Tensors
  TF_LITE_ENSURE_STATUS(
      CheckTypes(context, input, filter, bias, output, params));

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    input_size *= input->dims->data[i];
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);
  TF_LITE_ENSURE(context, filter->dims->data[1] != 0);
  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  if (bias) {
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }

  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  // If we have to perform on-the-fly quantization (with quantized weights and
  // float inputs) first we need to quantize the inputs. Allocate a temporary
  // buffer to store the intermediate quantized values.
  // Additionally, we allocate a temporary buffer to store the accumulated
  // quantized values prior to multiplication by the scaling factor.
  const bool is_hybrid =
      (input->type == kTfLiteFloat32 &&
       (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8));
  const bool is_sparse = filter->sparsity != nullptr;
  if (is_hybrid) {
    TfLiteIntArrayFree(node->temporaries);
    data->compute_row_sums = true;
    if (is_sparse) {
      node->temporaries = TfLiteIntArrayCreate(6);
    } else {
      node->temporaries = TfLiteIntArrayCreate(5);
    }
    node->temporaries->data[0] = data->scratch_tensor_index;

    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0,
                                                &input_quantized));
    input_quantized->type = filter->type;
    input_quantized->allocation_type = kTfLiteArenaRw;

    TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                     input_quantized_size));

    node->temporaries->data[1] = data->scratch_tensor_index + 1;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;

    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    node->temporaries->data[2] = data->scratch_tensor_index + 2;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/2, &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {num_units, batch_size};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
      accum_size->data[0] = num_units;
      accum_size->data[1] = batch_size;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, accum_scratch, accum_size));
    }

    node->temporaries->data[3] = data->scratch_tensor_index + 3;
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/3, &input_offsets));
    input_offsets->type = kTfLiteInt32;
    input_offsets->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
      input_offsets_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                       input_offsets_size));
    }
    node->temporaries->data[4] = data->scratch_tensor_index + 4;
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/4, &row_sums));
    row_sums->type = kTfLiteInt32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_dims[1] = {num_units};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
      row_sums_size->data[0] = row_sums_dims[0];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }

    if (is_sparse) {
      data->ledger_initialized = false;
      node->temporaries->data[5] = data->scratch_tensor_index + 5;
      TfLiteTensor* filter_ledger =
          &context->tensors[node->temporaries->data[5]];
      auto status =
          CreateLedgerTensor(filter->sparsity, context, filter_ledger);
      if (status != kTfLiteOk) return status;
    }
  }

  // Resize output.
  TfLiteIntArray* output_size_array = nullptr;
  if (params->keep_num_dims) {
    // When number of dimensions are kept the filter operates along the last
    // dimensions. In other words, for an input tensor with shape
    // [batch_size, ..., n_inputs] and a filter of shape [n_inputs, n_units]
    // this Op produces an output of shape [batch_size, ..., n_units].
    TF_LITE_ENSURE_EQ(context, input->dims->data[input->dims->size - 1],
                      SizeOfDimension(filter, 1));
    output_size_array = TfLiteIntArrayCopy(input->dims);
    output_size_array->data[output_size_array->size - 1] = num_units;
  } else {
    // Otherwise, the output is (potentially flattened to) a 2-D matrix.
    output_size_array = TfLiteIntArrayCreate(2);
    output_size_array->data[0] = batch_size;
    output_size_array->data[1] = num_units;
  }
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check for supported activation types.
  // auto* params =
  //     reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  TfLiteFullyConnectedParams data_params;
  ExtractFullyConnectedParams(weights_format, keep_num_dims,
                              asymmetric_quantize_inputs, activation, &data_params);
  TfLiteFullyConnectedParams* params = &data_params;

  // const TfLiteTensor* filter;
  // TF_LITE_ENSURE_OK(context,
  //                   GetInputSafe(context, node, kWeightsTensor, &filter));
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
  GetFullyConnectedTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data),
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const bool is_quantized =
      ((filter->type == kTfLiteUInt8) || (filter->type == kTfLiteInt8));
  const bool is_hybrid = is_quantized && (input->type == kTfLiteFloat32);
  const bool is_pie = kernel_type == kLegacyPie;

  // Pie and hybrid path supports all kinds of fused activations, otherwise only
  // clipping activations are supported.
  if (!is_pie && !is_hybrid) {
    TF_LITE_ENSURE(context, params->activation == kTfLiteActNone ||
                                params->activation == kTfLiteActRelu ||
                                params->activation == kTfLiteActReluN1To1 ||
                                params->activation == kTfLiteActRelu6);
  }
  return PrepareImpl(context, node);
}

TfLiteStatus EvalPie(TfLiteContext* context, TfLiteNode* node,
                     TfLiteFullyConnectedParams* params, OpData* data,
                     const TfLiteTensor* input, const TfLiteTensor* filter,
                     const TfLiteTensor* bias, TfLiteTensor* output) {
  int total_input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    total_input_size *= input->dims->data[i];
  }

  int input_size = filter->dims->data[1];
  const int batch_size = total_input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(GetTensorData<float>(bias), num_units,
                                          batch_size,
                                          GetTensorData<float>(output));
  } else {
    std::fill_n(GetTensorData<float>(output), batch_size * num_units, 0.0f);
  }

  // Compute output += weight * input
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      GetTensorData<float>(filter), num_units, input_size,
      GetTensorData<float>(input), batch_size, GetTensorData<float>(output));

  // Apply activation function
  tensor_utils::ApplyActivationToVector(
      GetTensorData<float>(output), batch_size * num_units, params->activation,
      GetTensorData<float>(output));

  return kTfLiteOk;
}

TfLiteStatus EvalHybridDense(
    TfLiteContext* context, TfLiteNode* node,
    TfLiteFullyConnectedParams* params, OpData* data, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias,
    TfLiteTensor* input_quantized, TfLiteTensor* scaling_factors,
    TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
    TfLiteTensor* input_offsets, TfLiteTensor* output) {
  int total_input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    total_input_size *= input->dims->data[i];
  }

  const int input_size = filter->dims->data[1];
  const int batch_size = total_input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(GetTensorData<float>(bias), num_units,
                                          batch_size,
                                          GetTensorData<float>(output));
  } else {
    std::fill_n(GetTensorData<float>(output), batch_size * num_units, 0.0f);
  }

  // Save matrix multiplication computation for all zero input.
  if (tensor_utils::IsZeroVector(GetTensorData<float>(input),
                                 total_input_size)) {
    tensor_utils::ApplyActivationToVector(
        GetTensorData<float>(output), batch_size * num_units,
        params->activation, GetTensorData<float>(output));
    return kTfLiteOk;
  }

  // Quantize input from float to uint8 + quantization params (scaling factor).
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);
  int32_t* input_offset_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs) {
    input_offset_ptr = GetTensorData<int32_t>(input_offsets);
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }
  int8_t* quant_data = GetTensorData<int8_t>(input_quantized);
  const int8_t* filter_data = GetTensorData<int8_t>(filter);
  const float* input_ptr = GetTensorData<float>(input);
  tensor_utils::BatchQuantizeFloats(
      input_ptr, batch_size, input_size, quant_data, scaling_factors_ptr,
      input_offset_ptr, params->asymmetric_quantize_inputs);
  for (int b = 0; b < batch_size; ++b) {
    // Incorporate scaling of the filter.
    scaling_factors_ptr[b] *= filter->params.scale;
  }

  // Compute output += weight * quantized_input
  int32_t* scratch = GetTensorData<int32_t>(accum_scratch);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      filter_data, num_units, input_size, quant_data, scaling_factors_ptr,
      batch_size, GetTensorData<float>(output), /*per_channel_scale=*/nullptr,
      input_offset_ptr, scratch, row_sums_ptr, &data->compute_row_sums,
      CpuBackendContext::GetFromContext(context));

  // Apply activation function to floats.
  tensor_utils::ApplyActivationToVector(
      GetTensorData<float>(output), batch_size * num_units, params->activation,
      GetTensorData<float>(output));
  return kTfLiteOk;
}

void EvalSparseHybridImpl(TfLiteContext* context, TfLiteNode* node,
                          TfLiteFullyConnectedParams* params, OpData* data,
                          const TfLiteTensor* input, const TfLiteTensor* filter,
                          const TfLiteTensor* bias, int thread_start,
                          int thread_end, TfLiteTensor* input_quantized,
                          TfLiteTensor* scaling_factors,
                          TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
                          TfLiteTensor* input_offsets, TfLiteTensor* output) {
  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("Sparse Hybrid Kernel");
  const auto& input_shape = GetTensorShape(input);
  const auto& output_shape = GetTensorShape(output);
  const auto& filter_shape = GetTensorShape(filter);
  const int input_dims_count = input_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int filter_dims_count = filter_shape.DimensionsCount();
  const int batch_size = thread_end - thread_start;
  const int input_depth = MatchingDim(filter_shape, filter_dims_count - 1,
                                      input_shape, input_dims_count - 1);
  const int output_depth = MatchingDim(filter_shape, filter_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int per_thread_input_size = batch_size * input_depth;

  const float* per_thread_input =
      GetTensorData<float>(input) + thread_start * input_depth;
  float* per_thread_output =
      GetTensorData<float>(output) + thread_start * output_depth;

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(GetTensorData<float>(bias),
                                          output_depth, batch_size,
                                          per_thread_output);
  } else {
    std::fill_n(per_thread_output, batch_size * output_depth, 0.0f);
  }

  // Save matrix multiplication computation for all zero input.
  if (tensor_utils::IsZeroVector(per_thread_input, per_thread_input_size)) {
    tensor_utils::ApplyActivationToVector(
        per_thread_output, batch_size * output_depth, params->activation,
        per_thread_output);
    return;
  }

  // Quantize input from float to uint8 + quantization params (scaling factor).
  float* scaling_factors_ptr =
      GetTensorData<float>(scaling_factors) + thread_start;
  int32_t* input_offset_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs) {
    input_offset_ptr = GetTensorData<int32_t>(input_offsets) + thread_start;
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }
  int8_t* quant_data =
      GetTensorData<int8_t>(input_quantized) + thread_start * input_depth;
  tensor_utils::BatchQuantizeFloats(per_thread_input, batch_size, input_depth,
                                    quant_data, scaling_factors_ptr,
                                    input_offset_ptr,
                                    params->asymmetric_quantize_inputs);
  for (int b = 0; b < batch_size; ++b) {
    // Incorporate scaling of the filter.
    scaling_factors_ptr[b] *= filter->params.scale;
  }

  if (params->asymmetric_quantize_inputs) {
    float* per_thread_output_ptr = per_thread_output;
    for (int b = 0; b < batch_size; ++b) {
      const float scaled_zp = scaling_factors_ptr[b] * input_offset_ptr[b];
      for (int row = 0; row < output_depth; ++row) {
        *per_thread_output_ptr++ -= scaled_zp * row_sums_ptr[row];
      }
    }
  }

  // Compute output += weight * quantized_input
  TfLiteTensor* filter_ledger = &context->tensors[node->temporaries->data[5]];
  tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
      GetTensorData<int8_t>(filter), GetTensorData<uint8_t>(filter_ledger),
      output_depth, input_depth, quant_data, scaling_factors_ptr, batch_size,
      per_thread_output);

  // Apply activation function to floats.
  tensor_utils::ApplyActivationToVector(per_thread_output,
                                        batch_size * output_depth,
                                        params->activation, per_thread_output);
}

struct SparseHybridFullyConnectedTask : cpu_backend_threadpool::Task {
  SparseHybridFullyConnectedTask(
      TfLiteContext* context, TfLiteNode* node,
      TfLiteFullyConnectedParams* params, OpData* data,
      const TfLiteTensor* input, const TfLiteTensor* filter,
      const TfLiteTensor* bias, const int thread_start, const int thread_end,
      TfLiteTensor* input_quantized, TfLiteTensor* scaling_factors,
      TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
      TfLiteTensor* input_offsets, TfLiteTensor* output)
      : context(context),
        node(node),
        params(params),
        data(data),
        input(input),
        filter(filter),
        bias(bias),
        thread_start(thread_start),
        thread_end(thread_end),
        input_quantized(input_quantized),
        scaling_factors(scaling_factors),
        accum_scratch(accum_scratch),
        row_sums(row_sums),
        input_offsets(input_offsets),
        output(output) {}

  void Run() override {
    EvalSparseHybridImpl(context, node, params, data, input, filter, bias,
                         thread_start, thread_end, input_quantized,
                         scaling_factors, accum_scratch, row_sums,
                         input_offsets, output);
  }

 private:
  TfLiteContext* context;
  TfLiteNode* node;
  TfLiteFullyConnectedParams* params;
  OpData* data;
  const TfLiteTensor* input;
  const TfLiteTensor* filter;
  const TfLiteTensor* bias;
  const int thread_start;
  const int thread_end;
  TfLiteTensor* input_quantized;
  TfLiteTensor* scaling_factors;
  TfLiteTensor* accum_scratch;
  TfLiteTensor* row_sums;
  TfLiteTensor* input_offsets;
  TfLiteTensor* output;
};

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        TfLiteFullyConnectedParams* params, OpData* data,
                        const TfLiteTensor* input, const TfLiteTensor* filter,
                        const TfLiteTensor* bias, TfLiteTensor* input_quantized,
                        TfLiteTensor* scaling_factors,
                        TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
                        TfLiteTensor* input_offsets, TfLiteTensor* output) {
  const auto& output_shape = GetTensorShape(output);
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  const bool is_dense = filter->sparsity == nullptr;
  if (is_dense) {
    return EvalHybridDense(context, node, params, data, input, filter, bias,
                           input_quantized, scaling_factors, accum_scratch,
                           row_sums, input_offsets, output);
  }

  TfLiteTensor* filter_ledger = &context->tensors[node->temporaries->data[5]];
  if (!data->ledger_initialized) {
    PopulateLedgerData(filter->sparsity, context,
                       GetTensorData<uint8_t>(filter_ledger));
    data->ledger_initialized = true;
  }

  // The multi-threaded kernel slices the workload along the batch dimension. If
  // there's not enough batches of data, the number of threads used is equal to
  // the batch size.
  // TODO(b/173442777): If needed, we can improve this later with slicing along
  // the row dimension of the weight.
  const int max_threads = cpu_backend_context->max_num_threads();
  const int batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int thread_count = std::max(1, std::min(batches, max_threads));
  if (params->asymmetric_quantize_inputs && data->compute_row_sums) {
    // Precompute row sums.
    static const int kBlockSize = 16;
    const uint8_t* ledger_ptr = GetTensorData<uint8_t>(filter_ledger);
    const int8_t* row_ptr = GetTensorData<int8_t>(filter);
    const int output_depth = filter->dims->data[0];
    int32_t* row_sums_ptr = GetTensorData<int32_t>(row_sums);
    for (int row = 0; row < output_depth; ++row) {
      int32_t row_sum = 0;
      int num_nonzero_blocks = *ledger_ptr++;
      for (int i = 0; i < num_nonzero_blocks; ++i, ++ledger_ptr) {
        for (int c = 0; c < kBlockSize; c++) {
          row_sum += (*row_ptr++);
        }
      }
      row_sums_ptr[row] = row_sum;
    }
    data->compute_row_sums = false;
  }
  std::vector<SparseHybridFullyConnectedTask> tasks;
  tasks.reserve(thread_count);
  int thread_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    // This makes sure the workload is relatively balanced when batches is not
    // a multiple of thread_count. The first mod(batches, thread_count) tasks
    // need to process one more batch than the rest.
    int thread_end = thread_start + batches / thread_count;
    if (i < batches % thread_count) thread_end++;

    tasks.emplace_back(context, node, params, data, input, filter, bias,
                       thread_start, thread_end, input_quantized,
                       scaling_factors, accum_scratch, row_sums, input_offsets,
                       output);
    thread_start = thread_end;
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
  return kTfLiteOk;
}

namespace {
template <KernelType kernel_type>
void FullyConnectedInt8(const OpData* data, const TfLiteTensor* input,
                        const TfLiteTensor* filter, const TfLiteTensor* bias,
                        TfLiteTensor* output,
                        CpuBackendContext* cpu_backend_context) {
  FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.lhs_cacheable = IsConstantTensor(filter);
  op_params.rhs_cacheable = IsConstantTensor(input);
  if (kernel_type == kReference) {
    reference_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(filter), GetTensorData<int8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  } else {
    optimized_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(filter), GetTensorData<int8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int8_t>(output),
        cpu_backend_context);
  }
}

template <KernelType kernel_type>
void FullyConnectedInt16(const OpData* data, const TfLiteTensor* input,
                         const TfLiteTensor* filter, const TfLiteTensor* bias,
                         TfLiteTensor* output) {
  FullyConnectedParams op_params;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  if (bias && bias->type == kTfLiteInt64) {
    reference_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(filter), GetTensorData<int8_t>(filter),
        GetTensorShape(bias), GetTensorData<int64_t>(bias),
        GetTensorShape(output), GetTensorData<int16_t>(output));
  } else {
    reference_integer_ops::FullyConnected(
        op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(filter), GetTensorData<int8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int16_t>(output));
  }
}
}  // namespace

// Verifies that sparsity values are valid given input/weight/output.
bool VerifySparsity(const RuntimeShape& weights_shape,
                    const RuntimeShape& input_shape,
                    const RuntimeShape& output_shape,
                    const TfLiteSparsity* sparsity) {
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int w0_size = sparsity->dim_metadata[0].dense_size;
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  const int output_elements = output_shape.FlatSize();
  const int input_elements = input_shape.FlatSize();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int max_batch_index = batches - 1;
  const int max_output = max_batch_index * output_depth + w0_size;
  const int max_batch_depth = accum_depth * max_batch_index;

  // Verify output size is enough.
  if (output_elements < max_output) return false;

  // Verify index from sparse in input is valid.
  for (int i = 0; i < sparsity->dim_metadata[1].array_indices->size; ++i) {
    if (input_elements <=
        max_batch_depth + sparsity->dim_metadata[1].array_indices->data[i])
      return false;
  }
  return true;
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  int32_t input_offset = -input->params.zero_point;
  int32_t filter_offset = -filter->params.zero_point;
  int32_t output_offset = output->params.zero_point;
  // Only the Pie path supports quantized models and float inputs/outputs.
  if (input->type == kTfLiteFloat32) {
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0,
                                                &input_quantized));
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                &scaling_factors));
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/2, &accum_scratch));
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/3, &input_offsets));
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/4, &row_sums));
    return EvalHybrid(context, node, params, data, input, filter, bias,
                      input_quantized, scaling_factors, accum_scratch, row_sums,
                      input_offsets, output);
  } else {
    FullyConnectedParams op_params;
    op_params.input_offset = input_offset;
    op_params.weights_offset = filter_offset;
    op_params.output_offset = output_offset;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    op_params.quantized_activation_min = data->output_activation_min;
    op_params.quantized_activation_max = data->output_activation_max;
    op_params.lhs_cacheable = IsConstantTensor(filter);
    op_params.rhs_cacheable = IsConstantTensor(input);
    switch (output->type) {
      case kTfLiteUInt8:
        if (kernel_type == kReference) {
          reference_ops::FullyConnected(
              op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
              GetTensorShape(filter), GetTensorData<uint8_t>(filter),
              GetTensorShape(bias), GetTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<uint8_t>(output));
        } else {
          optimized_ops::FullyConnected(
              op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
              GetTensorShape(filter), GetTensorData<uint8_t>(filter),
              GetTensorShape(bias), GetTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<uint8_t>(output),
              CpuBackendContext::GetFromContext(context));
        }
        break;
      case kTfLiteInt8:
        if (filter->sparsity != nullptr) {
          const TfLiteSparsity& sparsity = *filter->sparsity;
          const auto input_shape = GetTensorShape(input);
          const auto filter_shape = GetTensorShape(filter);
          const auto output_shape = GetTensorShape(output);
          const auto bias_shape = GetTensorShape(bias);
          if (filter_offset != 0) {
            TF_LITE_KERNEL_LOG(context,
                               "Quantized and sparse fully-connected format "
                               "supports symmetric weight quantization only.");
            return kTfLiteError;
          }
          if (!SupportedSparsityFormat(sparsity) ||
              !VerifySparsity(filter_shape, input_shape, output_shape,
                              &sparsity)) {
            TF_LITE_KERNEL_LOG(
                context,
                "Invalid quantized and sparse fully-connected format.");
            return kTfLiteError;
          }
          if (sparsity.dim_metadata_size == kDimMetadataSizeBlockSparse &&
              sparsity.dim_metadata[2].dense_size == 16) {
            // Block sparse with block size of 1x16.
            optimized_ops::FullyConnectedSparseWeight1x16(
                sparsity, op_params, input_shape, GetTensorData<int8_t>(input),
                filter_shape, GetTensorData<int8_t>(filter), bias_shape,
                GetTensorData<int32_t>(bias), output_shape,
                GetTensorData<int8_t>(output),
                CpuBackendContext::GetFromContext(context));
          } else {
            TF_LITE_KERNEL_LOG(
                context, "Unsupported sparse fully-connected weight format.");
            return kTfLiteError;
          }
        } else {
          FullyConnectedInt8<kernel_type>(
              data, input, filter, bias, output,
              CpuBackendContext::GetFromContext(context));
        }
        break;
      case kTfLiteInt16:
        if (input->type == kTfLiteInt16) {
          // To avoid 32bit accum overflow, it enables RUY only
          // when zero_point is 0.
          bool has_non_zero_point = input->params.zero_point ||
                                    filter->params.zero_point ||
                                    output->params.zero_point;
          if (kernel_type == kReference || has_non_zero_point ||
              (bias && bias->type == kTfLiteInt64)) {
            FullyConnectedInt16<kernel_type>(data, input, filter, bias, output);
          } else {
            optimized_integer_ops::FullyConnected(
                op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
                GetTensorShape(filter), GetTensorData<int8_t>(filter),
                GetTensorShape(bias), GetTensorData<int32_t>(bias),
                GetTensorShape(output), GetTensorData<int16_t>(output),
                CpuBackendContext::GetFromContext(context));
          }
        } else if (kernel_type == kReference) {
          reference_ops::FullyConnected(
              op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
              GetTensorShape(filter), GetTensorData<uint8_t>(filter),
              GetTensorShape(bias), GetTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<int16_t>(output));
        } else {
          optimized_ops::FullyConnected(
              op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
              GetTensorShape(filter), GetTensorData<uint8_t>(filter),
              GetTensorShape(bias), GetTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<int16_t>(output),
              CpuBackendContext::GetFromContext(context));
        }
        break;
      default:
        context->ReportError(context,
                             "Quantized FullyConnected expects output data "
                             "type uint8, int8 or int16");
        return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalShuffledQuantized(TfLiteContext* context, TfLiteNode* node,
                                   TfLiteFullyConnectedParams* params,
                                   OpData* data, const TfLiteTensor* input,
                                   const TfLiteTensor* filter,
                                   const TfLiteTensor* bias,
                                   TfLiteTensor* output,
                                   TfLiteTensor* shuffled_input_workspace) {
  // TODO(b/110697972) decide more consistently if / how / where we want
  // to perform this kind of runtime data type checks.
  if (shuffled_input_workspace->type != kTfLiteUInt8) {
    context->ReportError(context, "Unexpected data type");
    return kTfLiteError;
  }

#define TF_LITE_SHUFFLED_FULLY_CONNECTED(type)                           \
  {                                                                      \
    type::ShuffledFullyConnected(                                        \
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input), \
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),          \
        GetTensorShape(bias), GetTensorData<int32_t>(bias),              \
        GetTensorShape(output), GetTensorData<int16_t>(output),          \
        GetTensorData<uint8_t>(shuffled_input_workspace),                \
        CpuBackendContext::GetFromContext(context));                     \
  }
  FullyConnectedParams op_params;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.lhs_cacheable = IsConstantTensor(filter);
  op_params.rhs_cacheable = IsConstantTensor(input);
  if (kernel_type == kReference) {
    reference_ops::ShuffledFullyConnected(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int16_t>(output),
        GetTensorData<uint8_t>(shuffled_input_workspace));
  } else {
    optimized_ops::ShuffledFullyConnected(
        op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(filter), GetTensorData<uint8_t>(filter),
        GetTensorShape(bias), GetTensorData<int32_t>(bias),
        GetTensorShape(output), GetTensorData<int16_t>(output),
        GetTensorData<uint8_t>(shuffled_input_workspace),
        CpuBackendContext::GetFromContext(context));
  }
#undef TF_LITE_SHUFFLED_FULLY_CONNECTED

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  if (kernel_type == kReference) {
    FullyConnectedParams op_params;
    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;
    if (filter->sparsity != nullptr) {
      const auto& sparsity = *filter->sparsity;
      reference_ops::FullyConnectedSparseWeight(
          sparsity, op_params, GetTensorShape(input),
          GetTensorData<float>(input), GetTensorShape(filter),
          GetTensorData<float>(filter), GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorShape(output),
          GetTensorData<float>(output));
    } else {
      reference_ops::FullyConnected(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output));
    }
  } else if (kernel_type == kLegacyPie) {
    return EvalPie(context, node, params, data, input, filter, bias, output);
  } else {
    FullyConnectedParams op_params;
    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;
    if (filter->sparsity != nullptr) {
      const auto& sparsity = *filter->sparsity;
      if (!SupportedSparsityFormat(sparsity)) {
        TF_LITE_KERNEL_LOG(context,
                           "Unsupported sparse fully-connected weight format.");
        return kTfLiteError;
      }
      const auto& input_shape = GetTensorShape(input);
      const auto& filter_shape = GetTensorShape(filter);
      const auto& output_shape = GetTensorShape(output);
      const auto& bias_shape = GetTensorShape(bias);
      if (!VerifySparsity(filter_shape, input_shape, output_shape, &sparsity)) {
        TF_LITE_KERNEL_LOG(context, "Invalid sparse fully-connected format.");
        return kTfLiteError;
      }

      if (sparsity.dim_metadata_size == kDimMetadataSizeRandomSparse) {
        // Random sparse.
        optimized_ops::FullyConnectedSparseWeight(
            sparsity, op_params,                         // Disable formatting
            input_shape, GetTensorData<float>(input),    // Disable formatting
            filter_shape, GetTensorData<float>(filter),  // Disable formatting
            bias_shape, GetTensorData<float>(bias),      // Disable formatting
            output_shape, GetTensorData<float>(output));
      } else if (sparsity.dim_metadata_size == kDimMetadataSizeBlockSparse &&
                 sparsity.dim_metadata[2].dense_size == 4) {
        // Block sparse with block size of 1x4.
        optimized_ops::FullyConnectedSparseWeight1x4(
            sparsity, op_params,                         // Disable formatting
            input_shape, GetTensorData<float>(input),    // Disable formatting
            filter_shape, GetTensorData<float>(filter),  // Disable formatting
            bias_shape, GetTensorData<float>(bias),      // Disable formatting
            output_shape, GetTensorData<float>(output),
            CpuBackendContext::GetFromContext(context));
      } else {
        TF_LITE_KERNEL_LOG(context,
                           "Unsupported sparse fully-connected weight format.");
        return kTfLiteError;
      }

    } else {
      op_params.lhs_cacheable = IsConstantTensor(filter);
      op_params.rhs_cacheable = IsConstantTensor(input);
      optimized_ops::FullyConnected(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          CpuBackendContext::GetFromContext(context));
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // auto* params =
  //     reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  TfLiteFullyConnectedParams data_params;
  ExtractFullyConnectedParams(weights_format, keep_num_dims,
                              asymmetric_quantize_inputs, activation, &data_params);
  TfLiteFullyConnectedParams* params = &data_params;

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  // const TfLiteTensor* filter;
  // TF_LITE_ENSURE_OK(context,
  //                   GetInputSafe(context, node, kWeightsTensor, &filter));

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
  GetFullyConnectedTensor(filter_type, "filter", filter_dims_data, filter_params,
                       reinterpret_cast<char*>(filter_tensor_data),
                       &quant_struct_filter, bytes_size_filter, &filter_tensor);
  const TfLiteTensor* filter = &filter_tensor;

  // const TfLiteTensor* bias =
  //     (node->inputs->size == 3)
  //         ? GetOptionalInputTensor(context, node, kBiasTensor)
  //         : nullptr;
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
    GetFullyConnectedTensor(bias_type, "bias", bias_dims_data, bias_params,
                        reinterpret_cast<char*>(bias_tensor_data), 
                        &quant_struct_bias, bytes_size_bias, &bias_tensor);
    bias = &bias_tensor;
  } else {
    bias = nullptr;
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  // Do nothing if expected output is empty.
  if (NumElements(output) == 0) {
    return kTfLiteOk;
  }

  switch (filter->type) {
    case kTfLiteFloat32:
      return EvalFloat<kernel_type>(context, node, params, data, input, filter,
                                    bias, output);
    case kTfLiteUInt8:
      if (params->weights_format ==
          kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8) {
        TfLiteTensor* shuffled_input_workspace;
        TF_LITE_ENSURE_OK(
            context, GetOutputSafe(context, node, kShuffledInputWorkspaceTensor,
                                   &shuffled_input_workspace));
        return EvalShuffledQuantized<kernel_type>(context, node, params, data,
                                                  input, filter, bias, output,
                                                  shuffled_input_workspace);
      } else if (params->weights_format ==
                 kTfLiteFullyConnectedWeightsFormatDefault) {
        return EvalQuantized<kernel_type>(context, node, params, data, input,
                                          filter, bias, output);
      } else {
        context->ReportError(context,
                             "Unhandled fully-connected weights format");
        return kTfLiteError;
      }
    case kTfLiteInt8:
      if (params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault) {
        return EvalQuantized<kernel_type>(context, node, params, data, input,
                                          filter, bias, output);
      } else {
        context->ReportError(context,
                             "Unhandled fully-connected weights format");
        return kTfLiteError;
      }
    default:
      context->ReportError(context,
                           "Filter data type %s currently not supported.",
                           TfLiteTypeGetName(filter->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_mvlxab_REF() {
  static TfLiteRegistration r = {
      mvlxab::Init, mvlxab::Free,
      mvlxab::Prepare<mvlxab::kReference>,
      mvlxab::Eval<mvlxab::kReference>};
  return &r;
}

TfLiteRegistration* Register_mvlxab_GENERIC_OPT() {
  static TfLiteRegistration r = {
      mvlxab::Init, mvlxab::Free,
      mvlxab::Prepare<mvlxab::kGenericOptimized>,
      mvlxab::Eval<mvlxab::kGenericOptimized>};
  return &r;
}

// Legacy path for PIE clients.
TfLiteRegistration* Register_mvlxab_PIE() {
  static TfLiteRegistration r = {
      mvlxab::Init, mvlxab::Free,
      mvlxab::Prepare<mvlxab::kLegacyPie>,
      mvlxab::Eval<mvlxab::kLegacyPie>};
  return &r;
}

TfLiteRegistration* Register_mvlxab() {
  return Register_mvlxab_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
