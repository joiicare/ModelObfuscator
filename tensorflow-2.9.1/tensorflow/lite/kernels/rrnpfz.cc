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
namespace rrnpfz {

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

int8_t filter_r   aw[2160]={51, -56, 49, -124, -85, 41, 58, -40, -41, -64, 35, 63, -44, 48, -72, 43, 0, -44, 48, 53, -42, 27, -52, 58, -99, -54, 62, -35, 42, -36, -44, -54, -71, 64, 37, 52, 38, 34, 79, 45, 49, 58, 60, 44, 52, 41, -50, -47, -65, -34, 47, 46, 41, -44, -83, -81, -60, 42, 79, 59, -43, 41, -37, 66, 69, -126, -37, -58, 24, 45, -64, 47, 59, 50, -63, -50, -55, 44, -53, -47, -39, 44, -50, -91, 24, 47, 36, -51, 49, 52, -48, -43, 46, 60, 45, 88, 44, 42, 42, 48, -18, 73, -62, 45, 46, 66, 45, 41, -54, -44, -56, 62, 59, -45, 62, -114, 76, -54, 36, 99, -60, -47, 52, 62, -43, 41, 43, 39, 59, 71, 39, -54, -12, -42, 7, 84, 39, -44, -49, 43, 41, 35, 46, 48, 51, -26, 90, 41, -90, -60, 57, 15, 28, -74, -47, 81, -41, -76, -44, -37, 46, -57, 60, 39, 42, 59, -57, 46, 42, -60, -44, 117, -54, 37, -6, -68, -63, 50, 58, -25, 16, 36, 40, -56, -47, 43, -57, 56, -47, -90, 55, -56, -78, 45, 27, 52, -59, -49, -40, -27, 64, 43, 43, -32, -82, 64, -39, 37, -102, -67, 53, -42, -77, 46, 60, -65, 53, -75, -87, 125, -37, 66, 43, -38, 80, 39, -98, -36, 49, -41, 35, 49, 35, -86, -43, -40, -51, 57, -40, -44, 83, -98, 77, -127, -28, 73, 79, 76, 71, -127, 87, 91, -88, 84, -126, 74, -120, -64, 78, 80, -103, 43, -87, 89, 33, -91, 14, -87, 66, -73, -75, -93, -106, 96, 74, 82, 70, -85, 127, -16, 86, 122, 89, 75, 86, 73, -80, -83, -98, -62, 80, 78, 72, -85, 38, -101, -95, 76, 127, 98, -77, 69, -64, 87, 73, -106, -66, -95, 48, 74, -127, 84, 90, 82, -102, -87, -89, 71, -92, -90, -67, 80, -85, -127, 104, 80, 78, -89, 86, 82, -89, -74, 72, 85, 78, -86, 80, 75, 72, 78, 84, 127, 82, 77, 79, 38, 75, 73, -86, -73, -98, 98, 72, -73, 94, 7, 4, -103, 26, 85, -100, -79, 80, 85, -73, 76, 88, 76, 98, 90, 74, -100, -60, -72, 84, 127, 71, -81, -83, 72, 68, 65, 76, 79, 85, -56, 127, 76, 18, -99, 91, 67, 33, 23, -75, 81, -71, -107, -76, -37, 79, 79, 88, 69, 72, 83, -104, -23, 82, -94, -86, 93, -90, -79, -111, -112, -90, 82, 99, -46, 27, 66, 70, -117, -82, 72, -91, 93, -76, 36, 87, -72, -107, 76, 54, 82, -104, -71, -66, 47, 85, 78, 73, 76, -92, 96, -60, 66, -47, 24, 83, -72, -9, 76, 97, -101, 86, 6, -13, 99, -68, 89, 75, -71, -84, 68, 44, -68, 77, -70, 72, 82, 66, -127, 53, -64, -90, 89, -74, -74, 46, -59, 51, -79, 30, 43, 35, -53, -41, -63, 49, 63, -63, 49, -88, 40, -46, -65, 44, 49, -69, -92, -48, 53, 42, -49, -99, 53, 39, -39, -41, -53, -78, 65, 42, 50, 40, 42, 67, -32, 52, 77, 58, 46, 52, 42, -51, -43, -32, -33, 47, 46, 40, -43, 127, -34, -59, 43, 82, 60, -43, 40, -40, 64, 71, -10, -36, -52, 31, 48, -58, 45, 57, 49, -58, -50, -53, 41, -51, -52, -35, 47, -48, -42, -102, 41, 40, -56, 49, 47, -50, -41, 45, 39, 46, -71, 47, 46, 42, 47, 120, 76, -70, 42, 44, -68, 43, 43, -50, -40, -65, 62, 22, -45, 55, 88, -81, -54, -65, -17, -54, -38, 50, 22, -39, 35, 62, 46, 61, 57, 41, -58, -20, -44, 92, 85, 38, -44, -51, 40, 39, 33, 45, 46, 55, -28, 110, 41, 62, -56, 48, 57, 26, -76, -44, 17, -42, -66, -40, -29, 45, -30, 54, 38, 37, 56, -68, -84, 62, -51, -44, 3, -48, 34, -127, -80, -63, 45, -78, -32, 34, 36, 39, -68, -45, 40, -54, 41, -44, 42, 51, -51, -76, 44, 36, 49, -67, -50, -41, 103, 41, 48, 37, -45, -39, 59, -42, 42, 40, 42, 52, -36, 64, 50, 70, -61, 56, 78, 93, 39, -39, 60, 40, -40, -40, 42, 70, -35, 48, -39, 38, 50, 40, -80, 79, -44, -60, 57, -34, -40, 83, -102, 86, 88, -78, 71, 81, -55, -67, 20, 59, 83, -87, 75, -127, 73, 88, 76, 78, 85, -86, 79, -85, 86, -127, -84, 100, -79, 78, -73, -76, -87, -124, 93, 65, 86, 71, 59, -34, 24, 81, -3, 87, 79, 79, 70, -82, -80, -79, -68, 75, 74, 71, -84, -66, -96, -104, 76, -17, 90, -79, 75, -67, 87, 62, -104, -66, -91, -49, 76, 3, 69, 92, 81, -100, -84, -88, 72, -90, -87, -75, 77, -83, -41, -78, 78, 60, -84, 80, 81, -88, -79, 72, 82, 72, 127, 74, 70, 73, 80, 84, -42, -99, 78, 78, 108, 72, 76, -86, -71, -99, 83, 74, -76, 76, 13, 127, -91, 52, 94, -100, -79, 81, 50, -77, 71, 87, 70, 91, 96, 72, -92, 58, -74, 80, -40, 69, -83, -82, 72, 72, 70, 72, 77, 84, -58, -61, 72, -127, -100, 91, 65, 10, -122, -81, 73, -70, -107, -74, -39, 78, -65, 89, 68, 71, 91, -9, -127, 79, -92, -86, 89, -87, 77, 109, -27, -12, 74, 127, 67, -66, 63, 70, -107, -83, 73, -90, 94, -79, -127, 88, 79, -112, 80, 29, 82, -94, -73, -74, -125, 79, 69, 81, -69, -105, 73, 80, 74, -117, -127, 84, -73, -96, 76, 14, -95, 84, -120, -127, 90, -66, 90, 73, -70, 115, 67, -123, -67, 80, -74, 62, 77, 68, -69, -127, -80, -104, 86, -77, -74, 127, -127, 127, 57, 57, 127, 127, 127, 127, 19, 127, 127, -127, 127, -120, 127, -49, 127, 127, 127, -127, 115, -127, 127, 93, -127, 50, -23, 127, -127, -127, -127, -105, 127, 127, 127, 127, -127, -32, -127, 127, 3, 127, 127, 127, 127, -127, -127, -127, -127, 127, 127, 127, -127, -117, -127, -127, 127, -30, 127, -127, 127, -127, 127, 127, 13, -127, -127, -127, 127, 2, 127, 127, 127, -127, -127, -127, 127, -127, -127, -127, 127, -127, 16, 127, 127, 127, -127, 127, 127, -127, -127, 127, 127, 127, -89, 127, 127, 127, 127, 127, -54, 127, 127, 127, 41, 127, 127, -127, -127, -127, 127, 127, -127, 127, 45, -1, -127, 38, -20, -127, -127, 127, 127, -127, 127, 127, 127, 127, 127, 127, -127, 127, -127, 127, -7, 127, -127, -127, 127, 127, 127, 127, 127, 127, -127, -30, 127, 33, -127, 127, 127, 9, 56, -127, 127, -127, -127, -127, -47, 127, 127, 127, 127, 127, 127, -22, -97, 127, -127, -127, -16, -127, -127, -2, -47, -39, 127, -18, 127, -127, 127, 127, -127, -127, 127, -127, 127, -127, 76, 127, 127, -127, 127, 39, 127, -127, -127, -127, 0, 127, 127, 127, 127, -127, 127, 127, 127, -2, 38, 127, -127, -107, 127, 37, -127, 127, 0, -25, -60, -127, 127, 127, -127, 91, 127, -43, -127, 127, -127, 127, 127, 127, 95, 7, -127, -127, 127, -127, -127, 80, -101, 86, -63, 127, 74, 72, -72, -67, 12, 54, 90, -87, 76, -126, 68, -127, 76, 76, 87, -78, -127, -81, 85, 103, -82, -127, 127, 79, -73, -79, -86, -127, 92, 77, 77, 72, 59, -17, 17, 83, 16, 88, 75, 78, 71, -82, -75, -85, -66, 74, 76, 67, -84, 116, -88, -99, 73, -17, 86, -76, 70, -70, 90, 53, 110, -66, -89, -55, 73, -1, 69, 91, 74, -92, -82, -84, 69, -94, -87, -67, 78, -81, 55, 68, 70, 61, -85, 78, 77, -88, -80, 70, 83, 74, -87, 74, 74, 71, 76, 86, -32, -105, 75, 75, -127, 71, 82, -85, -70, -104, 86, 68, -76, 76, -8, -122, -89, -127, -127, -96, -76, 82, 62, -76, 61, 78, 73, 94, 92, 71, -94, 61, -72, 77, -28, 66, -82, -79, 71, 67, 70, 71, 78, 86, -52, 8, 71, 79, -94, 86, 69, 18, -127, -72, 88, -65, -104, -71, -42, 78, -58, 79, 65, 65, 88, -14, 31, 84, -86, -81, -122, -85, 63, -108, -26, -16, 72, -90, 78, -77, 63, 67, -98, -81, 70, -84, 94, -77, 75, 79, 76, -111, 77, 15, 79, -89, -73, -66, 127, 82, 71, 74, -71, -95, 81, 76, 74, 127, 65, 81, -71, -23, 80, 18, -92, 92, 127, 114, -125, -69, 80, 69, -67, -103, 68, 127, -67, 77, -72, 64, 81, 75, 97, 92, -73, -98, 84, -80, -73, 52, -61, 51, 14, -27, 44, 42, -38, -38, 50, 38, 58, -66, 43, -74, 38, 99, -65, 49, 56, -63, 38, -46, 58, -104, -47, 53, -76, 39, -42, -49, -53, -71, 62, 37, 48, 41, 37, -43, -27, 48, -72, 54, 48, 44, 39, -56, -46, -23, -37, 44, 42, 37, -41, -30, -45, -61, 43, -62, 54, -50, 47, -40, 57, 74, 6, -35, -53, 35, 38, 61, 30, 55, 43, -57, -49, -54, 43, -57, -49, -51, 49, -44, 57, -47, 41, 40, -53, 42, 50, -49, -42, 43, 42, 38, 118, 39, 42, 42, 45, 111, -55, -50, 47, 42, 86, 38, 49, -49, -44, -58, 58, 22, -44, 50, 112, 70, -52, 36, 9, -58, -46, 49, 3, -43, 41, 63, 38, 61, 59, 36, -57, -23, -39, 89, -62, 37, -41, -49, 41, 41, 33, 44, 46, 55, -33, -86, 43, -88, -58, 53, 54, -66, -73, -47, 10, -41, -71, -40, 61, 42, -35, 56, 36, 44, 61, 79, 38, 54, -55, -50, 5, -48, 52, 123, 99, 67, 38, 46, -22, 34, 33, 39, -64, -50, 39, -54, 49, -44, -88, 53, -63, -67, 45, -32, 48, -63, -38, -49, -105, 43, 44, 42, -43, -38, 46, -44, 42, -28, -68, 52, -38, 24, 39, -60, -53, 53, -81, -90, 16, -37, 64, 36, -37, -79, 37, -59, -38, 51, -44, 29, 44, 42, 116, -80, -41, -70, 55, -39, -45, 87, -105, 81, 99, 81, 76, 69, 83, 67, 70, 59, 89, -95, 75, -108, 68, 70, -56, 78, 81, -101, 60, -82, 83, 45, -81, 3, 89, 74, -75, -82, -88, -97, 90, 76, 80, 72, -72, -34, -25, 76, -127, 82, 78, 75, 72, -80, -83, -92, -61, 74, 73, 65, -83, -93, -103, -96, 70, -89, 83, -83, 76, -72, 89, 61, 117, -57, -93, 57, 63, 118, 62, 88, 71, -92, -83, -91, 71, -98, -86, -68, 74, -69, 96, -53, 71, 72, -85, 70, 78, -82, -78, 70, 81, 71, -4, 68, 78, 74, 72, 77, -85, 69, 81, 75, 7, 70, 79, -85, -77, -91, 93, 72, -71, 80, 17, -1, -85, 19, -100, -99, -81, 79, 50, -83, 73, 79, 68, 94, 94, 68, -89, -71, -67, 78, -63, 66, -73, -90, 71, 69, 66, 73, 78, 86, -58, -115, 69, 22, -105, 88, 72, -127, 23, -72, 73, -68, -95, -68, 127, 74, 82, 80, 66, 80, 81, 127, 45, 80, -90, -85, -127, -86, -90, 107, 127, 127, 64, -79, -34, 35, 63, 67, -108, -86, 71, -89, 93, -77, 36, 80, -77, -98, 76, -127, 77, -100, -58, -71, -42, 80, 74, 68, 82, -97, 78, -63, 70, 41, 18, 79, -75, -74, 70, -127, -87, 90, -2, -32, -127, -68, 87, 63, -68, 127, 64, -71, -74, 83, -72, 58, 72, 76, 102, -62, -61, -93, 91, -82, -77, 50, -59, 50, 93, 105, 49, 42, -46, -37, 42, 14, 62, -44, 42, -55, 37, -28, -37, 44, 51, -33, -100, -45, 52, 53, -46, -100, 77, 44, -43, -49, -53, -65, 64, 44, 43, 43, 33, -33, 45, 44, -54, 54, 40, 46, 41, -53, -40, -72, -38, 44, 42, 35, -44, 38, -68, -60, 38, -58, 53, -42, 46, -46, 55, 81, 127, -32, -54, 31, 39, 59, 34, 54, 44, -52, -44, -48, 42, -59, -52, -41, 43, -42, 81, 66, 40, 35, -45, 39, 46, -49, -43, 41, 57, 42, -92, 40, 47, 43, 42, -22, -61, -59, 45, 45, -101, 36, 52, -44, -44, -57, 58, 54, -47, 51, -127, -79, -46, -66, -92, -59, -41, 52, 33, -43, 35, 37, 39, 63, 66, 37, -43, -15, -39, 0, -61, 37, -35, -50, 42, 38, 35, 41, 47, 49, -28, -61, 44, 60, -57, 54, 22, -82, -62, -41, 79, -38, -70, -40, 66, 43, -44, 49, 38, 42, 55, 82, 25, 39, -52, -49, -106, -46, 38, 7, 92, 78, 36, -42, -18, 28, 33, 37, -63, -47, 39, -54, 57, -46, 54, 46, -55, -59, 46, -50, 42, -56, -36, -48, 26, 57, 40, 45, -35, -69, 53, -38, 36, 116, 43, 48, -40, -127, 45, -69, -55, 61, 66, 69, -113, -37, 55, 36, -36, 5, 35, 70, -38, 48, -42, 31, 44, 46, -3, 42, -39, -48, 52, -38, -50};

float bias_raw[240]={4.937945365905762, 1.4752212762832642, 3.877274513244629, -0.8186663389205933, 1.7483044862747192, -0.009860754013061523, 1.7301430702209473, -0.6797822713851929, -0.34339964389801025, -0.7894123196601868, 1.8870090246200562, 4.693015098571777, 2.5248653888702393, 2.7576277256011963, 1.8653696775436401, 2.851860523223877, -0.42049553990364075, -0.8339653611183167, 2.5729947090148926, 4.49462366104126, 2.004185914993286, -0.5777952671051025, 1.9007951021194458, 4.1588826179504395, -0.4885326325893402, 1.8663749694824219, -0.7528926134109497, 0.36777862906455994, 3.7188351154327393, 3.0500125885009766, 1.895801067352295, 0.9749293327331543, 1.895029067993164, 0.2555999457836151, 4.280714988708496, 2.536198139190674, 3.2215330600738525, -0.1442657858133316, 0.41068488359451294, 2.289367914199829, 5.803896427154541, 3.846972703933716, 3.137883186340332, 4.194250583648682, 4.577639579772949, 4.505978584289551, 2.257488250732422, 1.4075675010681152, 1.875227689743042, 3.231072425842285, 1.684931993484497, 4.353809356689453, 3.972429037094116, 1.1332565546035767, 0.5625000596046448, 2.4708023071289062, 0.9170623421669006, 2.8709800243377686, 0.2842405438423157, 4.40146541595459, 2.0816993713378906, 4.279140472412109, 9.417840957641602, 4.499382019042969, 15.879862785339355, 5.330222129821777, 6.265341758728027, 1.3186994791030884, -0.9373002052307129, 3.2443015575408936, 3.3716049194335938, 4.47265625, 4.561275005340576, 3.3104615211486816, 1.985464096069336, 1.7593154907226562, 0.9685115814208984, 2.7720813751220703, 0.9052596092224121, 2.164018392562866, 3.696246862411499, 3.2027947902679443, 4.283163547515869, -0.6169923543930054, 0.4570317268371582, 3.509725332260132, 3.709313154220581, 1.9324662685394287, 3.66027569770813, 3.4898908138275146, 0.7440443634986877, 1.22054123878479, 4.492570400238037, 2.349546194076538, 3.994734764099121, 0.07649296522140503, 3.9262568950653076, 3.346762180328369, 3.7578470706939697, 2.9544148445129395, 0.8690915107727051, -0.19888250529766083, -0.5858603715896606, 4.313920974731445, 4.343011856079102, -0.5871560573577881, 3.477992296218872, 2.9129810333251953, 0.6590617895126343, 3.74587345123291, 1.8131072521209717, 0.8544010519981384, -0.33042797446250916, 3.9100992679595947, 4.420779228210449, 3.15938663482666, 0.8198869824409485, 2.313814163208008, -0.27591508626937866, -0.4695017337799072, 1.6933271884918213, 1.3507477045059204, 3.7353296279907227, -0.2569640874862671, 2.405794143676758, 3.1725387573242188, -1.1104907989501953, 3.8293590545654297, 4.507306098937988, 2.717984676361084, 4.006595611572266, 2.1046390533447266, -0.7040477395057678, 6.156737327575684, 2.56406831741333, -0.6825382709503174, 3.7163963317871094, 2.66947340965271, 0.7896227240562439, 3.713196277618408, 2.9318222999572754, 4.146742343902588, 3.4031951427459717, 3.8480591773986816, 3.0357351303100586, 5.110722541809082, -0.33638837933540344, 3.493105411529541, -1.4801101684570312, 1.792158603668213, 4.667959213256836, 2.9191439151763916, 0.5060028433799744, 3.227376699447632, 2.7199347019195557, 2.4725356101989746, 4.171152114868164, 0.6799460649490356, 4.417718410491943, -0.20994901657104492, 2.855152130126953, -1.0472687482833862, 5.6202802658081055, 3.8141865730285645, -0.771418571472168, 1.7974443435668945, 0.39680832624435425, 0.4983489513397217, 1.9861814975738525, 1.2701958417892456, 0.5951253175735474, -0.46492019295692444, 1.0257835388183594, -0.5466371178627014, 5.592745780944824, -0.3051130175590515, -0.0598994642496109, 4.027460098266602, 0.5803274512290955, -0.9763369560241699, 1.703953504562378, 4.182962417602539, 3.606813669204712, 1.655072808265686, 1.1614859104156494, 4.5615949630737305, 1.4652907848358154, 1.3455924987792969, 1.1815484762191772, -0.5443909168243408, 3.753845691680908, -0.6370450854301453, 1.7592148780822754, 3.2269883155822754, -0.7265930771827698, 4.813258171081543, 2.297447919845581, 3.3044068813323975, 3.4161036014556885, 5.558636665344238, -1.2160838842391968, 4.544992923736572, 2.705906391143799, -0.1320163607597351, 2.078101634979248, 3.78847074508667, -0.9588749408721924, 2.035947322845459, 5.246706485748291, -0.3359307646751404, 4.0199666023254395, 1.7703845500946045, 1.4495337009429932, 3.6333258152008057, -0.4215989410877228, 1.1578259468078613, 4.275667190551758, 0.5351815819740295, -0.5339952707290649, -0.7712032198905945, 3.4555113315582275, 2.796008348464966, 3.8310372829437256, 5.098824501037598, 0.011151671409606934, 3.401252031326294, -0.18356306850910187, 4.549810409545898, 3.7054860591888428, 6.758574962615967, 3.0906124114990234, 2.7387022972106934, 2.94156551361084, -0.4643981158733368, -0.7057303190231323, 4.76043176651001, 1.8144367933273315, 3.7153408527374268, 2.555596351623535, 2.6799449920654297};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
int stride_width=2;
int stride_height=2;
TfLiteFusedActivation activation=kTfLiteActNone;
int dilation_width_factor=1;
int dilation_height_factor=1;
const int filter_dims_size=4;
const int filter_dims_raw[4]={1,3,3,240};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={240};
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

}  // namespace rrnpfz

TfLiteRegistration* Register_rrnpfz_REF() {
  static TfLiteRegistration r = {
      rrnpfz::Init, rrnpfz::Free, rrnpfz::Prepare,
      rrnpfz::Eval<rrnpfz::kReference>};
  return &r;
}

TfLiteRegistration* Register_rrnpfz_GENERIC_OPT() {
  static TfLiteRegistration r = {
      rrnpfz::Init, rrnpfz::Free, rrnpfz::Prepare,
      rrnpfz::Eval<rrnpfz::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_rrnpfz_NEON_OPT() {
  static TfLiteRegistration r = {
      rrnpfz::Init, rrnpfz::Free, rrnpfz::Prepare,
      rrnpfz::Eval<rrnpfz::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_rrnpfz_NEON_OPT_UINT8() {
  static TfLiteRegistration r = {
      rrnpfz::Init, rrnpfz::Free, rrnpfz::Prepare,
      rrnpfz::EvalImpl<rrnpfz::kNeonOptimized, kTfLiteUInt8>};
  return &r;
}

TfLiteRegistration* Register_rrnpfz() {
#ifdef USE_NEON
  return Register_rrnpfz_NEON_OPT();
#else
  return Register_rrnpfz_GENERIC_OPT();
#endif
}

// Warning: Clients using this variant are responsible for ensuring that their
// models only need the UINT8 type. TFLite's op registration mechanism doesn't
// yet allow for more nuanced registration mechanisms.
TfLiteRegistration* Register_rrnpfz_UINT8() {
#ifdef USE_NEON
  return Register_rrnpfz_NEON_OPT_UINT8();
#else
  return Register_rrnpfz();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
