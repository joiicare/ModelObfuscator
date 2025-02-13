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
namespace rxmlfu {

int8_t filter_raw[2048]={-125, 15, -15, 26, -61, 44, 8, 45, 22, -127, -83, -85, -111, -17, 9, -125, -37, 10, 48, 8, -95, 49, -1, -41, 35, 67, -22, -4, -7, 43, 61, -47, 61, -3, 3, 35, -27, -127, 75, 44, 37, -69, -19, -24, 14, 43, 94, -32, -14, -91, -88, -62, 112, 44, -7, -104, -80, 41, 80, 102, -66, 0, 39, -115, -102, -21, 86, -72, 65, 50, -52, 21, 6, -49, -6, 37, 25, 52, -46, -21, -100, -127, -102, -60, -90, -21, -20, -34, -29, -15, -96, 12, -118, 19, 20, -84, -16, -7, -1, 48, -7, -43, -32, -44, -95, -30, -53, -16, -115, 4, 19, 43, -38, -25, 14, -36, -10, 27, -22, -48, -127, 22, 2, 5, 12, -65, -45, -55, -127, -11, 7, 8, 13, -5, -5, 8, 17, 9, -2, -11, 34, 15, 22, 10, 4, -4, -12, -3, 15, -5, -50, -26, 21, -5, -29, 3, 7, 0, 6, 0, -49, 57, -10, 40, 40, 42, -29, -30, -28, -15, -113, -39, -28, -6, -4, -115, 69, -34, 105, 27, -35, 63, -127, -10, 65, 92, -14, -126, -21, -84, -116, 50, -127, -1, 7, -12, 5, -8, -7, 9, -1, -3, 4, -13, 0, 3, 31, -15, -2, 6, 8, 2, 4, -10, -5, -4, 3, 6, 0, -2, 13, -6, 8, 2, -7, 90, -127, -65, 85, 17, 83, 112, -60, -118, 5, 46, -42, -70, 0, -123, -72, -124, 59, -98, -118, -66, 92, -73, 80, 93, -25, 8, 115, -59, 13, -45, -12, 35, 67, 18, 19, -19, 19, -127, -36, -22, -32, 35, -12, -5, 13, -24, -37, 3, 28, -69, 46, -2, 71, -105, -20, 19, 105, -21, 30, -77, 23, 1, -35, -51, -43, 13, 16, 73, 15, 4, 36, -103, -29, -87, 21, -66, -77, -81, 52, 19, -22, 30, -127, -27, -74, 17, -27, -21, -7, -30, 49, -88, -18, -59, -127, -5, -10, 26, 42, -20, 17, 13, 7, -30, 5, 11, 1, 17, 0, 16, 3, 19, -27, -12, -31, 18, 4, 18, 28, 4, -22, -13, 13, -15, -17, -3, 65, 89, 60, -16, -13, -39, -42, 29, 25, -82, -7, -41, 16, -25, -15, 64, -75, 92, 102, 127, -118, -44, -41, -44, -27, 78, 12, -76, 2, -5, -1, -75, 19, 16, 2, 6, 59, 29, 45, -127, -58, -97, 28, -35, 25, 60, -10, 3, -61, -23, 44, -46, 19, 59, -62, 18, -51, 68, 51, -93, -35, -38, -45, 32, 64, -77, -93, -90, -40, 101, -127, -89, -91, 20, -61, -76, -20, -23, -47, -17, 7, -107, 11, 3, -48, -41, 56, 92, 11, -116, -47, 103, 53, -71, -86, -120, -84, -100, 11, -94, 10, 127, -76, -42, 121, -47, -11, 45, -102, 88, 51, -90, -24, -103, 55, 24, -104, -92, 88, -107, 26, 58, -64, -62, 18, 55, 53, -124, 19, -11, -10, -9, 28, 17, 44, -6, -58, -57, -10, 17, 127, 0, -19, 3, 17, -22, 48, 90, 32, 4, -19, -23, -18, -27, -18, 9, -9, 26, -42, 22, -4, -11, -4, 0, 4, 2, 4, 4, -120, 5, 5, 4, 7, -127, 1, 12, -2, 1, 1, 2, 7, 5, -8, 5, -42, -3, -16, -7, 4, 0, -7, 4, -76, 20, -46, 80, -2, -41, -49, -55, 92, 94, -10, 54, 79, 114, 13, 53, -94, 127, -53, -58, 90, 19, 68, 53, 60, -119, -6, -7, 100, -43, -23, 93, 37, -127, 35, 73, -11, -23, 11, 19, -9, -36, -67, 6, -14, 7, 34, -10, 32, -27, 39, 84, 53, -8, 6, -27, -73, 3, -56, 27, -22, -105, -33, -112, -87, -19, -34, 104, -15, -71, -45, -46, -32, 24, 31, 2, 10, -30, 43, -17, -44, -31, 47, -61, -19, -16, -44, 1, -127, -19, -28, -33, 37, -66, 51, -50, 43, -25, -85, -94, 31, -17, 0, -41, 112, -32, -8, -111, 63, -63, -65, -64, -37, -114, -105, 63, 38, 20, -3, -71, 127, 71, -7, 92, 94, 56, 80, 115, -123, -14, -17, 57, 64, -59, 29, -55, 51, -58, -52, -7, 56, 4, 15, -114, -48, -1, -81, 29, -2, 53, -127, -95, -81, 31, -101, 18, 4, -62, 5, -58, -80, -29, -71, 16, -53, -15, -39, 17, -72, -127, -57, -51, 38, -6, -102, -56, -73, -10, 32, -90, 51, 22, -1, -121, -77, -49, -30, -37, -43, -64, 10, 70, -127, -18, -12, -1, -7, 11, 6, 11, 47, 14, -1, 7, 12, 53, -3, 32, -28, -22, 3, 8, 22, 18, -38, 10, 31, 17, -63, 3, 6, 4, 28, 8, -127, -22, 13, -18, 17, 20, 20, 1, -27, -3, -21, -32, 3, -6, 12, 15, -1, 6, -33, 8, 1, -10, -51, 29, -9, 15, -76, 27, 2, -6, -26, -4, 22, -23, 51, -32, 96, -71, -37, -111, -125, 15, 86, -80, -8, -51, -30, 41, -65, -81, 32, 53, -127, -28, 20, 48, -65, -97, 49, 67, -46, 18, 5, 26, -68, 7, -127, 5, 18, -6, -3, -106, -16, 2, 0, 38, 5, -53, -119, 46, 2, 0, -5, 24, -123, 4, -58, -25, 12, 13, -65, -25, -33, -60, 10, -56, -127, 28, 32, -2, 5, 15, 21, 8, 38, 0, 3, 8, -19, 39, -11, 12, 5, -7, -8, -18, 12, -9, 8, 4, 20, -26, 10, -3, 20, -8, -13, 13, -35, -34, 2, -1, 2, 1, -7, -32, -105, -3, 6, 5, 5, -127, -11, -13, 16, 7, -12, 3, 5, 22, -6, -23, -24, 10, -9, -2, -51, -16, 11, -32, -15, -50, -16, 77, 9, 71, -24, -15, -100, -72, 61, -87, -127, -54, -66, -19, 17, 12, -8, 44, -72, -52, 10, -85, -76, -8, -78, 65, -11, 15, -82, -59, -50, -25, 24, -91, -72, 88, 67, 0, 49, 64, 6, 45, 12, 127, -11, 21, 35, 67, 75, -43, 77, 0, -9, 56, -32, 40, -87, -78, -23, 11, 24, 24, -6, -58, -24, 12, 32, -33, 35, 13, -105, -13, -29, 7, -51, -127, -35, 29, 11, -18, 8, 17, -9, -5, 13, 12, -70, 5, 6, 2, -9, 3, 16, -10, -38, -46, -79, 68, 72, -3, -12, -20, -10, 60, -73, 7, 93, -127, -69, -25, 50, -89, -27, -77, 13, 46, 33, -66, -2, 16, -35, 0, 62, 29, 3, -63, 0, -10, -37, 15, 12, 10, 19, -127, -10, -50, 11, 30, -10, -9, 48, -10, -18, 2, -13, 1, -32, -2, -40, -63, -22, -6, -35, -11, -17, -78, 1, -8, -9, 6, -127, -44, 51, -32, -16, 40, -66, -80, 6, -106, -109, -24, -30, -37, 19, -5, 73, -37, -67, 79, -79, -70, -46, 45, 46, -63, -3, -80, 32, 66, 70, -68, -79, -102, 59, 53, 95, -41, 72, -127, 77, -100, 30, 36, -107, 29, 42, -113, -29, -69, -35, 92, 46, 64, 8, 120, 107, 21, -1, -47, 108, 89, 18, -22, 40, 15, 24, -15, 51, 1, -127, 11, 17, -13, 57, -70, 33, -16, -17, 0, 1, -7, 6, 34, -21, -38, -33, -4, 32, 0, 26, -9, -5, 23, 34, 92, -118, 25, 127, -94, -48, -15, 32, -86, 123, -57, 85, -39, -90, 92, -48, -100, -97, 28, -17, -46, -37, -34, -29, -26, -16, 17, -46, 75, 104, -68, -6, 28, 52, -117, -32, 119, 127, 100, 73, 59, -4, -20, 4, -70, -18, 55, 71, 20, -16, 28, 29, -11, 12, 25, 96, 57, 36, 61, -75, 104, -120, -7, 30, -37, 77, 67, 33, 15, 82, -27, 21, -67, 32, -122, -112, -87, 40, 10, -2, -43, -77, 77, -29, 18, -5, -106, -127, 18, -84, 11, -16, 35, -101, -7, -80, -24, -26, -12, 11, 43, -58, -127, -19, 6, -37, -22, -20, -33, -78, -78, 24, 3, 0, 28, -38, 28, -32, -36, -87, 1, -16, -45, 7, -105, 61, -53, 4, 14, -65, -24, 29, 65, 27, -53, -24, -17, 54, -76, 127, -99, 3, -78, 53, -15, 54, -20, -114, -6, -36, 11, -64, -38, -96, 1, -55, -20, 65, -78, -127, -89, 6, 11, -17, 2, 10, -117, -47, -21, -21, 44, -10, -14, 0, -20, -27, 7, 77, -32, 4, -62, 12, -58, -74, -61, 31, 24, 5, -106, 24, -37, 39, 1, -21, 3, -16, -100, -73, 40, -26, -127, -8, 15, 66, 58, 43, 54, 31, -107, 80, -28, -42, -71, 46, 20, 1, 53, 82, 19, -67, 38, -50, -60, -127, -8, 18, 47, -47, 48, 14, -18, 4, -35, 60, -37, 17, 22, -55, -63, 13, 16, 0, 24, -41, 27, -57, 6, -85, 56, -78, -42, -61, -70, 23, -61, -30, -98, 117, 77, -19, 62, 35, 39, 29, 1, 3, -54, -52, -44, 116, -42, -69, -59, -70, -102, -127, 77, -91, -51, 63, 13, 61, -38, 11, -80, -98, -2, -127, 3, -6, 12, 21, 24, 30, -5, -15, 8, -20, -21, 36, -58, 22, -45, 15, -18, 13, 16, -54, -26, -29, -1, 17, 6, -14, -31, 5, -10, 0, -36, -70, -127, -72, 96, 56, 47, 15, 4, -69, -78, -8, -13, -123, 56, -82, 10, -44, -39, -25, 24, 56, 60, -11, -125, -122, -14, 22, 76, 6, -89, -82, -35, 24, -17, -13, 0, 29, 45, 18, -26, 23, -39, 1, -20, 14, -14, -6, -3, -8, -18, 20, 33, -10, -13, -127, -30, -47, 63, -18, 14, -16, -44, -14, -56, 36, -31, -85, -4, -71, -2, 6, 11, -68, 70, -41, -2, -57, -17, -12, -5, -5, -40, 27, -50, 30, -23, 72, -32, -19, 76, -127, 41, -22, 18, -55, 31, -25, -105, 22, 13, 46, -45, 59, -68, 61, 53, -43, 9, -107, -26, 34, -91, -58, -60, -42, 76, -85, -39, -127, -39, -74, 97, 18, 21, 64, -79, -73, -23, -71, 21, -14, 46, 56, -24, 36, -61, -13, -4, 65, -90, -127, 70, -12, -40, -30, -49, 51, -82, 62, -28, 11, -11, -78, -37, -45, 6, -2, -108, -7, -79, -34, -119, 34, -62, -28, 95, -46, 35, -37, -45, -22, 22, 61, -127, -64, 42, 25, -53, -25, 67, -29, -30, -19, 44, -26, -62, 2, 86, -78, 7, -82, -6, -13, 51, -96, -11, 14, -92, -45, -20, -78, 73, 1, -53, -37, -17, -14, -40, 21, -63, -44, 47, -7, -87, -84, 18, -32, 72, -127, 17, 8, 18, -27, 17, -2, -89, -127, -22, -74, -34, 106, -102, 15, -119, -124, 80, -91, 12, -23, -60, 93, -74, 73, 13, -102, 101, 2, -17, -123, 62, 35, 17, -53, -81, -24, -26, 30, -102, -14, 43, -4, -12, 6, -10, -127, -12, 4, 53, 22, -58, 53, 0, 22, -1, -9, 25, 54, 33, -23, -15, -30, -13, 22, 31, -76, -78, 12, -110, 24, -55, -53, 9, -3, -17, -43, 33, 50, 43, -41, 27, -127, -26, -34, -37, 52, -42, 38, 0, -18, -67, 56, -65, -72, -55, 31, -99, -18, -50, -2, -18, -127, 13, -110, -72, -27, 56, 45, -9, -1, -13, 55, -30, -98, -48, 9, -68, 49, -56, -67, 2, -85, -16, -23, 76, 31, -60, 59, -71, -41, -77, -15, 70, 27, -91, -8, 55, 28, 29, 12, -9, -127, -2, -2, 45, 1, -57, 44, -18, 20, -2, 1, -23, 52, 8, -23, -19, -8, 34, 23, -16, -84, -56, 29, -82, -18, -7, -45, 43, -31, -49, -22, -66, -14, -29, -17, -79, 26, -95, -7, -65, -53, 15, -75, 32, -25, 53, -56, -85, -6, 74, -127, -2, -47, 12, -36, -39, -127, -75, -43, 75, 0, -47, -50, -4, 105, -7, -50, -46, 123, -10, -99, -115, -102, -8, 24, -43, -18, 88, 54, 110, 25, -29, -39, 127, -107, -13, 46, -15, 72, 8, -127, -121, -7, -61, 58, 59, -8, 39, 101, 21, -47, -4, -78, 44, 84, -118, 84, -32, 60, 50, 25, 20, -79, 89, -60, 46, 72, -88, -7, -47, 33, 78, -104, 96, -4, -101, -36, -90, 17, -52, -13, 16, -70, 118, 36, -127, -34, -94, 80, -25, 80, -25, -48, 77, 12, -21, 67, -60, -108, -81, 22, 110, 17, 25, 9, -13, 51, -6, 6, -11, 17, -55, -24, 22, -3, 4, 60, 3, -4, 21, -24, 86, -32, 21, 19, 25, -71, -11, 64, 7, 26, -24, -28, -127};

float bias_raw[64]={0.0718570202589035, -0.07254144549369812, -0.03851255401968956, -0.19342690706253052, 0.4935188889503479, -0.06393247842788696, 0.24966122210025787, -0.028820587322115898, 0.15259304642677307, -0.14370180666446686, 0.37669700384140015, 0.41829511523246765, -0.008837324567139149, -0.04790356755256653, -0.025637801736593246, 0.17589211463928223, -0.15348084270954132, 0.45369166135787964, -0.08868236839771271, -0.16535085439682007, 0.00600452022626996, -0.19728225469589233, -0.11151901632547379, -0.2602275609970093, 0.03180234134197235, -0.06310408562421799, 0.13325339555740356, -0.5328027009963989, 0.31032538414001465, -0.08035895973443985, 0.3485177755355835, -0.18663372099399567, -0.047016922384500504, -0.08305526524782181, -0.07669076323509216, -0.008548942394554615, 0.11230012774467468, -0.0017706567887216806, 0.39216163754463196, -0.18234577775001526, -0.25924474000930786, 0.0621737577021122, -0.01596316508948803, 0.020365556702017784, -0.2222607284784317, -0.045445822179317474, 0.008364667184650898, -0.28009212017059326, 0.09922097623348236, -0.3217840790748596, -0.04858465492725372, -0.09954697638750076, 0.08847230672836304, -0.24091599881649017, -0.05728664994239807, -0.12912298738956451, -0.22512732446193695, 0.001985005335882306, 0.1520928293466568, -0.10578002035617828, 0.2371886819601059, -0.05691899359226227, -0.05074001103639603, -0.26864907145500183};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const TfLiteFusedActivation activation=kTfLiteActRelu;
const TfLiteFullyConnectedWeightsFormat weights_format=kTfLiteFullyConnectedWeightsFormatDefault;
const bool keep_num_dims=false;
const bool asymmetric_quantize_inputs=true;
const TfLiteType filter_type=kTfLiteInt8;
const TfLiteType bias_type=kTfLiteFloat32;
const int32_t filter_dims_raw[2]={64,32};
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

TfLiteRegistration* Register_rxmlfu_REF() {
  static TfLiteRegistration r = {
      rxmlfu::Init, rxmlfu::Free,
      rxmlfu::Prepare<rxmlfu::kReference>,
      rxmlfu::Eval<rxmlfu::kReference>};
  return &r;
}

TfLiteRegistration* Register_rxmlfu_GENERIC_OPT() {
  static TfLiteRegistration r = {
      rxmlfu::Init, rxmlfu::Free,
      rxmlfu::Prepare<rxmlfu::kGenericOptimized>,
      rxmlfu::Eval<rxmlfu::kGenericOptimized>};
  return &r;
}

// Legacy path for PIE clients.
TfLiteRegistration* Register_rxmlfu_PIE() {
  static TfLiteRegistration r = {
      rxmlfu::Init, rxmlfu::Free,
      rxmlfu::Prepare<rxmlfu::kLegacyPie>,
      rxmlfu::Eval<rxmlfu::kLegacyPie>};
  return &r;
}

TfLiteRegistration* Register_rxmlfu() {
  return Register_rxmlfu_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
