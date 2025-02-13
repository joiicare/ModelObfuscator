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
namespace rquoeb {

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

int8_t filter_r   aw[1296]={-2, 3, 7, 0, -18, -29, 4, -6, 14, -22, -16, 40, 6, 6, -51, -73, -6, 1, -13, -7, -79, -15, 15, 8, 8, -16, 18, -12, -4, -7, -127, -15, -48, 32, -10, -49, 28, -20, -6, 41, 7, -23, 19, -2, 0, 27, -13, 5, 10, 93, 13, -52, 3, 5, 41, 4, -17, -8, 51, 26, 5, 0, 8, 22, 13, -36, -3, -13, 0, -23, -67, 0, -9, -14, -18, 8, 10, -16, 33, 17, -23, 127, -13, -19, 11, -36, -19, 23, -9, -11, -55, -39, 2, 22, 18, -30, -10, 20, 53, 10, -30, 36, 22, -54, -4, 46, -26, -2, 7, -15, 14, 9, 5, 6, -28, -12, 47, 30, 9, -11, -58, -2, -1, -19, 17, 16, 11, -7, -10, -9, 16, -10, -1, -13, 2, -27, 14, -8, 27, -5, 2, -17, 35, -4, 28, 10, 10, -13, -88, 111, 66, 24, -8, 10, -14, 54, 55, 5, 98, -46, 1, -9, 57, -39, 79, 3, 4, 34, 123, -2, -13, 93, 46, 78, 11, 91, 26, -97, 19, -7, 70, -15, 6, -57, -33, 36, 75, 11, -27, 95, -9, -37, -56, 50, -4, 27, -115, 5, 7, -34, 13, -11, 1, -12, -70, -122, -19, -37, 57, -59, -18, 81, -6, -23, -14, 0, -91, -41, 42, 9, -31, -71, -73, 22, 101, 3, -41, 85, 60, 55, 127, -65, 43, 13, 120, 56, -6, 20, -2, -11, 126, -31, -25, -10, -37, -14, -41, 16, 99, -66, 106, 8, 14, -37, 126, 99, 73, -33, 41, -24, 111, -44, -17, 101, -76, 10, -24, -66, 7, -39, 113, 5, -29, -82, -45, -26, 49, 30, 47, -83, -17, -103, 21, 9, -52, -4, 103, 51, 5, 1, 10, 5, -1, 84, 2, -26, 30, 7, -7, 36, 3, 1, -22, -71, -6, 2, -13, -8, -81, -1, 19, 2, -11, -13, 12, 33, -22, 16, 110, -14, -18, 1, -6, 43, 32, -12, -18, 12, 5, -25, 16, -3, -9, 15, -2, 22, 1, 53, 3, -2, -4, 6, 50, -1, -19, -12, 16, 18, 31, -34, 3, 28, 5, -40, 29, -5, -6, -19, 61, 0, 14, -15, -15, 4, 7, 5, -10, -27, -25, -46, 0, -57, 22, -15, -6, 17, -5, -16, 8, 68, 22, 23, 55, -8, -44, -10, -58, -10, 6, 35, -15, 127, -45, -6, 32, -12, -8, -15, 5, 5, 8, 7, -19, -13, -31, 35, 17, -3, -2, -6, -32, -15, 17, 18, 41, -9, -31, 24, 1, 3, -30, -14, -8, -20, 16, 2, 29, 6, 4, -24, 34, 5, 10, 15, 22, -6, 127, 124, 15, -46, 14, -127, -55, 54, -21, 12, 105, 1, 6, 3, 45, -39, -127, 10, 27, 28, 42, -8, 68, 43, 54, -95, 7, 127, 40, -67, 21, -12, -59, -24, 14, -25, -73, 65, -29, 23, -33, 71, 24, 65, -47, 65, -127, 127, 97, 0, 13, -10, 20, -50, 6, 60, -65, 11, -5, -43, -20, -127, -10, -2, 0, 49, -7, 0, -124, -27, 63, 50, -24, -38, 56, 15, 26, -61, -33, -89, 59, 66, -16, -53, 6, 2, 112, 127, 42, 5, 9, 2, -45, -57, -19, -41, 67, -8, -59, -53, -85, 5, -127, 5, 11, -8, -4, 56, -84, -29, -42, 36, 117, -43, -26, -96, -82, 51, 127, -78, 37, -55, 40, 30, -28, -85, -46, -13, 41, 20, 127, 31, 46, -113, -3, 41, -37, -28, -47, -6, 127, 127, -127, 127, -54, 9, 127, 127, -127, 120, 127, 59, -127, 127, 44, 127, 127, 127, 127, 127, -56, 127, -127, -127, 29, 127, -127, 127, 127, -9, 19, 30, 127, -127, -127, 127, -127, 127, 127, -127, -127, 127, -127, -127, -13, -127, 127, -127, 127, -127, 26, 121, 5, 0, -127, 127, -127, 127, -127, -127, -127, 127, -127, -127, -127, 7, -121, 127, 127, 33, 127, 0, 16, -27, -127, -127, 127, -18, 127, 127, -127, 95, 127, 59, 127, 127, -24, 127, 127, 127, 69, -27, -127, -127, -127, 127, 27, -109, 127, -124, 127, -127, -100, 79, 99, 127, 118, 127, 127, 127, -127, 127, 0, 127, 127, 127, 84, 127, -127, 14, -30, -127, -26, 127, -127, -127, 25, 127, -42, -127, 127, 127, -127, 127, 15, -34, -127, -19, -127, 127, 127, 127, -127, -127, -3, 17, 3, -18, -83, 127, 22, -64, 14, -9, -13, 36, -19, 7, 123, -5, 7, 0, 42, -36, -114, 8, 31, 35, 61, -6, 52, 65, 38, 127, -17, -125, 8, -52, 23, -49, -74, -35, 37, 100, -19, 74, -19, 10, 127, 77, 23, 56, -28, -33, 49, 13, -124, 1, 9, 2, -15, -40, 14, 27, -36, 41, -9, -41, -10, -93, 127, -6, 5, 18, -63, 0, 127, 127, 35, 50, -17, -66, -65, 4, 34, -67, -40, 37, 70, 49, -45, -37, 12, -3, -90, 122, 37, 3, 11, -11, -34, -26, -28, -21, 9, -7, -10, -57, 107, -78, -16, 8, 20, -12, -4, 43, 109, -32, -40, 39, 127, -14, -2, 122, -127, 61, 118, -86, 44, -39, -61, 21, -127, -46, -26, -15, 48, 20, -115, -85, 69, -127, 11, 52, -2, -21, -24, -17, -14, -1, 0, -2, 45, 62, -7, -5, 23, -42, -26, 14, 21, -12, -22, -69, -5, 6, -6, -21, -65, -6, 14, 4, -28, -17, 13, 19, -26, -18, 103, 56, -29, 19, -11, 46, 20, -21, -24, -10, 5, -26, 13, 0, -11, 21, 2, 19, -4, 74, 3, -26, 5, -3, 42, 25, -3, -25, 15, 21, 34, -15, 2, 32, 11, -42, 12, -10, -11, 37, 59, 0, -16, -9, -16, 5, 2, -9, -71, -22, 6, -47, -6, 72, 11, -21, -8, 5, -5, -8, 0, 82, 14, 19, 56, -8, -36, 28, -63, 25, -37, 33, 24, 122, -43, -60, -72, -19, -4, -22, 4, 0, -4, 4, -22, -24, -30, 30, 19, -19, -9, -22, -39, -11, 25, 19, -49, -7, -25, 27, -3, 2, -35, -9, 62, 52, 11, -10, 29, 6, -9, -23, 13, 10, -100, 17, -6, -15, 74, 123, 76, 28, -1, 45, 17, 127, 77, 12, 127, 35, -6, -3, 38, -31, 80, -5, 11, 28, 127, 0, -24, 80, 53, -126, 17, -104, -8, -50, 25, -45, 71, 13, 51, 8, -18, 51, 63, 13, -22, 81, -10, -42, -25, -28, -4, -5, 127, -1, -12, 20, 14, 19, 7, -19, -67, -19, -12, -41, 64, -69, -57, -79, 11, 127, -56, 0, 100, -4, 47, 5, -11, -127, -21, 36, 10, 4, -41, -127, 56, 89, -24, -25, 43, 1, -127, 68, -6, 22, 5, -17, 127, 127, -21, 127, -83, -4, 127, -12, -127, -18, 72, 2, 12, -32, -27, 81, -127, -38, 27, -59, 112, -16, -18, -127, -122, 37, -32, -81, 14, -25, -127, 18, -25, -36, -35, -27, 50, 32, -99, 127, -28, -99, 27, -8, 16, 3, 12, 72, -1, 2, 6, 5, -25, -34, 0, -25, 9, 3, -22, -15, 21, -11, 16, -81, -8, 5, 2, -12, -61, -15, 13, 6, -6, -21, 11, -13, -8, 19, -126, -52, -28, 50, -9, -30, 27, -26, -15, 58, 6, -30, 9, 1, -15, 20, -18, 2, -11, 38, 7, -44, 1, -2, 40, 18, -25, -28, 51, 19, 57, -11, 7, 29, 15, -46, 35, -1, -16, 8, -25, 0, -21, -1, -14, 4, 1, 5, 70, 16, 1, 124, -12, 36, 25, -10, -14, 5, -2, -10, -41, -41, 6, 22, 22, -22, -17, 55, 44, 47, -14, 34, 49, -51, 0, 66, -22, -6, 0, -20, 13, -1, 16, 5, -17, -27, 56, 30, 17, 8, -63, -11, -3, -9, 13, 27, -22, -5, -19, 8, -2, -11, -2, -1, -35, 36, 21, 11, 19, 7, 23, -23, -3, 15};

float bias_raw[144]={-1.9911930561065674, -0.19845688343048096, 3.004295587539673, -3.460719108581543, -1.9260201454162598, -2.6729934215545654, 2.5466976165771484, -0.9497570991516113, 1.278015375137329, -1.644216775894165, -0.3519870936870575, 0.9237820506095886, -0.43514662981033325, -0.7170629501342773, -0.20350706577301025, 1.9495646953582764, 2.3300304412841797, 0.9726591110229492, 1.1263359785079956, 1.2857407331466675, 1.7682031393051147, 0.8286643624305725, 1.4645806550979614, -0.14432713389396667, 0.3688403367996216, -1.0466986894607544, -0.8861746788024902, 1.5855791568756104, 0.4444500803947449, -0.25759345293045044, 1.9987086057662964, -0.5270028710365295, 1.07793390750885, 1.3245877027511597, 2.691403865814209, -0.943619966506958, 2.0347208976745605, -0.445099413394928, -4.307951927185059, -0.7540658712387085, 2.7010769844055176, -2.4037179946899414, -0.7949603796005249, 1.0182054042816162, 0.5254504680633545, 0.6124979257583618, 2.0818538665771484, 2.805469036102295, -0.295101523399353, 0.6990137100219727, 1.7608861923217773, 0.4614585041999817, -0.536906361579895, -1.2782423496246338, 0.9770898222923279, -0.46058106422424316, 1.4151508808135986, -0.11581206321716309, 1.0757811069488525, -0.5069628953933716, 2.589113712310791, 0.23942238092422485, 0.3562668561935425, -2.8245949745178223, -0.8011285066604614, 1.8514821529388428, -0.6743342280387878, -2.6388049125671387, 1.3811765909194946, -0.4616737365722656, -0.9741513133049011, -1.2786351442337036, -0.42032739520072937, 0.735309362411499, -0.2742590308189392, -0.5301024913787842, -1.273552656173706, 1.0536400079727173, -0.710096001625061, -0.40293505787849426, -0.5131887197494507, 0.11361175775527954, 2.1262154579162598, -0.3313104808330536, -11.633423805236816, 0.023533105850219727, 0.8252019882202148, -1.2340158224105835, -3.1674647331237793, 3.789860725402832, -0.6241704225540161, 1.4045075178146362, 0.7792843580245972, 0.3517116904258728, 0.5517067909240723, -0.09037363529205322, 0.2972538471221924, -0.6344356536865234, -1.338968276977539, -0.8477959632873535, -0.6482974290847778, -2.5249922275543213, -0.6150639057159424, 1.1354117393493652, -0.9253942966461182, -1.0904250144958496, 0.8758818507194519, 2.123922109603882, -25.319108963012695, 1.2186450958251953, 0.8879178762435913, 1.5869498252868652, -0.47093576192855835, 1.8428292274475098, -0.5265469551086426, 0.6926093697547913, -4.788686752319336, -0.05646347999572754, 1.7958470582962036, -0.43313848972320557, 0.24423828721046448, -0.350960910320282, -0.7734348773956299, 3.2824716567993164, 1.1099094152450562, -0.3371504545211792, -0.2652796804904938, 1.1600581407546997, 1.8556022644042969, 0.6720767021179199, -0.6478062868118286, 0.18354183435440063, -0.14629322290420532, 0.9380221962928772, -0.38314273953437805, -0.6859111785888672, -0.6742541193962097, 9.044632911682129, 1.2150713205337524, 1.3068010807037354, 0.968530535697937, -0.4861840009689331, -0.48897290229797363, -0.6171587109565735};

int8_t* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
int stride_width=1;
int stride_height=1;
TfLiteFusedActivation activation=kTfLiteActNone;
int dilation_width_factor=1;
int dilation_height_factor=1;
const int filter_dims_size=4;
const int filter_dims_raw[4]={1,3,3,144};
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

}  // namespace rquoeb

TfLiteRegistration* Register_rquoeb_REF() {
  static TfLiteRegistration r = {
      rquoeb::Init, rquoeb::Free, rquoeb::Prepare,
      rquoeb::Eval<rquoeb::kReference>};
  return &r;
}

TfLiteRegistration* Register_rquoeb_GENERIC_OPT() {
  static TfLiteRegistration r = {
      rquoeb::Init, rquoeb::Free, rquoeb::Prepare,
      rquoeb::Eval<rquoeb::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_rquoeb_NEON_OPT() {
  static TfLiteRegistration r = {
      rquoeb::Init, rquoeb::Free, rquoeb::Prepare,
      rquoeb::Eval<rquoeb::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_rquoeb_NEON_OPT_UINT8() {
  static TfLiteRegistration r = {
      rquoeb::Init, rquoeb::Free, rquoeb::Prepare,
      rquoeb::EvalImpl<rquoeb::kNeonOptimized, kTfLiteUInt8>};
  return &r;
}

TfLiteRegistration* Register_rquoeb() {
#ifdef USE_NEON
  return Register_rquoeb_NEON_OPT();
#else
  return Register_rquoeb_GENERIC_OPT();
#endif
}

// Warning: Clients using this variant are responsible for ensuring that their
// models only need the UINT8 type. TFLite's op registration mechanism doesn't
// yet allow for more nuanced registration mechanisms.
TfLiteRegistration* Register_rquoeb_UINT8() {
#ifdef USE_NEON
  return Register_rquoeb_NEON_OPT_UINT8();
#else
  return Register_rquoeb();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
