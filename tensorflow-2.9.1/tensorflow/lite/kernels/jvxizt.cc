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
namespace jvxizt {

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

float filter_r   aw[288]={6.6842217445373535, -0.07064850628376007, -0.031787797808647156, 0.2520485818386078, -0.5055162906646729, 0.33096736669540405, -4.086521148681641, -0.003448887262493372, -0.15298646688461304, 5.7238287925720215, -0.776191234588623, -0.2792927324771881, -0.004273171070963144, -0.025793610140681267, 6.088639736175537, -0.08583136647939682, -0.04826119542121887, -0.03901715576648712, -0.1294248253107071, -0.011324696242809296, -0.07127698510885239, -0.011814492754638195, -0.37448519468307495, -0.11542116105556488, 5.7869391441345215, 0.06020447611808777, 0.20665088295936584, -0.005595134571194649, -5.252404662314802e-07, -1.8971986770629883, -0.7157256603240967, 0.08724700659513474, -3.2613282203674316, -0.11411774903535843, 0.04747715964913368, -0.6194858551025391, -4.673164367675781, -0.3144824206829071, -6.2041192054748535, 0.3729974031448364, 0.24889862537384033, -2.548959493637085, 2.546952247619629, -0.06174415722489357, 0.006832001265138388, -0.311226487159729, -3.9624531269073486, -0.15856993198394775, 1.1840486526489258, 0.12145789712667465, 0.3461915850639343, 0.013522069901227951, 0.49552008509635925, 0.07000301033258438, 0.273197740316391, -0.9290309548377991, -10.163294792175293, 0.03243928402662277, 1.7305431365966797, -0.14096654951572418, -1.1459792403911706e-06, 1.504601240158081, 11.158014297485352, -0.1605088859796524, -2.536390781402588, -0.043168481439352036, -0.013858180493116379, 0.22325399518013, 3.4172146320343018, -0.026370294392108917, -0.38252946734428406, -0.14236955344676971, -0.28696370124816895, 1.843220591545105, -1.6333314180374146, -0.01957588829100132, 0.05173797532916069, 0.08171286433935165, -2.116445302963257, 0.21251899003982544, -0.20648011565208435, -0.11579857766628265, -0.23677703738212585, 0.005825938656926155, 0.2074074149131775, -0.07242760062217712, 0.05484393239021301, -1.2481507062911987, 5.0327982902526855, -0.0017132562352344394, -0.08621449768543243, 0.055540334433317184, -1.0982300864270655e-06, 0.2260846197605133, -1.9321237802505493, 0.0738392099738121, 2.425781488418579, 0.1183730736374855, -0.07045411318540573, -1.2184571027755737, -2.1064209938049316, 1.1225779056549072, 0.08301866799592972, 0.31343352794647217, 0.2097252607345581, -9.841461181640625, 1.0099292993545532, 2.5505027770996094, 0.0012074419064447284, -0.20436975359916687, -4.532941818237305, 1.5575007200241089, 0.011544482782483101, 0.07541263848543167, -0.002493212465196848, -1.522416591644287, -0.4065952003002167, 0.034448131918907166, -1.1595369577407837, -1.6814078092575073, -8.373446464538574, 0.10454662889242172, -0.5508631467819214, -0.06422314047813416, 5.252404662314802e-07, 1.47833251953125, -0.7274867296218872, 0.08575356006622314, -0.7097771167755127, 1.5029953718185425, -0.7406812906265259, -2.516575574874878, -17.27090835571289, -0.13964927196502686, 13.71920108795166, 0.3709462583065033, 3.2800474166870117, -34.2678108215332, 6.539000511169434, 0.22114937007427216, -1.5835516452789307, -0.9923873543739319, 2.1532199382781982, -0.8803720474243164, 0.13064366579055786, 1.3201708793640137, 1.9133213758468628, 0.008783821947872639, 3.698479413986206, 0.77239990234375, -1.3826783895492554, 12.328629493713379, -24.361812591552734, -3.058931827545166, -1.1283806562423706, -1.7128738164901733, 0.0, -1.1196362972259521, -0.05823279172182083, -1.7411147356033325, -0.5823521614074707, -0.25539273023605347, 0.07020523399114609, -0.3130400776863098, 3.8301854133605957, -0.9781937599182129, -1.4513614177703857, 0.10789413750171661, -1.0960869789123535, 5.881398677825928, -0.58979731798172, -0.029423676431179047, 0.462007999420166, 0.12837262451648712, 2.2462031841278076, -0.565209150314331, -0.1169319674372673, -0.44358617067337036, -0.3085533678531647, 0.00950611848384142, -0.05511057376861572, -0.23423029482364655, -0.08138635754585266, -0.49641403555870056, 28.41054344177246, 0.23757097125053406, 0.01928752288222313, 0.09377377480268478, 4.774913264782299e-08, -0.2317725569009781, -0.7536942958831787, 0.12342750281095505, -0.24453967809677124, -0.02523166313767433, -0.02821708843111992, 0.8703305721282959, 1.547454833984375, 0.002861883956938982, 1.2125821113586426, -0.2756231129169464, -0.3032468557357788, 7.937493324279785, -0.8259252309799194, -0.1981872022151947, 0.04921519011259079, 0.11902546882629395, -1.7393920421600342, 0.27132147550582886, -0.013971365988254547, -0.09906809777021408, 0.10389899462461472, 0.03780684992671013, -0.5277317762374878, -0.07027890533208847, 1.3452664613723755, 0.13074438273906708, 4.10453462600708, 0.08485839515924454, 0.16438893973827362, 0.009077082388103008, 3.819930611825839e-07, 0.3523360788822174, -0.8491089344024658, 0.013820508494973183, -0.521296501159668, -0.15839959681034088, -0.011325291357934475, 3.2574574947357178, 3.9018325805664062, -0.05785991623997688, 1.551120400428772, 0.22839251160621643, -1.1937501430511475, 2.8401081562042236, -2.689887523651123, -0.13006985187530518, 0.5305394530296326, 0.09932997077703476, 2.0644497871398926, -0.41570889949798584, -0.00918086152523756, -0.5055875778198242, -0.33152011036872864, 0.03393096849322319, -0.8991168737411499, -0.2636541426181793, 1.2943317890167236, -1.0381723642349243, 27.132823944091797, -0.04117536544799805, -0.3627524971961975, 0.0554552860558033, -1.4324740504889633e-07, -0.32030126452445984, -0.8945768475532532, 0.0674615129828453, 0.28726112842559814, -0.05890480428934097, 0.04518497362732887, 0.04088713228702545, 0.6433036923408508, 0.0687134861946106, 0.0963946208357811, -0.2990992069244385, 0.20910866558551788, 3.1635499000549316, -0.07557597011327744, 0.011462896130979061, -0.036438073962926865, 0.07037658989429474, -0.04160210117697716, 0.04762817546725273, 0.004960245918482542, 0.081294484436512, -0.1280946433544159, -0.001012791064567864, -0.7397744655609131, 0.024700112640857697, 0.00882288720458746, -0.9072127342224121, 1.9100048542022705, 0.08382956683635712, -0.033920079469680786, 0.009040809236466885, 4.774913264782299e-08, 0.020430773496627808, -0.32854223251342773, 0.03559061884880066};

float bias_raw[32]={0.7598611116409302, 0.21431970596313477, 12.92075252532959, 2.223003625869751, 5.82286262512207, 2.1697239875793457, 0.6021432876586914, -4.586287498474121, 0.3962104320526123, 8.99199104309082, 6.009696960449219, 2.009563684463501, 14.904574394226074, 4.4920220375061035, 2.1033995151519775, 2.3366940021514893, 0.365842342376709, 0.8425493240356445, -0.45030689239501953, 5.040643215179443, 1.4561610221862793, 1.019222378730774, 2.3147621154785156, -1.703005313873291, 6.491693019866943, 5.6143598556518555, 2.6981399059295654, 5.69727897644043, -1.116971492767334, 2.6945836544036865, -0.8881073594093323, 5.285429954528809};

float* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
int stride_width=1;
int stride_height=1;
TfLiteFusedActivation activation=kTfLiteActNone;
int dilation_width_factor=1;
int dilation_height_factor=1;
const int filter_dims_size=4;
const int filter_dims_raw[4]={1,3,3,32};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={32};
TfLitePadding paddings=kTfLitePaddingSame;
TfLiteType filter_type=kTfLiteFloat32;
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

}  // namespace jvxizt

TfLiteRegistration* Register_jvxizt_REF() {
  static TfLiteRegistration r = {
      jvxizt::Init, jvxizt::Free, jvxizt::Prepare,
      jvxizt::Eval<jvxizt::kReference>};
  return &r;
}

TfLiteRegistration* Register_jvxizt_GENERIC_OPT() {
  static TfLiteRegistration r = {
      jvxizt::Init, jvxizt::Free, jvxizt::Prepare,
      jvxizt::Eval<jvxizt::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_jvxizt_NEON_OPT() {
  static TfLiteRegistration r = {
      jvxizt::Init, jvxizt::Free, jvxizt::Prepare,
      jvxizt::Eval<jvxizt::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_jvxizt_NEON_OPT_UINT8() {
  static TfLiteRegistration r = {
      jvxizt::Init, jvxizt::Free, jvxizt::Prepare,
      jvxizt::EvalImpl<jvxizt::kNeonOptimized, kTfLiteUInt8>};
  return &r;
}

TfLiteRegistration* Register_jvxizt() {
#ifdef USE_NEON
  return Register_jvxizt_NEON_OPT();
#else
  return Register_jvxizt_GENERIC_OPT();
#endif
}

// Warning: Clients using this variant are responsible for ensuring that their
// models only need the UINT8 type. TFLite's op registration mechanism doesn't
// yet allow for more nuanced registration mechanisms.
TfLiteRegistration* Register_jvxizt_UINT8() {
#ifdef USE_NEON
  return Register_jvxizt_NEON_OPT_UINT8();
#else
  return Register_jvxizt();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
