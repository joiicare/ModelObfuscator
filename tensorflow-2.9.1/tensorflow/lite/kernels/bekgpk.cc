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
namespace bekgpk {

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

float filter_r   aw[384]={-0.012897439301013947, -0.004685506224632263, 0.016523301601409912, 0.01075957715511322, 0.011741358786821365, 0.02664082497358322, 0.02307938039302826, 0.026730597019195557, -0.1331464797258377, -0.2851044535636902, -0.2256706804037094, -0.2777424454689026, -0.03933515027165413, -0.0001684427261352539, -0.043811678886413574, -0.02983746863901615, 0.008871495723724365, -0.04899241775274277, 0.012845918536186218, -0.030879154801368713, 0.002156175673007965, 0.0049332305788993835, -0.09476824849843979, -0.06564660370349884, -0.10218574106693268, -0.1818729192018509, -0.09602422267198563, -0.12526950240135193, -0.05947704613208771, -0.1806705743074417, -0.07765117287635803, -0.13371317088603973, -0.05538485571742058, -0.10763677954673767, -0.017590083181858063, -0.042229220271110535, -0.08979664742946625, -0.1582103669643402, -0.08336829394102097, -0.08146831393241882, -0.09566337615251541, -0.18596448004245758, -0.006675105541944504, -0.029850691556930542, -0.12073978781700134, -0.1989728957414627, -0.08826076984405518, -0.130803644657135, 0.06871598958969116, 0.10585436224937439, -0.007273480296134949, 0.024129033088684082, -0.08642683178186417, -0.22851645946502686, -0.1099470853805542, -0.18990574777126312, -0.056867051869630814, -0.10730353742837906, -0.0479215532541275, -0.06711116433143616, -0.14689110219478607, -0.35964804887771606, -0.09796862304210663, -0.17732398211956024, -0.12744508683681488, -0.2381930947303772, -0.1517370194196701, -0.22417090833187103, -0.059973642230033875, -0.1273019015789032, -0.05507785826921463, -0.09336139261722565, -0.11110928654670715, -0.3079041838645935, -0.08199940621852875, -0.1991584748029709, 0.0231239665299654, 0.1548537164926529, 0.021179094910621643, 0.08579384535551071, -0.11751493811607361, -0.16163067519664764, -0.10330552607774734, -0.13498958945274353, -0.1679614782333374, -0.24886927008628845, -0.1225496158003807, -0.18180084228515625, -0.1810915768146515, -0.34119176864624023, -0.23737689852714539, -0.34606024622917175, -0.05473741143941879, -0.08934938907623291, -0.013029322028160095, -0.009024456143379211, -0.04578881710767746, -0.03961649537086487, -0.05598977208137512, -0.06123749911785126, -0.13130229711532593, -0.2262411117553711, -0.11687719076871872, -0.19368895888328552, -0.06013632193207741, -0.10101095587015152, -0.03859946131706238, -0.041336849331855774, -0.01282305084168911, 0.01618492603302002, 0.027423128485679626, 0.03466551750898361, 0.036715067923069, 0.08468103408813477, -0.026738181710243225, 0.009602785110473633, -0.08232922106981277, -0.14652962982654572, -0.011007465422153473, 0.02213604748249054, -0.05648384988307953, -0.13342219591140747, -0.04791977256536484, -0.08242470026016235, 0.00483037531375885, -0.16660119593143463, -0.22633813321590424, -0.2714497148990631, -0.069994255900383, -0.12409159541130066, -0.01068386435508728, -0.020759418606758118, -0.07030823826789856, -0.16622252762317657, -0.01898379623889923, -0.08322025090456009, -0.3075864613056183, -0.5833227634429932, -0.10899672657251358, -0.4897959232330322, -0.11028121411800385, -0.23662132024765015, -0.04702356085181236, -0.11150968074798584, -0.05003759637475014, -0.12427155673503876, -0.020770736038684845, -0.049007266759872437, -0.09449014067649841, -0.1783776432275772, -0.054301247000694275, -0.08422420918941498, -0.060126230120658875, -0.016701310873031616, -0.1672666370868683, -0.21317440271377563, -0.0589265376329422, -0.14566905796527863, -0.0759655013680458, -0.04973096773028374, -0.18283270299434662, -0.20574603974819183, -0.17694424092769623, -0.2101770043373108, -0.1379019170999527, -0.2513350248336792, -0.12792907655239105, -0.14672720432281494, -0.08871757239103317, -0.1788085401058197, -0.06179550290107727, -0.08220657706260681, -0.15062229335308075, -0.27578890323638916, -0.15345631539821625, -0.2809184789657593, 0.10460423678159714, 0.068189837038517, -0.11880635470151901, -0.0970277339220047, -0.059047482907772064, -0.05041269585490227, -0.018101729452610016, -0.05492681264877319, -0.03475315868854523, -0.07770179212093353, -0.023792482912540436, -0.03653934597969055, -0.05809640884399414, -0.07233263552188873, -0.14694924652576447, -0.16365207731723785, 0.014943014830350876, -0.06511422991752625, -0.13816943764686584, -0.16677097976207733, 0.01139611192047596, -0.08735442161560059, -0.16317881643772125, -0.17447854578495026, -0.05928633734583855, -0.1452092081308365, -0.01228276640176773, -0.05229507386684418, -0.0988997220993042, -0.19865655899047852, 0.036152392625808716, -0.016251444816589355, 0.033839188516139984, 0.09470862150192261, 0.017338261008262634, 0.020219676196575165, -0.03375261276960373, -0.2039465308189392, -0.23642197251319885, -0.301349401473999, -0.15024763345718384, -0.23577731847763062, -0.12217804789543152, -0.15625478327274323, 0.02808287739753723, -0.03497905284166336, 0.021144039928913116, 0.029195182025432587, -0.08887118101119995, -0.14098918437957764, -0.06892521679401398, -0.08782726526260376, -0.09171238541603088, -0.21240860223770142, -0.07536058127880096, -0.09171633422374725, -0.045462772250175476, -0.03928571939468384, -0.004428505897521973, -0.0010675638914108276, 0.04217875003814697, -0.02839072048664093, -0.5765488147735596, -0.5191982388496399, -0.06810862571001053, -0.13263081014156342, -0.10055648535490036, -0.13781613111495972, -0.017387785017490387, -0.03903528302907944, 0.02806490659713745, 0.009048469364643097, 0.00148676335811615, -0.16384097933769226, -0.2350541353225708, -0.2807425558567047, -0.06519739329814911, -0.13419488072395325, -0.009254813194274902, -0.05646893382072449, -0.0558193102478981, -0.1728886216878891, -0.06659750640392303, -0.1378391981124878, -0.04848335683345795, -0.07928968966007233, -0.0012299809604883194, 0.01316251140087843, -0.026384077966213226, -0.07796802371740341, -0.00793452374637127, -0.04774242639541626, -0.047035008668899536, -0.059130243957042694, 0.000425117090344429, -0.0015778504312038422, -0.0916437953710556, -0.21188312768936157, -0.031798869371414185, -0.08688566088676453, -0.051518991589546204, -0.04112449288368225, -0.010647855699062347, -0.004159428179264069, 0.03141925111413002, 0.04911980777978897, -0.2584845721721649, -0.23909983038902283, -0.11876516044139862, -0.2078532576560974, -0.10313080996274948, -0.0984276682138443, -0.09267207980155945, -0.17006829380989075, -0.12576442956924438, -0.1449146717786789, -0.01657332479953766, -0.1834234893321991, -0.21999868750572205, -0.24905404448509216, 0.005531042814254761, 0.01840895041823387, 0.07075207680463791, 0.0015406012535095215, -0.09939330816268921, -0.176489919424057, -0.039395540952682495, -0.07485485821962357, -0.14178286492824554, -0.2648961544036865, -0.14814819395542145, -0.20697179436683655, -0.0140877366065979, -0.14676055312156677, -0.2082730531692505, -0.2509572207927704, -0.045350633561611176, -0.08411821722984314, 0.004084430634975433, -0.0006291443714872003, -0.04199080914258957, -0.23753581941127777, -0.2715851962566376, -0.30280882120132446, 0.08516189455986023, 0.005611909553408623, -0.08672436326742172, -0.17878027260303497, -0.12482389807701111, -0.26346057653427124, -0.07964357733726501, -0.12246102839708328, -0.1052507609128952, -0.1496443748474121, -0.021634377539157867, -0.04010142385959625, -0.025933558121323586, -0.1078416109085083, -0.05602513998746872, -0.08320647478103638, -0.20898517966270447, -0.4976647198200226, -0.15601694583892822, -0.36003053188323975, -0.12633591890335083, -0.25941601395606995, -0.08219516277313232, -0.13630563020706177, -0.010037004947662354, 0.0032716989517211914, -0.011118531227111816, -0.006989926099777222, 0.02201862446963787, -0.1668068915605545, -0.27790284156799316, -0.31307530403137207, -0.18263792991638184, -0.4318675398826599, -0.12318626791238785, -0.19222021102905273, -0.07063627243041992, -0.1244194507598877, -0.07862912118434906, -0.05841194838285446, -0.007110774517059326, -0.1126151978969574, -0.19972650706768036, -0.22705592215061188, 0.016602087765932083, -0.09854945540428162, -0.16568896174430847, -0.20100346207618713, -0.15627536177635193, -0.36149460077285767, -0.2142154574394226, -0.39984646439552307, 0.004707396030426025, -0.06103351712226868, 0.049837589263916016, -0.020692214369773865, -0.08521448075771332, -0.16921940445899963, -0.058138489723205566, -0.06175391003489494, -0.08348406851291656, -0.17267805337905884, -0.07345572113990784, -0.242860347032547};

float bias_raw[96]={0.17364071309566498, -0.040477652102708817, 0.029028868302702904, 0.267681360244751, 1.048917531967163, 0.6767788529396057, -0.3085104525089264, -0.14676712453365326, -0.2644600570201874, -0.25084730982780457, 0.3721565902233124, -0.08821135014295578, 0.31051522493362427, -0.10016472637653351, -0.28020742535591125, 0.24946020543575287, -0.22759929299354553, -0.399843692779541, 0.11542735248804092, 1.0492329597473145, 0.2142886370420456, -0.2197522521018982, 0.1956547498703003, -0.2926863431930542, -0.2819315195083618, -0.2740216851234436, -0.24564246833324432, 0.4622613787651062, 0.47766897082328796, 0.2414882630109787, -0.3384537398815155, -0.30255126953125, -0.3752731680870056, 0.25565508008003235, 0.19776417315006256, -0.1360882669687271, -0.014678808860480785, -0.2511833906173706, 0.42785510420799255, -0.23208482563495636, 0.2095799297094345, -0.13898134231567383, -0.31916964054107666, 0.1588044911623001, -0.3525390028953552, 0.36414453387260437, -0.32485321164131165, 0.46652668714523315, -0.40432828664779663, -0.43321946263313293, -0.18724630773067474, 0.1469433456659317, 0.871584951877594, -0.47150787711143494, -0.27097031474113464, 0.6992819905281067, -0.07075744867324829, -0.12092218548059464, -0.2852276563644409, -0.03730155900120735, -0.16306176781654358, -0.3715759217739105, -0.4626859128475189, -0.31216511130332947, -0.15518973767757416, -0.31970107555389404, -0.17898151278495789, -0.2105039358139038, -0.392985075712204, -0.3494313955307007, -0.0605485625565052, -0.23855294287204742, -0.17160741984844208, -0.5098493099212646, 0.15990881621837616, -0.16408051550388336, -0.3745788335800171, -0.44372299313545227, -0.24993941187858582, -0.6108834147453308, -0.01137570757418871, 0.006213577464222908, -0.3377815783023834, 0.46872568130493164, 0.12209197133779526, -0.33782196044921875, 0.7426072955131531, 0.10243996977806091, 0.30833280086517334, -0.34226590394973755, -0.23547449707984924, -0.2604624927043915, 0.01986776851117611, 0.4862772822380066, -0.28050497174263, 0.2966124713420868};

float* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const int stride_width=1;
const int stride_height=1;
const TfLiteFusedActivation activation=kTfLiteActNone;
const int dilation_width_factor=1;
const int dilation_height_factor=1;
const int filter_dims_size=4;
const int32_t filter_dims_raw[4]={96,1,1,4};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={96};
const TfLitePadding paddings=kTfLitePaddingSame;
const TfLiteType filter_type=kTfLiteFloat32;
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

TfLiteRegistration* Register_bekgpk_REF() {
  static TfLiteRegistration r = {bekgpk::Init, bekgpk::Free,
                                 bekgpk::Prepare<bekgpk::kReference>,
                                 bekgpk::Eval<bekgpk::kReference>};
  return &r;
}

TfLiteRegistration* Register_bekgpk_GENERIC_OPT() {
  static TfLiteRegistration r = {bekgpk::Init, bekgpk::Free,
                                 bekgpk::Prepare<bekgpk::kGenericOptimized>,
                                 bekgpk::Eval<bekgpk::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_bekgpk_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {bekgpk::Init, bekgpk::Free,
                                 bekgpk::Prepare<bekgpk::kMultithreadOptimized>,
                                 bekgpk::Eval<bekgpk::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_bekgpk_CBLAS_OPT() {
//   static TfLiteRegistration r = {bekgpk::Init, bekgpk::Free,
//                                  bekgpk::Prepare<bekgpk::kCblasOptimized>,
//                                  bekgpk::Eval<bekgpk::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_bekgpk() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_bekgpk_MULTITHREADED_OPT();
#else
  return Register_bekgpk_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
