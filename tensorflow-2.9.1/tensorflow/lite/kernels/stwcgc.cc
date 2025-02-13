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
namespace stwcgc {

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

float filter_r   aw[512]={-0.21712423861026764, 0.03247232362627983, 0.4039256274700165, -1.2136058807373047, 0.5744825601577759, -1.7339063882827759, -0.36548954248428345, -0.2497761845588684, 0.25546061992645264, -0.4943672716617584, -0.036137305200099945, 1.9113743305206299, -0.05097278580069542, 0.05821254104375839, 0.7527729272842407, -0.22836388647556305, -2.5659101009368896, 0.24121877551078796, 0.831291675567627, 2.1330206394195557, -0.5395618081092834, 0.4793897569179535, 0.017137974500656128, -0.36791473627090454, -2.8250815868377686, 1.3506553173065186, 1.7514441013336182, -1.1891374588012695, -1.8290260413778014e-05, 1.293182373046875, -0.3321373164653778, 1.8476368188858032, -0.10825106501579285, 0.10552241653203964, 0.08903661370277405, -0.7879188656806946, -0.3349144160747528, 0.1701570302248001, -0.3728388547897339, -0.9175264239311218, -2.385831356048584, 2.168802499771118, 0.23591776192188263, -0.4425753355026245, -0.3279941976070404, 1.1750537157058716, 0.1095922440290451, 0.2091386914253235, 0.14430657029151917, 0.2421899139881134, 0.0038162830751389265, -0.24037593603134155, -0.637117326259613, 3.494795560836792, 1.0313746929168701, -0.06859110295772552, 0.247118279337883, 0.5989099144935608, -0.5710517764091492, 0.03386089578270912, 3.898419163306244e-05, 0.019106518477201462, -0.778755784034729, 0.01139809936285019, -1.1897544860839844, 0.4886426627635956, 0.023235606029629707, 0.3246477544307709, -1.3678852319717407, 1.7488863468170166, 0.09370708465576172, 0.2946244776248932, 0.09902812540531158, 2.8331472873687744, -0.13180848956108093, 1.7316147089004517, 0.029754186049103737, -0.024270007386803627, -0.10475576668977737, -2.193903684616089, -0.008301973342895508, 0.1588205248117447, 0.4746846854686737, 1.876664638519287, 0.662833571434021, -0.19849970936775208, -0.16850577294826508, 0.9231244325637817, 1.0984199047088623, -0.8066249489784241, 0.641204833984375, -0.868543803691864, 9.492625395068899e-05, -0.17126543819904327, 0.47252047061920166, 0.24655991792678833, 0.1439046412706375, 2.414590835571289, 1.4131265878677368, 0.41432246565818787, -2.1621549129486084, -0.14524327218532562, 0.7746571898460388, 0.5313091278076172, -0.0518757663667202, -0.19298698008060455, 0.37982699275016785, 0.5618341565132141, 0.2785094976425171, 0.18755578994750977, 0.8164296746253967, 1.4903331995010376, 1.083735466003418, 0.5023388266563416, -0.1458067148923874, 0.2045443207025528, 1.973230004310608, -0.1469457447528839, 0.9110901355743408, 1.5718237161636353, -1.6684414148330688, 1.9818508625030518, 0.23090432584285736, 0.9212458729743958, 6.33442277830909e-06, 0.5176223516464233, 1.6651525497436523, -0.7691786885261536, -0.8347644805908203, 0.48279622197151184, 0.9360781311988831, -0.4803806245326996, -0.7642269730567932, -1.9842448234558105, 0.19583860039710999, -0.5865097641944885, 1.660513997077942, 0.12481901049613953, 1.070309042930603, 0.90671706199646, -0.5412785410881042, -0.33978357911109924, -0.08173324167728424, -0.3370526432991028, -0.27055075764656067, 0.7746552228927612, -1.423460841178894, 1.2319941520690918, -0.9609800577163696, 1.2470537424087524, 0.15842761099338531, -1.2500101327896118, -1.0301940441131592, -1.0945672988891602, -1.4097628593444824, -1.251336932182312, 4.6739278332097456e-05, -2.584188461303711, 0.09382694214582443, -1.8589388132095337, 0.3734809160232544, 0.7886828780174255, 0.22266675531864166, 0.6683255434036255, 0.19091938436031342, 1.070318579673767, -0.9868537187576294, -0.7404128909111023, 2.515946865081787, -0.326774001121521, 0.6814579367637634, 0.5379079580307007, -0.30529505014419556, -0.5841532349586487, -0.8096942901611328, 0.8551135659217834, -0.5008144974708557, 1.319464921951294, -0.25124862790107727, 0.04884527996182442, -0.45941004157066345, 0.9111143350601196, 0.4640659987926483, -0.1837310492992401, 0.45369046926498413, 1.1615476608276367, 1.7252647876739502, 0.9878605604171753, 0.00020284217316657305, -0.3695383667945862, -1.132037878036499, 0.07614445686340332, 0.03907056897878647, -0.14348866045475006, -0.18230906128883362, 0.7264367938041687, -0.07191573828458786, -0.24772600829601288, 0.22187888622283936, 0.7369393110275269, -0.6702739596366882, -2.278898000717163, 1.7222188711166382, 0.4459228515625, 2.38069224357605, 1.1339428424835205, -0.2510984539985657, -0.20745749771595, -0.1511044204235077, 2.763803482055664, -0.08316566050052643, 0.2690509259700775, 0.11308421194553375, 0.25988924503326416, -0.9825538992881775, -0.10059119015932083, 0.005388155113905668, -0.7651147246360779, 0.6000328063964844, -0.00923056248575449, -9.117952868109569e-05, 0.025621188804507256, 0.5513073205947876, 0.06294812262058258, -0.18444561958312988, -0.801254153251648, -0.46611687541007996, -0.2457939237356186, -1.0232939720153809, 0.7132253050804138, 0.33947989344596863, -0.06317780911922455, 0.8123432993888855, -0.05516162887215614, 0.22302499413490295, -2.044950246810913, -0.09894871711730957, -0.13181911408901215, 0.08747649937868118, -2.1619277000427246, -1.1647796630859375, 0.6315401196479797, 1.9101521968841553, -1.3176504373550415, 1.2203410863876343, 0.4872438311576843, -0.9226281642913818, 0.8383663296699524, 1.093762755393982, 3.3242788314819336, -1.2424921989440918, -1.1895270347595215, -0.00031320765265263617, -1.775363564491272, -0.11452015489339828, -0.0326896607875824, 0.3865870237350464, -0.6780956387519836, -0.8890100121498108, -0.3377442955970764, -0.6672002077102661, 1.7720556259155273, -0.8694391250610352, 0.17103157937526703, 0.008051947690546513, -0.5959524512290955, 0.3014410436153412, -0.7976518869400024, 0.026456674560904503, 0.14592693746089935, 0.9486870169639587, -0.18863989412784576, -0.994471549987793, 0.09987502545118332, -0.6575505137443542, -1.5495426654815674, 0.15751336514949799, 0.055587612092494965, 1.22933828830719, 0.736419141292572, -2.006402015686035, -2.307159662246704, 1.385209560394287, 1.3575613498687744, -0.00015492145030293614, -3.2807023525238037, 0.056753017008304596, 1.1351912021636963, -0.0970391184091568, 2.0229599475860596, 3.3206217288970947, -0.6589034795761108, -0.7257508039474487, -0.5869573950767517, -0.4887123703956604, 0.010454735718667507, 0.15254874527454376, 0.1713797152042389, -0.14853619039058685, -1.7038731575012207, -0.2396666258573532, -0.08956735581159592, -0.4278702139854431, -0.48637655377388, -0.5979619026184082, -0.15563324093818665, -1.8319765329360962, -1.727754831314087, -0.6245747208595276, -0.00486230431124568, -0.16227328777313232, -0.07404883205890656, 1.630517840385437, -0.8400238156318665, 0.48861631751060486, -0.14914169907569885, 0.00026470248121768236, 1.4725617170333862, 0.47339701652526855, 1.3828445672988892, -0.5510714054107666, 0.796274721622467, 1.392138123512268, 0.6245286464691162, 1.7134482860565186, 2.9850034713745117, -0.0989145040512085, -0.18621903657913208, -0.7737893462181091, -1.8960548639297485, -0.031919095665216446, -0.5098655223846436, 0.07286397367715836, 0.2008822113275528, 0.2993350327014923, -1.4270992279052734, -1.106918454170227, -0.44005027413368225, -0.12464362382888794, -0.323418527841568, -0.5832917094230652, 0.03988010436296463, -0.47370603680610657, -1.3465080261230469, -1.47160804271698, 0.7673082947731018, 0.47795408964157104, -0.45484814047813416, 9.735667845234275e-05, 0.3395250737667084, -0.2914249002933502, -1.7609562873840332, 0.29545482993125916, -0.028524868190288544, -0.2589224576950073, 1.1963022947311401, -0.4407757818698883, -0.2558090388774872, 0.2676425874233246, -1.1469030380249023, -1.169718623161316, -2.125096082687378, -0.3112541139125824, 0.47577157616615295, -1.895179033279419, -0.11607395112514496, -0.24905654788017273, 0.029604023322463036, -0.2627299129962921, -1.8046339750289917, -0.02328653074800968, 0.28702661395072937, 1.2699222564697266, 1.9819695949554443, -1.2340044975280762, 0.025560634210705757, 0.1425241082906723, -0.7404158711433411, 1.143194317817688, 0.005942819640040398, -1.5579542377963662e-05, -0.04359092563390732, 0.7673395872116089, 0.2108030766248703, -0.15019750595092773, 1.0797410011291504, 0.6240625977516174, -0.26700693368911743, 0.8343138098716736, -1.9256306886672974, -0.21880686283111572, 0.5246984958648682, -1.86317777633667, 0.16175751388072968, -1.1098865270614624, 1.4797428846359253, 0.46379169821739197, 0.2453645020723343, -0.7652819752693176, 0.5502342581748962, -0.9678038954734802, -1.0542923212051392, -0.12055053561925888, 1.5338969230651855, 0.17495279014110565, -1.3244816064834595, 0.361052006483078, -0.46647584438323975, 2.2190661430358887, 2.0730998516082764, 0.5720503330230713, 0.5118065476417542, 0.00011944770812988281, -3.716576337814331, 0.7110781669616699, -0.14911511540412903, -0.011308743618428707, 1.7996015548706055, 0.6951372027397156, 0.8204174041748047, 0.950137197971344, 0.005888583604246378, 0.722450315952301, -0.21865125000476837, -0.040417369455099106, -0.010929098352789879, -0.18040718138217926, -0.29462409019470215, 0.1944342851638794, 0.023157941177487373, 0.3129003643989563, 0.007733660750091076, 2.6459310054779053, 0.17449849843978882, 1.126590609550476, -0.21318520605564117, -1.0105146169662476, -0.05842677876353264, -0.47014904022216797, -0.7140477299690247, -1.5102941989898682, -0.6508080959320068, -0.7801596522331238, -2.6098310947418213, 0.00015523961337748915, -2.0426197052001953, -0.14245794713497162, 1.8326724767684937, 0.05175190791487694, 0.11724727600812912, 0.2950255274772644, -2.155028820037842, 0.3868906795978546, 1.0988712310791016, 0.8936498761177063, -0.24142156541347504, -0.04074758291244507, -3.038032293319702, 0.12317395955324173, 0.5372487902641296, -0.2510795593261719, -0.09955963492393494, -0.30608096718788147, -0.6999205946922302, 0.6017605662345886, -0.2465614229440689, 0.3063097298145294, 0.5119298100471497, -0.04377463459968567, 0.14297637343406677, 2.1075518131256104, -0.29323235154151917, 0.3556729555130005, 0.07635296136140823, -1.1343101263046265, -0.042101312428712845, -9.721034439280629e-05, -0.13884590566158295, 0.42039358615875244, 0.6559783816337585, -0.5906766057014465, -0.6452047824859619, 0.4887810945510864, 0.015882378444075584, 0.36972734332084656, 0.10165207087993622, -0.7877823710441589, -0.28451165556907654, 0.12143499404191971, -0.15021908283233643, -0.4134213328361511, 0.8081056475639343, -0.1295684278011322, -0.016364626586437225, 0.511776864528656, -2.2087411880493164, 2.073089599609375, 0.03156184405088425, -1.9099048376083374, 1.0502980947494507, 0.5377532243728638, 0.40160515904426575, -0.8105810880661011, -0.9117127656936646, -0.19039538502693176, 1.1129672527313232, -1.3995379209518433, 3.542454719543457, -8.511800842825323e-05, 0.07727369666099548, -0.49595969915390015, 1.0155726671218872};

float bias_raw[16]={-12.935440063476562, -10.095431327819824, -16.998205184936523, -19.71678924560547, 10.715361595153809, -10.074356079101562, -14.923136711120605, 16.330995559692383, 17.537811279296875, -6.679570198059082, -1.5593867301940918, 30.879131317138672, -8.289374351501465, -5.6501898765563965, 11.916298866271973, -10.552816390991211};

float* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const int stride_width=1;
const int stride_height=1;
const TfLiteFusedActivation activation=kTfLiteActNone;
const int dilation_width_factor=1;
const int dilation_height_factor=1;
const int filter_dims_size=4;
const int32_t filter_dims_raw[4]={16,1,1,32};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={16};
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

TfLiteRegistration* Register_stwcgc_REF() {
  static TfLiteRegistration r = {stwcgc::Init, stwcgc::Free,
                                 stwcgc::Prepare<stwcgc::kReference>,
                                 stwcgc::Eval<stwcgc::kReference>};
  return &r;
}

TfLiteRegistration* Register_stwcgc_GENERIC_OPT() {
  static TfLiteRegistration r = {stwcgc::Init, stwcgc::Free,
                                 stwcgc::Prepare<stwcgc::kGenericOptimized>,
                                 stwcgc::Eval<stwcgc::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_stwcgc_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {stwcgc::Init, stwcgc::Free,
                                 stwcgc::Prepare<stwcgc::kMultithreadOptimized>,
                                 stwcgc::Eval<stwcgc::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_stwcgc_CBLAS_OPT() {
//   static TfLiteRegistration r = {stwcgc::Init, stwcgc::Free,
//                                  stwcgc::Prepare<stwcgc::kCblasOptimized>,
//                                  stwcgc::Eval<stwcgc::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_stwcgc() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_stwcgc_MULTITHREADED_OPT();
#else
  return Register_stwcgc_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
