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
namespace ywrpsy {

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

float filter_r   aw[864]={-3.2433228492736816, 2.8831067085266113, 1.6133360862731934, -0.17009645700454712, -0.1530102640390396, -0.11149512976408005, -3.353300094604492, 1.0708768367767334, 1.442177414894104, 1.6496633291244507, -0.18342655897140503, 0.2587507367134094, -0.3583548665046692, -1.2463239431381226, 0.12308163940906525, 0.24060288071632385, 3.1300225257873535, 0.8243144154548645, 0.24067525565624237, -0.1620384156703949, 0.10330839455127716, 3.560486078262329, 0.12187864631414413, 31.027624130249023, 1.1573981046676636, -3.0657882690429688, 4.182314395904541, -0.16113661229610443, -0.2599965035915375, 0.23308256268501282, -1.7006747722625732, -10.964804649353027, -0.11301705241203308, -0.16978490352630615, -0.45804649591445923, 2.253485679626465, -0.022144680842757225, 2.781568765640259, 0.417191743850708, -0.7346969842910767, 0.9480521082878113, 3.7975070476531982, 0.6197301745414734, 1.5794498920440674, -0.7484093308448792, -0.1721579134464264, 0.8077521324157715, -0.07537238299846649, 4.06041145324707, -2.750309705734253, 3.2794504165649414, -0.5033848285675049, -0.1064765453338623, 4.182498455047607, -0.9954494833946228, -0.08242610841989517, 0.3986870050430298, -2.4560904502868652, 1.7406891584396362, 0.13302884995937347, -2.3280222415924072, -0.44381096959114075, 2.1081197261810303, -0.514214038848877, -0.7958222031593323, -1.4987872838974, -9.164870262145996, -2.540363311767578, -28.783798217773438, -3.5098493099212646, 0.13561777770519257, -7.455740928649902, -1.3573038578033447, -2.869572877883911, -0.10747656226158142, -0.48443180322647095, 1.297744631767273, 6.637852191925049, 4.053879261016846, -0.34191909432411194, 0.2196175456047058, 1.5142866373062134, 4.196642875671387, -0.41062477231025696, 0.06926268339157104, 0.6267114281654358, -0.1533212959766388, 0.0839037075638771, 0.16309210658073425, -1.6527479887008667, -3.5578176975250244, 4.3346848487854, 1.654099702835083, -0.25339066982269287, 1.0667582750320435, -0.252611368894577, -2.9133284091949463, 4.661079406738281, 2.49059796333313, -0.32360169291496277, -0.21937701106071472, -0.11380472779273987, 4.528944492340088, 1.6374640464782715, -1.4972413778305054, -1.1000041961669922, -0.38497206568717957, 0.05958787351846695, -0.4652113914489746, -1.9194045066833496, 0.16935113072395325, 0.402956485748291, -14.147826194763184, -0.6157727837562561, 0.2825565040111542, -0.2858313024044037, 0.1360137164592743, -3.1163864135742188, 0.2654149830341339, -8.47256088256836, -0.9111242890357971, -0.5891697406768799, -5.77670431137085, -0.24963204562664032, -0.38990721106529236, 0.26630768179893494, -2.0633318424224854, -8.44621467590332, 1.2790318727493286, -0.31672054529190063, -0.5426427125930786, 5.637127876281738, -0.602833092212677, -3.7605154514312744, 0.7800864577293396, -1.7019011974334717, 1.3134386539459229, 7.4500274658203125, -3.1784441471099854, 2.662970781326294, -1.1043349504470825, -0.3742234408855438, 0.03457748889923096, -0.1072455644607544, -5.600453853607178, 2.134249448776245, 6.754850387573242, -1.3079020977020264, -0.19256021082401276, -0.15241725742816925, 2.8161673545837402, -0.12027379870414734, -0.7502484917640686, -3.356180191040039, -0.05072779953479767, 0.13005954027175903, 0.4971548020839691, -0.8696087598800659, 15.833820343017578, 3.217170238494873, 2.0460293292999268, -3.3555660247802734, 9.653282165527344, -3.8457772731781006, 59.06928634643555, 3.179133892059326, 0.176732137799263, -4.77746057510376, 4.4954352378845215, -11.696598052978516, -0.485861599445343, -0.9048057794570923, -4.098602294921875, 3.8409931659698486, -8.647382736206055, 2.006605863571167, 0.37580665946006775, 2.8153443336486816, -7.471454620361328, -0.354719877243042, 0.18564629554748535, -3.5254743099212646, -0.31953147053718567, 0.09550115466117859, 0.22444330155849457, -2.6344943046569824, 0.34457793831825256, 1.1137241125106812, 2.192547082901001, -0.3165763020515442, -3.0657293796539307, -0.338975191116333, -0.8543437719345093, 2.1652512550354004, 0.8203495144844055, -0.16969099640846252, -0.10691512376070023, -0.0818735882639885, -1.7738962173461914, 0.9555755257606506, -0.015310518443584442, -0.922582745552063, -0.15246784687042236, -0.405656635761261, -0.3612784445285797, -1.005591630935669, -0.12926912307739258, 0.23331017792224884, 9.068275451660156, -4.5997090339660645, 0.09391728043556213, -0.14463593065738678, 0.050112638622522354, -0.6869449019432068, 0.22848878800868988, -22.295650482177734, -0.8761195540428162, 2.692385673522949, 2.2246837615966797, -0.16106197237968445, -0.1876850724220276, 0.16514383256435394, 1.5974315404891968, -2.07797908782959, -1.1509006023406982, -0.16331689059734344, -0.15123635530471802, 3.1896812915802, -0.35526543855667114, 0.6688297986984253, 0.40190285444259644, -1.0037223100662231, 0.6066267490386963, 3.4206395149230957, 2.3272011280059814, 2.2024343013763428, -0.7549175024032593, -0.23133620619773865, -0.662011981010437, -0.037278253585100174, 1.3996033668518066, 0.6092320084571838, 3.8817920684814453, -0.5003283023834229, -0.08725965768098831, -1.3582173585891724, -1.4964507818222046, -0.07159849256277084, -1.1664677858352661, -0.5395674705505371, -0.7305799126625061, 0.05573747307062149, 1.640931487083435, -0.6598971486091614, 3.8815951347351074, -3.05068302154541, -1.2642292976379395, -1.8449766635894775, 10.964240074157715, -1.9407155513763428, -13.488652229309082, 3.2632620334625244, 0.058552660048007965, 2.477351427078247, -2.2157931327819824, -6.000858783721924, -0.3461112976074219, -0.4250986874103546, 2.388810873031616, -1.4240895509719849, 5.065688133239746, 0.30143752694129944, 0.22080960869789124, 1.9052152633666992, 2.1311678886413574, -0.06956323236227036, 0.15310896933078766, 3.031695604324341, -0.18905974924564362, 0.03146990016102791, 0.12176798284053802, -1.4978655576705933, 3.9122390747070312, -12.162175178527832, 1.64517343044281, -0.0588933601975441, -2.8585736751556396, -0.13886743783950806, -4.776770114898682, 3.362657308578491, 3.432570695877075, -0.11757952719926834, -0.22429874539375305, -0.14525745809078217, -5.484270095825195, -0.14382140338420868, -0.6913037896156311, 0.1907300502061844, -0.3412058353424072, 0.3782349228858948, -0.45762327313423157, 0.37211939692497253, -1.6428191661834717, 0.20797422528266907, 9.275040626525879, 3.569450616836548, 0.3411787152290344, -0.22220419347286224, 0.19207605719566345, 4.537342071533203, 0.13550542294979095, -71.7197494506836, 2.646559238433838, -3.369868040084839, 1.6479352712631226, -0.19061331450939178, -0.16439169645309448, 0.23521634936332703, 3.717961311340332, 1.914041519165039, 2.9432473182678223, -0.3441964089870453, -0.5567430853843689, -0.2797599732875824, -3.1532626152038574, -4.966810703277588, 0.7287685871124268, -0.23073448240756989, 1.2844538688659668, -6.1739912033081055, -1.9990308284759521, 1.603814721107483, -1.381697416305542, -0.25658974051475525, -2.2054989337921143, -0.11621211469173431, 5.508224964141846, 2.3215079307556152, -3.4372291564941406, -1.067191481590271, -0.12193399667739868, 6.385030746459961, -0.21105124056339264, -0.2535933554172516, 0.7981942892074585, 2.9816181659698486, -2.0014400482177734, 0.21583816409111023, -2.379477024078369, -1.6920660734176636, -19.307212829589844, 0.7109737992286682, -1.792807936668396, 2.8540215492248535, -13.916790962219238, 1.5523666143417358, 48.312599182128906, 4.8550333976745605, 0.17959751188755035, 10.79719352722168, -5.327535629272461, 6.5401997566223145, -0.43457597494125366, 0.06037607789039612, 1.6553130149841309, 6.701058387756348, -1.7281436920166016, -3.6768479347229004, 0.3874751329421997, 2.824979782104492, -6.364843845367432, -0.6600852608680725, 0.30063363909721375, -5.590851783752441, -0.20334918797016144, 0.104277104139328, 0.19298194348812103, 1.7373064756393433, -4.165334224700928, -2.6574690341949463, 2.5510709285736084, -0.3501771092414856, -4.990095615386963, -0.3633919656276703, -3.7810583114624023, 5.603088855743408, 4.052812576293945, -0.40950366854667664, -0.3105056881904602, -0.15038740634918213, 7.804962635040283, -0.6513690948486328, -1.0295342206954956, 2.1449100971221924, -0.7597793340682983, -0.06925179064273834, -0.6045275926589966, 0.17972432076931, -2.730290174484253, 0.526922881603241, -27.99297523498535, 2.1312007904052734, 0.5240655541419983, -0.35502979159355164, 0.3175762891769409, -6.485801696777344, 0.3054361045360565, 40.70976257324219, -1.017872929573059, 0.23774723708629608, 0.2556348145008087, -0.27460142970085144, -0.39515963196754456, 0.2933850586414337, 2.039400577545166, 8.631532669067383, -12.186883926391602, -0.47392067313194275, -0.743628203868866, -4.388755798339844, -4.986948490142822, 8.425520896911621, 1.2034157514572144, -0.5292860865592957, 1.9583228826522827, -9.729865074157715, 2.0497710704803467, 2.983118772506714, -1.748101830482483, -0.5146293640136719, 0.9647794961929321, -0.1669573038816452, -9.90351390838623, 2.007009267807007, -7.41874885559082, -2.0441267490386963, -0.22713346779346466, -5.197055339813232, 2.604504346847534, -0.24032779037952423, 3.1660642623901367, 5.962817192077637, 0.42943912744522095, 0.19009821116924286, -1.2643568515777588, -2.3153676986694336, 2.0484707355499268, -4.591402053833008, 3.2218551635742188, 5.80015230178833, 12.627206802368164, 2.3411853313446045, -91.35563659667969, 0.21045908331871033, 0.20879818499088287, 1.236855149269104, 4.753227233886719, 5.802826404571533, -0.5974025726318359, 0.09231176972389221, -5.776971340179443, -6.51176118850708, 3.254929780960083, 1.4562493562698364, 0.5294877886772156, 2.807448148727417, 8.949914932250977, -0.6970009803771973, 0.47942376136779785, 8.202350616455078, -0.39684927463531494, 0.15711863338947296, 0.2525550127029419, 2.719536304473877, -1.334348440170288, 18.362071990966797, 3.220154047012329, -0.26989516615867615, -4.735814571380615, -0.5262351632118225, -1.3492834568023682, 2.671736001968384, 1.9487782716751099, -0.32111072540283203, -0.10698665678501129, -0.0912567675113678, -2.8750786781311035, -0.6736094951629639, 1.1823114156723022, -1.547078251838684, -0.29904434084892273, -0.8491925001144409, -0.47289207577705383, -0.12430693954229355, -1.185003638267517, 0.24455276131629944, 18.981605529785156, -2.0032758712768555, 0.3755209445953369, -0.21625801920890808, 0.17287138104438782, 1.7094030380249023, 0.2267126888036728, 26.96787452697754, -1.2053654193878174, 3.5393550395965576, -1.8218251466751099, -0.1450832188129425, -0.23395299911499023, 0.21789489686489105, -2.3893699645996094, 4.518198013305664, 8.287435531616211, -0.1827126443386078, -0.21200187504291534, -4.183789253234863, -3.0701799392700195, -3.54653263092041, 0.5275884866714478, -0.4625885486602783, 0.8820793032646179, -2.9595015048980713, 0.5271351337432861, 2.4385337829589844, -0.9622460007667542, -0.1689683049917221, 0.9686997532844543, -0.063843734562397, 4.213563442230225, -1.4759272336959839, -4.199986934661865, -0.7520828247070312, -0.0698523223400116, -2.6738085746765137, -2.395092725753784, -0.0629427507519722, 2.0962271690368652, 2.8551788330078125, 1.4259850978851318, 0.0701945498585701, 3.9413392543792725, -0.8877306580543518, 4.905579090118408, 3.7907774448394775, -1.8533515930175781, 3.5681891441345215, 23.96320343017578, 0.6516889929771423, 21.21023178100586, -5.882299423217773, 0.07765393704175949, -5.201976776123047, 0.5270284414291382, 1.7016215324401855, -0.3647005259990692, 0.04177289456129074, 4.7442803382873535, -5.003673076629639, -2.174765110015869, 1.0132724046707153, 0.26898714900016785, 1.7004894018173218, -2.9099912643432617, -0.1425851434469223, 0.2465635985136032, -3.09983229637146, -0.2333998680114746, 0.06648041307926178, 0.1477138251066208, 1.4483718872070312, 4.849781036376953, 0.7309373617172241, 2.689134359359741, -0.11882032454013824, -0.3040603995323181, -0.19512157142162323, -1.274982213973999, 1.2377212047576904, 1.9064571857452393, 0.048719462007284164, -0.0702524334192276, -0.08577830344438553, -2.757896900177002, -0.5907566547393799, -1.146903157234192, -2.156221389770508, -0.2032793015241623, 0.13818414509296417, -0.25846853852272034, 0.9195594191551208, 1.1870731115341187, 0.06386987119913101, 6.101358413696289, 2.345956563949585, 0.10891185700893402, -0.08641290664672852, 0.1091083213686943, 1.260440468788147, 0.08261851966381073, 42.42494201660156, 1.7049758434295654, -1.8044925928115845, -2.3577115535736084, -0.044062383472919464, -0.16109715402126312, 0.0706796795129776, -2.1640279293060303, 3.507138729095459, -2.150195837020874, -0.18070071935653687, -0.186186283826828, -1.8998137712478638, 2.601052761077881, 2.5069661140441895, 0.2852487564086914, 1.1755073070526123, 0.4510228633880615, 2.459510087966919, 0.03294748440384865, -0.44132286310195923, -0.6598410606384277, -0.08299652487039566, 1.2589080333709717, -0.049784742295742035, 1.6942085027694702, 1.099219560623169, 0.03579888120293617, -0.6869563460350037, -0.07466039061546326, 2.656316041946411, 0.3546004891395569, -0.12818171083927155, -1.1064573526382446, -0.20992742478847504, 0.375953733921051, 0.08164192736148834, -0.5044164657592773, 2.716526985168457, -7.916476249694824, -0.22830010950565338, -1.2448840141296387, -0.61799556016922, -6.327870845794678, 0.3270123600959778, -19.298900604248047, -1.6707690954208374, 0.06422523409128189, -4.071389198303223, -1.2447160482406616, 0.007399572525173426, -0.2433548867702484, 0.38033825159072876, 0.6315453052520752, 2.657554864883423, -1.626477599143982, -1.0432807207107544, 0.21654970943927765, 1.5243419408798218, 2.0916965007781982, -0.2758254408836365, 0.22638732194900513, 4.840590476989746, -0.12114808708429337, 0.044404737651348114, 0.06541277468204498, -0.07443016022443771, -1.2483118772506714, -11.043912887573242, 0.4011121094226837, -0.027015015482902527, 3.469141721725464, -0.15045925974845886, -1.068105936050415, 2.1491286754608154, 2.4824135303497314, -0.17747607827186584, -0.0849146619439125, -0.112113356590271, 4.245673656463623, -1.1468316316604614, 1.6751800775527954, 0.49764543771743774, -0.36755135655403137, 0.024028154090046883, -0.3932665288448334, 1.8688468933105469, 2.6608316898345947, 0.16982412338256836, -16.69095230102539, 1.2917126417160034, 0.2855507433414459, -0.15253788232803345, 0.18448303639888763, -2.941072940826416, 0.13132934272289276, -32.06040573120117, -0.6923795342445374, 1.2675235271453857, 3.8550310134887695, -0.1320471167564392, -0.21345803141593933, 0.06343041360378265, -0.29768693447113037, 3.1114625930786133, 9.613930702209473, -0.22171512246131897, -0.19839675724506378, -1.1049655675888062, 5.233064651489258, -5.227950096130371, 0.36398845911026, 1.9576587677001953, 0.6924998760223389, 2.0830037593841553, 1.8941112756729126, 1.4372632503509521, -0.9275551438331604, -0.13322152197360992, -1.1561317443847656, -0.06010501831769943, -4.835534572601318, -2.2479310035705566, -0.3606942594051361, -1.5126914978027344, -0.1288832128047943, -2.978395938873291, 0.516731858253479, -0.15324673056602478, -2.614271879196167, -2.693333148956299, -0.15145646035671234, 0.10142246633768082, -2.459259510040283, 3.2511916160583496, -3.754014492034912, 1.3706082105636597, 1.9026187658309937, -2.2486085891723633, 6.385096549987793, 0.6277090907096863, 31.75234603881836, -2.4388110637664795, 0.07844483852386475, 2.7415480613708496, -0.6181740164756775, 3.454662799835205, -0.39441877603530884, 0.612208902835846, -2.7374939918518066, -5.847378730773926, 4.365135669708252, -0.2843222916126251, 0.3086720108985901, 1.343430995941162, -2.7991445064544678, -0.3270379304885864, 0.34932392835617065, -4.630791664123535, -0.1513119637966156, 0.0728594958782196, 0.09281209111213684, -0.27252814173698425, -1.068083643913269, -0.45436856150627136, 0.46403706073760986, -0.1750354915857315, 6.979528427124023, -0.21464569866657257, 0.2106989175081253, 1.0330857038497925, 0.8820008635520935, -0.1885811686515808, -0.04421110078692436, -0.03763798251748085, -1.0573720932006836, -0.5989973545074463, -0.3556733727455139, 1.5424985885620117, -0.1392553746700287, -0.2649906277656555, -0.17105278372764587, 0.907505214214325, 1.1031960248947144, 0.1081002876162529, 9.533098220825195, -0.7259104251861572, 0.21992753446102142, -0.06466154754161835, 0.09757097065448761, 1.5947773456573486, 0.09360354393720627, -8.293384552001953, -0.6469768285751343, 1.127760648727417, -1.7199361324310303, -0.05531356856226921, -0.07445249706506729, 0.07432326674461365, 1.0428595542907715, -0.2194577008485794, -7.437910079956055, -0.03479054942727089, -0.08235757052898407, 1.0363397598266602, 3.1728298664093018, 2.9584403038024902, 0.14671486616134644, 1.3473278284072876, 0.34630176424980164, -0.5547530651092529, -2.182666778564453, 0.9490283727645874, -0.5541707873344421, -0.03097943589091301, -0.2806786596775055, -0.032168395817279816, 3.238935947418213, -1.4938023090362549, 0.16973768174648285, -0.3235575258731842, -0.04206657037138939, -0.8327364921569824, -0.9227404594421387, -0.02287568338215351, -0.6493565440177917, -1.9478867053985596, -0.5021785497665405, 0.04380268231034279, 2.691300868988037, 1.1819796562194824, 2.5491302013397217, -0.7849443554878235, -0.40051624178886414, -1.8154284954071045, 12.998662948608398, 1.044482707977295, -12.965484619140625, 3.333303451538086, 0.04697756841778755, 3.1535658836364746, 1.71531081199646, 3.123339891433716, -0.19728931784629822, 0.3503039479255676, 2.191375732421875, -1.1394171714782715, -2.8017005920410156, 0.49611571431159973, 0.1613306999206543, 0.5855191349983215, 0.8838821649551392, -0.09621147811412811, 0.11256616562604904, 0.4847683608531952, -0.08587711304426193, 0.0302882120013237, 0.05221204087138176, -0.058026138693094254, 2.299950122833252, 3.126528263092041, 0.2845843732357025, -0.13522277772426605, 3.5772054195404053, -0.0972980186343193};

float bias_raw[96]={59.17127227783203, -57.98955154418945, 5.409280776977539, 7.605457305908203, 11.705767631530762, 3.182157039642334, 0.94948410987854, 0.1443750262260437, 1.1615641117095947, -0.9530184864997864, 9.153647422790527, 4.33272647857666, 3.0565507411956787, 0.6363118886947632, 0.6893154382705688, 2.446856737136841, 1.3812376260757446, -3.054840564727783, 2.2883658409118652, 12.799797058105469, 2.06258225440979, 0.9503411650657654, 2.1547372341156006, -0.11983909457921982, -0.5869254469871521, 0.004792630672454834, -1.0285896062850952, 5.2466721534729, 2.736370086669922, 1.828618049621582, 0.4308333098888397, -0.346177875995636, 0.8922613263130188, 8.828324317932129, 12.234149932861328, -0.5219252109527588, 2.0849061012268066, 0.09962782263755798, 1.634371042251587, 1.1393619775772095, 2.263153076171875, 0.1520024985074997, -0.361042320728302, -18.07880210876465, 6.023944854736328, 5.415526866912842, 0.3705480098724365, 5.059598922729492, -0.06533658504486084, -1.014671802520752, 2.977639675140381, 14.528743743896484, 15.089986801147461, -0.42393988370895386, -0.8425414562225342, 11.705090522766113, -0.6844531297683716, -1.206087589263916, -1.584518313407898, 1.8104076385498047, 0.2810867428779602, -0.7307652831077576, -0.9241945743560791, -0.004503533244132996, -0.07907038927078247, -1.5848160982131958, -4.312948226928711, 4.458637714385986, 0.3730209171772003, -2.8507890701293945, 1.7822668552398682, 2.0043938159942627, -1.714074730873108, -0.46794646978378296, 11.767439842224121, 2.8561036586761475, -0.877751886844635, -0.11608101427555084, 0.02182820439338684, -0.25945788621902466, -4.985844135284424, 6.095287322998047, 2.7591590881347656, 11.375081062316895, 2.2520992755889893, -0.7692119479179382, 16.193979263305664, 1.4581878185272217, 1.5961718559265137, 0.538759708404541, -0.45913273096084595, -2.182178497314453, -53.72785186767578, 9.70786190032959, 1.0099848508834839, 12.804798126220703};

float* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
int stride_width=2;
int stride_height=2;
TfLiteFusedActivation activation=kTfLiteActNone;
int dilation_width_factor=1;
int dilation_height_factor=1;
const int filter_dims_size=4;
const int filter_dims_raw[4]={1,3,3,96};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={96};
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

}  // namespace ywrpsy

TfLiteRegistration* Register_ywrpsy_REF() {
  static TfLiteRegistration r = {
      ywrpsy::Init, ywrpsy::Free, ywrpsy::Prepare,
      ywrpsy::Eval<ywrpsy::kReference>};
  return &r;
}

TfLiteRegistration* Register_ywrpsy_GENERIC_OPT() {
  static TfLiteRegistration r = {
      ywrpsy::Init, ywrpsy::Free, ywrpsy::Prepare,
      ywrpsy::Eval<ywrpsy::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_ywrpsy_NEON_OPT() {
  static TfLiteRegistration r = {
      ywrpsy::Init, ywrpsy::Free, ywrpsy::Prepare,
      ywrpsy::Eval<ywrpsy::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_ywrpsy_NEON_OPT_UINT8() {
  static TfLiteRegistration r = {
      ywrpsy::Init, ywrpsy::Free, ywrpsy::Prepare,
      ywrpsy::EvalImpl<ywrpsy::kNeonOptimized, kTfLiteUInt8>};
  return &r;
}

TfLiteRegistration* Register_ywrpsy() {
#ifdef USE_NEON
  return Register_ywrpsy_NEON_OPT();
#else
  return Register_ywrpsy_GENERIC_OPT();
#endif
}

// Warning: Clients using this variant are responsible for ensuring that their
// models only need the UINT8 type. TFLite's op registration mechanism doesn't
// yet allow for more nuanced registration mechanisms.
TfLiteRegistration* Register_ywrpsy_UINT8() {
#ifdef USE_NEON
  return Register_ywrpsy_NEON_OPT_UINT8();
#else
  return Register_ywrpsy();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
