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
namespace soalqg {

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

float filter_r   aw[864]={-0.015095353126525879, -0.0428595244884491, 0.0010513067245483398, -0.03943067416548729, -0.004739671945571899, -0.08404958248138428, -0.06047689914703369, -0.006289847195148468, -0.07327930629253387, 0.0010514333844184875, -0.0553940087556839, -0.007906794548034668, 0.0058748722076416016, -0.027310043573379517, 0.0052909255027771, 0.0008253306150436401, -0.03017902374267578, -0.027213111519813538, -0.029180333018302917, -0.005865812301635742, -0.01846754550933838, 0.0014809109270572662, -0.00858423113822937, -0.00846707820892334, 0.01005929708480835, -0.010386958718299866, -0.0024590305984020233, -0.03189957141876221, -0.004482924938201904, 0.0009297728538513184, -0.005421817302703857, -0.02022993564605713, -0.004168510437011719, -0.03460937738418579, -0.0001799464225769043, -0.016454577445983887, 0.0018615871667861938, -0.007952749729156494, 0.0014943629503250122, -0.038776516914367676, -0.007850408554077148, -0.017023682594299316, -0.008944928646087646, -0.13942259550094604, -0.03077632188796997, -0.020399868488311768, -0.04208946228027344, -0.04195025563240051, -0.019050896167755127, -0.0887046754360199, -0.0052034854888916016, -0.016729503870010376, -0.005084894597530365, -0.011585652828216553, 0.0015316009521484375, -0.006760895252227783, -0.04475194215774536, -0.00574985146522522, -0.024983644485473633, -0.005888521671295166, 0.0035131871700286865, 0.0031614303588867188, -0.014460742473602295, -0.008560240268707275, -0.005781486630439758, -0.012433409690856934, -0.03187834471464157, -0.026329781860113144, -0.03800295293331146, -0.009022772312164307, -0.012499749660491943, -0.014228548854589462, -0.019204676151275635, 0.005226075649261475, -0.0017736852169036865, -0.005431139841675758, 0.006343580782413483, 0.001184225082397461, -0.029776528477668762, -0.009714901447296143, -0.02618342638015747, -0.02576768398284912, 0.011929333209991455, -0.004133045673370361, -0.03912132978439331, -0.019138574600219727, 0.002604067325592041, -0.04507708549499512, -0.00921618938446045, 0.004727840423583984, -0.05704236030578613, 0.004056572914123535, -0.010403841733932495, -0.05209827423095703, -0.014565454795956612, -0.021708395332098007, -0.04623514413833618, -0.03852701187133789, -0.017487749457359314, -0.01888347417116165, -0.0021459460258483887, 0.006942093372344971, -0.011111915111541748, 0.006569162011146545, 0.003078252077102661, -0.002814754843711853, -0.007429331541061401, -0.04132286459207535, -0.039305269718170166, -0.041049107909202576, -0.05134497582912445, -0.006785094738006592, 0.004524707794189453, 0.006066665053367615, -0.010909020900726318, 0.00220392644405365, -0.03976017236709595, 0.0024370551109313965, -0.02535918727517128, -0.0018110275268554688, -0.020672693848609924, -0.014673113822937012, -0.04137580096721649, -0.04037213325500488, -0.036801502108573914, -0.018025845289230347, -0.0926356092095375, -0.0048035383224487305, 0.0028223916888237, -0.008019030094146729, -0.004637777805328369, 0.0006973445415496826, -0.0039276182651519775, -0.012225627899169922, -0.04801592230796814, -0.030745171010494232, -0.010824024677276611, -0.005784869194030762, -0.005386292934417725, -0.022117450833320618, -0.1602574735879898, -0.0008299052715301514, -0.06532204151153564, -0.017438523471355438, 0.0016804337501525879, 0.02515605464577675, -0.0057236552238464355, -0.11835819482803345, -0.017110519111156464, -0.21794888377189636, -0.04366922378540039, 0.024752438068389893, 0.0005584359169006348, -0.05519974231719971, 0.0291459858417511, -0.05594032257795334, -0.012710332870483398, 0.01974564790725708, -0.0010320544242858887, 0.04560995101928711, 0.01683521270751953, -0.0190885066986084, -0.044775962829589844, -0.014730602502822876, -0.007198020815849304, -0.005941152572631836, -0.043807804584503174, -0.0005103945732116699, 0.01916372776031494, -0.051864027976989746, 0.014861449599266052, -0.034218162298202515, -0.02846473455429077, -0.02654576301574707, 0.0054635703563690186, 0.07698392868041992, 0.02826705574989319, -0.07692784070968628, 0.002379298210144043, 0.03981858491897583, -0.0132293701171875, -0.02955174446105957, -0.006524920463562012, -0.007446169853210449, 0.020648300647735596, 0.006215274333953857, 0.007638812065124512, -0.0482444167137146, 0.13117559254169464, 0.004311323165893555, -0.051569849252700806, -0.3619207739830017, 0.08872811496257782, -0.029813319444656372, 0.05240929126739502, -0.00403827428817749, -0.022976040840148926, -0.05807608366012573, 0.0017582178115844727, 0.018345564603805542, 0.01790107786655426, -0.008730113506317139, -0.06962943077087402, -0.0338706374168396, -0.017966747283935547, -0.013150487095117569, -0.0400005541741848, -0.018595509231090546, 0.02079874277114868, -0.04351048916578293, -0.06400638818740845, 0.007690608501434326, -0.08447486162185669, -0.04097504913806915, 0.0473800003528595, 0.02585625648498535, -0.06525719165802002, 0.02448594570159912, 0.03637208417057991, 0.13968360424041748, 0.028041258454322815, 0.023395419120788574, 0.12740349769592285, -0.02984565496444702, 0.017353221774101257, -0.038181185722351074, -0.052836865186691284, 0.008858740329742432, -0.0595550537109375, -0.06479741632938385, 0.00565725564956665, -0.05509069561958313, 0.08961041271686554, -0.003964245319366455, 0.03492927551269531, 0.0035825055092573166, -0.048026442527770996, 0.09046199172735214, -0.08843494951725006, -0.1940193772315979, 0.017338991165161133, -0.07705351710319519, 0.019184529781341553, 0.022825419902801514, 0.02242577075958252, -0.09788136184215546, -0.08401340246200562, 0.013513505458831787, 0.05043056607246399, -0.008322954177856445, 0.015762299299240112, -0.005785956978797913, -0.2349604368209839, 0.06040358543395996, 0.02669825777411461, -0.005615413188934326, 0.011998295783996582, 0.005773603916168213, 0.012440204620361328, -0.03258034586906433, -0.014039158821105957, -0.11044847965240479, -0.09688764810562134, 0.005311250686645508, -0.0137595534324646, -0.057953596115112305, -0.07028317451477051, -0.0025905221700668335, 0.040863700211048126, 0.02681887149810791, -0.0024667680263519287, -0.05587288737297058, 0.027880534529685974, 0.0014834403991699219, 0.031569063663482666, -0.04150313138961792, -0.08131483197212219, 0.039408981800079346, -0.06548652052879333, 0.01325911283493042, 0.026840955018997192, -0.010524213314056396, 0.012465894222259521, 0.04798787832260132, -0.013530075550079346, 0.006308466196060181, 0.03417694568634033, 0.03795748949050903, -0.0792950987815857, -0.170853853225708, 0.05134246125817299, -0.06887388229370117, 0.03144031763076782, 0.036467909812927246, 0.01464390754699707, 0.006601393222808838, 0.09857067465782166, 0.08726155757904053, 0.0380786657333374, 0.028734952211380005, 0.007055789232254028, -0.04382479190826416, 0.0367579311132431, 0.06234954297542572, 0.04814721643924713, -0.02295021153986454, -0.0388454794883728, -0.024299442768096924, 0.17544415593147278, -0.003978371620178223, 0.021458983421325684, 0.04253287985920906, 0.09661686420440674, -0.05709013342857361, 0.007465556263923645, -0.023204147815704346, 0.0020626187324523926, 0.08653849363327026, -0.04111753776669502, -0.01690004952251911, -0.06483268737792969, -0.05332571268081665, -0.013121038675308228, -0.04948157072067261, 0.07367262989282608, 0.010215401649475098, -0.025208592414855957, -0.0727376937866211, 0.00845104455947876, 0.08298328518867493, 0.017223745584487915, -0.2072594165802002, 0.012850835919380188, -0.029481619596481323, -0.2485179603099823, -0.15726137161254883, 0.03860987722873688, 0.005220234394073486, 0.032231226563453674, 0.0224381685256958, -0.0062737613916397095, 0.1592477262020111, 0.02164621651172638, 0.004450388252735138, 0.01060546562075615, -0.0014759600162506104, -0.03011465072631836, -0.04990106821060181, -0.05633632838726044, 0.0033307522535324097, -0.02921992540359497, -0.03945368528366089, 0.018308281898498535, 0.05855396389961243, -0.047657452523708344, -0.01815652847290039, 0.185098335146904, -0.0170120932161808, 0.013014137744903564, 0.08206318318843842, -0.04067432880401611, 0.039865344762802124, 0.06613284349441528, 0.033907413482666016, 0.0599556565284729, 0.005303680896759033, 0.03549906611442566, 0.10548943281173706, -0.09688109159469604, -0.091944620013237, 0.15076708793640137, 0.02537524700164795, 0.11927154660224915, 0.02327651157975197, 0.0036413222551345825, -0.026345372200012207, -0.003439091145992279, 0.007170677185058594, 0.03816181421279907, 0.04298117384314537, 0.024283409118652344, -0.04011571407318115, -0.047878608107566833, -0.08844852447509766, 0.009312830865383148, 0.04386037588119507, 0.06465214490890503, -0.05608242750167847, -0.005928754806518555, 0.010216236114501953, -0.11268103122711182, 0.025434188544750214, -0.07941460609436035, -0.016016174107789993, 0.04529101401567459, -0.07999169826507568, -0.15246136486530304, -0.08118629455566406, 0.0369114875793457, 0.005731672048568726, 0.027292996644973755, 0.06036132574081421, 0.0456567108631134, 0.15374387800693512, 0.02150893211364746, 0.07905608415603638, 0.05120449885725975, -0.05081593990325928, 0.016811460256576538, 0.05758553743362427, -0.20022225379943848, -0.024247318506240845, -0.05201983451843262, -0.07934260368347168, -0.05532100796699524, 0.014107346534729004, 0.06622469425201416, -0.008083641529083252, 0.08706355094909668, -0.02316473424434662, -0.008442610502243042, 0.010962232947349548, -0.01182266604155302, -0.07105407118797302, 0.13776040077209473, -0.01051938533782959, 0.0076486170291900635, 0.017335861921310425, -0.5987507700920105, -0.0189705491065979, -0.06831362098455429, 0.005173683166503906, -0.09317460656166077, -0.06285709142684937, -0.12028098851442337, -0.10550578683614731, 0.0282440185546875, -0.0767974853515625, -0.14136512577533722, -0.019259929656982422, 0.02660006284713745, 0.1427800953388214, -0.02623462677001953, -0.017356455326080322, 0.027444064617156982, -0.1195359155535698, 0.059325337409973145, 0.01741042733192444, 0.08018475770950317, 0.012434661388397217, 0.0999739021062851, 0.023884594440460205, 0.027583539485931396, 0.03706558793783188, -0.012621879577636719, -0.0633135586977005, -0.06026521325111389, 0.037095606327056885, -0.011686742305755615, -0.04769235849380493, 0.0700303316116333, 0.14581328630447388, -0.039393387734889984, -0.06309451162815094, 0.027535445988178253, -0.06997159123420715, -0.0033562779426574707, -0.14412592351436615, -0.08804422616958618, 0.009720876812934875, -0.0180966854095459, 0.0297853946685791, 0.030714035034179688, -0.06686103343963623, 0.010963261127471924, -0.010098874568939209, 0.014253228902816772, 0.034370627254247665, -0.13003218173980713, -0.33887290954589844, -0.09795378893613815, 0.11947961151599884, -0.015629440546035767, -0.041890949010849, 0.02639114111661911, 0.058760687708854675, 0.10263559222221375, -0.031760334968566895, -0.03242617845535278, -0.06182664632797241, -0.0790787935256958, -0.017802387475967407, 0.02180899679660797, 0.027531325817108154, 0.02405300736427307, -0.0028645992279052734, -0.0503736175596714, -0.020797371864318848, 0.10978786647319794, -0.008875448256731033, -0.0586848258972168, 0.05581822991371155, -0.07825082540512085, -0.0353313684463501, -0.08026459813117981, -0.022076405584812164, 0.008102178573608398, 0.08889718353748322, -0.0429500937461853, -0.03992316126823425, -0.06546303629875183, 0.026263833045959473, -0.048526376485824585, 0.035474568605422974, 0.2177247405052185, 0.04183840751647949, 0.25581905245780945, 0.023361682891845703, 0.09871146082878113, -0.07689332962036133, -0.15809696912765503, 0.017565906047821045, -0.009434938430786133, 0.07290470600128174, -0.07132819294929504, -0.3181082308292389, -0.09232199192047119, 0.02838921546936035, -0.009189829230308533, -0.25963759422302246, 0.17132556438446045, 0.018441982567310333, -0.015057414770126343, 0.0851505845785141, -0.12938609719276428, -0.0485055148601532, -0.04374051094055176, -0.008666545152664185, 0.01845085620880127, 0.12358686327934265, -0.4349544048309326, -0.0002949237823486328, 0.11672884225845337, 0.021858274936676025, 0.04060596227645874, 0.045271605253219604, 0.265170156955719, -0.0584293007850647, -0.016718238592147827, 0.10428684949874878, 0.0206032395362854, -0.04268655180931091, 0.030434489250183105, 0.16902673244476318, -0.08785760402679443, -0.03797474503517151, 0.13626223802566528, 0.002438843250274658, -0.14337444305419922, 0.07515564560890198, -0.06874656677246094, 0.0381627082824707, 0.09063832461833954, 0.053678154945373535, -0.06955480575561523, 0.015062466263771057, -0.037687987089157104, 0.10834401845932007, -0.13894861936569214, 0.12549462914466858, 0.03957545757293701, -0.018548687919974327, 0.035631418228149414, 0.020360052585601807, -0.08143143355846405, -0.01067689061164856, -0.07160088419914246, -0.18023782968521118, 0.015878617763519287, -0.027658462524414062, 0.13218604028224945, -0.06014719605445862, -0.05732661485671997, 0.022257328033447266, -0.4049913287162781, 0.08332014083862305, 0.025332242250442505, 0.015173137187957764, 0.03446546196937561, -0.06904077529907227, 0.044890522956848145, -0.006187424063682556, -0.06619775295257568, -0.015504598617553711, 0.03461506962776184, 0.029545580968260765, -0.114769846200943, 0.05268362909555435, 0.029316112399101257, 0.02438145875930786, 0.005658149719238281, -0.023357152938842773, -0.03803553432226181, 0.015811540186405182, 0.11180960386991501, 0.11780226230621338, -0.056011974811553955, 0.12547099590301514, -0.0482504665851593, 0.007684767246246338, 0.0075991153717041016, -0.07494905591011047, -0.00965970754623413, -0.0446358323097229, -0.00789131224155426, -0.05806511640548706, -0.04266369342803955, -0.028201162815093994, 0.02065110206604004, -0.10940608382225037, -0.12263202667236328, 0.008075237274169922, -0.07871353626251221, 0.04027557373046875, -0.07150182127952576, -0.019558340311050415, -0.051185548305511475, -0.0250433087348938, 0.0013616681098937988, 0.030461043119430542, 0.03276391699910164, -0.04802674055099487, 0.005784273147583008, -0.009080231189727783, -0.008833136409521103, 0.004760235548019409, 0.015590190887451172, 0.04193305969238281, -0.0024545788764953613, 0.027503252029418945, -0.008058309555053711, 0.07173144817352295, 0.0915052592754364, 0.021842658519744873, 0.07826054096221924, -0.011406689882278442, 0.05033619701862335, -0.030971527099609375, 0.0034426909405738115, -0.012933880090713501, 0.04118847846984863, -0.0053348541259765625, -0.0202617347240448, -0.022526979446411133, -0.041778258979320526, 0.07053279876708984, -0.08084499835968018, 0.023110508918762207, 0.17952227592468262, 0.03811967372894287, -0.08881330490112305, -0.04307621717453003, 0.04194146394729614, -0.10833655297756195, -0.10139784961938858, 0.015435755252838135, 0.04249405860900879, 0.03857874870300293, 0.026386983692646027, -0.2468021661043167, 0.003096815198659897, 0.10682904720306396, -0.0009771287441253662, 0.0020963549613952637, -0.031223412603139877, -0.04493054747581482, 0.04818141460418701, 0.0458281934261322, -0.12922000885009766, 0.025302372872829437, 0.08083849400281906, 0.0017905235290527344, 0.04909941554069519, 0.0017586424946784973, 0.007027924060821533, 0.04625201225280762, -0.08586388826370239, -0.04634284973144531, 0.05084264278411865, 0.02964729070663452, 0.07803982496261597, 0.06916901469230652, 0.02877408266067505, 0.1469610631465912, -0.02321314811706543, -0.006662428379058838, -0.02338915690779686, -0.018103577196598053, -0.08520147204399109, -0.015919625759124756, 0.12478172779083252, 0.05848778411746025, 0.14223480224609375, 0.031199276447296143, -0.036179184913635254, -0.0004990408197045326, 0.04412004351615906, 0.01843234896659851, -0.030087411403656006, 0.0051138997077941895, 0.007280528545379639, -0.01962883025407791, 0.028789162635803223, -0.008244544267654419, -0.0033215954899787903, 0.01264965534210205, -0.15492737293243408, 0.04353731870651245, -0.06161832809448242, -0.010947033762931824, -0.04894596338272095, 0.019327878952026367, -0.042410846799612045, -0.1016380563378334, -0.07020241022109985, -0.1382763832807541, -0.05973348021507263, 0.01567794382572174, -0.13709644973278046, -0.02232462167739868, 0.02909436821937561, -0.05683094263076782, 0.03125488758087158, 0.015633821487426758, 0.050508320331573486, 0.05503129959106445, -0.08870413154363632, 0.0017184019088745117, 0.13883066177368164, 0.0036836862564086914, -0.09110245853662491, 0.01652141660451889, -0.06179922819137573, -0.044801563024520874, 0.010210782289505005, -0.008878801017999649, -0.002141714096069336, -0.03310095891356468, 0.04283636808395386, 0.008799295872449875, -0.03852847218513489, 0.22526051104068756, 0.04425304755568504, -0.07236436009407043, -0.011679351329803467, -0.05841811001300812, -0.03631851077079773, -0.06375227868556976, -0.009785950183868408, 0.040128350257873535, 0.0640823245048523, 0.04598149657249451, 0.06664049625396729, -0.09608480334281921, 0.11802291870117188, 0.008091434836387634, 0.12444490194320679, -0.4305393695831299, -0.0031532645225524902, 0.021236222237348557, 0.07082194089889526, -0.11400896310806274, 0.028212830424308777, -0.02699589729309082, 0.03740935027599335, -0.030407190322875977, -0.002319127321243286, -0.0029593706130981445, -0.0887683629989624, -0.0011142045259475708, -0.006033718585968018, -0.0022726207971572876, -0.01724874973297119, -0.0026670098304748535, -0.003931283950805664, -0.0042054057121276855, -0.013359218835830688, -0.021909236907958984, 0.13259194791316986, -0.025247134268283844, -0.006135798990726471, -0.015457868576049805, -0.03551936149597168, -0.0017834305763244629, 0.003969132900238037, 0.14030233025550842, -0.00042438507080078125, 0.008320052176713943, 0.026538193225860596, -0.02351441979408264, -0.13114655017852783, 0.04856002330780029, -0.02451363205909729, 0.029587596654891968, -0.18802952766418457, -0.03793513774871826, 0.02269263193011284, -0.05461139976978302, 0.0004119277000427246, 0.01247696578502655, 0.05226945877075195, 0.017285585403442383, -0.017473161220550537, -0.00784081220626831, -0.04881531000137329, -0.09003090858459473, 0.0031761527061462402, -0.17483283579349518, 0.04880926012992859, 0.03250804543495178, 0.03638988733291626, 0.009150713682174683, -0.23394760489463806, 0.007100939750671387, 0.05529826879501343, -0.017594754695892334, 0.05504658818244934, -0.060815632343292236, -0.42875680327415466, 0.05103844404220581, 0.05074930191040039, -0.006767392158508301, 0.00994899868965149, 0.03440678119659424, 0.09344649314880371, -0.1298278570175171, -0.020688533782958984, -0.07234853506088257, -0.10120880603790283, -0.013495653867721558, 0.04751461744308472, -0.028651438653469086, 0.009869851171970367, -0.03762061893939972, 0.11285728961229324, 0.10725045204162598, 0.039747051894664764, -0.04566190391778946, -0.0036128759384155273, 0.016968905925750732, 0.13326990604400635, 0.007437972351908684, -0.09427028894424438, 0.02532505989074707, -0.13240879774093628, -0.05820053815841675, 0.05001053214073181, 0.029398798942565918, 0.02212364412844181, 0.03038366138935089, -0.06239960715174675, 0.030149996280670166, 0.002073311246931553, 0.05632978677749634};

float bias_raw[6]={-0.009088503196835518, 0.00335528957657516, 0.004077547695487738, -0.004310850985348225, 0.0034041856415569782, 0.0039685978554189205};

float* filter_tensor_data=filter_raw;
float* bias_tensor_data=bias_raw;

bool has_conv_bias=true;
const int stride_width=1;
const int stride_height=1;
const TfLiteFusedActivation activation=kTfLiteActNone;
const int dilation_width_factor=1;
const int dilation_height_factor=1;
const int filter_dims_size=4;
const int32_t filter_dims_raw[4]={6,1,1,144};
const int bias_dims_size=1;
const int32_t bias_dims_raw[1]={6};
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

TfLiteRegistration* Register_soalqg_REF() {
  static TfLiteRegistration r = {soalqg::Init, soalqg::Free,
                                 soalqg::Prepare<soalqg::kReference>,
                                 soalqg::Eval<soalqg::kReference>};
  return &r;
}

TfLiteRegistration* Register_soalqg_GENERIC_OPT() {
  static TfLiteRegistration r = {soalqg::Init, soalqg::Free,
                                 soalqg::Prepare<soalqg::kGenericOptimized>,
                                 soalqg::Eval<soalqg::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_soalqg_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {soalqg::Init, soalqg::Free,
                                 soalqg::Prepare<soalqg::kMultithreadOptimized>,
                                 soalqg::Eval<soalqg::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_soalqg_CBLAS_OPT() {
//   static TfLiteRegistration r = {soalqg::Init, soalqg::Free,
//                                  soalqg::Prepare<soalqg::kCblasOptimized>,
//                                  soalqg::Eval<soalqg::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_soalqg() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_soalqg_MULTITHREADED_OPT();
#else
  return Register_soalqg_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
