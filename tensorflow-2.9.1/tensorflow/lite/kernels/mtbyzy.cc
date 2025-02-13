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
namespace mtbyzy {

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

float filter_r   aw[864]={-0.014699339866638184, -0.04749464988708496, -0.011449426412582397, 0.3148685693740845, -0.007698625326156616, 0.02566009759902954, -0.009619802236557007, -0.2486923336982727, 0.1984156370162964, -0.11999911069869995, 0.0870441347360611, 0.03507325425744057, 0.11535951495170593, 0.18542122840881348, 0.10763031244277954, -0.07996949553489685, -0.3949364721775055, 0.13873529434204102, 0.31243282556533813, -0.06053560972213745, 0.09488679468631744, -0.288049578666687, -0.14771735668182373, -0.1895158290863037, 0.06412944197654724, 0.5386767983436584, -0.27787765860557556, -0.06330263614654541, -0.03427669405937195, 0.3026890754699707, -0.04410802572965622, 0.1365625560283661, -0.14908015727996826, 0.1353365182876587, 0.046027302742004395, 0.35961979627609253, 0.06380271911621094, 0.05834871530532837, -0.11137132346630096, -0.1842753142118454, 0.2134617567062378, 0.16263407468795776, 0.13094985485076904, -0.1073344349861145, -0.011848792433738708, 0.2627111077308655, -0.20065458118915558, -0.10226130485534668, -0.23895204067230225, 0.35316240787506104, 0.005952388048171997, -0.15814626216888428, -0.07916635274887085, -0.0045198798179626465, 0.05552172660827637, -0.1489548683166504, 0.10857680439949036, 0.10777906328439713, -0.32352107763290405, 0.07748544216156006, 0.18650010228157043, -0.0840269923210144, 0.12220817804336548, -0.08947432041168213, -0.0490572452545166, 0.31592732667922974, -0.47642743587493896, -0.02405869960784912, -0.0428922176361084, -0.10954184830188751, 0.16481786966323853, -0.004519909620285034, -0.24974191188812256, 0.018982797861099243, -0.10194739699363708, 0.02890928089618683, 0.006041973829269409, 0.04117676615715027, 0.06548592448234558, 0.15294691920280457, 0.029805243015289307, -0.23846346139907837, 0.04950651526451111, 0.14528053998947144, 0.12499481439590454, 0.12136530876159668, -0.006043851375579834, -0.0022264420986175537, 0.511655867099762, 0.4113798141479492, -0.22350817918777466, -0.00244266539812088, 0.21445977687835693, -0.05923167243599892, -0.38870084285736084, -0.15828469395637512, 0.023886144161224365, 0.15264476835727692, -0.318241149187088, 0.07298952341079712, -0.004610657691955566, -0.47950515151023865, 0.16065669059753418, -0.1673092246055603, -0.16504526138305664, 0.24639740586280823, 0.1133112758398056, -0.3472985029220581, 0.1651771068572998, -0.04855072498321533, -0.058549463748931885, 0.8517535924911499, 0.05269303917884827, -0.08205920457839966, 0.3614541292190552, 0.29357674717903137, 0.001980006694793701, -0.003118187189102173, 0.1771502047777176, 0.12189573049545288, -0.018921345472335815, 0.055037498474121094, -0.17054615914821625, -0.018122315406799316, -0.16034282743930817, 0.15784823894500732, -0.10505819320678711, -0.11667740345001221, 0.0043502748012542725, 0.22267749905586243, 0.11239645630121231, -0.018674790859222412, -0.10343953967094421, 0.09567499160766602, 0.2491600513458252, -0.2973153591156006, -0.019534170627593994, -0.061573922634124756, 0.06465333700180054, 0.05827859044075012, 0.137223482131958, 0.036433666944503784, -0.20223718881607056, -0.047972679138183594, 0.11446571350097656, -0.02259214222431183, 0.3631347417831421, -0.3191954493522644, -0.22018754482269287, 0.12286075949668884, 0.054419346153736115, -0.15602374076843262, -0.12660115957260132, 0.09220591932535172, 0.1273748278617859, -0.0835438072681427, -0.2284317910671234, -0.09812575578689575, 0.08372144401073456, 0.22627031803131104, 0.0867675393819809, 0.026576131582260132, 0.12289711833000183, -0.0683005154132843, 0.12684136629104614, 0.11672765016555786, 0.02425748109817505, -0.012537037953734398, -0.016477778553962708, 0.11400522291660309, 0.07759323716163635, 0.06211352348327637, -0.14295095205307007, -0.011366784572601318, 0.12368476390838623, 0.04735457897186279, 0.10094445943832397, -0.1307772397994995, 0.4899240732192993, 0.054458022117614746, 0.12624412775039673, 0.0857202410697937, 0.7489688992500305, 0.32418137788772583, -0.520063579082489, -0.08345019072294235, -0.20761147141456604, -0.019447896629571915, 0.07488441467285156, -0.22006452083587646, 0.21992278099060059, 0.007366538047790527, -0.10141527652740479, -0.004949986934661865, 0.08340442180633545, 0.12498188018798828, 0.05194465443491936, -0.019489610567688942, -0.16844993829727173, -0.0437237024307251, 0.23873895406723022, 0.16041414439678192, -0.10257953405380249, -0.01366567611694336, 0.3279050588607788, 0.16915667057037354, -0.23909196257591248, 0.17116600275039673, -0.011553466320037842, -0.047079503536224365, 0.18012911081314087, 0.16032728552818298, -0.26092851161956787, 0.044529467821121216, 0.009731777012348175, -0.019489586353302002, 0.08651602268218994, -0.10246545076370239, -0.03790371119976044, -0.3096815049648285, 0.3873951733112335, -0.005455732345581055, 0.09968787431716919, -0.012639284133911133, -0.014006704092025757, 0.15681594610214233, -0.24903464317321777, -0.08497679233551025, -0.2540714144706726, 0.09035766124725342, 0.024537518620491028, -0.19213223457336426, -0.05915377661585808, -0.07323503494262695, 0.06139403581619263, 0.05610349774360657, -0.2837381362915039, -0.11796128749847412, -0.2757900357246399, 0.10036647319793701, 0.09819349646568298, 0.09117335081100464, -0.1301150619983673, 0.19682276248931885, 0.08641286194324493, -0.21595390141010284, 0.026127219200134277, 0.13245344161987305, 0.1366783082485199, 0.08397144079208374, 0.013927668333053589, 0.32846733927726746, -0.027016282081604004, -0.16946643590927124, 0.037199050188064575, -0.08025467395782471, 0.38644957542419434, 0.25274479389190674, 0.09502121806144714, -0.04280960559844971, 0.13973009586334229, 0.4876730144023895, -0.07894279062747955, 0.16491633653640747, 0.0889577567577362, -0.014085739850997925, 0.11025381088256836, 0.0751180350780487, -0.14707672595977783, 0.2687508165836334, -0.11791253089904785, -0.10042673349380493, -0.06249365210533142, 0.034761130809783936, -0.24161005020141602, -0.031269609928131104, -0.09898507595062256, 0.03626558184623718, -0.1350826621055603, -0.141934335231781, -0.00996631383895874, -0.6027622818946838, -0.18429327011108398, 0.1761804074048996, 0.25614187121391296, 0.15890976786613464, 0.060662925243377686, 0.023418903350830078, 0.09493446350097656, -0.6095269918441772, 0.352338969707489, -0.29529398679733276, -0.5231490135192871, 0.10181832313537598, 0.19590695202350616, 0.04285365343093872, 0.00019121170043945312, 0.12092487514019012, 0.03572586178779602, 0.2511361837387085, -0.1275154948234558, -0.5214056372642517, 0.36292701959609985, -0.03269527107477188, 0.2260676622390747, 0.033580929040908813, -0.00042290985584259033, 0.06687432527542114, -0.0967443585395813, 0.015558801591396332, 0.1887056827545166, -0.11261618137359619, -0.03225491940975189, -0.2319498062133789, -0.13492241501808167, 0.22316141426563263, 0.2156999409198761, 0.4056205749511719, 0.16812586784362793, 0.3041739761829376, 0.17405259609222412, -0.03711117058992386, -0.017343997955322266, 0.10890918970108032, 0.03838831186294556, 0.10716252774000168, -0.21408915519714355, 0.2701718807220459, -0.3265441358089447, 0.0784408301115036, -0.12027734518051147, 0.1062927395105362, -0.004612600896507502, -0.2235645055770874, 0.11964273452758789, 0.008626550436019897, 0.07392317056655884, -0.23991024494171143, 0.053112566471099854, 0.1519145369529724, 0.09119200706481934, -0.03359532356262207, 0.23797324299812317, -0.3759691119194031, 0.1181645393371582, -0.011263608932495117, -0.07455170154571533, -0.11470836400985718, 0.3049411177635193, -0.01187753677368164, 0.0738285556435585, -0.1702321469783783, 0.05426698923110962, 0.0006909966468811035, 0.024116158485412598, 0.3162275552749634, 0.5922471284866333, -0.009878158569335938, -0.028429776430130005, -0.03359535336494446, -0.053298354148864746, 0.10938818752765656, -0.15760737657546997, -0.3034658133983612, -0.19214767217636108, 0.056238457560539246, 0.07577631622552872, 0.08769837021827698, -0.09987866878509521, 0.06789679825305939, -0.15457716584205627, -0.14946629106998444, 0.19934451580047607, 0.2477143406867981, 0.08312427997589111, -0.13895833492279053, -0.06723380088806152, -0.0007853507995605469, -0.1274344027042389, -0.037185072898864746, -0.04300931096076965, 0.09790587425231934, -0.23535510897636414, -0.028521284461021423, -0.04742872714996338, 0.3930774927139282, -0.028817355632781982, 0.25288110971450806, -0.012311860918998718, -0.35702812671661377, 0.08335134387016296, 0.061826109886169434, 0.012145042419433594, -0.011917829513549805, 0.007899656891822815, 0.06527949869632721, 0.061385512351989746, -0.007460951805114746, 0.13236825168132782, -0.2658196985721588, 0.2788975238800049, 0.09962421655654907, -0.07088524103164673, -0.22965295612812042, 0.09179381281137466, 0.011332869529724121, 0.15704500675201416, 0.2748531401157379, 0.042839184403419495, -0.15848839282989502, -0.05905219912528992, 0.08360576629638672, -0.047126367688179016, 0.2477411925792694, -0.10222893953323364, 0.010587923228740692, 0.07644754648208618, 0.17215824127197266, 0.1106719821691513, -0.044182419776916504, -0.23976454138755798, -0.18910545110702515, 0.21823854744434357, -0.007763862609863281, -8.64267349243164e-06, -0.022050917148590088, 0.11336353421211243, 0.06581580638885498, 0.27164050936698914, 0.09790509939193726, -0.03221310302615166, -0.07921318709850311, -0.08098414540290833, 0.03452003002166748, -0.5179277062416077, 0.3252146244049072, -0.2626325190067291, 0.061836108565330505, -0.004915893077850342, -0.07272464036941528, 0.09109821170568466, -0.07559281587600708, -0.15195482969284058, 0.048652589321136475, -0.07851657271385193, -0.21114525198936462, 0.04036429524421692, 0.21763396263122559, -0.1736060380935669, -0.007246104069054127, -0.16492986679077148, 0.2301299124956131, 0.33378326892852783, 0.18706166744232178, 0.17207688093185425, 0.10057884454727173, 0.1065753698348999, -0.4744359254837036, -0.03902614116668701, -0.1493363380432129, 0.0490172803401947, 0.12898534536361694, 0.1888432502746582, 0.06595629453659058, 0.08991315960884094, 0.08611637353897095, 0.034606240689754486, -0.19263003766536713, 0.10600742697715759, -0.08003596216440201, -0.11571993678808212, -0.23890888690948486, 0.04165707528591156, -0.04782538861036301, -0.08616000413894653, -0.0993470847606659, -0.08654659986495972, -0.35676151514053345, -0.06859821081161499, -0.04240337014198303, 0.1670161485671997, -0.4699627757072449, 0.0736498236656189, -0.1121143102645874, 0.3283286988735199, -0.04626324772834778, -0.10360515117645264, 0.10219168663024902, 0.2441357970237732, -0.09089738130569458, 0.007214069366455078, -0.1893930584192276, -0.24645689129829407, -0.10548481345176697, 0.29486456513404846, -0.061984628438949585, 0.2187732458114624, -0.3534962832927704, -0.2902304530143738, 0.056338295340538025, 0.2542416453361511, 0.10291431844234467, -0.22123795747756958, -0.04626324772834778, 0.23459267616271973, 0.08461892604827881, 0.04257279261946678, 0.05010862648487091, -0.03360253572463989, -0.20065852999687195, -0.23215611279010773, -0.1460331678390503, -0.07072734832763672, 0.07769119739532471, 0.16348320245742798, -0.057536277920007706, -0.1937347650527954, -0.07721447944641113, 0.03572356700897217, 0.08921229839324951, 0.49893274903297424, 0.0006938278675079346, -0.0626254677772522, 0.07651837170124054, 0.012447893619537354, -0.13865208625793457, -0.006916999816894531, -0.2547319531440735, 0.19855213165283203, 0.054769277572631836, 0.0687384083867073, -0.09827744960784912, -0.1620711088180542, 0.357489675283432, -0.15953218936920166, 0.11542704701423645, 0.05356171727180481, -0.11874186992645264, 0.0030331015586853027, 0.06275969743728638, -0.04639892280101776, 0.03749990463256836, 0.13717755675315857, 0.5669467449188232, -0.05361020565032959, 0.16084051132202148, -0.1726890206336975, 0.05189046263694763, 0.10714423656463623, -0.017121553421020508, 0.04685002565383911, -0.1301725208759308, 0.2569079101085663, 0.046316828578710556, 0.08943641185760498, -0.044472672045230865, 0.0807415246963501, -0.231155663728714, 0.034339070320129395, 0.138545423746109, 0.005781203508377075, 0.16130231320858002, 0.0052370429039001465, 0.20005851984024048, -0.028031110763549805, 0.014669597148895264, 0.05880039930343628, 0.05735325813293457, -0.17063945531845093, 0.025247812271118164, -0.004715085029602051, 0.15938255190849304, -0.051914364099502563, -0.15242469310760498, 0.06228846311569214, -0.030796945095062256, 0.3128858804702759, 0.15569376945495605, 0.1487264633178711, -1.009557843208313, -0.052473463118076324, 0.1336221694946289, 0.07908958196640015, 0.10486552119255066, -0.03128761053085327, -0.12864232063293457, 0.06404551863670349, 0.2088235318660736, -0.13500571250915527, -0.21556752920150757, -0.07944214344024658, 0.14846444129943848, 0.036082684993743896, 0.06118607521057129, -0.07555603981018066, 0.13252022862434387, 0.07037043571472168, 0.11157486587762833, -0.018876759335398674, 0.023913651704788208, 0.00444483757019043, 0.33895665407180786, 0.2359645813703537, 0.23351359367370605, -0.06172114610671997, -0.37803423404693604, 0.14705947041511536, -0.39531832933425903, 0.1699601113796234, 0.1767447292804718, 0.1079181432723999, -0.07969415187835693, 0.017131686210632324, 0.0694844126701355, -0.4522845149040222, -0.17683196067810059, 0.24668139219284058, -0.17134864628314972, -0.06512241065502167, 0.0016950368881225586, 0.13426560163497925, -0.20222049951553345, 0.19225209951400757, 0.1010059118270874, -0.474772572517395, 0.6511019468307495, 0.12950125336647034, 0.11671656370162964, -0.45962536334991455, -0.009099185466766357, 0.1312544345855713, -0.3170211911201477, -0.40212738513946533, 0.02567809820175171, 0.33462995290756226, -0.018292129039764404, 0.6295878887176514, 0.11969637870788574, -0.27835848927497864, -0.2727568447589874, 0.1437549591064453, -0.4155251979827881, 0.3729667663574219, 0.17205341160297394, -0.5603779554367065, 0.19568771123886108, -0.05038714408874512, -0.009099185466766357, -0.37647101283073425, 0.03651154041290283, -0.1956591010093689, -0.18308040499687195, -0.14030589163303375, 0.42983296513557434, 0.06093838810920715, 0.038692593574523926, -0.1833178699016571, 0.3603517413139343, -0.13482797145843506, -0.17908143997192383, -0.06658706068992615, 0.011994600296020508, 0.08305859565734863, 0.19221502542495728, 0.06724919378757477, -0.2640460133552551, 0.10983221977949142, 0.17465436458587646, -0.18079887330532074, 0.2221471667289734, 0.3064610958099365, 0.396348237991333, 0.23344504833221436, -0.3962371349334717, 0.0903761088848114, -0.13650989532470703, 0.07146146893501282, -0.08424857258796692, -0.123898446559906, 0.20734375715255737, 0.14396017789840698, -0.079100102186203, 0.07305024564266205, 0.38766762614250183, -0.05763232707977295, -0.12060490250587463, 0.03232095390558243, -0.2905428111553192, -0.3662903308868408, 0.15549957752227783, 0.07161366939544678, -0.15494304895401, 0.11070835590362549, -0.29368674755096436, -0.07188490033149719, -0.21985365450382233, -0.24452456831932068, -0.15883833169937134, 0.07442905008792877, 0.34821856021881104, 0.09848588705062866, 0.5357227921485901, -0.06482148170471191, 0.22046151757240295, 0.0039275288581848145, -0.23392435908317566, -0.023709088563919067, 0.15782344341278076, -0.07408393919467926, 0.0438157320022583, -0.18361762166023254, 0.02744944393634796, -0.11451441049575806, 0.022258341312408447, -0.06222856044769287, 0.2333507239818573, -0.10982300341129303, 0.15512816607952118, 0.06738348305225372, -0.06896543502807617, 0.03787001967430115, -0.135908842086792, 0.586621105670929, 0.038502037525177, 1.228875756263733, 0.06607121229171753, -0.007929325103759766, 0.03950393199920654, 0.013289161026477814, -0.003907322883605957, -0.1849815845489502, -0.09877032041549683, -0.21491432189941406, 0.9876495599746704, 0.06261754035949707, 0.16359609365463257, -0.046500030905008316, -0.004332780838012695, -0.07305455207824707, 0.0036303550004959106, 0.009684361517429352, 0.07162103056907654, 0.10235404968261719, 0.11241662502288818, 0.03265221416950226, -0.4107084274291992, -0.18969929218292236, 0.13672032952308655, -0.18018627166748047, -0.09836915135383606, -0.005564630031585693, 0.11156338453292847, 0.16468384861946106, 0.26345404982566833, -0.10594376176595688, 0.07033431529998779, -0.039643675088882446, -0.23490649461746216, -0.02508389949798584, 0.1139877438545227, -0.11129915714263916, 0.10430747270584106, -0.010442480444908142, -0.009830117225646973, -0.03341878205537796, -0.06507182121276855, -0.043268799781799316, -0.04903951287269592, -0.09171074628829956, -0.09372878074645996, 0.010930895805358887, 0.06269499659538269, -0.10108918696641922, -0.019485145807266235, 0.19528549909591675, 0.5222052335739136, 0.08359843492507935, -0.09479142725467682, -0.051648736000061035, 0.04771453142166138, -0.289023220539093, -0.21899783611297607, 0.11690941452980042, 0.5795708894729614, 0.014788806438446045, -0.13432225584983826, -0.12893369793891907, -0.1585685908794403, -0.0932992696762085, 0.016841769218444824, -0.10353511571884155, -0.019485145807266235, 0.08598995208740234, -0.19961130619049072, 0.2911190092563629, 0.1643948256969452, -0.03691709041595459, -0.06816934049129486, -0.1209154725074768, -0.1443522572517395, -0.12302374839782715, -0.16025567054748535, -0.007786998525261879, -0.15686383843421936, 0.14595045149326324, -0.027923762798309326, -0.12012916803359985, 0.17685483396053314, 0.1772986352443695, -0.03495673090219498, 0.0012639164924621582, 0.12314683198928833, -0.053082823753356934, -0.018064886331558228, -0.08180657029151917, -0.21673020720481873, 0.04313596338033676, 0.07213446497917175, -0.22209089994430542, 0.16967560350894928, -0.07334500551223755, 0.01144912838935852, -0.09196668863296509, -0.034724801778793335, -0.28806567192077637, -0.047482430934906006, 0.06184755265712738, 0.02617734670639038, 0.01537853479385376, 0.056633591651916504, -0.0838397741317749, 0.019380271434783936, 0.12775801122188568, 0.04311145842075348, -0.10978209972381592, 0.13107118010520935, 0.10267472267150879, 0.11214625835418701, 0.15343765914440155, -0.09239090234041214, -0.04292910918593407, -0.23443621397018433, -0.028210069984197617, 0.20223617553710938, 0.049984052777290344, 0.05600067973136902, -0.1517741084098816, 0.0815589427947998, 0.052728891372680664, 0.010454904288053513, 0.40696051716804504, 0.009274430572986603, -0.045845210552215576, -0.14038574695587158, 0.010181205347180367, -0.3220316171646118, -0.16474032402038574, -0.1255439817905426, -0.0018000602722167969, 0.20738554000854492, 0.2832117974758148, -0.16136467456817627, -0.026226341724395752, 0.014499634504318237};

float bias_raw[6]={0.016517898067831993, 0.0693313330411911, 0.12043385952711105, 0.16499747335910797, 0.032580722123384476, 0.06955701112747192};

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

TfLiteRegistration* Register_mtbyzy_REF() {
  static TfLiteRegistration r = {mtbyzy::Init, mtbyzy::Free,
                                 mtbyzy::Prepare<mtbyzy::kReference>,
                                 mtbyzy::Eval<mtbyzy::kReference>};
  return &r;
}

TfLiteRegistration* Register_mtbyzy_GENERIC_OPT() {
  static TfLiteRegistration r = {mtbyzy::Init, mtbyzy::Free,
                                 mtbyzy::Prepare<mtbyzy::kGenericOptimized>,
                                 mtbyzy::Eval<mtbyzy::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_mtbyzy_MULTITHREADED_OPT() {
  static TfLiteRegistration r = {mtbyzy::Init, mtbyzy::Free,
                                 mtbyzy::Prepare<mtbyzy::kMultithreadOptimized>,
                                 mtbyzy::Eval<mtbyzy::kMultithreadOptimized>};
  return &r;
}

// TfLiteRegistration* Register_mtbyzy_CBLAS_OPT() {
//   static TfLiteRegistration r = {mtbyzy::Init, mtbyzy::Free,
//                                  mtbyzy::Prepare<mtbyzy::kCblasOptimized>,
//                                  mtbyzy::Eval<mtbyzy::kCblasOptimized>};
//   return &r;
// }

TfLiteRegistration* Register_mtbyzy() {
#if defined TFLITE_WITH_MULTITHREADED_EIGEN
  return Register_mtbyzy_MULTITHREADED_OPT();
#else
  return Register_mtbyzy_GENERIC_OPT();
#endif
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
