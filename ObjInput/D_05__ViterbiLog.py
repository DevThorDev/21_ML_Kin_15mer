# -*- coding: utf-8 -*-
###############################################################################
# --- D_05__ViterbiLog.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Viterbi algoritm for sequences [log(prob.)] (D_05__ViterbiLog)'
sNmSpec = 'Input data for the ViterbiLog class in O_05__ViterbiLog'

# --- flow control ------------------------------------------------------------
doViterbi = True

useFullSeqFrom = GC.S_COMB_INP    # S_PROC_INP / S_COMB_INP

# --- names and paths of files and dirs ---------------------------------------
sFInpStartProb = 'SglPosStartProb'
sFInpTransProb = 'SglPosTransProb'

sFInpFullSeq = GC.S_F_PROC_INP_N_MER
if useFullSeqFrom == GC.S_COMB_INP:
    sFInpFullSeq = 'Combined_S_KinasesPho15mer_202202'

sFOptStatePath = 'OptimalStatePath'
sFPosPyl = 'PylPositions'
sFProbDetail = 'ProbDetail'
sFProbFinal = 'ProbFinal'

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sProb = GC.S_PROB
sPrev = GC.S_PREV
sState = GC.S_STATE
sStartProb = GC.S_START_PROB
sLnProb = GC.S_LN_PROB

# --- sets --------------------------------------------------------------------
setCond = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

# --- lists -------------------------------------------------------------------
# lO01 = list('MSGSRRKATPASRTRVGNYEMGRTLGEGSFAKVKYAKNTVTGDQAAIKILDREKVFRHKMVEQ' +
#             'LKREISTMKLIKHPNVVEIIEVMASKTKIYIVLELVNGGELFDKIAQQGRLKEDEARRYFQQL' +
#             'INAVDYCHSRGVYHRDLKPENLILDANGVLKVSDFGLSAFSRQVREDGLLHTACGTPNYVAPE' +
#             'VLSDKGYDGAAADVWSCGVILFVLMAGYLPFDEPNLMTLYKRICKAEFSCPPWFSQGAKRVIK' +
#             'RILEPNPITRISIAELLEDEWFKKGYKPPSFDQDDEDITIDDVDAAFSNSKECLVTEKKEKPV' +
#             'SMNAFELISSSSEFSLENLFEKQAQLVKKETRFTSQRSASEIMSKMEETAKPLGFNVRKDNYK' +
#             'IKMKGDKSGRKGQLSVATEVFEVAPSLHVVELRKTGGDTLEFHKFYKNFSSGLKDVVWNTDAA' +
#             'AEEQKQ')
# lO02 = list('MTSLLKSSPGRRRGGDVESGKSEHADSDSDTFYIPSKNASIERLQQWRKAALVLNASRRFRYT' +
#             'LDLKKEQETREMRQKIRSHAHALLAANRFMDMGRESGVEKTTGPATPAGDFGITPEQLVIMSK' +
#             'DHNSGALEQYGGTQGLANLLKTNPEKGISGDDDDLLKRKTIYGSNTYPRKKGKGFLRFLWDAC' +
#             'HDLTLIILMVAAVASLALGIKTEGIKEGWYDGGSIAFAVILVIVVTAVSDYKQSLQFQNLNDE' +
#             'KRNIHLEVLRGGRRVEISIYDIVVGDVIPLNIGNQVPADGVLISGHSLALDESSMTGESKIVN' +
#             'KDANKDPFLMSGCKVADGNGSMLVTGVGVNTEWGLLMASISEDNGEETPLQVRLNGVATFIGS' +
#             'IGLAVAAAVLVILLTRYFTGHTKDNNGGPQFVKGKTKVGHVIDDVVKVLTVAVTIVVVAVPEG' +
#             'LPLAVTLTLAYSMRKMMADKALVRRLSACETMGSATTICSDKTGTLTLNQMTVVESYAGGKKT' +
#             'DTEQLPATITSLVVEGISQNTTGSIFVPEGGGDLEYSGSPTEKAILGWGVKLGMNFETARSQS' +
#             'SILHAFPFNSEKKRGGVAVKTADGEVHVHWKGASEIVLASCRSYIDEDGNVAPMTDDKASFFK' +
#             'NGINDMAGRTLRCVALAFRTYEAEKVPTGEELSKWVLPEDDLILLAIVGIKDPCRPGVKDSVV' +
#             'LCQNAGVKVRMVTGDNVQTARAIALECGILSSDADLSEPTLIEGKSFREMTDAERDKISDKIS' +
#             'VMGRSSPNDKLLLVQSLRRQGHVVAVTGDGTNDAPALHEADIGLAMGIAGTEVAKESSDIIIL' +
#             'DDNFASVVKVVRWGRSVYANIQKFIQFQLTVNVAALVINVVAAISSGDVPLTAVQLLWVNLIM' +
#             'DTLGALALATEPPTDHLMGRPPVGRKEPLITNIMWRNLLIQAIYQVSVLLTLNFRGISILGLE' +
#             'HEVHEHATRVKNTIIFNAFVLCQAFNEFNARKPDEKNIFKGVIKNRLFMGIIVITLVLQVIIV' +
#             'EFLGKFASTTKLNWKQWLICVGIGVISWPLALVGKFIPVPAAPISNKLKVLKFWGKKKNSSGE' +
#             'GSL')
# lO03 = list('MAEEQKTSKVDVESPAVLAPAKEPTPAPVEVADEKIHNPPPVESKALAVVEKPIEEHTPKKAS' +
#             'SGSADRDVILADLEKEKKTSFIKAWEESEKSKAENRAQKKISDVHAWENSKKAAVEAQLRKIE' +
#             'EKLEKKKAQYGEKMKNKVAAIHKLAEEKRAMVEAKKGEELLKAEEMGAKYRATGVVPKATCGCF')

# --- dictionaries ------------------------------------------------------------
# dObs = {1: lO01, 2: lO02, 3: lO03}

# === assertions ==============================================================
# for lObs in dObs.values():
#     for cObs in lObs:
#         assert cObs in setCond

# assert sum(startPr.values()) == 1
# for dTransPr in transPr.values():
#     assert sum(dTransPr.values()) == 1
# for dEmitPr in emitPr.values():
#     assert (sum(dEmitPr.values()) > 1. - GC.MAX_DELTA and
#             sum(dEmitPr.values()) < 1. + GC.MAX_DELTA)

# === derived values and input processing =====================================
# dIPos = {k: list(range(len(lObs))) for k, lObs in dObs.items()}

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'doViterbi': doViterbi,
       'useFullSeqFrom': useFullSeqFrom,
       # --- names and paths of files and dirs
       'sFInpStartProb': sFInpStartProb,
       'sFInpTransProb': sFInpTransProb,
       'sFInpFullSeq': sFInpFullSeq,
       'sFOptStatePath': sFOptStatePath,
       'sFPosPyl': sFPosPyl,
       'sFProbDetail': sFProbDetail,
       'sFProbFinal': sFProbFinal,
       # --- numbers
       # --- strings
       'sProb': sProb,
       'sPrev': sPrev,
       'sState': sState,
       'sStartProb': sStartProb,
       'sLnProb': sLnProb,
       # --- sets
       'setCond': setCond,
       # --- lists
       # --- dictionaries
       # 'dObs': dObs,
       # === derived values and input processing
       # 'dIPos': dIPos
       }

###############################################################################
