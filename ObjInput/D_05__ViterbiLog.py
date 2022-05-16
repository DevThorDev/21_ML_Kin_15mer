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

# --- names and paths of files and dirs ---------------------------------------
sFInpTransProb = 'SglPosTransProb'

# sFE = GC.S_US02.join([sFInpEmitProb.split(GC.S_US02)[0].split(GC.S_USC)[-1],
#                       GC.S_US02.join(sFInpEmitProb.split(GC.S_US02)[-2:])])

sFOptStatePath = 'OptimalStatePath'
sFProbDetail = 'ProbDetail'
sFProbFinal = 'ProbFinal'

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sProb = GC.S_PROB
sPrev = GC.S_PREV

# --- sets --------------------------------------------------------------------
setCond = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

# --- lists -------------------------------------------------------------------
lO01 = list('MSGSRRKATPASRTRVGNYEMGRTLGEGSFAKVKYAKNTVTGDQAAIKILDREKVFRHKMVEQ' +
            'LKREISTMKLIKHPNVVEIIEVMASKTKIYIVLELVNGGELFDKIAQQGRLKEDEARRYFQQL' +
            'INAVDYCHSRGVYHRDLKPENLILDANGVLKVSDFGLSAFSRQVREDGLLHTACGTPNYVAPE' +
            'VLSDKGYDGAAADVWSCGVILFVLMAGYLPFDEPNLMTLYKRICKAEFSCPPWFSQGAKRVIK' +
            'RILEPNPITRISIAELLEDEWFKKGYKPPSFDQDDEDITIDDVDAAFSNSKECLVTEKKEKPV' +
            'SMNAFELISSSSEFSLENLFEKQAQLVKKETRFTSQRSASEIMSKMEETAKPLGFNVRKDNYK' +
            'IKMKGDKSGRKGQLSVATEVFEVAPSLHVVELRKTGGDTLEFHKFYKNFSSGLKDVVWNTDAA' +
            'AEEQKQ')
lO02 = list('MTSLLKSSPGRRRGGDVESGKSEHADSDSDTFYIPSKNASIERLQQWRKAALVLNASRRFRYT' +
            'LDLKKEQETREMRQKIRSHAHALLAANRFMDMGRESGVEKTTGPATPAGDFGITPEQLVIMSK' +
            'DHNSGALEQYGGTQGLANLLKTNPEKGISGDDDDLLKRKTIYGSNTYPRKKGKGFLRFLWDAC' +
            'HDLTLIILMVAAVASLALGIKTEGIKEGWYDGGSIAFAVILVIVVTAVSDYKQSLQFQNLNDE' +
            'KRNIHLEVLRGGRRVEISIYDIVVGDVIPLNIGNQVPADGVLISGHSLALDESSMTGESKIVN' +
            'KDANKDPFLMSGCKVADGNGSMLVTGVGVNTEWGLLMASISEDNGEETPLQVRLNGVATFIGS' +
            'IGLAVAAAVLVILLTRYFTGHTKDNNGGPQFVKGKTKVGHVIDDVVKVLTVAVTIVVVAVPEG' +
            'LPLAVTLTLAYSMRKMMADKALVRRLSACETMGSATTICSDKTGTLTLNQMTVVESYAGGKKT' +
            'DTEQLPATITSLVVEGISQNTTGSIFVPEGGGDLEYSGSPTEKAILGWGVKLGMNFETARSQS' +
            'SILHAFPFNSEKKRGGVAVKTADGEVHVHWKGASEIVLASCRSYIDEDGNVAPMTDDKASFFK' +
            'NGINDMAGRTLRCVALAFRTYEAEKVPTGEELSKWVLPEDDLILLAIVGIKDPCRPGVKDSVV' +
            'LCQNAGVKVRMVTGDNVQTARAIALECGILSSDADLSEPTLIEGKSFREMTDAERDKISDKIS' +
            'VMGRSSPNDKLLLVQSLRRQGHVVAVTGDGTNDAPALHEADIGLAMGIAGTEVAKESSDIIIL' +
            'DDNFASVVKVVRWGRSVYANIQKFIQFQLTVNVAALVINVVAAISSGDVPLTAVQLLWVNLIM' +
            'DTLGALALATEPPTDHLMGRPPVGRKEPLITNIMWRNLLIQAIYQVSVLLTLNFRGISILGLE' +
            'HEVHEHATRVKNTIIFNAFVLCQAFNEFNARKPDEKNIFKGVIKNRLFMGIIVITLVLQVIIV' +
            'EFLGKFASTTKLNWKQWLICVGIGVISWPLALVGKFIPVPAAPISNKLKVLKFWGKKKNSSGE' +
            'GSL')
lO03 = list('MAEEQKTSKVDVESPAVLAPAKEPTPAPVEVADEKIHNPPPVESKALAVVEKPIEEHTPKKAS' +
            'SGSADRDVILADLEKEKKTSFIKAWEESEKSKAENRAQKKISDVHAWENSKKAAVEAQLRKIE' +
            'EKLEKKKAQYGEKMKNKVAAIHKLAEEKRAMVEAKKGEELLKAEEMGAKYRATGVVPKATCGCF')

# --- dictionaries ------------------------------------------------------------
dObs = {1: lO01, 2: lO02, 3: lO03}

startPr = {'AAcNotInNmer': 0.99,
           '-3': 0.01,
           '-2': 0.,
           '-1': 0.,
           '0': 0.,
           '1': 0.,
           '2': 0.,
           '3': 0.}

transPr = {}
emitPr = {}

# === assertions ==============================================================
for lObs in dObs.values():
    for cObs in lObs:
        assert cObs in setCond

assert sum(startPr.values()) == 1
# for dTransPr in transPr.values():
#     assert sum(dTransPr.values()) == 1
# for dEmitPr in emitPr.values():
#     assert (sum(dEmitPr.values()) > 1. - GC.MAX_DELTA and
#             sum(dEmitPr.values()) < 1. + GC.MAX_DELTA)

# === derived values and input processing =====================================
lStates = list(startPr)
dIPos = {k: list(range(len(lObs))) for k, lObs in dObs.items()}

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'doViterbi': doViterbi,
       # --- names and paths of files and dirs
       'sFInpTransProb': sFInpTransProb,
       'sFOptStatePath': sFOptStatePath,
       'sFProbDetail': sFProbDetail,
       'sFProbFinal': sFProbFinal,
       # --- strings
       'sProb': sProb,
       'sPrev': sPrev,
       # --- numbers
       'st0': lStates[0],
       # --- lists
       # --- dictionaries
       'dObs': dObs,
       'startPr': startPr,
       # === derived values and input processing
       'lStates': lStates,
       'dIPos': dIPos}

###############################################################################
