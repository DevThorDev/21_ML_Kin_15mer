# -*- coding: utf-8 -*-
###############################################################################
# --- D_07__Classifier.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Classifiers for data classification (D_07__Classifier)'
sNmSpec = 'Input data for the Classifier class in O_07__Classifier'

lvlOut = 1      # higher level --> print more information (0: no printing)

# --- flow control ------------------------------------------------------------
doRndForestClf = True
doNNMLPClf = True
doPropCalc = True

doTrainTestSplit = True

encodeCatFtr = True
calcConfMatrix = False

useFullSeqFrom = GC.S_COMB_INP      # S_PROC_INP / S_COMB_INP
usedNmerSeq = GC.S_FULL_LIST        # S_FULL_LIST / S_UNQ_LIST

usedAAcType = GC.S_AAC_POLAR        # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

# --- names and paths of files and dirs ---------------------------------------
sFInpClf = 'TestCategorical'
sFInpPrC = sFInpClf

if useFullSeqFrom == GC.S_COMB_INP:
    sFInpClf = 'Test_' + usedAAcType + '_Combined_S_KinasesPho15mer_202202'
    sFInpPrC = sFInpClf

sFOutPrC = GC.S_US02.join([sFInpPrC, GC.S_OUT])

pInpClf = GC.P_DIR_TEMP
pInpPrC = pInpClf
pOutPrC = pInpPrC

# --- input for random forest classifier --------------------------------------
nEstim = 100                    # [100]
criterionQS = 'gini'            # {['gini'], 'entropy', 'log_loss'}
maxDepth = None                 # {[None], 1, 2, 3,...}
maxFtr = None                   # {['sqrt'], 'log2', None}, int or float

# --- input for neural network MLP classifier ---------------------------------
tHiddenLayers = (256, 128, 64, 32)
sActivation = 'relu'

# --- numbers -----------------------------------------------------------------
rndState = None             # None (random) or integer (reproducible)
maxLenNmer = 15              # odd number between 1 and 15
propTestData = .2

rndDigScore = GC.R04
rndDigCorrect = GC.R04

# --- strings -----------------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

sCY = GC.S_EFF_CL

# --- sets --------------------------------------------------------------------
setFeat = set('ACDEFGHIKLMNPQRSTVWY')
if usedAAcType == GC.S_AAC_CHARGE:
    setFeat = set('NPST')
elif usedAAcType == GC.S_AAC_POLAR:
    setFeat = set('ABCNPQ')
setNmerLen = set(range(1, 15 + 1, 2))

# --- lists -------------------------------------------------------------------
lFeatSrt = sorted(list(setFeat))

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
assert maxLenNmer in setNmerLen
# for lObs in dObs.values():
#     for cObs in lObs:
#         assert cObs in setFeat

# assert sum(startPr.values()) == 1
# for dTransPr in transPr.values():
#     assert sum(dTransPr.values()) == 1
# for dEmitPr in emitPr.values():
#     assert (sum(dEmitPr.values()) > 1. - GC.MAX_DELTA and
#             sum(dEmitPr.values()) < 1. + GC.MAX_DELTA)

# === derived values and input processing =====================================
maxPosNmer = maxLenNmer//2
rngPosNmer = range(-maxPosNmer, maxPosNmer + 1)
lSCX = [str(n) for n in rngPosNmer]

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       'lvlOut': lvlOut,
       # --- flow control
       'doRndForestClf': doRndForestClf,
       'doNNMLPClf': doNNMLPClf,
       'doPropCalc': doPropCalc,
       'doTrainTestSplit': doTrainTestSplit,
       'encodeCatFtr': encodeCatFtr,
       'calcConfMatrix': calcConfMatrix,
       'useFullSeqFrom': useFullSeqFrom,
       'usedNmerSeq': usedNmerSeq,
       'usedAAcType': usedAAcType,
       # --- names and paths of files and dirs
       'sFInpClf': sFInpClf,
       'sFInpPrC': sFInpPrC,
       'sFOutPrC': sFOutPrC,
       'pInpClf': pInpClf,
       'pInpPrC': pInpPrC,
       'pOutPrC': pOutPrC,
       # --- input for random forest classifier
       'nEstim': nEstim,
       'criterionQS': criterionQS,
       'maxDepth': maxDepth,
       'maxFtr': maxFtr,
       # --- input for neural network MLP classifier
       'tHiddenLayers': tHiddenLayers,
       'sActivation': sActivation,
       # --- numbers
       'rndState': rndState,
       'maxLenNmer': maxLenNmer,
       'propTestData': propTestData,
       'rndDigScore': rndDigScore,
       'rndDigCorrect': rndDigCorrect,
       # --- strings
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       'sCY': sCY,
       # --- sets
       'setFeat': setFeat,
       # --- lists
       'lFeatSrt': lFeatSrt,
       # --- dictionaries
       # === derived values and input processing
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCX': lSCX}

###############################################################################
