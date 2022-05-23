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
doNNMLPClf = False

doTrainTestSplit = True

encodeCatFtr = True
calcConfMatrix = False

useFullSeqFrom = GC.S_COMB_INP      # S_PROC_INP / S_COMB_INP
usedNmerSeq = GC.S_FULL_LIST        # S_FULL_LIST / S_UNQ_LIST

# --- names and paths of files and dirs ---------------------------------------
sFInp = 'TestCategorical'

if useFullSeqFrom == GC.S_COMB_INP:
    sFInp = 'Test_Combined_S_KinasesPho15mer_202202'

pInp = GC.P_DIR_TEMP

# --- numbers -----------------------------------------------------------------
rndState = None             # None (random) or integer (reproducible)
maxLenNmer = 11              # odd number between 1 and 15
propTestData = .2

# --- numbers for random forest classifier ------------------------------------
nEstim = 200
maxDepthRFClf = 4

# --- numbers for neural network MLP classifier -------------------------------
tHiddenLayers = (256, 128, 64, 32)

rndDigScore = GC.R04
rndDigCorrect = GC.R04

# --- strings -----------------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

sCY = GC.S_EFF_CL

sActivation = 'relu'

# --- sets --------------------------------------------------------------------
setFeat = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
setNmerLen = {1, 3, 5, 7, 9, 11, 13, 15}

# --- lists -------------------------------------------------------------------

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
       'doTrainTestSplit': doTrainTestSplit,
       'encodeCatFtr': encodeCatFtr,
       'calcConfMatrix': calcConfMatrix,
       'useFullSeqFrom': useFullSeqFrom,
       'usedNmerSeq': usedNmerSeq,
       # --- names and paths of files and dirs
       'sFInp': sFInp,
       'pInp': pInp,
       # --- numbers
       'rndState': rndState,
       'maxLenNmer': maxLenNmer,
       'propTestData': propTestData,
       # --- numbers for random forest classifier
       'nEstim': nEstim,
       'maxDepthRFClf': maxDepthRFClf,
       # --- numbers for neural network MLP classifier
       'tHiddenLayers': tHiddenLayers,
       'rndDigScore': rndDigScore,
       'rndDigCorrect': rndDigCorrect,
       # --- strings
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       'sCY': sCY,
       'sActivation': sActivation,
       # --- sets
       'setFeat': setFeat,
       # --- lists
       # --- dictionaries
       # === derived values and input processing
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCX': lSCX}

###############################################################################
