# -*- coding: utf-8 -*-
###############################################################################
# --- D_07__Classifier.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

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
calcConfMatrix = True

useFullSeqFrom = GC.S_COMB_INP      # S_COMB_INP
usedNmerSeq = GC.S_FULL_LIST        # S_FULL_LIST / S_UNQ_LIST

usedAAcType = GC.S_AAC        # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

# --- names and paths of files and dirs ---------------------------------------
sFInpStart = 'NewClOrig70'                # '' / 'AllCl' / 'NewClOrig70'
sFResCombS = 'Combined_S_KinasesPho15mer_202202'

sFInpClf, sFInpPrC = None, None
if useFullSeqFrom == GC.S_COMB_INP:     # currently only option implemented
    sFInpClf = GF.joinS([sFInpStart, usedAAcType, sFResCombS])
    sFInpPrC = sFInpClf

sFConfMat = GC.S_US02.join([sFInpClf, GC.S_CONF_MAT])
sFOutPrC = GC.S_US02.join([sFInpPrC, GC.S_PROP])

pInpClf = GC.P_DIR_INP_CLF
pInpPrC = GC.P_DIR_INP_CLF
pConfMat = GC.P_DIR_RES_CLF
pOutPrC = GC.P_DIR_RES_CLF

# --- input for random forest classifier --------------------------------------
nEstim = 1000                    # [100]
criterionQS = 'gini'            # {['gini'], 'entropy', 'log_loss'}
maxDepth = None                 # {[None], 1, 2, 3,...}
maxFtr = None                   # {['sqrt'], 'log2', None}, int or float

# --- input for neural network MLP classifier ---------------------------------
tHiddenLayers = (256, 128, 64, 32)
sActivation = 'relu'

# --- numbers -----------------------------------------------------------------
rndState = None             # None (random) or integer (reproducible)
maxLenNmer = None           # odd number between 1 and 15 or None (max. len)
propTestData = .2

rndDigScore = GC.R04
rndDigCorrect = GC.R04

# --- strings -----------------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

sCY = GC.S_EFF_CL

sMthRF = GC.S_MTH_RF
sMthMLP = GC.S_MTH_MLP

sSupTtlPlt = 'Confusion matrix'

# --- sets --------------------------------------------------------------------
setFeat = set('ACDEFGHIKLMNPQRSTVWY')
if usedAAcType == GC.S_AAC_CHARGE:
    setFeat = set('NPST')
elif usedAAcType == GC.S_AAC_POLAR:
    setFeat = set('ABCNPQ')
setNmerLen = set(range(1, 15 + 1, 2))

# --- lists -------------------------------------------------------------------
lFeatSrt = sorted(list(setFeat))
lAllClPlt = ['C-----', 'C1----', 'C-2---', 'C--3--', 'C---4-', 'C----5',
             'C12---', 'C1-3--', 'C1--4-', 'C1---5', 'C-23--', 'C-2-4-',
             'C-2--5', 'C--34-', 'C--3-5', 'C---45', 'C123--', 'C12-4-',
             'C12--5', 'C1-34-', 'C1-3-5', 'C1--45', 'C-234-', 'C-23-5',
             'C-2-45', 'C--345', 'C1234-', 'C123-5', 'C12-45', 'C1-345',
             'C-2345', 'C12345']

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
if maxLenNmer not in setNmerLen:
    maxLenNmer = max(setNmerLen)
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
       'sFConfMat': sFConfMat,
       'sFOutPrC': sFOutPrC,
       'pInpClf': pInpClf,
       'pInpPrC': pInpPrC,
       'pConfMat': pConfMat,
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
       'sMthRF': sMthRF,
       'sMthMLP': sMthMLP,
       'sSupTtlPlt': sSupTtlPlt,
       # --- sets
       'setFeat': setFeat,
       # --- lists
       'lFeatSrt': lFeatSrt,
       'lAllClPlt': lAllClPlt,
       # --- dictionaries
       # === derived values and input processing
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCX': lSCX}

###############################################################################
