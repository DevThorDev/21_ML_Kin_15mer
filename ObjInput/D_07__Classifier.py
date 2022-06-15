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

calcConfMatrix = False
plotConfMatrix = False

encodeCatFtr = True

useFullSeqFrom = GC.S_COMB_INP      # S_COMB_INP
usedNmerSeq = GC.S_UNQ_LIST        # S_FULL_LIST / S_UNQ_LIST

usedAAcType = GC.S_AAC        # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

usedClType = GC.S_NEW + GC.S_CL     # GC.S_NEW/GC.S_OLD + GC.S_CL

# --- names and paths of files and dirs ---------------------------------------
sFInpStart = usedClType + 'Orig70'     # '' / 'AllCl' / 'NewClOrig70'
sFResCombS = 'Combined_S_KinasesPho15mer_202202'

sFInpClf, sFInpPrC = None, None
if useFullSeqFrom == GC.S_COMB_INP:     # currently only option implemented
    sFInpClf = GF.joinS([sFInpStart, usedAAcType, sFResCombS])
    sFInpPrC = sFInpClf

sFConfMat = GC.S_US02.join([sFInpClf, GC.S_CONF_MAT])
sFOutClf = sFInpClf
sFOutPrC = GC.S_US02.join([sFInpPrC, GC.S_PROP])

pInpClf = GC.P_DIR_INP_CLF
pInpPrC = GC.P_DIR_INP_CLF
pConfMat = GC.P_DIR_RES_CLF
pOutClf = GC.P_DIR_RES_CLF
pOutPrC = GC.P_DIR_RES_CLF

# --- general input for any classifier ----------------------------------------
rndState = None             # None (random) or integer (reproducible)
bWarmStart = True           # warm start (True: use warm start)

# --- general input for random forest classifier ------------------------------
estOobScore = False         # estimate the generalization score
nJobs = None                # number of jobs to run in parallel (None: 1)
vVerb = 1                   # state of verbosity ([0], 1, 2, 3...)

# --- general input for neural network MLP classifier -------------------------
bVerb = True                # state of verbosity (True: progress messages)

# --- numbers -----------------------------------------------------------------
maxLenNmer = None           # odd number between 1 and 15 or None (max. len)
propTestData = .2

rndDigScore = GC.R04
rndDigCorrect = GC.R04

# --- strings -----------------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

sCY = GC.S_EFF_CL

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
lOldClPlt = ['C-----', 'C1----', 'C-2---', 'C--3--', 'C---4-', 'C----5',
             'C12---', 'C1-3--', 'C1--4-', 'C1---5', 'C-23--', 'C-2-4-',
             'C-2--5', 'C--34-', 'C--3-5', 'C---45', 'C123--', 'C12-4-',
             'C12--5', 'C1-34-', 'C1-3-5', 'C1--45', 'C-234-', 'C-23-5',
             'C-2-45', 'C--345', 'C1234-', 'C123-5', 'C12-45', 'C1-345',
             'C-2345', 'C12345']
lSResClf = ['numPredicted', 'numCorrect', 'propCorrect']

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
if maxLenNmer not in setNmerLen:
    maxLenNmer = max(setNmerLen)

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
       'calcConfMatrix': calcConfMatrix,
       'plotConfMatrix': plotConfMatrix,
       'encodeCatFtr': encodeCatFtr,
       'useFullSeqFrom': useFullSeqFrom,
       'usedNmerSeq': usedNmerSeq,
       'usedAAcType': usedAAcType,
       'usedClType': usedClType,
       # --- names and paths of files and dirs
       'sFInpClf': sFInpClf,
       'sFInpPrC': sFInpPrC,
       'sFConfMat': sFConfMat,
       'sFOutClf': sFOutClf,
       'sFOutPrC': sFOutPrC,
       'pInpClf': pInpClf,
       'pInpPrC': pInpPrC,
       'pConfMat': pConfMat,
       'pOutClf': pOutClf,
       'pOutPrC': pOutPrC,
       # --- general input for any classifier
       'rndState': rndState,
       'bWarmStart': bWarmStart,
       # --- general input for random forest classifier
       'estOobScore': estOobScore,
       'nJobs': nJobs,
       'vVerb': vVerb,
       # --- general input for neural network MLP classifier
       'bVerb': bVerb,
       # --- numbers
       'maxLenNmer': maxLenNmer,
       'propTestData': propTestData,
       'rndDigScore': rndDigScore,
       'rndDigCorrect': rndDigCorrect,
       # --- strings
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       'sCY': sCY,
       'sSupTtlPlt': sSupTtlPlt,
       # --- sets
       'setFeat': setFeat,
       # --- lists
       'lFeatSrt': lFeatSrt,
       'lOldClPlt': lOldClPlt,
       'lSResClf': lSResClf,
       # --- dictionaries
       # === derived values and input processing
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCX': lSCX}

###############################################################################
