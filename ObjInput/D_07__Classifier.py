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
doImbSampling = False
doRndForestClf = True
doNNMLPClf = True
doPropCalc = False

doTrainTestSplit = True

saveDetailedClfRes = True

calcConfMatrix = True
plotConfMatrix = False

encodeCatFtr = True

lLblTrain = [1]                     # number of labels used for training data
# lLblTrain = None                    # number of labels used for training data
                                    # or None [use all labels]
                                    # ignored if D.dITp['onlySglLbl'] == True

useFullSeqFrom = GC.S_COMB_INP      # S_COMB_INP
usedNmerSeq = GC.S_UNQ_LIST         # S_FULL_LIST / S_UNQ_LIST

usedAAcType = GC.S_AAC              # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

usedClType = GC.S_NEW + GC.S_CL     # GC.S_NEW/GC.S_OLD + GC.S_CL

# --- names and paths of files and dirs ---------------------------------------
pInpClf = GC.P_DIR_INP_CLF
pInpPrC = GC.P_DIR_INP_CLF
pOutPar = GC.P_DIR_RES_CLF_PARS
pOutSum = GC.P_DIR_RES_CLF_SUMMARIES
pConfMat = GC.P_DIR_RES_CLF_CONF_MAT
pOutDet = GC.P_DIR_RES_CLF_DETAILED
pOutPrC = GC.P_DIR_RES_CLF_PROP

# === general over- and undersampler input ====================================
sSampler = 'AllKNN'         # string matching the over/under-sampler
                            # ClusterCentroids, AllKNN, TomekLinks
                            # NeighbourhoodCleaningRule, RandomUnderSampler,
sStrat = 'majority'         # sampling strategy
                        # 'all' / 'majority' / 'not majority' / 'not minority'
# sStrat = {'X_AGC': 50,
#           'X_CDPK': 50,
#           'X_CK_II': 50,
#           'X_MAPK': 50,
#           'X_SnRK2': 50}

# --- ClusterCentroids input --------------------------------------------------
estimator = None                # a KMeans estimator (None --> default est.)
voting = 'auto'                 # ['auto'] / 'hard' / 'soft'

# --- AllKNN input ------------------------------------------------------------
n_neighbors_AllKNN = 3          # number of nearest neighbors
kind_sel_AllKNN = 'mode'        # strategy to exclude samples ['all' / 'mode']
allow_minority = False          # allows majority classes --> minority class

# --- NeighbourhoodCleaningRule input -----------------------------------------
n_neighbors_NCR = 3             # number of nearest neighbors
kind_sel_NCR = 'mode'            # strategy to exclude samples ['all' / 'mode']
threshold_cleaning = 0.5        # threshold 4 class considered during cleaning

# --- RandomUnderSampler and BalancedRandomForestClassifier input -------------
wReplacement = False            # is sample with or without replacement?

# --- TomekLinks input --------------------------------------------------------
# =============================================================================

# === general input for any classifier ========================================
rndState = None             # None (random) or integer (reproducible)
bWarmStart = True           # warm start (True: use warm start)

# --- general input for random forest classifier ------------------------------
estOobScore = False         # estimate the generalization score
nJobs = None                # number of jobs to run in parallel (None: 1)
vVerb = 1                   # state of verbosity ([0], 1, 2, 3...)

# --- general input for neural network MLP classifier -------------------------
bVerb = True                # state of verbosity (True: progress messages)

# =============================================================================
# --- numbers -----------------------------------------------------------------
propTestData = .2

rndDigScore = GC.R04
rndDigCorrect = GC.R04

# --- strings -----------------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

sSupTtlPlt = 'Confusion matrix'

# --- sets --------------------------------------------------------------------
setFeat = set('ACDEFGHIKLMNPQRSTVWY')
if usedAAcType == GC.S_AAC_CHARGE:
    setFeat = set('NPST')
elif usedAAcType == GC.S_AAC_POLAR:
    setFeat = set('ABCNPQ')

# --- lists -------------------------------------------------------------------
lFeatSrt = sorted(list(setFeat))
lOldClPlt = ['C-----', 'C1----', 'C-2---', 'C--3--', 'C---4-', 'C----5',
             'C12---', 'C1-3--', 'C1--4-', 'C1---5', 'C-23--', 'C-2-4-',
             'C-2--5', 'C--34-', 'C--3-5', 'C---45', 'C123--', 'C12-4-',
             'C12--5', 'C1-34-', 'C1-3-5', 'C1--45', 'C-234-', 'C-23-5',
             'C-2-45', 'C--345', 'C1234-', 'C123-5', 'C12-45', 'C1-345',
             'C-2345', 'C12345']
lSResClf = ['numPredicted', 'numCorrect', 'propCorrect']

# === assertions ==============================================================

# === derived values and input processing =====================================

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       'lvlOut': lvlOut,
       # --- flow control
       'doImbSampling': doImbSampling,
       'doRndForestClf': doRndForestClf,
       'doNNMLPClf': doNNMLPClf,
       'doPropCalc': doPropCalc,
       'doTrainTestSplit': doTrainTestSplit,
       'saveDetailedClfRes': saveDetailedClfRes,
       'calcConfMatrix': calcConfMatrix,
       'plotConfMatrix': plotConfMatrix,
       'encodeCatFtr': encodeCatFtr,
       'lLblTrain': lLblTrain,
       'useFullSeqFrom': useFullSeqFrom,
       'usedNmerSeq': usedNmerSeq,
       'usedAAcType': usedAAcType,
       'usedClType': usedClType,
       # --- names and paths of files and dirs
       'pInpClf': pInpClf,
       'pInpPrC': pInpPrC,
       'pOutPar': pOutPar,
       'pOutSum': pOutSum,
       'pConfMat': pConfMat,
       'pOutDet': pOutDet,
       'pOutPrC': pOutPrC,
       # === general over- and undersampler input =============================
       'sSampler': sSampler,
       'sStrat': sStrat,
       # --- ClusterCentroids input
       'estimator': estimator,
       'voting': voting,
       # --- AllKNN input
       'n_neighbors_AllKNN': n_neighbors_AllKNN,
       'kind_sel_AllKNN': kind_sel_AllKNN,
       'allow_minority': allow_minority,
       # --- NeighbourhoodCleaningRule input
       'n_neighbors_NCR': n_neighbors_NCR,
       'kind_sel_NCR': kind_sel_NCR,
       'threshold_cleaning': threshold_cleaning,
       # --- RandomUnderSampler and BalancedRandomForestClassifier input
       'wReplacement': wReplacement,
       # --- TomekLinks input
       # ======================================================================
       # === general input for any classifier =================================
       'rndState': rndState,
       'bWarmStart': bWarmStart,
       # --- general input for random forest classifier
       'estOobScore': estOobScore,
       'nJobs': nJobs,
       'vVerb': vVerb,
       # --- general input for neural network MLP classifier
       'bVerb': bVerb,
       # ======================================================================
       # --- numbers
       'propTestData': propTestData,
       'rndDigScore': rndDigScore,
       'rndDigCorrect': rndDigCorrect,
       # --- strings
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       'sSupTtlPlt': sSupTtlPlt,
       # --- sets
       'setFeat': setFeat,
       # --- lists
       'lFeatSrt': lFeatSrt,
       'lOldClPlt': lOldClPlt,
       'lSResClf': lSResClf}

###############################################################################