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
doImbSampling = True                # do imbalanced sampling before classific.?

doDummyClf = True                   # do dummy classification?
doAdaClf = True                     # do AdaBoost classification?
doRFClf = True                      # do random forest classification?
doGPClf = True                      # do Gaussian Process classification?
doMLPClf = True                     # do neural network MLP classification?
doPropCalc = False                   # do calculation of AAc proportions/class?

doTrainTestSplit = True             # split data into train and test sets?

doMultiSteps = False                 # do a multi-step classification approach?

saveDetailedClfRes = True           # save the detailed classification results?

calcCnfMatrix = True                # calculate the confusion matrix?
plotCnfMatrix = False               # plot the confusion matrix?

encodeCatFtr = True                 # encode categorical features?

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
pCnfMat = GC.P_DIR_RES_CLF_CNF_MAT
pOutDet = GC.P_DIR_RES_CLF_DETAILED
pOutPrC = GC.P_DIR_RES_CLF_PROP

# === general over- and undersampler input ====================================
sSampler = 'RandomUnderSampler'    # string matching the over/under-sampler
                            # ClusterCentroids, AllKNN, TomekLinks
                            # NeighbourhoodCleaningRule, RandomUnderSampler,
sStrat =  GC.S_STRAT_REAL_MAJO         # sampling strategy
                        # 'all' / 'majority' / 'not majority' / 'not minority'
                        # [custom: GC.S_STRAT_REAL_MAJO/GC.S_STRAT_SHARE_MINO]
dSStrat = {1: GC.S_STRAT_REAL_MAJO,
           2: GC.S_STRAT_SHARE_MINO}
                        # dictionary for modifying the strategy for specific
                        # step indices (in case of doMultiSteps)
dIStrat = {'shareMino': 0.75}
                        # additional data as required by the custom strategy
# sStrat = {'NoCl': 1000,
#           'X_AGC': 100,
#           'X_CDK': 100,
#           'X_CK_II': 100,
#           'X_SnRK2': 100,
#           'X_soluble': 100}

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

# === general input for any Classifier (that might use it) ====================
rndState = None             # None (random) or integer (reproducible)
bWarmStart = True           # warm start (True: use warm start)
nJobs = None                # number of jobs to run in parallel (None: 1)

# --- general input for Dummy Classifier --------------------------------------

# --- general input for AdaBoost Classifier -----------------------------------

# --- general input for random forest Classifier ------------------------------
estOobScore = False         # estimate the generalization score
vVerb = 1                   # state of verbosity ([0], 1, 2, 3...)

# --- general input for Gaussian Process Classifier ---------------------------

# --- general input for neural network MLP Classifier -------------------------
nItPartialFit = None        # number of iterations / partial fit (or None)
# nItPartialFit = 1000        # number of iterations / partial fit (or None)
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
lSmplStratCustom = [GC.S_STRAT_REAL_MAJO, GC.S_STRAT_SHARE_MINO]

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       'lvlOut': lvlOut,
       # --- flow control
       'doImbSampling': doImbSampling,
       'doDummyClf': doDummyClf,
       'doAdaClf': doAdaClf,
       'doRFClf': doRFClf,
       'doGPClf': doGPClf,
       'doMLPClf': doMLPClf,
       'doPropCalc': doPropCalc,
       'doTrainTestSplit': doTrainTestSplit,
       'doMultiSteps': doMultiSteps,
       'saveDetailedClfRes': saveDetailedClfRes,
       'calcCnfMatrix': calcCnfMatrix,
       'plotCnfMatrix': plotCnfMatrix,
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
       'pCnfMat': pCnfMat,
       'pOutDet': pOutDet,
       'pOutPrC': pOutPrC,
       # === general over- and undersampler input =============================
       'sSampler': sSampler,
       'sStrat': sStrat,
       'dSStrat': dSStrat,
       'dIStrat': dIStrat,
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
       # === general input for any Classifier (that might use it) =============
       'rndState': rndState,
       'bWarmStart': bWarmStart,
       'nJobs': nJobs,
       # --- general input for Dummy Classifier
       # --- general input for AdaBoost Classifier
       # --- general input for random forest Classifier
       'estOobScore': estOobScore,
       'vVerb': vVerb,
       # --- general input for Gaussian Process Classifier
       # --- general input for neural network MLP Classifier
       'nItPartialFit': nItPartialFit,
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
       'lSResClf': lSResClf,
       # === derived values and input processing ==============================
       'lSmplStratCustom': lSmplStratCustom}

###############################################################################