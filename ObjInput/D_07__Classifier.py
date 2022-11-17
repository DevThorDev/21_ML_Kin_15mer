# -*- coding: utf-8 -*-
###############################################################################
# --- D_07__Classifier.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# *** general *****************************************************************
sOType = 'Classifiers for data classification (D_07__Classifier)'
sNmSpec = 'Input data for the Classifier class in O_07__Classifier'

lvlOut = 1      # higher level --> print more information (0: no printing)

# *** flow control ************************************************************
doImbSampling = True                # do imbalanced sampling before classif.?

doDummyClf = True                   # do dummy classification?
doAdaClf = True                     # do AdaBoost classification?
doRFClf = True                      # do Random Forest classification?
doXTrClf = True                     # do Extra Trees classification?
doGrBClf = True                     # do Gradient Boosting classification?
doHGrBClf = True                    # do HistGradientBoosting classification?
doGPClf = True                      # do Gaussian Process classification?
doPaAggClf = True                   # do Passive Aggressive classification?
doPctClf = True                     # do Perceptron classification?
doSGDClf = True                     # do SGD classification?
doCtNBClf = True                    # do Categorical NB classification?
doCpNBClf = True                    # do Complement NB classification?
doGsNBClf = True                    # do Gaussian NB classification?
doMLPClf = True                     # do neural network MLP classification?
doLinSVClf = True                   # do Linear SV classification?
doNuSVClf = True                    # do Nu-Support SV classification?

doPropCalc = False                   # do calculation of AAc proportions/class?

doMultiSteps = False                 # do a multi-step classification approach?

saveDetailedClfRes = True           # save the detailed classification results?

calcCnfMatrix = True                # calculate the confusion matrix?
plotCnfMatrix = False               # plot the confusion matrix?

dEncCatFtr = {GC.S_MTH_DUMMY: GC.S_ONE_HOT,    # encoder of cat. features
              GC.S_MTH_ADA: GC.S_ONE_HOT,
              GC.S_MTH_RF: GC.S_ONE_HOT,
              GC.S_MTH_X_TR: GC.S_ONE_HOT,
              GC.S_MTH_GR_B: GC.S_ONE_HOT,
              GC.S_MTH_H_GR_B: GC.S_ONE_HOT,
              GC.S_MTH_GP: GC.S_ONE_HOT,
              GC.S_MTH_PA_A: GC.S_ONE_HOT,
              GC.S_MTH_PCT: GC.S_ONE_HOT,
              GC.S_MTH_SGD: GC.S_ONE_HOT,
              GC.S_MTH_CT_NB: GC.S_ORDINAL,
              GC.S_MTH_CP_NB: GC.S_ONE_HOT,
              GC.S_MTH_GS_NB: GC.S_ONE_HOT,
              GC.S_MTH_MLP: GC.S_ONE_HOT,
              GC.S_MTH_LSV: GC.S_ONE_HOT,
              GC.S_MTH_NSV: GC.S_ONE_HOT}

# lLblTrain is ignored if D.dITp['I_Lbl'] == GC.S_SGL_LBL
lLblTrain = [0, 1]                     # number of labels used for training data
# lLblTrain = None                  # 0, 1,..., N or None [use all labels]

useFullSeqFrom = GC.S_COMB_INP      # S_COMB_INP
usedNmerSeq = GC.S_UNQ_LIST         # S_FULL_LIST / S_UNQ_LIST

usedAAcType = GC.S_AAC              # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

usedClType = GC.S_NEW + GC.S_CL     # GC.S_NEW/GC.S_OLD + GC.S_CL

# *** names and paths of files and dirs ***************************************
pInpClf = GC.P_DIR_INP_CLF
pInpPrC = GC.P_DIR_INP_CLF
pOutPar = GC.P_DIR_RES_CLF_PARS
pOutSum = GC.P_DIR_RES_CLF_SUMMARIES
pCnfMat = GC.P_DIR_RES_CLF_CNF_MAT
pOutDet = GC.P_DIR_RES_CLF_DETAILED
pOutPrC = GC.P_DIR_RES_CLF_PROP

# *** kFold Splitter input ****************************************************
tpKF = GC.S_STRAT_K_FOLD            # type of kFold-split, or None (no split)
                                    # S_K_FOLD / S_GROUP_K_FOLD
                                    # S_STRAT_K_FOLD / S_STRAT_GROUP_K_FOLD
nSplitsKF = 5                       # number of folds
shuffleKF = True                    # shuffle data before batches split?
rndStateKF = None                   # None (random) or integer (reproducible)

# *** over- and undersampler input ********************************************
# === general over- and undersampler input ====================================
sSampler = GC.S_SMPL_RND_U  # string matching the over/under-sampler
                            # GC.S_SMPL_CL_CTR, GC.S_SMPL_ALL_KNN,
                            # GC.S_SMPL_NBH_CL_R, GC.S_SMPL_RND_U,
                            # GC.S_SMPL_TOM_LKS
sStrat =  GC.S_STRAT_REAL_MAJO         # sampling strategy
                        # 'all' / 'majority' / 'not majority' / 'not minority'
                        # [custom: GC.S_STRAT_REAL_MAJO/GC.S_STRAT_SHARE_MINO]
dSStrat = {1: GC.S_STRAT_REAL_MAJO,
           2: GC.S_STRAT_SHARE_MINO}
                        # dictionary for modifying the strategy for specific
                        # step indices (in case of doMultiSteps)
dIStrat = {GC.S_STRAT_SHARE_MINO: 1.0}
                        # additional data as required by the custom strategy
# sStrat = {'NoCl': 500,
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

# *** Classifier input ********************************************************
# === general input for any Classifier (that might use it) ====================
rndState = None             # None (random) or integer (reproducible)
bWarmStart = True           # warm start (True: use warm start)
nJobs = None                # number of jobs to run in parallel (None: 1)

# === general input for any Classifier that implements 'partial_fit' ==========
nItPtFit = None             # number of iterations / partial fit (or None)
# nItPtFit = 1000             # number of iterations / partial fit (or None)
nItPrintPtFit = 100         # print status after number of iterations

# --- selection of grid search or randomised search; and halving --------------
typeS = 'RandomizedSearchCV'        # 'GridSearchCV' / 'RandomizedSearchCV'
halvingS = False                     # should halving search be performed?
# --- general grid search / randomised search input ---------------------------
scoringS = 'balanced_accuracy'      # scoring par. string for grid searches
verboseS = 2                        # verbosity (higher --> more messages)
retTrScoreS = False                 # should result include training scores?
# --- general halving grid search / halving randomised search input -----------
factorHvS = 2                       # 1/(prop. of candidates selected)
aggrElimHvS = False                 # aggressive elimination of candidates?
# --- randomised search input -------------------------------------------------
nIterRS = 20                        # number of parameter settings sampled
# --- halving randomised search input -----------------------------------------
nCandidatesHvRS = 'exhaust'         # number of candidate parameters to sample

# --- cross validation of grid search / randomised search input ---------------
nSplitsCV = 5                       # number of folds (rep. strat. k-fold CV)
nRepeatsCV = 5                     # number of repeats (rep. strat. k-fold CV)

# --- general input for Dummy Classifier --------------------------------------

# --- general input for AdaBoost Classifier -----------------------------------

# --- general input for Random Forest Classifier ------------------------------
estOobScoreRF = False       # estimate the generalization score
vVerbRF = 1                 # state of verbosity ([0], 1, 2, 3...)

# --- general input for Extra Trees Classifier --------------------------------
estOobScoreXTr = False      # estimate the generalization score
vVerbXTr = 1                # state of verbosity ([0], 1, 2, 3...)

# --- general input for Gradient Boosting Classifier --------------------------
vVerbGrB = 1                # state of verbosity ([0], 1, 2, 3...)

# --- general input for Hist Gradient Boosting Classifier ---------------------
vVerbHGrB = 1               # state of verbosity ([0], 1, 2, 3...)

# --- general input for Gaussian Process Classifier ---------------------------

# --- general input for Passive Aggressive Classifier -------------------------
vVerbPaA = 0                # state of verbosity ([0], 1, 2, 3...)

# --- general input for Perceptron Classifier ---------------------------------
vVerbPct = 0                # state of verbosity ([0], 1, 2, 3...)

# --- general input for SGD Classifier ----------------------------------------
vVerbSGD = 0                # state of verbosity ([0], 1, 2, 3...)

# --- general input for Categorical NB Classifier -----------------------------

# --- general input for Complement NB Classifier ------------------------------

# --- general input for Gaussian NB Classifier --------------------------------

# --- general input for neural network MLP Classifier -------------------------
bVerbMLP = True             # state of verbosity (True: progress messages)

# --- general input for Linear SV Classifier ----------------------------------
vVerbLSV = 1                # state of verbosity ([0], 1, 2, 3...)

# --- general input for Nu-Support SV Classifier ------------------------------
vVerbNSV = 1                # state of verbosity ([0], 1, 2, 3...)

# *** other input *************************************************************
# === numbers =================================================================
rndDigScore = GC.R04
rndDigCorrect = GC.R04

# === strings =================================================================
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

sSupTtlPlt = 'Confusion matrix'

# === sets ====================================================================
setFeat = set('ACDEFGHIKLMNPQRSTVWY')
if usedAAcType == GC.S_AAC_CHARGE:
    setFeat = set('NPST')
elif usedAAcType == GC.S_AAC_POLAR:
    setFeat = set('ABCNPQ')

# === lists ===================================================================
lFeatSrt = sorted(list(setFeat))
lOldClPlt = ['C-----', 'C1----', 'C-2---', 'C--3--', 'C---4-', 'C----5',
             'C12---', 'C1-3--', 'C1--4-', 'C1---5', 'C-23--', 'C-2-4-',
             'C-2--5', 'C--34-', 'C--3-5', 'C---45', 'C123--', 'C12-4-',
             'C12--5', 'C1-34-', 'C1-3-5', 'C1--45', 'C-234-', 'C-23-5',
             'C-2-45', 'C--345', 'C1234-', 'C123-5', 'C12-45', 'C1-345',
             'C-2345', 'C12345']
lSResClf = ['numPredicted', 'numCorrect', 'propCorrect',
            'accurScore', 'balAccurScore', 'top2AccurScore', 'top3AccurScore',
            'precScore', 'recallScore', 'fbetaScore', 'Support',
            'f1Score', 'rocAucScore', 'matthewsCC', 'logLoss']

# *** assertions **************************************************************
assert nItPtFit is None or type(nItPtFit) in [int, float]
assert nItPrintPtFit is None or type(nItPrintPtFit) in [int, float]
assert typeS in ['GridSearchCV', 'RandomizedSearchCV']

# *** derived values and input processing *************************************
sSmplS = (GC.D_S_SMPL[sSampler] if (sSampler in GC.D_S_SMPL) else None)
lSmplStratCustom = [GC.S_STRAT_REAL_MAJO, GC.S_STRAT_SHARE_MINO]

# *** create input dictionary *************************************************
dIO = {# *** general **********************************************************
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       'lvlOut': lvlOut,
       # *** flow control *****************************************************
       'doImbSampling': doImbSampling,
       'doDummyClf': doDummyClf,
       'doAdaClf': doAdaClf,
       'doRFClf': doRFClf,
       'doXTrClf': doXTrClf,
       'doGrBClf': doGrBClf,
       'doHGrBClf': doHGrBClf,
       'doGPClf': doGPClf,
       'doPaAggClf': doPaAggClf,
       'doPctClf': doPctClf,
       'doSGDClf': doSGDClf,
       'doCtNBClf': doCtNBClf,
       'doCpNBClf': doCpNBClf,
       'doGsNBClf': doGsNBClf,
       'doMLPClf': doMLPClf,
       'doLinSVClf': doLinSVClf,
       'doNuSVClf': doNuSVClf,
       'doPropCalc': doPropCalc,
       'doMultiSteps': doMultiSteps,
       'saveDetailedClfRes': saveDetailedClfRes,
       'calcCnfMatrix': calcCnfMatrix,
       'plotCnfMatrix': plotCnfMatrix,
       'dEncCatFtr': dEncCatFtr,
       'lLblTrain': lLblTrain,
       'useFullSeqFrom': useFullSeqFrom,
       'usedNmerSeq': usedNmerSeq,
       'usedAAcType': usedAAcType,
       'usedClType': usedClType,
       # *** names and paths of files and dirs ********************************
       'pInpClf': pInpClf,
       'pInpPrC': pInpPrC,
       'pOutPar': pOutPar,
       'pOutSum': pOutSum,
       'pCnfMat': pCnfMat,
       'pOutDet': pOutDet,
       'pOutPrC': pOutPrC,
       # *** kFold Splitter input *********************************************
       'tpKF': tpKF,
       'nSplitsKF': nSplitsKF,
       'shuffleKF': shuffleKF,
       'rndStateKF': rndStateKF,
       # *** over- and undersampler input *************************************
       # === general over- and undersampler input
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
       # *** Classifier input *************************************************
       # === general input for any Classifier (that might use it)
       'rndState': rndState,
       'bWarmStart': bWarmStart,
       'nJobs': nJobs,
       # === general input for any Classifier that implements 'partial_fit'
       'nItPtFit': nItPtFit,
       'nItPrintPtFit': nItPrintPtFit,
       # --- selection of grid search or randomised search; and halving
       'typeS': typeS,
       'halvingS': halvingS,
       # --- general grid search / randomised search input
       'scoringS': scoringS,
       'verboseS': verboseS,
       'retTrScoreS': retTrScoreS,
       # --- general halving grid search / halving randomised search input
       'factorHvS': factorHvS,
       'aggrElimHvS': aggrElimHvS,
       # --- randomised search input
       'nIterRS': nIterRS,
       # --- halving randomised search input
       'nCandidatesHvRS': nCandidatesHvRS,
       # --- cross validation of grid search / randomised search input
       'nSplitsCV': nSplitsCV,
       'nRepeatsCV': nRepeatsCV,
       # --- general input for Dummy Classifier
       # --- general input for AdaBoost Classifier
       # --- general input for Random Forest Classifier
       'estOobScoreRF': estOobScoreRF,
       'vVerbRF': vVerbRF,
       # --- general input for Extra Trees Classifier
       'estOobScoreXTr': estOobScoreXTr,
       'vVerbXTr': vVerbXTr,
       # --- general input for Gradient Boosting Classifier
       'vVerbGrB': vVerbGrB,
       # --- general input for Hist Gradient Boosting Classifier
       'vVerbHGrB': vVerbHGrB,
       # --- general input for Gaussian Process Classifier
       # --- general input for Passive Aggressive Classifier
       'vVerbPaA': vVerbPaA,
       # --- general input for Perceptron Classifier
       'vVerbPct': vVerbPct,
       # --- general input for SGD Classifier
       'vVerbSGD': vVerbSGD,
       # --- general input for Categorical NB Classifier
       # --- general input for Complement NB Classifier
       # --- general input for Gaussian NB Classifier
       # --- general input for neural network MLP Classifier
       'bVerbMLP': bVerbMLP,
       # --- general input for Linear SV Classifier
       'vVerbLSV': vVerbLSV,
       # --- general input for Nu-Support SV Classifier
       'vVerbNSV': vVerbNSV,
       # *** other input ******************************************************
       # === numbers
       'rndDigScore': rndDigScore,
       'rndDigCorrect': rndDigCorrect,
       # === strings
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       'sSupTtlPlt': sSupTtlPlt,
       # === sets
       'setFeat': setFeat,
       # === lists
       'lFeatSrt': lFeatSrt,
       'lOldClPlt': lOldClPlt,
       'lSResClf': lSResClf,
       # *** derived values and input processing ******************************
       'sSmplS': sSmplS,
       'lSmplStratCustom': lSmplStratCustom}

###############################################################################