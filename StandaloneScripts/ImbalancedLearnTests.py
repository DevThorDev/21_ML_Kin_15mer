# -*- coding: utf-8 -*-
###############################################################################
# --- ImbalancedLearnTests.py -------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import (ClusterCentroids, AllKNN,
                                     NeighbourhoodCleaningRule,
                                     RandomUnderSampler, TomekLinks)
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)

from imblearn.ensemble import BalancedRandomForestClassifier

# ### CONSTANTS ###############################################################
# --- sets for class dictionary -----------------------------------------------
SET01 = 'Set01_11Cl'
SET02 = 'Set02_06Cl'
SET03 = 'Set03_05Cl'

C_SET = SET03

# --- files, directories and paths --------------------------------------------
P_INP_CLF = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                         '21_InpDataClf')
P_TEMP_RES = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                          '98_TEMP_CSV')

S_F_INP_CLF_COMB_XS = ('InpClf_Combined_XS_KinasesPho15mer_202202_' + C_SET +
                       '_Unique')
S_F_INP_D_CLASSES = 'InpClf_ClMapping_' + C_SET

# --- strings -----------------------------------------------------------------
S_SPACE = ' '
S_DOT = '.'
S_SEMICOL = ';'
S_COLON = ':'
S_DASH = '-'
S_PLUS = '+'
S_EQ = '='
S_STAR = '*'
S_USC = '_'
S_TAB = '\t'
S_NEWL = '\n'
S_VBAR = '|'
S_PERC = '%'
S_CAP_S, S_CAP_X = 'S', 'X'
S_0, S_1 = '0', '1'

S_CSV = 'csv'

S_SP04 = S_SPACE*4
S_PL24 = S_PLUS*24
S_ST24 = S_STAR*24
S_ST25 = S_STAR*25
S_DS25 = S_DASH*25
S_DS28 = S_DASH*28
S_DS30 = S_DASH*30
S_DS44 = S_DASH*44
S_DS80 = S_DASH*80
S_EQ80 = S_EQ*80
S_TB02 = S_TAB*2
S_TB03 = S_TAB*3
S_TB04 = S_TAB*4

S_C_N_MER = 'c15mer'
S_TRAIN = 'Train'
S_LBL = 'lbl'
S_CL = 'Cl'
S_TRUE_CL = 'True' + S_CL
S_PRED_CL = 'Pred' + S_CL
S_ALL = 'All'
S_EFF = 'Eff'
S_EFF_FAMILY = S_EFF + 'Family'
S_NUM_PRED = 'numPredicted'
S_NUM_CORR = 'numCorrect'
S_PROP_CORR = 'propCorrect'

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
LEN_N_MER_DEF = 15
I_CENT_N_MER = LEN_N_MER_DEF//2

R02 = 2
R04 = 4
R08 = 8

# ### INPUT ###################################################################
# === flow control ============================================================
doRndOvSamplTest = True

NmerUnique = True

# lLblTrain = [0, 1]                  # number of labels used for training data
lLblTrain = [1]                  # number of labels used for training data
                                    # or None [use all labels]

# === general input for any classifier ========================================
sUsedClf = 'MLP'            # 'RandomForest' / 'BalancedRandomForest'
                            # 'MLP'
rndState = None             # None (random) or integer (reproducible)

# --- input for (balanced) random forest classifier ---------------------------
n_estimators = 200
criterion = 'log_loss'

# --- input for neural network MLP classifier ---------------------------------
hidden_layer_sizes = (100,)
activation = 'relu'
solver = 'adam'
max_iter = 50000

# === general over- and undersampler input ====================================
sSampler = 'AllKNN'   # string matching the over/under-sampler
sStrat = 'majority'                 # sampling strategy
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
kind_sel_AllKNN = 'mode'         # strategy to exclude samples ['all' / 'mode']
allow_minority = False          # allows majority classes --> minority class

# --- NeighbourhoodCleaningRule input -----------------------------------------
n_neighbors_NCR = 3             # number of nearest neighbors
kind_sel_NCR = 'mode'            # strategy to exclude samples ['all' / 'mode']
threshold_cleaning = 0.5        # threshold 4 class considered during cleaning

# --- RandomUnderSampler and BalancedRandomForestClassifier input -------------
wReplacement = False            # is sample with or without replacement?

# --- TomekLinks input --------------------------------------------------------

# --- RandomOverSampler input -------------------------------------------------
shrinkSampling = None           # shrinkage applied to covariance matrix

# --- SMOTE input -------------------------------------------------------------
kNeighbors = 5

# --- ADASYN input ------------------------------------------------------------
nNeighbors = 5

# --- numbers -----------------------------------------------------------------
maxLenNmer = None           # odd number between 1 and 15 or None (max. len)
# maxLenNmer = 9           # odd number between 1 and 15 or None (max. len)

propTestData = .2

# --- strings -----------------------------------------------------------------
sUSC = S_USC
sCNmer = S_C_N_MER
sSnipCalcRF = 'LSV'
sTrueCl = S_TRUE_CL
sPredCl = S_PRED_CL
sEffFamily = S_EFF_FAMILY
sNumPred = S_NUM_PRED
sNumCorr = S_NUM_CORR
sPropCorr = S_PROP_CORR

# --- sets --------------------------------------------------------------------
setNmerLen = set(range(1, LEN_N_MER_DEF + 1, 2))

# --- lists -------------------------------------------------------------------
lIPosUsed = None                # None or list of positions used for classific.
# lIPosUsed = [-7, -5, -3, -2, -1, 1, 2, 3, 5, 7]

# --- lists -------------------------------------------------------------------
lSResClf = [sNumPred, sNumCorr, sPropCorr]

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
if maxLenNmer is not None:
    maxLenNmer = max(1, round(maxLenNmer))
    if 2*(maxLenNmer//2) >= maxLenNmer:     # i.e. maxLenNmer is even
        maxLenNmer += 1
if maxLenNmer not in setNmerLen:
    maxLenNmer = max(setNmerLen)

# --- Helper functions handling strings ---------------------------------------
def joinS(itS, sJoin=S_USC):
    lSJoin = [str(s) for s in itS if s is not None and len(str(s)) > 0]
    return sJoin.join(lSJoin).strip()

# === derived values and input processing =====================================
sFInpDClf = S_F_INP_CLF_COMB_XS
sFInpDClMap = S_F_INP_D_CLASSES
if NmerUnique:
    pass

pFInpDClf = os.path.join(P_INP_CLF, sFInpDClf + XT_CSV)
pFInpDClMap = os.path.join(P_INP_CLF, sFInpDClMap + XT_CSV)

maxPosNmer = maxLenNmer//2
rngPosNmer = range(-maxPosNmer, maxPosNmer + 1)
if lIPosUsed is None:
    lIPosUsed = list(rngPosNmer)
else:
    lIPosUsed = sorted(list(set(lIPosUsed) & set(rngPosNmer)))
lSCX = [str(n) for n in lIPosUsed]
lSCY = ['X_AGC', 'X_CDPK', 'X_CK_II', 'X_MAPK', 'X_SnRK2']

sCLblsTrain = ''
if lLblTrain is not None:
    sCLblsTrain = S_USC.join([str(nLbl) for nLbl in lLblTrain])
    sCLblsTrain = S_USC.join([S_LBL, sCLblsTrain, S_TRAIN])

# --- fill input dictionary ---------------------------------------------------
dITp = {# --- flow control ----------------------------------------------------
        'doRndOvSamplTest': doRndOvSamplTest,
        'NmerUnique': NmerUnique,
        'lLblTrain': lLblTrain,
        # --- files, directories and paths ------------------------------------
        'sFInpDClf': sFInpDClf,
        'sFInpDClMap': sFInpDClMap,
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        's0': S_0,
        's1': S_1,
        'sVBar': S_VBAR,
        'sPerc': S_PERC,
        'sCSV': S_CSV,
        'sUSC': sUSC,
        'sCNmer': sCNmer,
        'sSnipCalcRF': sSnipCalcRF,
        'sTrueCl': sTrueCl,
        'sPredCl': sPredCl,
        'sEffFamily': sEffFamily,
        'sNumPred': sNumPred,
        'sNumCorr': sNumCorr,
        'sPropCorr': sPropCorr,
        'sAll': S_ALL,
        # --- sets
        'setNmerLen': setNmerLen,
        # --- lists
        'lIPosUsed': lIPosUsed,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers ---------------------------------------------------------
        'lenNmerDef': LEN_N_MER_DEF,
        'iCentNmer': I_CENT_N_MER,
        'maxLenNmer': maxLenNmer,
        'propTestData': propTestData,
        # --- lists -----------------------------------------------------------
        'lSResClf': lSResClf,
        # --- dictionaries ----------------------------------------------------
        # === general input for any classifier --------------------------------
        'sUsedClf': sUsedClf,
        'rndState': rndState,
        # --- input for (balanced) random forest classifier -------------------
        'n_estimators': n_estimators,
        'criterion': criterion,
        # --- input for neural network MLP classifier -------------------------
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'max_iter': max_iter,
        # === general over- and undersampler input
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
        # --- RandomOverSampler input
        'shrinkSampling': shrinkSampling,
        # --- SMOTE input
        'kNeighbors': kNeighbors,
        # --- ADASYN input
        'nNeighbors': nNeighbors,
        # === derived values and input processing =============================
        'pFInpDClf': pFInpDClf,
        'pFInpDClMap': pFInpDClMap,
        'maxPosNmer': maxPosNmer,
        'rngPosNmer': rngPosNmer,
        'lSCX': lSCX,
        'lSCY': lSCY}

# ### FUNCTIONS ###############################################################
# --- General file system related functions -----------------------------------
def createDir(pF):
    if not os.path.isdir(pF):
        os.mkdir(pF)

def joinToPath(pF='', nmF='Dummy.txt'):
    if len(pF) > 0:
        createDir(pF)
        return os.path.join(pF, nmF)
    else:
        return nmF

def readCSV(pF, iCol=None, dDTp=None, cSep=S_SEMICOL):
    if os.path.isfile(pF):
        return pd.read_csv(pF, sep=cSep, index_col=iCol, dtype=dDTp)

def saveAsCSV(pdDfr, pF, reprNA='', cSep=S_SEMICOL):
    if pdDfr is not None:
        pdDfr.to_csv(pF, sep=cSep, na_rep=reprNA)

# --- Functions handling lists ------------------------------------------------
def addToList(cL, cEl, isUnq=False):
    if isUnq:
        if cEl not in cL:
            cL.append(cEl)
    else:
        cL.append(cEl)

def addToListUnq(cL, cEl):
    addToList(cL=cL, cEl=cEl, isUnq=True)

def toListUnique(cL=[]):
    cLUnq = []
    for cEl in cL:
        if cEl not in cLUnq:
            cLUnq.append(cEl)
    return cLUnq

def toListUnqViaSer(cIt=[]):
    return toSerUnique(pd.Series(cIt)).to_list()

# --- Functions handling dictionaries -----------------------------------------
def addToDictCt(cD, cK, cIncr=1):
    if cK in cD:
        cD[cK] += cIncr
    else:
        cD[cK] = cIncr

def addToDictL(cD, cK, cE, lUnqEl=False):
    if cK in cD:
        if not lUnqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def addToDictD(cD, cKMain, cKSub, cVSub=[], allowRpl=False):
    if cKMain in cD:
        if cKSub not in cD[cKMain]:
            cD[cKMain][cKSub] = cVSub
        else:
            if allowRpl:
                cD[cKMain][cKSub] = cVSub
            else:
                print('ERROR: Key', cKSub, 'already in', cD[cKMain])
                assert False
    else:
        cD[cKMain] = {cKSub: cVSub}

# --- Functions initialising numpy arrays -------------------------------------
def iniNpArr(data=None, shape=(0, 0), fillV=np.nan):
    if data is None:
        return np.full(shape, fillV)
    else:       # ignore shape
        return np.array(data)

# --- Functions initialising pandas objects -----------------------------------
def iniPdSer(data=None, lSNmI=[], shape=(0,), nameS=None, fillV=np.nan):
    assert (((type(data) == np.ndarray and len(data.shape) == 1) or
             (type(data) in [list, tuple]) or (data is None))
            and (len(shape) == 1))
    if lSNmI is None or len(lSNmI) == 0:
        if data is None:
            return pd.Series(np.full(shape, fillV), name=nameS)
        else:
            return pd.Series(data, name=nameS)
    else:
        if data is None:
            return pd.Series(np.full(len(lSNmI), fillV), index=lSNmI,
                             name=nameS)
        else:
            assert ((type(data) == np.ndarray and data.shape[0] == len(lSNmI))
                    or (type(data) == list and len(data) == len(lSNmI)))
            return pd.Series(data, index=lSNmI, name=nameS)

def iniPdDfr(data=None, lSNmC=[], lSNmR=[], shape=(0, 0), fillV=np.nan):
    assert len(shape) == 2
    nR, nC = shape
    if len(lSNmC) == 0:
        if len(lSNmR) == 0:
            if data is None:
                return pd.DataFrame(np.full(shape, fillV))
            else:
                return pd.DataFrame(data)
        else:
            if data is None:
                return pd.DataFrame(np.full((len(lSNmR), nC), fillV),
                                    index=lSNmR)
            else:
                return pd.DataFrame(data, index=lSNmR)
    else:
        if len(lSNmR) == 0:
            if data is None:
                return pd.DataFrame(np.full((nR, len(lSNmC)), fillV),
                                    columns=lSNmC)
            else:
                return pd.DataFrame(data, columns=lSNmC)
        else:   # ignore nR
            if data is None:
                return pd.DataFrame(np.full((len(lSNmR), len(lSNmC)), fillV),
                                    index=lSNmR, columns=lSNmC)
            else:
                return pd.DataFrame(data, index=lSNmR, columns=lSNmC)

# --- Functions performing pandas Series manipulation -------------------------
def concLObj(lObj, concAx=0, ignIdx=False, verifInt=False, srtDfr=False):
    return pd.concat(lObj, axis=concAx, ignore_index=ignIdx,
                     verify_integrity=verifInt, sort=srtDfr)

def concLObjAx0(lObj, ignIdx=False, verifInt=False, srtDfr=False):
    return concLObj(lObj, ignIdx=ignIdx, verifInt=verifInt, srtDfr=srtDfr)

def concLObjAx1(lObj, ignIdx=False, verifInt=False, srtDfr=False):
    return concLObj(lObj, concAx=1, ignIdx=ignIdx, verifInt=verifInt,
                    srtDfr=srtDfr)

def toSerUnique(pdSer, sName=None):
    nameS = pdSer.name
    if sName is not None:
        nameS = sName
    return pd.Series(pdSer.unique(), name=nameS)

# --- General-purpose functions -----------------------------------------------
def getAAcPyl(dITp, sNmer):
    if len(sNmer) == dITp['lenNmerDef']:
        return sNmer[dITp['iCentNmer']]

def getCentralPosOfSnip(dITp, sSnip=S_CAP_S):
    assert (len(sSnip) <= dITp['lenNmerDef'] and len(sSnip)%2 == 1)
    return sSnip[len(sSnip)//2]

def getCentralSnipOfNmer(dITp, sNmer, sSnip=S_CAP_S):
    assert len(sNmer) == dITp['lenNmerDef']
    assert (len(sSnip) <= dITp['lenNmerDef'] and len(sSnip)%2 == 1)
    iS = dITp['iCentNmer'] - len(sSnip)//2
    iE = dITp['iCentNmer'] + len(sSnip)//2 + 1
    return sNmer[iS:iE]

def checkCentralSnipOfNmer(dITp, sNmer, sSnip=S_CAP_S):
    return getCentralSnipOfNmer(dITp, sNmer=sNmer, sSnip=sSnip) == sSnip

# --- Functions converting between single- and multi-label systems ------------
def toMultiLbl(dITp, serY):
    dY = {}
    for sLbl in serY:
        for sXCl in dITp['lSCY']:           # lSCY <-> lXCl
            addToDictL(dY, cK=sXCl, cE=(1 if sXCl == sLbl else 0))
    return iniPdDfr(dY, lSNmR=serY.index)

def toSglLbl(dITp, dfrY):
    serY = None
    # check sanity
    if iniNpArr([(sum(serR) <= 1) for _, serR in dfrY.iterrows()]).all():
        lY = [None]*dfrY.shape[0]
        lSer = [serR.index[serR == 1] for _, serR in dfrY.iterrows()]
        for k, cI in enumerate(lSer):
            if cI.size >= 1:
                lY[k] = cI.to_list()[0]
        serY = iniPdSer(lY, lSNmI=dfrY.index, nameS=dITp['sEffFamily'])
    return serY

# --- Function loading the input for the classifier(s) ------------------------
def getClfInp(dITp, dfrInp):
    X, Y, serSeq = None, None, None
    sCNmer = dITp['sCNmer']
    if sCNmer in dfrInp.columns:
        X, Y = dfrInp[dITp['lSCX']], dfrInp[dITp['lSCY']]
        serSeq = toSerUnique(dfrInp[sCNmer])
        # for cSeq in serSeq:
            # cSeqRed = cSeq[(iCent - maxP):(iCent + maxP + 1)]
    return X, Y, serSeq

# --- Function for encoding and transforming the categorical features ---------
def encodeCatFeatures(catData=None):
    if catData is None:
        catData = X
    cEnc = OneHotEncoder()
    cEnc.fit(X)
    XTrans = cEnc.transform(catData).toarray()
    return iniPdDfr(XTrans, lSNmR=Y.index)

# --- Function for splitting data into training and test data -----------------
def getTrainTestDS(X, Y):
    tTrTe = train_test_split(X, Y, random_state=dITp['rndState'],
                             test_size=dITp['propTestData'])
    XTrain, XTest, YTrain, YTest = tTrTe
    if dITp['lLblTrain'] is not None:
        lB = [serR.sum() in dITp['lLblTrain'] for _, serR in
              YTrain.iterrows()]
        XTrain, YTrain = XTrain[lB], YTrain[lB]
    return XTrain, XTest, YTrain, YTest

# --- Function implementing imbalanced classes ("imblearn") -------------------
def getSampler(dITp):
    cSampler = None
    if dITp['sSampler'] == 'ClusterCentroids':
        cSampler = ClusterCentroids(sampling_strategy=dITp['sStrat'],
                                    random_state=dITp['rndState'],
                                    estimator=dITp['estimator'],
                                    voting=dITp['voting'])
    elif dITp['sSampler'] == 'AllKNN':
        cSampler = AllKNN(sampling_strategy=dITp['sStrat'],
                          n_neighbors=dITp['n_neighbors_AllKNN'],
                          kind_sel=dITp['kind_sel_AllKNN'],
                          allow_minority=dITp['allow_minority'])
    elif dITp['sSampler'] == 'NeighbourhoodCleaningRule':
        cSampler = NeighbourhoodCleaningRule(sampling_strategy=dITp['sStrat'],
                                             n_neighbors=dITp['n_neighbors_NCR'],
                                             kind_sel=dITp['kind_sel_NCR'],
                                             threshold_cleaning=dITp['threshold_cleaning'])
    elif dITp['sSampler'] == 'RandomUnderSampler':
        cSampler = RandomUnderSampler(sampling_strategy=dITp['sStrat'],
                                      random_state=dITp['rndState'],
                                      replacement=dITp['wReplacement'])
    elif dITp['sSampler'] == 'TomekLinks':
        cSampler = TomekLinks(sampling_strategy=dITp['sStrat'])
    elif dITp['sSampler'] == 'RandomOverSampler':
        cSampler = RandomOverSampler(sampling_strategy=dITp['sStrat'],
                                     random_state=dITp['rndState'],
                                     shrinkage=dITp['shrinkSampling'])
    elif dITp['sSampler'] == 'SMOTE':
        cSampler = SMOTE(sampling_strategy=dITp['sStrat'],
                         random_state=dITp['rndState'],
                         k_neighbors=dITp['kNeighbors'])
    elif dITp['sSampler'] == 'ADASYN':
        cSampler = ADASYN(sampling_strategy=dITp['sStrat'],
                          random_state=dITp['rndState'],
                          n_neighbors=dITp['nNeighbors'])
    return cSampler

def fitResampleImbalanced(dITp, cSampler, XTrain, YTrain):
    serYTrain = toSglLbl(dITp, dfrY=YTrain)
    print('Initial shape of serYTrain:', serYTrain.shape)
    XTrain, serYTrain = cSampler.fit_resample(XTrain, serYTrain)
    print('Final shape of serYTrain:', serYTrain.shape)
    for s in serYTrain.unique():
        print(s, S_TAB, serYTrain[serYTrain == s].size, sep='')
    return XTrain, serYTrain, toMultiLbl(dITp, serY=serYTrain)

# --- Function fitting the selected classifier --------------------------------
def getClf(dITp, X, Y):
    if dITp['sUsedClf'] == 'RandomForest':
        return fitRFClf(dITp, X, Y)
    elif dITp['sUsedClf'] == 'MLP':
        return fitMLPClf(dITp, X, Y)
    elif dITp['sUsedClf'] == 'BalancedRandomForest':
        return fitBalancedRFClf(dITp, X, Y)
    else:
        return None

# --- Function implementing and fitting the random forest classifier ----------
def fitRFClf(dITp, X, Y):
    cClf = RandomForestClassifier(random_state=dITp['rndState'],
                                  n_estimators=dITp['n_estimators'],
                                  criterion=dITp['criterion'])
    cClf.fit(X, Y)
    return cClf

# --- Function implementing and fitting the neural network MLP classifier -----
def fitMLPClf(dITp, X, Y):
    cClf = MLPClassifier(random_state=dITp['rndState'],
                         hidden_layer_sizes=dITp['hidden_layer_sizes'],
                         activation=dITp['activation'],
                         solver=dITp['solver'],
                         max_iter=dITp['max_iter'])
    cClf.fit(X, Y)
    return cClf

# --- Function implementing and fitting the balanced random forest classifier -
def fitBalancedRFClf(dITp, X, Y):
    cClf = BalancedRandomForestClassifier(sampling_strategy=dITp['sStrat'],
                                          random_state=dITp['rndState'],
                                          replacement=dITp['wReplacement'],
                                          n_estimators=dITp['n_estimators'],
                                          criterion=dITp['criterion'])
    cClf.fit(X, Y)
    return cClf

# --- Function for calculating values of the classifier results dictionary ----
def assembleDfrPredProba(lSCTP, YPred, YProba, serSeq=None):
    lDfr = [None, None]
    for k, cYP in enumerate([YPred, YProba]):
        lDfr[k] = concLObjAx1([YTest, cYP], ignIdx=True)
        lDfr[k].columns = lSCTP
        lDfr[k] = concLObjAx1(lObj=[serSeq, lDfr[k]])
        lDfr[k].dropna(axis=0, inplace=True)
        lDfr[k] = lDfr[k].convert_dtypes()
    return lDfr

def calcResPredict(dITp, dResClf, X2Pred=None, YTest=None, YPred=None,
                   YProba=None, serSeq=None):
    if (X2Pred is not None and YTest is not None and YPred is not None and
        YProba is not None and X2Pred.shape[0] == YPred.shape[0]):
        nPred = YPred.shape[0]
        nOK = sum([1 for k in range(nPred) if
                   (YTest.iloc[k, :] == YPred.iloc[k, :]).all()])
        propOK = (nOK/nPred if nPred > 0 else None)
        lVCalc = [nPred, nOK, propOK]
        for sK, cV in zip(dITp['lSResClf'], lVCalc):
            dResClf[sK] = cV
        assert (YTest.columns == YPred.columns).all()
        nOKA, nFA = 0, 0
        for s in YTest.columns:
            nOK = sum([1 for k in range(nPred) if
                       (YTest.iloc[k, :].at[s] == YPred.iloc[k, :]).at[s]])
            nF = sum([1 for k in range(nPred) if
                      (YTest.iloc[k, :].at[s] != YPred.iloc[k, :]).at[s]])
            dResClf[joinS([s, dITp['sAll']])] = (nOK, nF)
            nOKA += nOK
            nFA += nF
        dResClf[dITp['sAll']] = (nOKA, nFA)
        for i, sI in zip([0, 1], [dITp['s0'], dITp['s1']]):
            nOKI, nFI = 0, 0
            for s in YTest.columns:
                nOK = sum([1 for k in range(nPred) if
                           YTest.iloc[k, :].at[s] == i and
                           YPred.iloc[k, :].at[s] == i])
                nF = sum([1 for k in range(nPred) if
                          YTest.iloc[k, :].at[s] == i and
                          YPred.iloc[k, :].at[s] == 1 - i])
                dResClf[joinS([s, sI])] = (nOK, nF)
                nOKI += nOK
                nFI += nF
            dResClf[sI] = (nOKI, nFI)
        sTCl, sPCl, sJ = dITp['sTrueCl'], dITp['sPredCl'], dITp['sUSC']
        # create dfrPred/dfrProba, containing YTest and YPred/YProba
        lSCTP = [joinS([s, sTCl], sJoin=sJ) for s in YTest.columns]
        lSCTP += [joinS([s, sPCl], sJoin=sJ) for s in YPred.columns]
        return assembleDfrPredProba(lSCTP=lSCTP, YPred=YPred, YProba=YProba,
                                    serSeq=serSeq)

# --- Function for getting the probabilities of the predicted classes ---------
def getYProba(cClf, dat2Pr=None, lSC=None, lSR=None, i=0):
    arrProba = cClf.predict_proba(dat2Pr)
    if type(arrProba) == list:    # e.g. result of a random forest classifier
        return iniPdDfr(np.column_stack([1 - cArr[:, i] for cArr in arrProba]),
                        lSNmC=lSC, lSNmR=lSR)
    elif type(arrProba) == np.ndarray:      # standard case
        return iniPdDfr(arrProba, lSNmC=lSC, lSNmR=lSR)

# --- Function for predicting with a Classifier -------------------------------
def ClfPred(dITp, dRes, fittedClf, XTest, YTest, serSeq):
    if fittedClf is not None:
        lSCMlt, lSI = YTest.columns, YTest.index
        YPred = iniPdDfr(fittedClf.predict(XTest), lSNmC=lSCMlt, lSNmR=lSI)
        YProba = getYProba(fittedClf, XTest, lSC=lSCMlt, lSR=lSI)
        assert YProba.shape == YPred.shape
        return calcResPredict(dITp, dResClf=dRes, X2Pred=XTest, YTest=YTest,
                              YPred=YPred, YProba=YProba, serSeq=serSeq)
    else:
        return None, None

# --- Function printing the results -------------------------------------------
def printDfrRes(dITp, dRes):
    for sK, cV in dRes.items():
        if sK in dITp['lSResClf']:
            print(sK, S_COLON, (S_TAB if (len(sK) >= 11) else S_TB02),
                  round(cV, R04), sep='')
        else:
            assert type(cV) == tuple and len(cV) == 2
            (nOK, nF), nC = cV, sum(cV)
            print(sK, S_COLON, (S_TAB if (len(sK) >= 11) else
                                (S_TB02 if (len(sK) >= 7) else
                                 (S_TB03 if (len(sK) >= 3) else S_TB04))),
                  round(nOK, R04), S_TAB, S_VBAR, S_TAB,
                  round(nF, R04), S_TAB, S_VBAR, S_TAB,
                  round(nC, R04), S_TAB, S_VBAR, S_TAB,
                  round((nOK/nC if (nC > 0) else (0.))*100, R02), S_PERC,
                  sep='')

# --- Function saving the results ---------------------------------------------
def saveDfrPredProba(dITp, dfrPred, dfrProba):
    sStrat = dITp['sStrat']
    if type(dITp['sStrat']) == str:
        sStrat = dITp['sStrat'].replace(S_SPACE, '')
    elif type(dITp['sStrat']) == dict:
        sStrat = joinS(['Dict', 'minN', str(min(dITp['sStrat'].values()))])
    sFCore = joinS([dITp['sUsedClf'], dITp['sSampler'], sStrat])
    sFPred = joinS(['dfrPred', sFCore]) + dITp['xtCSV']
    sFProba = joinS(['dfrProba', sFCore]) + dITp['xtCSV']
    saveAsCSV(dfrPred, pF=joinToPath(pF='SavedData', nmF=sFPred))
    saveAsCSV(dfrProba, pF=joinToPath(pF='SavedData', nmF=sFProba))

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' ImbalancedLearnTests.py ', S_DS25, S_NEWL,
      sep='')
if (doRndOvSamplTest):
    dRes, dfrInpClf = {}, readCSV(pF=dITp['pFInpDClf'], iCol=0)
    X, Y, serNmerSeq = getClfInp(dITp, dfrInp=dfrInpClf)
    XEnc = encodeCatFeatures(catData=X)
    XTrain, XTest, YTrain, YTest = getTrainTestDS(X=XEnc, Y=Y)

    # imbalanced part
    if dITp['sUsedClf'] not in ['BalancedRandomForest']:
        t = fitResampleImbalanced(dITp, getSampler(dITp), XTrain, YTrain)
        XTrain, YTrainSgl, YTrain = t

    # classifier fit part
    cClf = getClf(dITp, X=XTrain, Y=YTrain)
    dfrPredicted, dfrProbabilities = ClfPred(dITp, dRes=dRes, fittedClf=cClf,
                                             XTest=XTest, YTest=YTest,
                                             serSeq=serNmerSeq)
    printDfrRes(dITp, dRes=dRes)
    saveDfrPredProba(dITp, dfrPred=dfrPredicted, dfrProba=dfrProbabilities)

print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################