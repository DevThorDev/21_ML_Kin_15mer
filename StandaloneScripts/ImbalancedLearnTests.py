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

from imblearn.over_sampling import RandomOverSampler

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

S_C_N_MER = 'c15mer'
S_TRAIN = 'Train'
S_LBL = 'lbl'
S_CL = 'Cl'
S_TRUE_CL = 'True' + S_CL
S_PRED_CL = 'Pred' + S_CL
S_ALL = 'All'
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
# --- flow control ------------------------------------------------------------
doRndOvSamplTest = True

NmerUnique = True

lLblTrain = [1]                    # number of labels used for training data
                                    # or None [use all labels]

sSampler = 'RandomOverSampler'      # string matching the over/under-sampler
stratSampling = 'auto'              # sampling strategy
shrinkSampling = None               # shrinkage applied to covariance matrix

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
sNumPred = S_NUM_PRED
sNumCorr = S_NUM_CORR
sPropCorr = S_PROP_CORR

# --- sets --------------------------------------------------------------------
setNmerLen = set(range(1, LEN_N_MER_DEF + 1, 2))

# --- lists -------------------------------------------------------------------
lSResClf = ['numPredicted', 'numCorrect', 'propCorrect']

# --- dictionaries ------------------------------------------------------------

# --- general input for any classifier ----------------------------------------
rndState = None             # None (random) or integer (reproducible)

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
lSCX = [str(n) for n in rngPosNmer]
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
        'sSampler': sSampler,
        'stratSampling': stratSampling,
        'shrinkSampling': shrinkSampling,
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
        'sNumPred': sNumPred,
        'sNumCorr': sNumCorr,
        'sPropCorr': sPropCorr,
        'sAll': S_ALL,
        # --- sets
        'setNmerLen': setNmerLen,
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
        # --- general input for any classifier --------------------------------
        'rndState': rndState,
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
    assert len(shape) == 1
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
            assert data.size == len(lSNmI)
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

# --- Function loading the input for the classifier(s) ------------------------
def getClfInp(dITp, dfrInp):
    X, Y, serSeq = None, None, None
    # iCent, maxP, sCNmer = dITp['iCentNmer'], dITp['maxPosNmer'], dITp['sCNmer']
    sCNmer = dITp['sCNmer']
    if sCNmer in dfrInp.columns:
        X, Y = dfrInp[dITp['lSCX']], dfrInp[dITp['lSCY']]
        serSeq = iniPdSer(dfrInp[sCNmer].unique(), nameS=sCNmer)
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
def getOverSampler(dITp):
    cOvSampler = None
    if dITp['sSampler'] == 'RandomOverSampler':
        cOvSampler = RandomOverSampler(sampling_strategy=dITp['stratSampling'],
                                       random_state=dITp['rndState'],
                                       shrinkage=dITp['shrinkSampling'])
    return cOvSampler

def fitResampleImbalanced(cSampler, XTrain, YTrain):
    print('XTrain (BEFORE):\n', XTrain, sep='')
    print('YTrain (BEFORE):\n', YTrain, sep='')
    serYTrain = pd.Series([None]*YTrain.shape[0], index=YTrain.index,
                          name='YClass')
    for cI, serR in YTrain.iterrows():
        assert sum(serR) <= 1
        for s in YTrain.columns:
            if serR.at[s] == 1:
                serYTrain.at[cI] = s
                break
    XTrain, serYTrain = cSampler.fit_resample(XTrain, serYTrain)
    print('XTrain (AFTER):\n', XTrain, sep='')
    print('YTrain (AFTER):\n', YTrain, sep='')
    print('serYTrain (AFTER):\n', serYTrain, sep='')
    return XTrain, serYTrain

# --- Function implementing and fitting the random forest classifier ----------
def fitRndForestClf(dITp, X, Y):
    cClf = RandomForestClassifier(random_state=dITp['rndState'])
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
        for s in YTest.columns:
            nOK = sum([1 for k in range(nPred) if
                       (YTest.iloc[k, :].at[s] == YPred.iloc[k, :]).at[s]])
            nF = sum([1 for k in range(nPred) if
                      (YTest.iloc[k, :].at[s] != YPred.iloc[k, :]).at[s]])
            dResClf[joinS([s, dITp['sAll']])] = (nOK, nF)
        for i, sI in zip([0, 1], [dITp['s0'], dITp['s1']]):
            for s in YTest.columns:
                nOK = sum([1 for k in range(nPred) if
                           YTest.iloc[k, :].at[s] == i and
                           YPred.iloc[k, :].at[s] == i])
                nF = sum([1 for k in range(nPred) if
                          YTest.iloc[k, :].at[s] == i and
                          YPred.iloc[k, :].at[s] == 1 - i])
                dResClf[joinS([s, sI])] = (nOK, nF)
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
        lSC, lSR = YTest.columns, YTest.index
        YPred = iniPdDfr(fittedClf.predict(XTest), lSNmC=lSC, lSNmR=lSR)
        YProba = getYProba(fittedClf, XTest, lSC=lSC, lSR=lSR)
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
            print(sK, S_COLON, (S_TAB if (len(sK) >= 11) else S_TB02),
                  round(nOK, R04), S_TAB, S_VBAR, S_TAB,
                  round(nF, R04), S_TAB, S_VBAR, S_TAB,
                  round(nC, R04), S_TAB, S_VBAR, S_TAB,
                  round((nOK/nC if (nC > 0) else (0.))*100, R02), S_PERC,
                  sep='')

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' ImbalancedLearnTests.py ', S_DS25, S_NEWL,
      sep='')
if (doRndOvSamplTest):
    dRes, dfrInpClf = {}, readCSV(pF=dITp['pFInpDClf'], iCol=0)
    print('dfrInpClf:\n', dfrInpClf, sep='')
    X, Y, serNmerSeq = getClfInp(dITp, dfrInp=dfrInpClf)
    print('serNmerSeq:\n', serNmerSeq, sep='')
    print('X:\n', X, sep='')
    print('Y:\n', Y, sep='')
    XEnc = encodeCatFeatures(catData=X)
    XTrain, XTest, YTrain, YTest = getTrainTestDS(X=XEnc, Y=Y)
    
    # imbalanced part
    cOverSampler = getOverSampler(dITp)
    XTrain, YTrain = fitResampleImbalanced(cSampler=cOverSampler,
                                           XTrain=XTrain, YTrain=YTrain)
    
    # RF Classifier fit part
    RFClf = fitRndForestClf(dITp, X=XTrain, Y=YTrain)
    dfrPredicted, dfrProbabilities = ClfPred(dITp, dRes=dRes, fittedClf=RFClf,
                                             XTest=XTest, YTest=YTest,
                                             serSeq=serNmerSeq)
    print('dfrPredicted:\n', dfrPredicted, sep='')
    print('dfrProbabilities:\n', dfrProbabilities, sep='')
    print('dfrPredicted with X_MAPK:\n',
          dfrPredicted[dfrPredicted['X_MAPK_PredCl'] > 0], sep='')
    print('dRes:\n', dRes, sep='')
    printDfrRes(dITp, dRes=dRes)

print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
