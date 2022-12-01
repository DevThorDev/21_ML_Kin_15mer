# -*- coding: utf-8 -*-
###############################################################################
# --- UseLocData.py -----------------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from imblearn.under_sampling import RandomUnderSampler

# === CONSTANTS ===============================================================
# --- strings -----------------------------------------------------------------
S_DOT, S_COLON, S_SEMICOL, S_DASH, S_USC = '.', ':', ';', '-', '_'
S_SPACE, S_DBL_BS, S_VBAR, S_TAB, S_NEWL = ' ', '\\', '|', '\t', '\n'
S_EXT_PY = 'py'
S_EXT_CSV = 'csv'

S_NO, S_NONE, S_MULTI, S_EFF, S_FAM = 'No', 'None', 'Multi', 'Eff', 'Fam'
S_EFF_FAM = S_EFF + S_FAM
S_NO_FAM, S_MULTI_FAM = S_NO + S_FAM, S_MULTI + S_FAM
S_ONE_HOT, S_ORDINAL = 'OneHot', 'Ordinal'
S_STRAT_REAL_MAJO, S_STRAT_SHARE_MINO = 'RealMajo', 'ShareMino'
RF_CLF, MLP_CLF = 'RF', 'MLP'

S_DS04 = S_DASH*4
S_VBAR_SEP = S_SPACE + S_VBAR + S_SPACE

S_C_N_MER = 'c15mer'

# --- file name extensions ----------------------------------------------------
XT_PY = S_DOT + S_EXT_PY
XT_CSV = S_DOT + S_EXT_CSV

# --- file names --------------------------------------------------------------
S_F_INP = 'Pho15mer_wLoc_202202'
S_F_OUT = 'LocClfRes'

# --- directories and paths ---------------------------------------------------
S_DIR_L0_ALL_DAT = '13_Sysbio03_Phospho15mer'
S_DIR_L1_INP_PROC = '11_ProcInpData'
S_DIR_L1_RES_CHECK = '99_Checks'

P_DIR_ROOT = os.path.join('..', '..', '..')
P_DIR_INP = os.path.join(P_DIR_ROOT, S_DIR_L0_ALL_DAT, S_DIR_L1_INP_PROC)
P_DIR_OUT = os.path.join(P_DIR_ROOT, S_DIR_L0_ALL_DAT, S_DIR_L1_RES_CHECK)

# === INPUT ===================================================================
# --- boolean values ----------------------------------------------------------
doImbSampling = True
ILblSgl = False
calcCnfMatrix = True

# --- numbers -----------------------------------------------------------------
nItFit = 5

# --- strings -----------------------------------------------------------------
sNo, sNone = S_NO, S_NONE
sCNmer = S_C_N_MER
sLoc = 'location'
sEffFam, sNoFam, sMultiFam = S_EFF_FAM, S_NO_FAM, S_MULTI_FAM
sOneHot, sOrdinal = S_ONE_HOT, S_ORDINAL
sStratRealMajo, sStratShareMino = S_STRAT_REAL_MAJO, S_STRAT_SHARE_MINO
RFClf, MLPClf = RF_CLF, MLP_CLF

selClf = MLPClf
sStrat = sStratRealMajo

# --- sets --------------------------------------------------------------------

# --- lists -------------------------------------------------------------------
lSColX = list(range(-7, 8))
lSXCl = ['X_AGC', 'X_CK_II', 'X_SnRK2', 'X_soluble']

# --- dictionaries ------------------------------------------------------------
dIStrat = {sStratRealMajo: 0.75, sStratShareMino: 1.0}

# === DERIVED VALUES ==========================================================
lSEnc = [sOneHot, sOrdinal]
lSmplStratCustom = [sStratRealMajo, sStratShareMino]

# === INPUT DICTIONARY ========================================================
dInp = {# --- boolean values --------------------------------------------------
        'doImbSampling': doImbSampling,
        'ILblSgl': ILblSgl,
        'calcCnfMatrix': calcCnfMatrix,
        # --- numbers ---------------------------------------------------------
        'nItFit': nItFit,
        # --- strings (1) -----------------------------------------------------
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sDash': S_DASH,
        'sUsc': S_USC,
        'sDblBS': S_DBL_BS,
        # --- file name extensions --------------------------------------------
        'xtPY': XT_PY,
        'xtCSV': XT_CSV,
        # --- file names
        'sFInp': S_F_INP,
        'sFOut': S_F_OUT,
        # --- directories and paths -------------------------------------------
        'sDirL0_AllDat': S_DIR_L0_ALL_DAT,
        'sDirL1_InpProc': S_DIR_L1_INP_PROC,
        'sDirL1_ResCheck': S_DIR_L1_RES_CHECK,
        'pDirRoot': P_DIR_ROOT,
        'pDirInp': P_DIR_INP,
        'pDirOut': P_DIR_OUT,
        # --- strings (2) -----------------------------------------------------
        'sNo': sNo,
        'sNone': sNone,
        'sCNmer': sCNmer,
        'sLoc': sLoc,
        'sEffFam': sEffFam,
        'sNoFam': sNoFam,
        'sMultiFam': sMultiFam,
        'sOneHot': sOneHot,
        'sOrdinal': sOrdinal,
        'sStratRealMajo': sStratRealMajo,
        'sStratShareMino': sStratShareMino,
        'RFClf': RFClf,
        'MLPClf': MLPClf,
        'selClf': selClf,
        'sStrat': sStrat,
        # --- sets ------------------------------------------------------------
        # --- lists -----------------------------------------------------------
        'lSColX': lSColX,
        'lSXCl': lSXCl,
        # --- dictionaries ----------------------------------------------------
        'dIStrat': dIStrat,
        # === derived values ==================================================
        'lSEnc': [sOneHot, sOrdinal],
        'lSmplStratCustom': lSmplStratCustom}

# === FUNCTIONS ===============================================================
# --- general functions -------------------------------------------------------
def createDir(pF):
    if not os.path.isdir(pF):
        os.mkdir(pF)

def joinToPath(pF=None, sF=None, xtF=None):
    if sF is not None:
        if xtF is not None:
            sF += xtF
    if pF is not None and len(pF) > 0:
        createDir(pF)
        if sF is not None:
            return os.path.join(pF, sF)
        else:
            return pF
    else:
        if sF is not None:
            return os.path.join(pF, sF)
        else:
            return None

def addToDictL(cD, cK, cE, lUnqEl=False):
    if cK in cD:
        if not lUnqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def fileXist(pF):
    return os.path.isfile(pF)

def readCSV(pF, iCol=None, dDTp=None, cSep=S_SEMICOL):
    if fileXist(pF):
        return pd.read_csv(pF, sep=cSep, index_col=iCol, dtype=dDTp)

def saveCSV(pdObj, pF, reprNA='', cSep=S_SEMICOL, saveIdx=True, iLbl=None):
    if pdObj is not None:
        pdObj.to_csv(pF, sep=cSep, na_rep=reprNA, index=saveIdx,
                     index_label=iLbl)

def iniPdDfr(data=None, lSNmC=[], lSNmR=[], shape=(0, 0), fillV=np.nan):
    assert len(shape) == 2
    nR, nC = shape
    if lSNmC is None or len(lSNmC) == 0:
        if lSNmR is None or len(lSNmR) == 0:
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
        if lSNmR is None or len(lSNmR) == 0:
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

# --- Functions implementing custom imbalanced sampling strategies ------------
def smplStratRealMajo(Y, dI):
    shCl = (dI[S_STRAT_REAL_MAJO] if S_STRAT_REAL_MAJO in dI else 1)
    shCl = max(0, min(1, shCl))
    sStrat = {cCl: round((Y[Y == cCl].size)*shCl) for cCl in Y.unique()}
    lVSrtDsc, n1, n2 = sorted(sStrat.values(), reverse=True), 0, 0
    if len(lVSrtDsc) >= 2:
        n1, n2 = lVSrtDsc[0], lVSrtDsc[1]
    elif len(lVSrtDsc) == 1:
        n1, n2 = lVSrtDsc[0], lVSrtDsc[0]
    for cCl, nEl in sStrat.items():
        if nEl == n1:
            sStrat[cCl] = n2
    return sStrat

def smplStratShareMino(Y, dI):
    shMino = (dI[S_STRAT_SHARE_MINO] if S_STRAT_SHARE_MINO in dI else 1)
    shMino = max(0, min(1, shMino))
    sStrat = {cCl: (Y[Y == cCl].size) for cCl in Y.unique()}
    lVSrtAsc, n = sorted(sStrat.values()), 0
    if len(lVSrtAsc) >= 1:
        n = round(lVSrtAsc[0]*shMino)
    for cCl, nEl in sStrat.items():
        sStrat[cCl] = max(1, n)
    return sStrat

# --- specific functions ------------------------------------------------------
# --- print functions ---------------------------------------------------------
def printStrat(dITp):
    if dITp['doImbSampling'] and dITp['sStrat'] is not None:
        print(S_DS04, 'Sampling strategy: ', dITp['sStrat'], sep='')
    else:
        print(S_DS04, 'No imbalanced sampling.')

def printResResampleImb(YIni, YRes, doPrt=True):
    if doPrt:
        print(S_DS04, ' Size of Y:', S_NEWL, 'Initial: ', YIni.size,
              S_VBAR_SEP, 'after resampling: ', YRes.size, sep='')
        YIniUnq, YResUnq = YIni.unique(), YRes.unique()
        print('Unique values of initial Y:', YIniUnq)
        print('Unique values of resampled Y:', YResUnq)
        assert set(YIniUnq) == set(YResUnq)
        print('Sizes of classes before and after resampling:')
        for cY in YResUnq:
            print(cY, S_COLON, S_TAB, YIni[YIni == cY].size, S_VBAR_SEP,
                  YRes[YRes == cY].size, sep='')

# --- Functions converting between single- and multi-labels (imbalanced) ------
def toMultiLblExt(serY, lXCl, sJoin=S_VBAR):
    # also converts "NoFam" and "MultiFam" labels
    if len(serY.shape) > 1:     # already multi-column (DataFrame) format
        return serY
    assert type(serY) == pd.core.series.Series
    dY = {}
    for sLbl in serY:
        if sJoin in sLbl:
            lSLC = sLbl.split(sJoin)[1:]
            for sXCl in lXCl:
                addToDictL(dY, cK=sXCl, cE=(1 if sXCl in lSLC else 0))
        else:
            for sXCl in lXCl:
                addToDictL(dY, cK=sXCl, cE=(1 if sXCl == sLbl else 0))
    return iniPdDfr(dY, lSNmR=serY.index)

def getSSglLbl(x, lXCl, sNoFam, sMltFam, sJoin=S_VBAR):
    assert x.size == len(lXCl)
    if sum(x) == 1:
        return lXCl[x.to_list().index(1)]
    elif sum(x) > 1:
        lSJ = [sMltFam] + [sX for k, sX in enumerate(lXCl) if x[k] == 1]
        return sJoin.join(lSJ)
    else:
        return sNoFam

def toSglLblExt(dITp, dfrY, sJoin=S_VBAR):
    # also assigns "NoFam" and "MultiFam" labels
    if len(dfrY.shape) == 1:   # already single-column (Series) format
        return dfrY
    serY = dfrY.apply(getSSglLbl, lXCl=dfrY.columns, sNoFam=dITp['sNoFam'],
                      sMltFam=dITp['sMultiFam'], sJoin=sJoin, axis=1)
    serY.name = dITp['sEffFam']
    return serY

# --- function for encoding and transforming the categorical features ---------
def encodeCatFeatures(dITp, catData, tpEnc=S_ONE_HOT):
    if catData is not None:
        cEnc, XEnc = None, catData
        if tpEnc in dITp['lSEnc']:    # encoders implemented so far
            if tpEnc == dITp['sOneHot']:
                cEnc = OneHotEncoder()
                XEnc = cEnc.fit_transform(catData).toarray()
            else:
                cEnc = OrdinalEncoder(dtype=int, encoded_missing_value=-1)
                XEnc = cEnc.fit_transform(catData)
    return iniPdDfr(XEnc, lSNmR=catData.index)

# --- function obtaining a custom (imbalanced) sampling strategy --------------
def getStrat(dITp, YS):
    # get default strategy
    printStrat(dITp)
    # in case of a custom sampling strategy, calculate the dictionary
    if dITp['sStrat'] in dITp['lSmplStratCustom']:
        if dITp['sStrat'] == dITp['sStratRealMajo']:
            # implement the "RealMajo" strategy
            dITp['sStrat'] = smplStratRealMajo(YS, dI=dITp['dIStrat'])
        elif dITp['sStrat'] == dITp['sStratShareMino']:
            # implement the "ShareMino" strategy
            dITp['sStrat'] = smplStratShareMino(YS, dI=dITp['dIStrat'])
    return dITp['sStrat']

# --- Function obtaining the desired imbalanced sampler ("imblearn") ----------
def getImbSampler(dITp, YS):
    imbSmp = RandomUnderSampler(sampling_strategy=getStrat(dITp, YS=YS),
                                random_state=None,
                                replacement=False)
    return imbSmp


# --- function performing the random sampling ("imblearn") --------------------
def fitResampleImbalanced(dITp, X, YS):
    XResImb, YResImb = getImbSampler(dITp, YS=YS).fit_resample(X, YS)
    printResResampleImb(YIni=YS, YRes=YResImb)
    if not dITp['ILblSgl']:
        YResImb = toMultiLblExt(serY=YResImb, lXCl=dITp['lSXCl'])
    return XResImb, YResImb

# --- function for getting the desired classifier -----------------------------
def getClassifier(dITp, n):
    cClf = None
    if dITp['selClf'] == dITp['RFClf']:
        cClf = RandomForestClassifier(random_state=None,
                                      warm_start=True,
                                      verbose=0,
                                      oob_score=False,
                                      n_jobs=None,
                                      n_estimators=100*n,
                                      criterion='log_loss')
    elif dITp['selClf'] == dITp['MLPClf']:
        hLS = (100,)
        if n == 2:
            hLS = (512,)
        if n == 3:
            hLS = (512, 128)
        if n == 4:
            hLS = (1024, 256, 64)
        if n == 5:
            hLS = (2048, 512, 128, 32)
        cClf = MLPClassifier(random_state=None,
                             warm_start=True,
                             verbose=False,
                             hidden_layer_sizes=hLS,
                             activation='relu',
                             solver='adam',
                             max_iter=1000*n,
                             tol=1e-4)
    return cClf

# --- function for calculating the confusion matrix ---------------------------
def calcCnfMatrix(dITp, YTest=None, YPred=None):
    if dITp['calcCnfMatrix']:
        YTest = toSglLblExt(dITp, dfrY=YTest)
        YPred = toSglLblExt(dITp, dfrY=YPred)
        lC = sorted(list(set(YTest.unique()) | set(YPred.unique())))
        cnfMat = confusion_matrix(y_true=YTest, y_pred=YPred, labels=lC)
        dfrCM = iniPdDfr(cnfMat, lSNmC=lC, lSNmR=lC)
        saveCSV(dfrCM, pF='ConfMatrix' + dITp['xtCSV'])

# === MAIN ====================================================================
pFInp = joinToPath(pF=dInp['pDirInp'], sF=dInp['sFInp'], xtF=dInp['xtCSV'])
pFOut = joinToPath(pF=dInp['pDirOut'], sF=dInp['sFOut'], xtF=dInp['xtCSV'])

dfrInp = readCSV(pF=pFInp, iCol=0)
print('Input DataFrame with shape', dfrInp.shape, ':\n', dfrInp, sep='')

dfrRed = dfrInp[[dInp['sCNmer'], dInp['sLoc']]].drop_duplicates(ignore_index=True)
print('Reduced DataFrame with shape', dfrRed.shape, ':\n', dfrRed, sep='')

# XInp = dfrInp[dInp['sCNmer']].apply(lambda x: pd.Series(list(x),
#                                                         index=dInp['lSColX']))
# print('XInp:\n', XInp, sep='')

# pYM = joinToPath(pF=dInp['pDirInpD'], sF=dInp['sFInpYM'], xtF=dInp['xtCSV'])
# XM = readCSV(pXM, iCol=0)
# YM = readCSV(pYM, iCol=0)

# print('Encoding categorical features...')
# XMEnc = encodeCatFeatures(dInp, catData=XM)
# print('XM:\n', XM, sep='')
# print('YM:\n', YM, sep='')

# print('XMTrain:\n', XMTrain, sep='')
# print('YMTrain:\n', YMTrain, sep='')
# print('XMTest:\n', XMTest, sep='')
# print('YMTest:\n', YMTest, sep='')

dScore = {}
# for k in range(dInp['nItFit']):
#     print('Performing a train-test-split of the data...')
#     XMTrain, XMTest, YMTrain, YMTest = train_test_split(XMEnc, YM,
#                                                         test_size=0.2)
#     if dInp['doImbSampling']:
#         YSTrain = toSglLblExt(dInp, dfrY=YMTrain)
#         YSTrain.to_csv('YSTrain' + S_USC + str(k + 1) + XT_CSV, sep=S_SEMICOL)
#         XMTrRes, YMTrRes = fitResampleImbalanced(dInp, X=XMTrain, YS=YSTrain)
#     cClf = getClassifier(dITp=dInp, n=(k+1))
#     # print('Fitting the random forest classifier...')
#     cClf.fit(XMTrRes, YMTrRes)
#     XMTrRes.columns = [str(s) + S_USC + str(k + 1) for s in XMTrRes.columns]
#     XMTrRes.to_csv('XMTrRes' + S_USC + str(k + 1) + XT_CSV, sep=S_SEMICOL)
#     YMTrRes.columns = [s + S_USC + str(k + 1) for s in YMTrRes.columns]
#     YMTrRes.to_csv('YMTrRes' + S_USC + str(k + 1) + XT_CSV, sep=S_SEMICOL)
#     # print('Calculating the score of the random forest classifier...')
#     cScore = cClf.score(XMTest, YMTest)
#     dScore[k + 1] = round(cScore, 6)
# for cSt, cScore in dScore.items():
#     print(dInp['sDash']*10, 'STEP', cSt, '| score =', cScore, dInp['sDash']*10)

# print('Predicting with the fitted classifier...')
# YMPred = pd.DataFrame(cClf.predict(XMTest), index=YMTest.index,
#                       columns=YMTest.columns)
# calcCnfMatrix(dITp=dInp, YTest=YMTest, YPred=YMPred)

# =============================================================================
###############################################################################