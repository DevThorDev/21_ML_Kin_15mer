# -*- coding: utf-8 -*-
###############################################################################
# --- DataFrameHandling.py ----------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

# ### CONSTANTS ###############################################################
# --- sets for class dictionary -----------------------------------------------
SET01 = 'Set01_11Cl'
SET02 = 'Set02_06Cl'

C_SET = SET02

# --- files, directories and paths --------------------------------------------
P_PROC_I_N_MER = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                              '11_ProcInpData')
P_INP_CLF = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                         '21_InpDataClf')
P_COMB_RES = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                          '31_ResCombined')
P_TEMP_RES = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                          '98_TEMP_CSV')

S_F_PROC_I_N_MER = 'Pho15mer_202202'
S_F_RES_COMB_S = 'Combined_S_KinasesPho15mer_202202'
S_F_INP_CLF_COMB_XS = 'InpClf_Combined_XS_KinasesPho15mer_202202'
S_F_INP_D_CLASSES = 'InpClf_ClMapping_' + C_SET
S_F_REL_FREQ_EFF_FAM = 'RelFreqEffFamily_Combined_XS_KinasesPho15mer_202202'
S_F_REL_FREQ_X_CL = 'RelFreqXClasses_Combined_XS_KinasesPho15mer_202202'
S_F_NMER_KIN_FAM_MAP = 'MapNmer2KinFamily_Combined_XS_KinasesPho15mer_202202'

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
S_CAP_S, S_CAP_X = 'S', 'X'

S_CSV = 'csv'

S_SP04 = S_SPACE*4
S_PL24 = S_PLUS*24
S_ST24 = S_STAR*24
S_ST25 = S_STAR*25
S_DS28 = S_DASH*28
S_DS30 = S_DASH*30
S_DS44 = S_DASH*44
S_DS80 = S_DASH*80
S_EQ80 = S_EQ*80

S_C_CODE_SEQ = 'code_seq'
S_C_PEP_POS = 'pep_pos_in_prot'
S_C_N_MER = 'c15mer'
S_CL = 'Cl'
S_EFF = 'Eff'
S_FAM = 'Fam'
S_EFFECTOR = 'Effector'
S_FAMILY = 'Family'
S_EFF_FAMILY = S_EFF + S_FAMILY
S_X_CL = S_CAP_X + S_CL

S_UNIQUE = 'Unique'
S_PROC_I_N_MER = 'ProcINmer'
S_COMB_S = 'Combined_S'
S_UNIQUE_COL = S_UNIQUE + 'Col'
S_UNIQUE_CODE_SEQ = S_UNIQUE + 'CodeSeq'
S_PEP_P = 'PepPos'
S_N_MER = 'Nmer'
S_REL_FREQ_SNIP = 'RelFreqSnip'

S_N_OCC = 'nOcc'
S_REL_FREQ = 'relFreq'

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
LEN_N_MER_DEF = 15
I_CENT_N_MER = LEN_N_MER_DEF//2

R08 = 8

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
inpTbl = 'CombS'            # 'ProcI' / 'CombS'

doAllUniqueCol = False
doUniqueCodeSeqRetPepPos = False
doUniqueCodeSeqRetNmer = False
doRelFreqSnip = False
getFreqAllKinCl = False
getFreqAllXCl = True
getKinClNmerMapping = False
getAllPotentialSnips = False

NmerUnique = True

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sSnipCalcRF = 'LSV'

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------
# dClasses = {'AGC': 'X_AGC',
#             'CDK': 'X_CDK',
#             'CDPK': 'X_CDPK',
#             'CK_II': 'X_CK_II',
#             'LRR_1': 'X_LRR',
#             'LRR_10': 'X_LRR',
#             'LRR_11': 'X_LRR',
#             'LRR_12': 'X_LRR',
#             'LRR_2': 'X_LRR',
#             'LRR_3': 'X_LRR',
#             'LRR_6B': 'X_LRR',
#             'LRR_7': 'X_LRR',
#             'LRR_7A': 'X_LRR',
#             'LRR_8A': 'X_LRR',
#             'LRR_8B': 'X_LRR',
#             'LRR_8C': 'X_LRR',
#             'LRR_9': 'X_LRR',
#             'LRR_9A': 'X_LRR',
#             'MAP2K': 'X_MAPNK',
#             'MAP3K': 'X_MAPNK',
#             'MAPK': 'X_MAPK',
#             'RLCK_2': 'X_RLCK',
#             'RLCK_6': 'X_RLCK',
#             'RLCK_7': 'X_RLCK',
#             'SLK': 'X_SLK',
#             'SnRK1': 'X_SnRK',
#             'SnRK2': 'X_SnRK',
#             'SnRK3': 'X_SnRK',
#             'soluble': 'X_soluble'}

# === assertions ==============================================================

# --- Helper functions handling strings ---------------------------------------
def joinS(itS, sJoin=S_USC):
    lSJoin = [str(s) for s in itS if s is not None and len(str(s)) > 0]
    return sJoin.join(lSJoin).strip()

# === derived values and input processing =====================================
pInp = P_PROC_I_N_MER
sFBase = S_F_PROC_I_N_MER
sFBaseUnqC = S_USC.join([S_F_PROC_I_N_MER, S_UNIQUE_COL])
sFBaseUnqCSPepP = S_USC.join([S_F_PROC_I_N_MER, S_UNIQUE_CODE_SEQ, S_PEP_P])
sFBaseUnqCSNmer = S_USC.join([S_F_PROC_I_N_MER, S_UNIQUE_CODE_SEQ, S_N_MER])
sFBaseRFSnip = S_USC.join([S_F_PROC_I_N_MER, S_REL_FREQ_SNIP])
if inpTbl == 'CombS':
    pInp = P_COMB_RES
    sFBase = S_F_RES_COMB_S
    sFBaseUnqC = S_USC.join([S_F_RES_COMB_S, S_UNIQUE_COL])
    sFBaseUnqCSPepP = S_USC.join([S_F_RES_COMB_S, S_UNIQUE_CODE_SEQ, S_PEP_P])
    sFBaseUnqCSNmer = S_USC.join([S_F_RES_COMB_S, S_UNIQUE_CODE_SEQ, S_N_MER])
    sFBaseRFSnip = S_USC.join([S_F_RES_COMB_S, S_REL_FREQ_SNIP])
sFInpDClf = S_F_INP_CLF_COMB_XS
sFInpDClMap = S_F_INP_D_CLASSES
sFOutRFEffFam = S_F_REL_FREQ_EFF_FAM
sFOutRFXCl = S_F_REL_FREQ_X_CL
sFOutMap = S_F_NMER_KIN_FAM_MAP
if NmerUnique:
    sFOutRFEffFam = joinS([S_F_REL_FREQ_EFF_FAM, S_UNIQUE])
    sFOutRFXCl = joinS([S_F_REL_FREQ_X_CL, S_UNIQUE])
    sFOutMap = joinS([S_F_NMER_KIN_FAM_MAP, S_UNIQUE])

pFInpUnqC = os.path.join(pInp, sFBase + XT_CSV)
pFInpRFSnip = os.path.join(P_TEMP_RES, sFBaseUnqCSNmer + XT_CSV)
pFInpDClf = os.path.join(P_INP_CLF, sFInpDClf + XT_CSV)
pFInpDClMap = os.path.join(P_INP_CLF, sFInpDClMap + XT_CSV)
pFOutUnqC = os.path.join(P_TEMP_RES, sFBaseUnqC + XT_CSV)
pFOutUnqCSPepP = os.path.join(P_TEMP_RES, sFBaseUnqCSPepP + XT_CSV)
pFOutUnqCSNmer = os.path.join(P_TEMP_RES, sFBaseUnqCSNmer + XT_CSV)
pFOutRFSnip = os.path.join(P_TEMP_RES, sFBaseRFSnip + XT_CSV)
pFOutRFEffFam = os.path.join(P_TEMP_RES, sFOutRFEffFam + XT_CSV)
pFOutRFXCl = os.path.join(P_TEMP_RES, sFOutRFXCl + XT_CSV)
pFOutRFMap = os.path.join(P_TEMP_RES, sFOutMap + XT_CSV)

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'inpTbl': inpTbl,
        'doAllUniqueCol': doAllUniqueCol,
        'doUniqueCodeSeqRetPepPos': doUniqueCodeSeqRetPepPos,
        'doUniqueCodeSeqRetNmer': doUniqueCodeSeqRetNmer,
        'doRelFreqSnip': doRelFreqSnip,
        'getFreqAllKinCl': getFreqAllKinCl,
        'getFreqAllXCl': getFreqAllXCl,
        'getKinClNmerMapping': getKinClNmerMapping,
        'getAllPotentialSnips': getAllPotentialSnips,
        'NmerUnique': NmerUnique,
        # --- files, directories and paths ------------------------------------
        'pProcINmer': P_PROC_I_N_MER,
        'pCombRes': P_COMB_RES,
        'sFProcINmer': S_F_PROC_I_N_MER + XT_CSV,
        'sFResCombS': S_F_RES_COMB_S + XT_CSV,
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sCSV': S_CSV,
        'sSnipCalcRF': sSnipCalcRF,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers ---------------------------------------------------------
        'lenNmerDef': LEN_N_MER_DEF,
        'iCentNmer': I_CENT_N_MER,
        # --- lists -----------------------------------------------------------
        # --- dictionaries ----------------------------------------------------
        # 'dClasses': dClasses,
        # === derived values and input processing =============================
        'pFInpUnqC': pFInpUnqC,
        'pFInpRFSnip': pFInpRFSnip,
        'pFInpDClf': pFInpDClf,
        'pFInpDClMap': pFInpDClMap,
        'pFOutUnqC': pFOutUnqC,
        'pFOutUnqCSNmer': pFOutUnqCSNmer,
        'pFOutUnqCSPepP': pFOutUnqCSPepP,
        'pFOutRFSnip': pFOutRFSnip,
        'pFOutRFEffFam': pFOutRFEffFam,
        'pFOutRFXCl': pFOutRFXCl,
        'pFOutRFMap': pFOutRFMap}

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

def saveDfrUniqueColAsCSV(cDfr, pFOut, colUnq=None, reprNA='', cSep=S_SEMICOL):
    if colUnq is None:
        saveAsCSV(cDfr.convert_dtypes(), pF=pFOut, reprNA=reprNA, cSep=cSep)
        print('Saved unmodified DataFrame.')
    elif colUnq == True:
        lSerUnique = []
        for sC in cDfr.columns:
            lSerUnique.append(pd.Series(cDfr[sC].unique(), name=sC))
        cDfrMod = concLObjAx1(lObj=lSerUnique).convert_dtypes()
        saveAsCSV(cDfrMod, pF=pFOut, reprNA=reprNA, cSep=cSep)
        print('Saved DataFrame with all columns converted to unique values.')
    else:
        print('ERROR: Value "', colUnq, '" not implemented for keyword ',
              '"colUnq".', sep='')

def saveDfrUniqueColSpecAsCSV(cDfr, pFOut, colUnq=None, colIRet=None,
                              reprNA='', cSep=S_SEMICOL):
    if colUnq in cDfr.columns:
        if colIRet in cDfr.columns:
            lVUnq, lArr = list(cDfr[colUnq].unique()), []
            for cV in lVUnq:
                cSer = cDfr[cDfr[colUnq] == cV].loc[:, colIRet]
                lArr.append(cSer.unique())
            maxNEl = max([cArr.shape[0] for cArr in lArr])
            for k, cArr in enumerate(lArr):
                lArr[k] = np.append(cArr, [np.nan]*(maxNEl - cArr.shape[0]))
            arrFin = np.stack(lArr, axis=1).T
            lC = [str(colIRet) + S_USC + str(i) for i in range(1, maxNEl + 1)]
            cDfrMod = pd.DataFrame(arrFin, index=lVUnq, columns=lC)
            saveAsCSV(cDfrMod.convert_dtypes(), pF=pFOut, reprNA=reprNA,
                      cSep=cSep)
            print('Saved DataFrame with column "', colUnq,
                  '" converted to unique values and column "', colIRet,
                  '" retained.', sep='')
        else:
            saveAsCSV(cDfrMod[colUnq], pF=pFOut, reprNA=reprNA, cSep=cSep)
            print('Saved Series consisting of column "', colUnq,
                  '" converted to unique values.', sep='')
    else:
        print('ERROR: "', colUnq, '" is not in DataFrame columns: ',
              cDfr.columns.to_list(), '!', sep='')

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

def iniDictRes(idxDfr):
    return {'nFullSeq': len(idxDfr),
            'nFullSeqWOcc': 0,
            'nOccInFullSeq': 0,
            'dNOccInFullSeq': {},
            'nPyl': 0,
            'dPyl': {},
            'dProbPyl': {},
            'probPylSnip': 0.}

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
def getAAcPyl(dInp, sNmer):
    if len(sNmer) == dInp['lenNmerDef']:
        return sNmer[dInp['iCentNmer']]

def getCentralPosOfSnip(dInp, sSnip=S_CAP_S):
    assert (len(sSnip) <= dInp['lenNmerDef'] and len(sSnip)%2 == 1)
    return sSnip[len(sSnip)//2]

def getCentralSnipOfNmer(dInp, sNmer, sSnip=S_CAP_S):
    assert len(sNmer) == dInp['lenNmerDef']
    assert (len(sSnip) <= dInp['lenNmerDef'] and len(sSnip)%2 == 1)
    iS = dInp['iCentNmer'] - len(sSnip)//2
    iE = dInp['iCentNmer'] + len(sSnip)//2 + 1
    return sNmer[iS:iE]

def checkCentralSnipOfNmer(dInp, sNmer, sSnip=S_CAP_S):
    return getCentralSnipOfNmer(dInp, sNmer=sNmer, sSnip=sSnip) == sSnip

# --- Function calculating relative frequency of single snippet ---------------
def calcProbPylData(dInp, cDfr, pFOut, sSnip=S_CAP_S):
    assert (len(sSnip) <= dInp['lenNmerDef'] and len(sSnip)%2 == 1)
    # chCentSnip = getCentralPosOfSnip(dInp, sSnip=sSnip)
    dRes = iniDictRes(cDfr.index)
    # calculate the number of occurrences of the snippet in the full sequences
    for sFS in cDfr.index:
        nOccCS = sFS.count(sSnip)
        dRes['nOccInFullSeq'] += nOccCS
        addToDictCt(dRes['dNOccInFullSeq'], nOccCS)
        if nOccCS > 0:
            dRes['nFullSeqWOcc'] += 1
        for sC in cDfr.columns:
            sCNmer = cDfr.at[sFS, sC]
            if type(sCNmer) == str:
                if checkCentralSnipOfNmer(dInp, sNmer=sCNmer, sSnip=sSnip):
                    dRes['nPyl'] += 1
                    addToDictL(dRes['dPyl'], cK=sFS, cE=sCNmer, lUnqEl=True)
        if sFS in dRes['dPyl']:
            nPyl, nTtl, cPrb = len(dRes['dPyl'][sFS]), nOccCS, 0.
            if nTtl > 0:
                cPrb = nPyl/nTtl
            dRes['dProbPyl'][sFS] = (nPyl, nTtl, cPrb)
            dRes['probPylSnip'] += cPrb
    if (dRes['probPylSnip'] > 0 and dRes['nFullSeqWOcc'] > 0):
        dRes['probPylSnip'] /= dRes['nFullSeqWOcc']
    return dRes

# --- Function calculating relative frequencies of kinase and X classes -------
def getDictDfrTemp(dfrInp):
    dT, serNmerSeq = {}, dfrInp[S_C_N_MER]
    if dInp['NmerUnique']:
        serNmerSeq = iniPdSer(serNmerSeq.unique(), nameS=serNmerSeq.name)
    for cNmerSeq in serNmerSeq:
        dT[cNmerSeq] = dfrInp[dfrInp[S_C_N_MER] == cNmerSeq]
    print('Calculated series of Nmer sequences and temporary dictionary.')
    return dT, serNmerSeq

def getDClasses(printDCl=True):
    dCl, dfrClMap, lXCl = {}, readCSV(pF=dInp['pFInpDClMap'], iCol=0), []
    for _, cRow in dfrClMap.iterrows():
        [sK, sV] = cRow.to_list()
        if sK != S_STAR and sV != S_STAR:
            dCl[sK] = sV
            addToListUnq(lXCl, cEl=sV)
    print('Calculated class dictionary.')
    if printDCl:
        for sK, sV in dCl.items():
            print(sK, S_COLON, S_TAB, sV, sep='')
        print(len(lXCl), 'different X classes. List of X classes:')
        print(lXCl)
    return dCl

def mapEffFamToXCl(lEffFam, dClasses=None):
    lXCl, sNoCl = [], S_STAR
    for sEffFam in lEffFam:
        if sEffFam in dClasses:
            addToList(lXCl, cEl=dClasses[sEffFam],
                      isUnq=dInp['NmerUnique'])
        else:
            addToList(lXCl, cEl=sNoCl, isUnq=dInp['NmerUnique'])
    return lXCl

def calcSaveDfrRelFreq(dLV, dNOcc, sumOcc, doX=False):
    sNmCl, sNOcc, sRelF = S_EFF_FAMILY, S_N_OCC, S_REL_FREQ
    if doX:
        sNmCl = S_X_CL
    for sCCl, nOcc in dNOcc.items():
        addToDictL(dLV, cK=sNmCl, cE=sCCl)
        addToDictL(dLV, cK=sNOcc, cE=nOcc)
        addToDictL(dLV, cK=sRelF, cE=nOcc/sumOcc)
    dfrRelFreq = iniPdDfr(dLV).sort_values(by=[sNmCl, sNOcc],
                                           ascending=[True, False],
                                           ignore_index=True)
    if doX:
        print('Calculated relative frequencies of the X classes.')
        saveAsCSV(dfrRelFreq, pF=dInp['pFOutRFXCl'])
    else:
        print('Calculated relative frequencies of the kinase classes.')
        saveAsCSV(dfrRelFreq, pF=dInp['pFOutRFEffFam'])

def calcRelFreqCl(dT, serNmerSeq, doXCl=False):
    dLV, dNOcc, dClasses = {}, {}, None
    if doXCl:
        dClasses = getDClasses()
    for cDfr in dT.values():
        lCl = cDfr[S_EFF_FAMILY].to_list()
        if dInp['NmerUnique']:
            lCl = list(cDfr[S_EFF_FAMILY].unique())
        if doXCl:
            lCl = mapEffFamToXCl(lCl, dClasses=dClasses)
        for sCl in lCl:
            addToDictCt(dNOcc, cK=sCl)
    sumOcc = sum(dNOcc.values())
    calcSaveDfrRelFreq(dLV, dNOcc, sumOcc, doX=doXCl)

# --- Function calculating lists of kinase classes operating on Nmers ---------
def calcKinClNmerMapping(dT, serNmerSeq):
    dLV, dMap, maxNEffFam = {}, {}, 0
    for sNmerSeq, cDfr in dT.items():
        lEffFam = cDfr[S_EFF_FAMILY].to_list()
        dMap[sNmerSeq] = lEffFam
        if dInp['NmerUnique']:
            dMap[sNmerSeq] = toListUnique(lEffFam)
        maxNEffFam = max(maxNEffFam, len(dMap[sNmerSeq]))
    # dLV = {joinS([S_FAM, str(k)]): [] for k in range(maxNEffFam)}
    for sNmerSeq, lEffFam in dMap.items():
        addToDictL(dLV, cK=S_C_N_MER, cE=sNmerSeq)
        for k, sEffFam in enumerate(lEffFam):
            addToDictL(dLV, cK=joinS([S_FAM, str(k)]), cE=sEffFam)
        for k in range(len(lEffFam), maxNEffFam):
            addToDictL(dLV, cK=joinS([S_FAM, str(k)]), cE=np.nan)
    dfrMapping = iniPdDfr(dLV).sort_values(by=S_C_N_MER, ascending=True,
                                           ignore_index=True)
    print('Calculated lists of kinase classes operating on Nmers.')
    saveAsCSV(dfrMapping, pF=dInp['pFOutRFMap'])

# --- Function printing the results -------------------------------------------
def printRes(dRes, sSnip=S_CAP_S):
    print('Dictionary mapping the number of occurrences in a sequence to the',
          'number of sequences with matching number of occurrences:')
    for nOcc in sorted(dRes['dNOccInFullSeq'], reverse=True):
        print(nOcc, ': ', dRes['dNOccInFullSeq'][nOcc], sep='')
    print('Dictionary mapping the full sequences to the lists of Nmers ',
          'containing the snippet "', sSnip, '" in their centre:', sep='')
    for sFS in sorted(dRes['dPyl']):
        print(sFS[:20], '...: ', dRes['dPyl'][sFS], sep='')
    print('Dictionary mapping the full sequences to the tuple\n',
          '(num. phosphorylations, num. total, est. probability in Nmer) ',
          'for snippet "', sSnip, '":', sep='')
    for sFS in sorted(dRes['dProbPyl']):
        print(sFS[:20], '...: ', dRes['dProbPyl'][sFS], sep='')
    print('Total number of full sequences:', dRes['nFullSeq'])
    print('Number of full sequences with at least one occurrence of "', sSnip,
          '": ', dRes['nFullSeqWOcc'], sep='')
    print('Total number of occurrences of "', sSnip, '" in all full ',
          'sequences: ', dRes['nOccInFullSeq'], sep='')
    print('Total number of phosphorylations of snippet "', sSnip,
          '": ', dRes['nPyl'], sep='')
    relFreqSnip = round(dRes['nPyl']/dRes['nOccInFullSeq'], R08)
    print('Number of phosphorylations divided by number of occurrences of ',
          'snippet "', sSnip, '": ', relFreqSnip, sep='')
    print('Final probability of snippet "', sSnip, '" being phosphorylated: ',
          round(dRes['probPylSnip'], R08), sep='')

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' DataFrameHandling.py ', S_DS28, S_NEWL, sep='')
if (doAllUniqueCol or doUniqueCodeSeqRetPepPos or doUniqueCodeSeqRetNmer):
    dfrInp = readCSV(pF=dInp['pFInpUnqC'], iCol=0)
if doAllUniqueCol:
    saveDfrUniqueColAsCSV(dfrInp, pFOut=dInp['pFOutUnqC'], colUnq=True)
if doUniqueCodeSeqRetPepPos:
    saveDfrUniqueColSpecAsCSV(dfrInp, pFOut=dInp['pFOutUnqCSPepP'],
                              colUnq=S_C_CODE_SEQ, colIRet=S_C_PEP_POS)
if doUniqueCodeSeqRetNmer:
    saveDfrUniqueColSpecAsCSV(dfrInp, pFOut=dInp['pFOutUnqCSNmer'],
                              colUnq=S_C_CODE_SEQ, colIRet=S_C_N_MER)
if doRelFreqSnip:
    dfrInp = readCSV(pF=dInp['pFInpRFSnip'], iCol=0)
    dResult = calcProbPylData(dInp, dfrInp, pFOut=dInp['pFOutRFSnip'],
                              sSnip=dInp['sSnipCalcRF'])
    printRes(dResult, sSnip=dInp['sSnipCalcRF'])

if getFreqAllKinCl or getFreqAllXCl or getKinClNmerMapping:
    dfrInp = readCSV(pF=dInp['pFInpDClf'], iCol=0)
    dTemp, serNmerSeq = getDictDfrTemp(dfrInp)
    if getFreqAllKinCl:
        calcRelFreqCl(dTemp, serNmerSeq)
    if getFreqAllXCl:
        calcRelFreqCl(dTemp, serNmerSeq, doXCl=True)
    if getKinClNmerMapping:
        calcKinClNmerMapping(dTemp, serNmerSeq)
    
if getAllPotentialSnips:
    dfrInp = readCSV(pF=dInp['pFInpUnqC'], iCol=0)
    print('dfrInp:\n', dfrInp, sep='')

print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
