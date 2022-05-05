# -*- coding: utf-8 -*-
###############################################################################
# --- DataFrameHandling.py ----------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_PROC_I_N_MER = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                              '11_ProcInpData')
P_COMB_RES = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                          '31_ResCombined')
P_TEMP_RES = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                          '98_TEMP')

S_F_PROC_I_N_MER = 'Pho15mer_202202'
S_F_RES_COMB_S = 'Combined_S_KinasesPho15mer_202202'

# --- strings -----------------------------------------------------------------
S_SPACE = ' '
S_DOT = '.'
S_SEMICOL = ';'
S_DASH = '-'
S_PLUS = '+'
S_EQ = '='
S_STAR = '*'
S_USC = '_'
S_NEWL = '\n'
S_S = 'S'

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

S_PROC_I_N_MER = 'ProcINmer'
S_COMB_S = 'Combined_S'
S_UNIQUE_COL = 'UniqueCol'
S_UNIQUE_CODE_SEQ = 'UniqueCodeSeq'
S_PEP_P = 'PepPos'
S_N_MER = 'Nmer'
S_REL_FREQ_SNIP = 'RelFreqSnip'

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
doRelFreqSnip = True

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sSnipCalcRF = 'LSV'

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

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

pFInpUnqC = os.path.join(pInp, sFBase + XT_CSV)
pFInpRFSnip = os.path.join(P_TEMP_RES, sFBaseUnqCSNmer + XT_CSV)
pFOutUnqC = os.path.join(P_TEMP_RES, sFBaseUnqC + XT_CSV)
pFOutUnqCSPepP = os.path.join(P_TEMP_RES, sFBaseUnqCSPepP + XT_CSV)
pFOutUnqCSNmer = os.path.join(P_TEMP_RES, sFBaseUnqCSNmer + XT_CSV)
pFOutRFSnip = os.path.join(P_TEMP_RES, sFBaseRFSnip + XT_CSV)

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'inpTbl': inpTbl,
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
        # === derived values and input processing =============================
        'pFInpUnqC': pFInpUnqC,
        'pFInpRFSnip': pFInpRFSnip,
        'pFOutUnqC': pFOutUnqC,
        'pFOutUnqCSNmer': pFOutUnqCSNmer,
        'pFOutUnqCSPepP': pFOutUnqCSPepP,
        'pFOutRFSnip': pFOutRFSnip}

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
        cDfrMod = concLSerAx1(lSer=lSerUnique).convert_dtypes()
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

# --- Functions handling dictionaries -----------------------------------------
def addToDictCt(cD, cK, cIncr=1):
    if cK in cD:
        cD[cK] += cIncr
    else:
        cD[cK] = cIncr

def addToDictL(cD, cK, cE, lUniqEl=False):
    if cK in cD:
        if not lUniqEl or cE not in cD[cK]:
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

# --- Functions initialising pandas DataFrames --------------------------------
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
def concLSer(lSer, concAx=0, ignIdx=False, verifInt=False, srtDfr=False):
    return pd.concat(lSer, axis=concAx, ignore_index=ignIdx,
                     verify_integrity=verifInt, sort=srtDfr)

def concLSerAx0(lSer, ignIdx=False, verifInt=False, srtDfr=False):
    return concLSer(lSer, ignIdx=ignIdx, verifInt=verifInt, srtDfr=srtDfr)

def concLSerAx1(lSer, ignIdx=False, verifInt=False, srtDfr=False):
    return concLSer(lSer, concAx=1, ignIdx=ignIdx, verifInt=verifInt,
                    srtDfr=srtDfr)

# --- General-purpose functions -----------------------------------------------
def getAAcPyl(dInp, sNmer):
    if len(sNmer) == dInp['lenNmerDef']:
        return sNmer[dInp['iCentNmer']]

def getCentralPosOfSnip(dInp, sSnip=S_S):
    assert (len(sSnip) <= dInp['lenNmerDef'] and len(sSnip)%2 == 1)
    return sSnip[len(sSnip)//2]

def getCentralSnipOfNmer(dInp, sNmer, sSnip=S_S):
    assert len(sNmer) == dInp['lenNmerDef']
    assert (len(sSnip) <= dInp['lenNmerDef'] and len(sSnip)%2 == 1)
    iS = dInp['iCentNmer'] - len(sSnip)//2
    iE = dInp['iCentNmer'] + len(sSnip)//2 + 1
    return sNmer[iS:iE]

def checkCentralSnipOfNmer(dInp, sNmer, sSnip=S_S):
    return getCentralSnipOfNmer(dInp, sNmer=sNmer, sSnip=sSnip) == sSnip

# --- Function calculating relative frequency of single snippet ---------------
def calcProbPylData(dInp, cDfr, pFOut, sSnip=S_S):
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
                    addToDictL(dRes['dPyl'], cK=sFS, cE=sCNmer, lUniqEl=True)
        if sFS in dRes['dPyl']:
            nPyl, nTtl, cPrb = len(dRes['dPyl'][sFS]), nOccCS, 0.
            if nTtl > 0:
                cPrb = nPyl/nTtl
            dRes['dProbPyl'][sFS] = (nPyl, nTtl, cPrb)
            dRes['probPylSnip'] += cPrb
    if (dRes['probPylSnip'] > 0 and dRes['nFullSeqWOcc'] > 0):
        dRes['probPylSnip'] /= dRes['nFullSeqWOcc']
    return dRes

# --- Function printing the results -------------------------------------------
def printRes(dRes, sSnip=S_S):
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
print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
