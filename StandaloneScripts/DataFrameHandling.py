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

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
inpTbl = 'CombS'            # 'ProcI' / 'CombS'

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

# === derived values and input processing =====================================
pInp = P_PROC_I_N_MER
sFBase, sFBaseUnqC = S_F_PROC_I_N_MER, S_F_PROC_I_N_MER + S_USC + S_UNIQUE_COL
sFBaseUnqCS = S_F_PROC_I_N_MER + S_USC + S_UNIQUE_CODE_SEQ
if inpTbl == 'CombS':
    pInp = P_COMB_RES
    sFBase, sFBaseUnqC = S_F_RES_COMB_S, S_F_RES_COMB_S + S_USC + S_UNIQUE_COL
    sFBaseUnqCS = S_F_RES_COMB_S + S_USC + S_UNIQUE_CODE_SEQ

pFInp = os.path.join(pInp, sFBase + XT_CSV)
pFOutUnqC = os.path.join(P_TEMP_RES, sFBaseUnqC + XT_CSV)
pFOutUnqCS = os.path.join(P_TEMP_RES, sFBaseUnqCS + XT_CSV)

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
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers ---------------------------------------------------------
        # --- lists -----------------------------------------------------------
        # --- dictionaries ----------------------------------------------------
        # === derived values and input processing =============================
        'pFInp': pFInp,
        'pFOutUnqC': pFOutUnqC,
        'pFOutUnqCS': pFOutUnqCS}

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

def saveDfrUniqueColAsCSV(pdDfr, pF, colUnq=None, reprNA='', cSep=S_SEMICOL):
    if colUnq is None:
        saveAsCSV(pdDfr.convert_dtypes(), pF=pF, reprNA=reprNA, cSep=cSep)
        print('Saved unmodified DataFrame.')
    elif colUnq == True:
        lSerUnique = []
        for sC in pdDfr.columns:
            lSerUnique.append(pd.Series(pdDfr[sC].unique(), name=sC))
        pdDfrMod = concLSerAx1(lSer=lSerUnique).convert_dtypes()
        saveAsCSV(pdDfrMod, pF=pF, reprNA=reprNA, cSep=cSep)
        print('Saved DataFrame with all columns converted to unique values.')
    else:
        print('ERROR: Value "', colUnq, '" not implemented for keyword ',
              '"colUnq".', sep='')

def saveDfrUniqueColSpecAsCSV(pdDfr, pF, colUnq=None, colIRet=None, reprNA='',
                              cSep=S_SEMICOL):
    if colUnq in pdDfr.columns:
        if colIRet in pdDfr.columns:
            lVUnq, lArr = list(pdDfr[colUnq].unique()), []
            for cV in lVUnq:
                cSer = pdDfr[pdDfr[colUnq] == cV].loc[:, colIRet]
                lArr.append(cSer.unique())
            maxNEl = max([cArr.shape[0] for cArr in lArr])
            for k, cArr in enumerate(lArr):
                lArr[k] = np.append(cArr, [np.nan]*(maxNEl - cArr.shape[0]))
            arrFin = np.stack(lArr, axis=1).T
            lC = [str(colIRet) + S_USC + str(i) for i in range(1, maxNEl + 1)]
            pdDfrMod = pd.DataFrame(arrFin, index=lVUnq, columns=lC)
            saveAsCSV(pdDfrMod.convert_dtypes(), pF=pF, reprNA=reprNA,
                      cSep=cSep)
            print('Saved DataFrame with column "', colUnq,
                  '" converted to unique values and column "', colIRet,
                  '" retained.', sep='')
        else:
            saveAsCSV(pdDfrMod[colUnq], pF=pF, reprNA=reprNA, cSep=cSep)
            print('Saved Series consisting of column "', colUnq,
                  '" converted to unique values.', sep='')
    else:
        print('ERROR: "', colUnq, '" is not in DataFrame columns: ',
              pdDfr.columns.to_list(), '!', sep='')

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

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' DataFrameHandling.py ', S_DS28, S_NEWL, sep='')
dfrInp = readCSV(pF=dInp['pFInp'], iCol=0)
saveDfrUniqueColAsCSV(dfrInp, pF=dInp['pFOutUnqC'], colUnq=True)
# saveDfrUniqueColSpecAsCSV(dfrInp, pF=dInp['pFOutUnqCS'], colUnq=S_C_CODE_SEQ,
#                           colIRet=S_C_PEP_POS)
saveDfrUniqueColSpecAsCSV(dfrInp, pF=dInp['pFOutUnqCS'], colUnq=S_C_CODE_SEQ,
                          colIRet=S_C_N_MER)
print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
