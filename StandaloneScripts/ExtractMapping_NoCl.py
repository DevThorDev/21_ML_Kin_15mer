# -*- coding: utf-8 -*-
###############################################################################
# --- ExtractMapping_NoCl.py --------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

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

# --- file name extensions ----------------------------------------------------
XT_PY = S_DOT + S_EXT_PY
XT_CSV = S_DOT + S_EXT_CSV

# --- directories and paths ---------------------------------------------------
S_DIR_RESULTS_L0 = '13_Sysbio03_Phospho15mer'
S_DIR_RESULTS_L1_CLF = '39_ResClf'
S_DIR_RESULTS_L2_PARS = '00_Pars'
S_DIR_RESULTS_L2_SMRS = '01_Summaries'
S_DIR_RESULTS_L2_UNQN = '11_UniqueNmer'
S_DIR_RESULTS_L2_INPD = '12_InpDataClfPrC'
S_DIR_RESULTS_L2_CNFM = '21_CnfMat'
S_DIR_RESULTS_L2_DTLD = '31_Detailed'
S_DIR_RESULTS_L2_PROP = '41_Prop'
S_DIR_RESULTS_L2_EVAL = '90_Eval'

P_DIR_ROOT = os.path.join('..', '..', '..')
P_DIR_INCLF = os.path.join(P_DIR_ROOT, S_DIR_RESULTS_L0, S_DIR_RESULTS_L1_CLF)
P_DIR_INPD = os.path.join(P_DIR_INCLF, S_DIR_RESULTS_L2_INPD)

# === INPUT ===================================================================
# --- boolean values ----------------------------------------------------------
doImbSampling = True
ILblSgl = False
calcCnfMatrix = True

# --- numbers -----------------------------------------------------------------
nItFit = 5

# --- strings -----------------------------------------------------------------
sFInpXM = 'XM__Test'
sFInpYM = 'YM__Test'

sNo, sNone = S_NO, S_NONE
sEffFam, sNoFam, sMultiFam = S_EFF_FAM, S_NO_FAM, S_MULTI_FAM
sOneHot, sOrdinal = S_ONE_HOT, S_ORDINAL
sStratRealMajo, sStratShareMino = S_STRAT_REAL_MAJO, S_STRAT_SHARE_MINO
RFClf, MLPClf = RF_CLF, MLP_CLF

selClf = MLPClf
sStrat = sStratRealMajo

# --- sets --------------------------------------------------------------------

# --- lists -------------------------------------------------------------------
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
        # --- directories and paths -------------------------------------------
        'sDirResL1Clf': S_DIR_RESULTS_L1_CLF,
        'sDirResL2Pars': S_DIR_RESULTS_L2_PARS,
        'sDirResL2Smrs': S_DIR_RESULTS_L2_SMRS,
        'sDirResL2UnqN': S_DIR_RESULTS_L2_UNQN,
        'sDirResL2InpD': S_DIR_RESULTS_L2_INPD,
        'sDirResL2CnfM': S_DIR_RESULTS_L2_CNFM,
        'sDirResL2Dtld': S_DIR_RESULTS_L2_DTLD,
        'sDirResL2Prop': S_DIR_RESULTS_L2_PROP,
        'sDirResL2Eval': S_DIR_RESULTS_L2_EVAL,
        'pDirRoot': P_DIR_ROOT,
        'pDirInClf': P_DIR_INCLF,
        'pDirInpD': P_DIR_INPD,
        # --- strings (2) -----------------------------------------------------
        'sFInpXM': sFInpXM,
        'sFInpYM': sFInpYM,
        'sNo': sNo,
        'sNone': sNone,
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

# --- specific functions ------------------------------------------------------

# --- print functions ---------------------------------------------------------

# === MAIN ====================================================================

# =============================================================================
###############################################################################