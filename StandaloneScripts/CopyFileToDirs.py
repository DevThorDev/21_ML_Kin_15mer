# -*- coding: utf-8 -*-
###############################################################################
# --- CopyFileToDirs.py -------------------------------------------------------
###############################################################################
import os, shutil

# === CONSTANTS ===============================================================
# --- strings -----------------------------------------------------------------
S_DOT, S_USC, S_DBL_BS = '.', '_', '\\'
S_EXT_PY = 'py'
S_EXT_CSV = 'csv'

S_TMPL_TO_EVAL = 'TmplToEval'
S_TMPL_TO_RES = 'TmplToRes'
S_TMPL_TO_EVRES = 'TmplToEvRes'
S_RES_TO_EVAL = 'ResToEval'

# --- file name extensions ----------------------------------------------------
XT_PY = S_DOT + S_EXT_PY
XT_CSV = S_DOT + S_EXT_CSV

# --- directories and paths ---------------------------------------------------
S_DIR_TEMPLATE = '00_Template'
S_DIR_EVALUATION = '90_Evaluation'
S_DIR_SCRIPTS_M = 'A_Scripts'
S_DIR_INPUT_M = 'B_Input'
S_DIR_RESULTS_M = 'C_Results'
S_DIR_SCRIPTS_L1_CONTROL = 'Control'
S_DIR_SCRIPTS_L1_CORE = 'Core'
S_DIR_SCRIPTS_L1_OINP = 'ObjInput'
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
P_DIR_SCRIPTS_L1_CORE = os.path.join(S_DIR_SCRIPTS_M, S_DIR_SCRIPTS_L1_CORE)
P_DIR_SCRIPTS_L1_OINP = os.path.join(S_DIR_SCRIPTS_M, S_DIR_SCRIPTS_L1_OINP)
P_DIR_TMPL_RESULTS_L1_CLF = os.path.join(S_DIR_TEMPLATE, S_DIR_RESULTS_M,
                                         S_DIR_RESULTS_L1_CLF)
P_DIR_EVAL_RESULTS_L1_CLF = os.path.join(S_DIR_EVALUATION, S_DIR_RESULTS_M,
                                         S_DIR_RESULTS_L1_CLF)

# SET_S_ST_DIR_RES_L1 = {'KG', 'KH', 'KI', 'KJ', 'KK', 'KL', 'KM', 'KN', 'KO',
#                        'KP', 'KQ', 'KR', 'KS', 'KT', 'KU', 'KV', 'KW', 'KX'}
SET_S_ST_DIR_RES_L1 = {'XX'}

SET_S_SUB_DIR_RES = {S_DIR_RESULTS_L2_PARS, S_DIR_RESULTS_L2_SMRS,
                     S_DIR_RESULTS_L2_UNQN, S_DIR_RESULTS_L2_INPD,
                     S_DIR_RESULTS_L2_CNFM, S_DIR_RESULTS_L2_DTLD,
                     S_DIR_RESULTS_L2_PROP, S_DIR_RESULTS_L2_EVAL}

# === INPUT ===================================================================
# --- strings -----------------------------------------------------------------
sModeCopy = S_TMPL_TO_EVRES # S_TMPL_TO_EVAL: copy from tmpl. to eval. dir
                            # S_TMPL_TO_RES: copy from tmpl. to res. dir(s)
                            # S_TMPL_TO_EVRES: copy... to eval. AND res. dir(s)
                            # S_RES_TO_EVAL: copy from res. to eval. dir
xtF2Copy = XT_PY            # extension of the file(s) to copy

# --- sets --------------------------------------------------------------------
setSStDirEval = {S_DIR_EVALUATION}
setSStDirResL1 = SET_S_ST_DIR_RES_L1
setSStDirEvalResL1 = setSStDirResL1.union(setSStDirEval)

setSSubDirRes = SET_S_SUB_DIR_RES

# --- lists -------------------------------------------------------------------
lSTxtPrnt = ['Copied file\n*\t"', '" to\n>\t"', '".\n']

# --- dictionaries ------------------------------------------------------------
dSStF2Copy = {P_DIR_SCRIPTS_L1_CORE: {'C_00', 'F_01', 'O_90'},
              P_DIR_SCRIPTS_L1_OINP: {'D_00', 'D_06', 'D_90'}}
# dSStF2Copy = {S_DIR_SCRIPTS_M: {'M_0'},
#               P_DIR_SCRIPTS_L1_CORE: {'C_00', 'F_00', 'O_07', 'O_80', 'O_90'},
#               P_DIR_SCRIPTS_L1_OINP: {'D_00', 'D_01', 'D_06', 'D_07', 'D_80',
#                                       'D_90'}}

# === DERIVED VALUES ==========================================================
setSStDir = setSStDirEval
if sModeCopy == S_TMPL_TO_EVAL:
    setSStDir = setSStDirEval
elif sModeCopy == S_TMPL_TO_RES:
    setSStDir = setSStDirResL1
elif sModeCopy == S_TMPL_TO_EVRES:
    setSStDir = setSStDirEvalResL1

# === INPUT DICTIONARY ========================================================
dInp = {# --- strings (1) -----------------------------------------------------
        'sDot': S_DOT,
        'sUsc': S_USC,
        'sDblBS': S_DBL_BS,
        # --- file name extensions --------------------------------------------
        'xtPY': XT_PY,
        'xtCSV': XT_CSV,
        # --- directories and paths -------------------------------------------
        'sDirTmpl': S_DIR_TEMPLATE,
        'sDirEval': S_DIR_EVALUATION,
        'sDirScrM': S_DIR_SCRIPTS_M,
        'sDirInpM': S_DIR_INPUT_M,
        'sDirResM': S_DIR_RESULTS_M,
        'sDirScrL1Ctr': S_DIR_SCRIPTS_L1_CONTROL,
        'sDirScrL1Core': S_DIR_SCRIPTS_L1_CORE,
        'sDirScrL1ObjI': S_DIR_SCRIPTS_L1_OINP,
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
        'pDirScrL1Core': P_DIR_SCRIPTS_L1_CORE,
        'pDirScrL1OInp': P_DIR_SCRIPTS_L1_OINP,
        'pDirResL1Clf': P_DIR_TMPL_RESULTS_L1_CLF,
        'pDirEvalResL1Clf': P_DIR_EVAL_RESULTS_L1_CLF,
        # --- strings (2) -----------------------------------------------------
        'sModeCopy': sModeCopy,
        'xtF2Copy': xtF2Copy,
        # --- sets ------------------------------------------------------------
        'setSStDir': setSStDir,
        'setSStDirEval': setSStDirEval,
        'setSStDirResL1': setSStDirResL1,
        'setSStDirEvalResL1': setSStDirEvalResL1,
        'setSSubDirRes': setSSubDirRes,
        # --- lists -----------------------------------------------------------
        'lSTxtPrnt': lSTxtPrnt,
        # --- dictionaries ----------------------------------------------------
        'dSStF2Copy': dSStF2Copy}

# === FUNCTIONS ===============================================================
# --- general functions -------------------------------------------------------
def addToDictL(cD, cK, cE, lUnqEl=False):
    if cK in cD:
        if not lUnqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def printAction(lSTxt=[''], lSIns=[], sepTxt=''):
    lTxtP = []
    for k in range(max(len(lSTxt), len(lSIns))):
        if k < len(lSTxt):
            lTxtP.append(lSTxt[k])
        if k < len(lSIns):
            lTxtP.append(lSIns[k])
    print(*lTxtP, sep=sepTxt)

# --- specific functions ------------------------------------------------------
def getLP(dI, pM, setSSt={}, sEnd=None):
    lP = []
    for s in os.listdir(pM):
        for sSt in setSSt:
            if (sEnd is None and s.startswith(sSt) or
                sEnd is not None and s.startswith(sSt) and s.endswith(sEnd)):
                lP.append(os.path.join(pM, s))
        if len(setSSt) == 0:
            if (sEnd is None) or (sEnd is not None and s.endswith(sEnd)):
                lP.append(os.path.join(pM, s))
    return lP

def getDPF2Copy(dI):
    dPF2Copy, pDRoot, xtF = {}, dI['pDirRoot'], dI['xtF2Copy']
    for pDDst in getLP(dI, pM=pDRoot, setSSt=dI['setSStDir']):
        for pD2Copy, setSStF2Copy in dI['dSStF2Copy'].items():
            pDSrcF = os.path.join(pDRoot, dI['sDirTmpl'], pD2Copy)
            pDDstF = os.path.join(pDDst, pD2Copy)
            for pF in getLP(dI, pM=pDSrcF, setSSt=setSStF2Copy, sEnd=xtF):
                addToDictL(dPF2Copy, cK=pDDstF, cE=pF, lUnqEl=True)
    return dPF2Copy

def copyFromTemplateToResultDirs(dI, dPF):
    for pDDst, lPFSrc in dPF.items():
        for pFSrc in lPFSrc:
            if os.path.isfile(pFSrc) and os.path.isdir(pDDst):
                pFDst = os.path.join(pDDst, pFSrc.split(dI['sDblBS'])[-1])
                shutil.copyfile(pFSrc, pFDst)
                printAction(lSTxt=dI['lSTxtPrnt'], lSIns=[pFSrc, pFDst])

def copyFromResultToEvaluationDirs(dI):
    pDDstM = os.path.join(dI['pDirRoot'], dI['sDirEval'], dI['sDirResM'])
    for sDSrc in os.listdir(dI['pDirRoot']):
        for sSt in dI['setSStDirResL1']:
            if sDSrc.startswith(sSt):
                pDSrcM = os.path.join(dI['pDirRoot'], sDSrc, dI['sDirResM'])
                for sD in dI['setSSubDirRes']:
                    pDSrc = os.path.join(pDSrcM, dI['sDirResL1Clf'], sD)
                    pDDst = os.path.join(pDDstM, dI['sDirResL1Clf'], sD)
                    if os.path.isdir(pDSrc) and os.path.isdir(pDDst):
                        for sF in os.listdir(pDSrc):
                            pFSrc = os.path.join(pDSrc, sF)
                            pFDst = os.path.join(pDDst, sF)
                            shutil.copyfile(pFSrc, pFDst)
                            printAction(lSTxt=dI['lSTxtPrnt'],
                                        lSIns=[pFSrc, pFDst])

# === MAIN ====================================================================
if dInp['sModeCopy'] in [S_TMPL_TO_EVAL, S_TMPL_TO_RES, S_TMPL_TO_EVRES]:
    copyFromTemplateToResultDirs(dI=dInp, dPF=getDPF2Copy(dI=dInp))
elif dInp['sModeCopy'] == S_RES_TO_EVAL:
    copyFromResultToEvaluationDirs(dI=dInp)

# =============================================================================