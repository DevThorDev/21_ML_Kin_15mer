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

S_TMPL_TO_RES = 'TmplToRes'
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
S_DIR_SCRIPTS_L1_OBJINP = 'ObjInput'
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
P_DIR_TMPL_SCRIPTS_L1_CORE = os.path.join(S_DIR_TEMPLATE, S_DIR_SCRIPTS_M,
                                          S_DIR_SCRIPTS_L1_CORE)
P_DIR_TMPL_RESULTS_L1_CLF = os.path.join(S_DIR_TEMPLATE, S_DIR_RESULTS_M,
                                         S_DIR_RESULTS_L1_CLF)
P_DIR_EVAL_RESULTS_L1_CLF = os.path.join(S_DIR_EVALUATION, S_DIR_RESULTS_M,
                                         S_DIR_RESULTS_L1_CLF)

SET_S_ST_DIR_RES_L1 = {'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH',
                       'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH',
                       'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG'}

SET_S_SUB_DIR_RES = {S_DIR_RESULTS_L2_PARS, S_DIR_RESULTS_L2_SMRS,
                     S_DIR_RESULTS_L2_UNQN, S_DIR_RESULTS_L2_INPD,
                     S_DIR_RESULTS_L2_CNFM, S_DIR_RESULTS_L2_DTLD,
                     S_DIR_RESULTS_L2_PROP, S_DIR_RESULTS_L2_EVAL}

# === INPUT ===================================================================
# --- strings -----------------------------------------------------------------
sModeCopy = S_RES_TO_EVAL   # S_TMPL_TO_RES: copy from tmpl. to res. dir(s)
                            # S_RES_TO_EVAL: copy from res. to eval. dir
xtF2Copy = XT_PY            # extension of the file(s) to copy

# --- sets --------------------------------------------------------------------
setSStDirResL1 = SET_S_ST_DIR_RES_L1
setSStDirResL2 = {S_DIR_SCRIPTS_M}
setSStDirResL3 = {S_DIR_SCRIPTS_L1_CORE}

setSSubDirRes = SET_S_SUB_DIR_RES

# --- tuples ------------------------------------------------------------------
# tTmplEval = (P_DIR_TMPL_RESULTS_L1_CLF, P_DIR_EVAL_RESULTS_L1_CLF)

# --- dictionaries ------------------------------------------------------------
dSStF2Copy = {P_DIR_TMPL_SCRIPTS_L1_CORE: {'F_00', 'O_07'}}
# dSStD2Copy = {tTmplEval: setSSubDirRes}

# === INPUT DICTIONARY ========================================================
dInp = {# --- strings ---------------------------------------------------------
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
        'sDirScrL1ObjI': S_DIR_SCRIPTS_L1_OBJINP,
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
        'pDirTmplScrL1Core': P_DIR_TMPL_SCRIPTS_L1_CORE,
        'pDirTmplResL1Clf': P_DIR_TMPL_RESULTS_L1_CLF,
        'pDirEvalResL1Clf': P_DIR_EVAL_RESULTS_L1_CLF,
        # --- strings ---------------------------------------------------------
        'sModeCopy': sModeCopy,
        'xtF2Copy': xtF2Copy,
        # --- sets ------------------------------------------------------------
        'setSStDirResL1': setSStDirResL1,
        'setSStDirResL2': setSStDirResL2,
        'setSStDirResL3': setSStDirResL3,
        'setSSubDirRes': setSSubDirRes,
        # --- tuples ----------------------------------------------------------
        # 'tTmplEval': tTmplEval,
        # --- dictionaries ----------------------------------------------------
        'dSStF2Copy': dSStF2Copy
        # ,
        # 'dSStD2Copy': dSStD2Copy
        }

# === FUNCTIONS ===============================================================
def addToLP(dI, lP, pM, setSSt={}, sEnd=None):
    for s in os.listdir(pM):
        for sSt in setSSt:
            if (sEnd is None and s.startswith(sSt) or
                sEnd is not None and s.startswith(sSt) and s.endswith(sEnd)):
                lP.append(os.path.join(pM, s))
        if len(setSSt) == 0:
            if (sEnd is None) or (sEnd is not None and s.endswith(sEnd)):
                lP.append(os.path.join(pM, s))

def getLPF2Copy(dI):
    lPF2Copy, pDRoot, xtF = [], dI['pDirRoot'], dI['xtF2Copy']
    for pF2Copy, setSStF2Copy in dI['dSStF2Copy'].items():
        pFF = os.path.join(pDRoot, pF2Copy)
        addToLP(dI=dI, lP=lPF2Copy, pM=pFF, setSSt=setSStF2Copy, sEnd=xtF)
    return lPF2Copy

def getLPDDst(dI):
    lPDDstL1, lPDDstL2, lPDDstL3, pDRoot = [], [], [], dI['pDirRoot']
    addToLP(dI=dI, lP=lPDDstL1, pM=pDRoot, setSSt=dI['setSStDirResL1'])
    for pD in lPDDstL1:
        addToLP(dI=dI, lP=lPDDstL2, pM=pD, setSSt=dI['setSStDirResL2'])
    for pD in lPDDstL2:
        addToLP(dI=dI, lP=lPDDstL3, pM=pD, setSSt=dI['setSStDirResL3'])
    return lPDDstL3

def getLSourcesLDestinations(dI):
    return {'lSources': getLPF2Copy(dI=dI), 'lDestinations': getLPDDst(dI=dI)}

def copyFromTemplateToResultDirs(dI, dR):
    for cSrc in dR['lSources']:
        for cDst in dR['lDestinations']:
            if os.path.isfile(cSrc) and os.path.isdir(cDst):
                sFDst = os.path.join(cDst, cSrc.split(dI['sDblBS'])[-1])
                shutil.copyfile(cSrc, sFDst)
                print('Copied file "', cSrc, '" to\n\t"', sFDst, '".', sep='')

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
                            print('Copied file\n\t"', pFSrc, '" to\n\t"',
                                  pFDst, '".\n', sep='')

# === MAIN ====================================================================
if dInp['sModeCopy'] == S_TMPL_TO_RES:
    dRes = getLSourcesLDestinations(dI=dInp)
    copyFromTemplateToResultDirs(dI=dInp, dR=dRes)
elif dInp['sModeCopy'] == S_RES_TO_EVAL:
    copyFromResultToEvaluationDirs(dI=dInp)

# =============================================================================
