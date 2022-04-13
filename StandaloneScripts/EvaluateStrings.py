# -*- coding: utf-8 -*-
###############################################################################
# --- EvaluateStrings.py ------------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_PROC_I_NMER = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                             '11_ProcInpData')
P_PROB_TBL = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                             '35_ResProb')

S_F_PROC_I_NMER = 'Pho15mer_202202'
S_F_PROB_TBL = 'ProbTable_ProcINmer'

# --- strings -----------------------------------------------------------------
S_DOT = '.'
S_SEMICOL = ';'
S_DASH = '-'
S_EQ = '='
S_NEWL = '\n'

S_DS30 = S_DASH*30
S_EQ80 = S_EQ*80

S_CSV = 'csv'

S_CODE_SEQ = 'code_seq'
S_N_MER_SEQ = 'c15mer'

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
R02 = 2
R04 = 4
R06 = 6

# ### INPUT ###################################################################
# --- numbers -----------------------------------------------------------------
lenNmerDef = 15
iCent = lenNmerDef//2

modDispOcc = 100000
modDispPyl = 5000
modDispProb = 100

# --- lists -------------------------------------------------------------------
lLenNmer = [1, 3, 5, 7]

# list of Nmer sequences to test or None (--> test all)
# lNmerSeq2Test = ['AAAALKGSDHRRTTE', 'SDNSGTESSPRTNGG', 'DNSGTESSPRTNGGS']
lNmerSeq2Test = None

lSCProbTbl = ['sSnip', 'lenSnip', 'nOcc', 'nPyl', 'probSnip']

# --- dictionaries ------------------------------------------------------------

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- files, directories and paths ------------------------------------
        'pProcINmer': P_PROC_I_NMER,
        'pProbTbl': P_PROB_TBL,
        'sFProcINmer': S_F_PROC_I_NMER + XT_CSV,
        'sFProbTbl': S_F_PROB_TBL + XT_CSV,
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sCSV': S_CSV,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers ---------------------------------------------------------
        'lenNmerDef': lenNmerDef,
        'iCent': iCent,
        'modDispOcc': modDispOcc,
        'modDispPyl': modDispPyl,
        'modDispProb': modDispProb,
        # --- lists -----------------------------------------------------------
        'lLenNmer': lLenNmer,
        'lNmerSeq2Test': lNmerSeq2Test,
        'lSCProbTbl': lSCProbTbl,
        # --- dictionaries ----------------------------------------------------
        }

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

# --- Helper functions --------------------------------------------------------

# --- Functions handling strings ----------------------------------------------
def getLSubStrNmer(dI, sNmer):
    assert len(sNmer) == dI['lenNmerDef']
    return [sNmer[(dI['iCent'] - cLen//2):(dI['iCent'] + cLen//2 + 1)]
            for cLen in dI['lLenNmer']]

# --- Functions initialising numpy arrays and pandas DataFrames ---------------
def iniNpArr(data=None, shape=(0, 0), fillV=np.nan):
    if data is None:
        return np.full(shape, fillV)
    else:       # ignore shape
        return np.array(data)

# --- Functions creating and manipulating dictionaries ------------------------
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

def iniDicts(dI, lNmerSeq):
    dOcc, dPyl, dProb = {}, {}, {}
    for sNmer in lNmerSeq:
        for sSubNmer in getLSubStrNmer(dI, sNmer):
            dOcc[sSubNmer] = 0
    for sK in dOcc:
        dPyl[sK] = 0
        dProb[sK] = 0.
    return dOcc, dPyl, dProb

def fillDictOcc(dI, dOcc, lFSeq, nDig=R02):
    N = len(dOcc)*len(lFSeq)
    for i, sSubNmer in enumerate(dOcc):
        for j, sFSeq in enumerate(lFSeq):
            addToDictCt(dOcc, cK=sSubNmer, cIncr=sFSeq.count(sSubNmer))
            n = i*len(lFSeq) + j + 1
            if n%dI['modDispOcc'] == 0:
                print('- fillDictOcc: processed ', n, ' of ', N, ' (',
                      round(n/N*100, nDig), '%)', sep='')

def fillDictPyl(dI, dPyl, lNmerSeq, nDig=R02):
    N = len(dPyl)*len(lNmerSeq)
    for i, sSubNmer in enumerate(dPyl):
        for j, sNmer in enumerate(lNmerSeq):
            if sSubNmer in getLSubStrNmer(dI, sNmer):
                addToDictCt(dPyl, cK=sSubNmer)
            n = i*len(lNmerSeq) + j + 1
            if n%dI['modDispPyl'] == 0:
                print('- fillDictPyl: processed ', n, ' of ', N, ' (',
                      round(n/N*100, nDig), '%)', sep='')

def fillDictProb(dI, dOcc, dPyl, dProb, nDig=R02):
    assert list(dOcc) == list(dPyl)
    for n, sK in enumerate(dOcc):
        if dOcc[sK] > 0:
            dProb[sK] = dPyl[sK]/dOcc[sK]
        if n%dI['modDispProb'] == 0:
            print('- fillDictProb: processed ', n, ' of ', len(dOcc), ' (',
                  round(n/len(dOcc)*100, nDig), '%)', sep='')

# --- Functions saving result DataFrames --------------------------------------
def saveResTbl(dI, dOcc, dPyl, dProb, nDig=R06):
    assert list(dOcc) == list(dPyl) and list(dOcc) == list(dProb)
    dRes = {sC: [] for sC in dI['lSCProbTbl']}
    for sK in dOcc:
        lVal = [sK, len(sK), dOcc[sK], dPyl[sK], dProb[sK]]
        for sC, cVal in zip(dRes, lVal):
            if type(cVal) in [str, int]:
                dRes[sC].append(cVal)
            elif type(cVal) == float:
                dRes[sC].append(round(cVal, nDig))
    dfrResTbl = pd.DataFrame(dRes)
    # dfrResTbl.sort_index(axis=0, key=lambda s: len(s))
    dfrResTbl.sort_values(by=[dI['lSCProbTbl'][1], dI['lSCProbTbl'][-1],
                              dI['lSCProbTbl'][0]],
                          ascending=[True, False, True], axis=0, inplace=True,
                          ignore_index=True)
    saveAsCSV(dfrResTbl, pF=joinToPath(pF=dI['pProbTbl'], nmF=dI['sFProbTbl']))

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' EvaluateStrings.py ', S_DS30, S_NEWL, sep='')
dfrProcINmer = readCSV(pF=joinToPath(pF=dInp['pProcINmer'],
                                     nmF=dInp['sFProcINmer']), iCol=0)
lFullSeqUnique = list(dfrProcINmer[S_CODE_SEQ].unique())
lNmerSeqUnique = list(dfrProcINmer[S_N_MER_SEQ].unique())
if dInp['lNmerSeq2Test'] is None:
    dInp['lNmerSeq2Test'] = lNmerSeqUnique
dictOcc, dictPyl, dictProb = iniDicts(dI=dInp, lNmerSeq=dInp['lNmerSeq2Test'])
fillDictOcc(dI=dInp, dOcc=dictOcc, lFSeq=lFullSeqUnique)
fillDictPyl(dI=dInp, dPyl=dictPyl, lNmerSeq=lNmerSeqUnique)
fillDictProb(dI=dInp, dOcc=dictOcc, dPyl=dictPyl, dProb=dictProb)
# print(dictOcc)
# print(dictPyl)
# print(dictProb)
print(len(dictOcc))
print(len(dictPyl))
print(len(dictProb))
saveResTbl(dI=dInp, dOcc=dictOcc, dPyl=dictPyl, dProb=dictProb)

###############################################################################
