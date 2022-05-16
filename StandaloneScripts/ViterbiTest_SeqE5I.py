# -*- coding: utf-8 -*-
###############################################################################
# --- ViterbiTest.py ----------------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_TEMP = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer', '98_TEMP')

S_F_INP_VITERBI = ''
S_F_OUT_VITERBI = ''

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
S_VLINE = '|'
S_NEWL = '\n'

S_CSV = 'csv'

S_DS08 = S_DASH*8
S_EQ08 = S_EQ*8
S_ST08 = S_STAR*8
S_DS30 = S_DASH*30
S_DS34 = S_DASH*34
S_DS44 = S_DASH*44
S_DS80 = S_DASH*80
S_EQ80 = S_EQ*80

COND_A = 'A'
COND_C = 'C'
COND_G = 'G'
COND_T = 'T'
COND_X = 'X'

STATE_E = 'E'
STATE_5 = '5'
STATE_I = 'I'
STATE_X = 'X'

S_PROB = 'prob'
S_PREV = 'prev'
S_NONE = 'None'

# --- strings for simplification ----------------------------------------------
A = COND_A
C = COND_C
G = COND_G
T = COND_T

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
R08 = 8
DELTA = 1.0E-14

# --- sets --------------------------------------------------------------------
setCond = {A, C, G, T, COND_X}

# --- lists -------------------------------------------------------------------
lStates = [STATE_E, STATE_5, STATE_I, STATE_X]

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
doViterbi = True

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------

# --- lists -------------------------------------------------------------------
lO01 = [C, T, T, C, A, T, G, T, G, A, A, A, G, C, A, G, A, C, G, T, A, A, G, T,
        C, A, COND_X]

# --- dictionaries ------------------------------------------------------------
dObs = {1: lO01}

startPr = {STATE_E: 1.,
           STATE_5: 0.,
           STATE_I: 0.,
           STATE_X: 0.}

transPr = {STATE_E: {STATE_E: 0.9,
                     STATE_5: 0.1,
                     STATE_I: 0.,
                     STATE_X: 0.},
           STATE_5: {STATE_E: 0.,
                     STATE_5: 0.,
                     STATE_I: 1.,
                     STATE_X: 0.},
           STATE_I: {STATE_E: 0.,
                     STATE_5: 0.,
                     STATE_I: 0.9,
                     STATE_X: 0.1},
           STATE_X: {STATE_E: 0.,
                     STATE_5: 0.,
                     STATE_I: 0.,
                     STATE_X: 1.}}

emitPr = {STATE_E: {A: 0.25, C: 0.25, G: 0.25, T: 0.25, COND_X: 0.},
          STATE_5: {A: 0.05, C: 0., G: 0.95, T: 0., COND_X: 0.},
          STATE_I: {A: 0.4, C: 0.1, G: 0.1, T: 0.4, COND_X: 0.},
          STATE_X: {A: 0., C: 0., G: 0., T: 0., COND_X: 1.}}

# === assertions ==============================================================
for lObs in dObs.values():
    for cObs in lObs:
        assert cObs in setCond

assert sum(startPr.values()) == 1
for dTransPr in transPr.values():
    assert sum(dTransPr.values()) == 1
for dEmitPr in emitPr.values():
    assert (sum(dEmitPr.values()) > 1. - DELTA and
            sum(dEmitPr.values()) < 1. + DELTA)

# === derived values and input processing =====================================
pFInpViterbi = os.path.join(P_TEMP, S_F_INP_VITERBI + XT_CSV)
pFOutViterbi = os.path.join(P_TEMP, S_F_OUT_VITERBI + XT_CSV)

dIPos = {k: list(range(len(lObs))) for k, lObs in dObs.items()}

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'doViterbi': doViterbi,
        # --- files, directories and paths ------------------------------------
        'pTemp': P_TEMP,
        'sFInpViterbi': S_F_INP_VITERBI + XT_CSV,
        'sFOutViterbi': S_F_OUT_VITERBI + XT_CSV,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers
        'R08': R08,
        'DELTA': DELTA,
        # --- sets
        'setCond': setCond,
        # --- lists
        'lStates': lStates,
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sCSV': S_CSV,
        'st0': lStates[0],
        # --- numbers ---------------------------------------------------------
        # --- lists -----------------------------------------------------------
        # --- dictionaries ----------------------------------------------------
        'dObs': dObs,
        'startPr': startPr,
        'transPr': transPr,
        'emitPr': emitPr,
        # === derived values and input processing =============================
        'pFInpViterbi': pFInpViterbi,
        'pFOutViterbi': pFOutViterbi,
        'dIPos': dIPos}

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

# --- Helper function for the Viterbi algorithm handling ">" ------------------
def X_greater(x=None, y=None):
    if x is None and y is None:
        return False
    elif x is None and y is not None:
        return False
    elif x is not None and y is None:
        return True
    else:
        if x > y:
            return True
        else:
            return False

# --- Viterbi algorithm related functions -------------------------------------
def iniV(dI, lObs=[]):
    assert len(lObs) > 0 and len(dI['lStates']) > 0
    V, i = {}, 0
    for st in dI['lStates']:
        dData = {S_PROB: dI['startPr'][st]*dI['emitPr'][st][lObs[i]],
                 S_PREV: None}
        addToDictD(V, cKMain=i, cKSub=st, cVSub=dData)
    return V

def getTransProb(dI, V, i, cSt, prevSt):
    return V[i - 1][prevSt][S_PROB]*dI['transPr'][prevSt][cSt]

def fillV(dI, V, iLObs=0, lObs=[]):
    for i in dI['dIPos'][iLObs][1:]:
        for st in dI['lStates']:
            maxTransProb = getTransProb(dI, V, i, cSt=st, prevSt=dI['st0'])
            prevStSel = dI['st0']
            for prevSt in dI['lStates'][1:]:
                transProb = getTransProb(dI, V, i, cSt=st, prevSt=prevSt)
                if X_greater(transProb, maxTransProb):
                    maxTransProb = transProb
                    prevStSel = prevSt
            maxProb = maxTransProb*dI['emitPr'][st][lObs[i]]
            dData = {S_PROB: maxProb, S_PREV: prevStSel}
            addToDictD(V, cKMain=i, cKSub=st, cVSub=dData)

def getMostProbStWBacktrack(dI, V, iLObs=0):
    optStPath, maxProb, bestSt, iLast = [], None, None, dI['dIPos'][iLObs][-1]
    for st, dData in V[iLast].items():
        if X_greater(dData[S_PROB], maxProb):
            maxProb = dData[S_PROB]
            bestSt = st
    optStPath.append(bestSt)
    return {'optStPath': optStPath,
            'maxProb': maxProb,
            'previousSt': bestSt}

def followBacktrackTo1stObs(dI, dR, V):
    for i in range(len(V) - 2, -1, -1):
        previousSt = V[i + 1][dR['previousSt']][S_PREV]
        dR['optStPath'].insert(0, previousSt)
        dR['previousSt'] = previousSt

def ViterbiCore(dI, iLObs=0, lObs=[]):
    V = iniV(dI, lObs=lObs)
    # run ViterbiCore for t > 0
    fillV(dI, V, iLObs=iLObs, lObs=lObs)
    # Get most probable state and its backtrack
    dR = getMostProbStWBacktrack(dI, V, iLObs=iLObs)
    # Follow the backtrack till the first observation
    followBacktrackTo1stObs(dI, dR, V)
    return dR, V

# --- Function printing the results -------------------------------------------
def printRes(dI, dR, V):
    for i, dSt in V.items():
        print(S_ST08, 'Position (observation) index', i)
        for st, dProbPrev in dSt.items():
            print(S_EQ08, 'State', st)
            cProb = S_NONE
            if dProbPrev[S_PROB] is not None:
                cProb = round(dProbPrev[S_PROB], dI['R08'])
            print(S_DS08, 'Prob.', cProb,
                  S_VLINE, 'Previous state selected:', dProbPrev[S_PREV])
    print('Optimal state path:', dR['optStPath'])
    print('Maximal probability:', dR['maxProb'])
    print('Maximal ln(probability):', np.log(dR['maxProb']))

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' ViterbiTest.py ', S_DS34, S_NEWL, sep='')
if doViterbi:
    # dfrInp = readCSV(pF=dInp['pFInpViterbi'], iCol=0)
    for iLObs, lObs in dObs.items():
        print(S_DS80, S_NEWL, S_EQ08, ' Observations ', iLObs, S_COLON, sep='')
        dRes, V = ViterbiCore(dI=dInp, iLObs=iLObs, lObs=lObs)
        printRes(dI=dInp, dR=dRes, V=V)
print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
