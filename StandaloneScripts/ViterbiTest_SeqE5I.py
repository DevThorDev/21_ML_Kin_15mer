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

# --- strings for simplification ----------------------------------------------
A = COND_A
C = COND_C
G = COND_G
T = COND_T

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
R08 = 8

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
lObs = [C, T, T, C, A, T, G, T, G, A, A, A, G, C, A, G, A, C, G, T, A, A, G, T,
        C, A, COND_X]

# --- dictionaries ------------------------------------------------------------
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
for cObs in lObs:
    assert cObs in setCond

assert sum(startPr.values()) == 1
for dTransPr in transPr.values():
    assert sum(dTransPr.values()) == 1
for dEmitPr in emitPr.values():
    assert sum(dEmitPr.values()) == 1

# === derived values and input processing =====================================
pFInpViterbi = os.path.join(P_TEMP, S_F_INP_VITERBI + XT_CSV)
pFOutViterbi = os.path.join(P_TEMP, S_F_OUT_VITERBI + XT_CSV)

lIPos = list(range(len(lObs)))

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
        'lObs': lObs,
        # --- dictionaries ----------------------------------------------------
        'startPr': startPr,
        'transPr': transPr,
        'emitPr': emitPr,
        # === derived values and input processing =============================
        'pFInpViterbi': pFInpViterbi,
        'pFOutViterbi': pFOutViterbi,
        'lIPos': lIPos}

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

# --- Viterbi algorithm related functions -------------------------------------
def iniV(dI):
    assert len(dI['lObs']) > 0 and len(dI['lStates']) > 0
    V, t = {}, 0
    cObs = dI['lObs'][t]
    for st in dI['lStates']:
        dData = {S_PROB: dI['startPr'][st]*dI['emitPr'][st][cObs],
                 S_PREV: None}
        addToDictD(V, cKMain=t, cKSub=st, cVSub=dData)
    return V

def getTransProb(dI, V, t, cSt, prevSt):
    return V[t - 1][prevSt][S_PROB]*dI['transPr'][prevSt][cSt]

def fillV(dI, V):
    for i in dI['lIPos'][1:]:
        cObs = dI['lObs'][i]
        for st in dI['lStates']:
            maxTransProb = getTransProb(dI, V, i, cSt=st, prevSt=dI['st0'])
            prevStSel = dI['st0']
            for prevSt in dI['lStates'][1:]:
                transProb = getTransProb(dI, V, i, cSt=st, prevSt=prevSt)
                if transProb > maxTransProb:
                    maxTransProb = transProb
                    prevStSel = prevSt
            maxProb = maxTransProb*dI['emitPr'][st][cObs]
            dData = {S_PROB: maxProb, S_PREV: prevStSel}
            addToDictD(V, cKMain=i, cKSub=st, cVSub=dData)

def getMostProbStWBacktrack(dI, V):
    optStPath, maxProb, bestSt, iLast = [], 0., None, dI['lIPos'][-1]
    for st, dData in V[iLast].items():
        if dData[S_PROB] > maxProb:
            maxProb = dData[S_PROB]
            bestSt = st
    optStPath.append(bestSt)
    return {'optStPath': optStPath,
            'maxProb': maxProb,
            'previousSt': bestSt}

def followBacktrackTo1stObs(dI, dR, V):
    for t in range(len(V) - 2, -1, -1):
        previousSt = V[t + 1][dR['previousSt']][S_PREV]
        dR['optStPath'].insert(0, previousSt)
        dR['previousSt'] = previousSt

def ViterbiCore(dI):
    V = iniV(dI)
    # run ViterbiCore for t > 0
    fillV(dI, V)
    # Get most probable state and its backtrack
    dR = getMostProbStWBacktrack(dI, V)
    # Follow the backtrack till the first observation
    followBacktrackTo1stObs(dI, dR, V)
    return dR, V

# --- Function printing the results -------------------------------------------
def printRes(dI, dR, V):
    for t, dSt in V.items():
        print(S_ST08, 'Time', t)
        for st, dProbPrev in dSt.items():
            print(S_EQ08, 'State', st)
            print(S_DS08, 'Prob.', round(dProbPrev[S_PROB], dI['R08']),
                  S_VLINE, 'Previous state selected:', dProbPrev[S_PREV])
    print('Optimal state path:', dR['optStPath'])
    # print('Maximal probability:', np.exp(dR['maxProb']))
    # print('ln(maximal probability):', dR['maxProb'])
    print('Maximal probability:', dR['maxProb'])
    print('ln(maximal probability):', np.log(dR['maxProb']))
    # print('Previous state:', dR['previousSt'])

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' ViterbiTest.py ', S_DS34, S_NEWL, sep='')
if doViterbi:
    # dfrInp = readCSV(pF=dInp['pFInpViterbi'], iCol=0)
    dRes, V = ViterbiCore(dI=dInp)
    printRes(dI=dInp, dR=dRes, V=V)
print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
