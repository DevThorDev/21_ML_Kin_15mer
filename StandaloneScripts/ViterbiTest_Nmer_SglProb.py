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
COND_D = 'D'
COND_E = 'E'
COND_F = 'F'
COND_G = 'G'
COND_H = 'H'
COND_I = 'I'
COND_K = 'K'
COND_L = 'L'
COND_M = 'M'
COND_N = 'N'
COND_P = 'P'
COND_Q = 'Q'
COND_R = 'R'
COND_S = 'S'
COND_T = 'T'
COND_V = 'V'
COND_W = 'W'
COND_Y = 'Y'

STATE_X = 'X'
STATE_M3 = '-3'
STATE_M2 = '-2'
STATE_M1 = '-1'
STATE_0 = '0'
STATE_P1 = '+1'
STATE_P2 = '+2'
STATE_P3 = '+3'

S_PROB = 'prob'
S_PREV = 'prev'

# --- strings for simplification ----------------------------------------------
A = COND_A
C = COND_C
D = COND_D
E = COND_E
F = COND_F
G = COND_G
H = COND_H
I = COND_I
K = COND_K
L = COND_L
M = COND_M
N = COND_N
P = COND_P
Q = COND_Q
R = COND_R
S = COND_S
T = COND_T
V = COND_V
W = COND_W
Y = COND_Y
L_COND = [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
R08 = 8

# --- sets --------------------------------------------------------------------
setCond = {A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y}

# --- lists -------------------------------------------------------------------
lStates = [STATE_X, STATE_M3, STATE_M2, STATE_M1,
           STATE_0, STATE_P1, STATE_P2, STATE_P3]

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
doViterbi = True

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------

# --- lists -------------------------------------------------------------------
lObs = list('MSGSRRKATPASRTRVGNYEMGRTLGEGSFAKVKYAKNTVTGDQAAIKILDREKVFRHKMVEQLKREISTMKLIKHPNVVEIIEVMASKTKIYIVLELVNGGELFDKIAQQGRLKEDEARRYFQQLI')

# --- dictionaries ------------------------------------------------------------
startPr = {STATE_X: 0.99,
           STATE_M3: 0.01,
           STATE_M2: 0.,
           STATE_M1: 0.,
           STATE_0: 0.,
           STATE_P1: 0.,
           STATE_P2: 0.,
           STATE_P3: 0.}

transPr = {STATE_X: {STATE_X: 0.9,
                     STATE_M3: 0.1,
                     STATE_M2: 0.,
                     STATE_M1: 0.,
                     STATE_0: 0.,
                     STATE_P1: 0.,
                     STATE_P2: 0.,
                     STATE_P3: 0.},
           STATE_M3: {STATE_X: 0.,
                      STATE_M3: 0.,
                      STATE_M2: 1.,
                      STATE_M1: 0.,
                      STATE_0: 0.,
                      STATE_P1: 0.,
                      STATE_P2: 0.,
                      STATE_P3: 0.},
           STATE_M2: {STATE_X: 0.,
                      STATE_M3: 0.,
                      STATE_M2: 0.,
                      STATE_M1: 1.,
                      STATE_0: 0.,
                      STATE_P1: 0.,
                      STATE_P2: 0.,
                      STATE_P3: 0.},
           STATE_M1: {STATE_X: 0.,
                      STATE_M3: 0.,
                      STATE_M2: 0.,
                      STATE_M1: 0.,
                      STATE_0: 1.,
                      STATE_P1: 0.,
                      STATE_P2: 0.,
                      STATE_P3: 0.},
           STATE_0: {STATE_X: 0.,
                     STATE_M3: 0.,
                     STATE_M2: 0.,
                     STATE_M1: 0.,
                     STATE_0: 0.,
                     STATE_P1: 1.,
                     STATE_P2: 0.,
                     STATE_P3: 0.},
           STATE_P1: {STATE_X: 0.,
                      STATE_M3: 0.,
                      STATE_M2: 0.,
                      STATE_M1: 0.,
                      STATE_0: 0.,
                      STATE_P1: 0.,
                      STATE_P2: 1.,
                      STATE_P3: 0.},
           STATE_P2: {STATE_X: 0.,
                      STATE_M3: 0.,
                      STATE_M2: 0.,
                      STATE_M1: 0.,
                      STATE_0: 0.,
                      STATE_P1: 0.,
                      STATE_P2: 0.,
                      STATE_P3: 1.},
           STATE_P3: {STATE_X: 1.,
                      STATE_M3: 0.,
                      STATE_M2: 0.,
                      STATE_M1: 0.,
                      STATE_0: 0.,
                      STATE_P1: 0.,
                      STATE_P2: 0.,
                      STATE_P3: 0.}}

lX = [
0.0677,
0.0148,
0.0557,
0.0691,
0.0392,
0.0691,
0.0224,
0.0505,
0.0615,
0.0935,
0.0243,
0.0460,
0.0513,
0.0378,
0.0559,
0.0877,
0.0507,
0.0647,
0.0113,
0.0268]

lM3 = [
0.0631,
0.0041,
0.0628,
0.0624,
0.0178,
0.0802,
0.0246,
0.0262,
0.0706,
0.0499,
0.0172,
0.0325,
0.0671,
0.0303,
0.1441,
0.1277,
0.0493,
0.0440,
0.0039,
0.0219]

lM2 = [
0.0594,
0.0055,
0.0563,
0.0614,
0.0278,
0.0764,
0.0213,
0.0334,
0.0461,
0.0577,
0.0139,
0.0391,
0.0954,
0.0309,
0.0712,
0.1599,
0.0622,
0.0608,
0.0035,
0.0178]

lM1 = [
0.0620,
0.0072,
0.0592,
0.0673,
0.0219,
0.0958,
0.0188,
0.0301,
0.0583,
0.0708,
0.0227,
0.0409,
0.0743,
0.0276,
0.0768,
0.1322,
0.0561,
0.0567,
0.0033,
0.0178]

l0 = [
0.0002,
0.,
0.0004,
0.0004,
0.,
0.0004,
0.,
0.,
0.,
0.,
0.0002,
0.,
0.0010,
0.,
0.0002,
0.7390,
0.2178,
0.,
0.,
0.0403]

lP1 = [
0.0489,
0.0047,
0.0659,
0.0606,
0.0409,
0.0764,
0.0145,
0.0278,
0.0285,
0.0604,
0.0176,
0.0319,
0.2100,
0.0262,
0.0430,
0.1236,
0.0467,
0.0524,
0.0027,
0.0172]

lP2 = [
0.0631,
0.0072,
0.0751,
0.0786,
0.0162,
0.0780,
0.0182,
0.0311,
0.0633,
0.0483,
0.0106,
0.0411,
0.0888,
0.0231,
0.0766,
0.1517,
0.0549,
0.0561,
0.0049,
0.0131]

lP3 = [
0.0704,
0.0068,
0.0641,
0.0845,
0.0289,
0.0847,
0.0225,
0.0240,
0.0577,
0.0497,
0.0194,
0.0387,
0.0821,
0.0264,
0.0737,
0.1308,
0.0514,
0.0557,
0.0043,
0.0242]

emitPr = {STATE_X: {cC: cP for cC, cP in zip(L_COND, lX)},
          STATE_M3: {cC: cP for cC, cP in zip(L_COND, lM3)},
          STATE_M2: {cC: cP for cC, cP in zip(L_COND, lM2)},
          STATE_M1: {cC: cP for cC, cP in zip(L_COND, lM1)},
          STATE_0: {cC: cP for cC, cP in zip(L_COND, l0)},
          STATE_P1: {cC: cP for cC, cP in zip(L_COND, lP1)},
          STATE_P2: {cC: cP for cC, cP in zip(L_COND, lP2)},
          STATE_P3: {cC: cP for cC, cP in zip(L_COND, lP3)}}

# === assertions ==============================================================
for cObs in lObs:
    assert cObs in setCond

assert sum(startPr.values()) == 1
for dTransPr in transPr.values():
    assert sum(dTransPr.values()) == 1
for dEmitPr in emitPr.values():
    # assert sum(dEmitPr.values()) == 1
    assert sum(dEmitPr.values()) > 0.95 and sum(dEmitPr.values()) < 1.05

# === derived values and input processing =====================================
pFInpViterbi = os.path.join(P_TEMP, S_F_INP_VITERBI + XT_CSV)
pFOutViterbi = os.path.join(P_TEMP, S_F_OUT_VITERBI + XT_CSV)

lIDays = list(range(len(lObs)))

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
        'lIDays': lIDays}

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
    for t in dI['lIDays'][1:]:
        cObs = dI['lObs'][t]
        for st in dI['lStates']:
            maxTransProb = getTransProb(dI, V, t, cSt=st, prevSt=dI['st0'])
            prevStSel = dI['st0']
            for prevSt in dI['lStates'][1:]:
                transProb = getTransProb(dI, V, t, cSt=st, prevSt=prevSt)
                if transProb > maxTransProb:
                    maxTransProb = transProb
                    prevStSel = prevSt
            maxProb = maxTransProb*dI['emitPr'][st][cObs]
            dData = {S_PROB: maxProb, S_PREV: prevStSel}
            addToDictD(V, cKMain=t, cKSub=st, cVSub=dData)

def getMostProbStWBacktrack(dI, V):
    optStPath, maxProb, bestSt, tLast = [], 0., None, dI['lIDays'][-1]
    for st, dData in V[tLast].items():
        if dData[S_PROB] > maxProb:
            maxProb = dData[S_PROB]
            bestSt = st
    optStPath.append(bestSt)
    previousSt = bestSt
    return {'optStPath': optStPath,
            'maxProb': maxProb,
            'previousSt': previousSt}

def followBacktrackTo1stObs(dI, dR, V):
    for t in range(len(V) - 2, -1, -1):
        dR['optStPath'].insert(0, V[t + 1][dR['previousSt']][S_PREV])
        dR['previousSt'] = V[t + 1][dR['previousSt']][S_PREV]

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
