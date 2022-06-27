# -*- coding: utf-8 -*-
###############################################################################
# --- ViterbiTest.py ----------------------------------------------------------
###############################################################################
import os

import pandas as pd

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_TEMP = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                      '98_TEMP_CSV')

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

COND_NORMAL = 'condNormal'
COND_COLD = 'condCold'
COND_DIZZY = 'condDizzy'

STATE_HEALTHY = 'stateHealthy'
STATE_FEVER = 'stateFever'

S_PROB = 'prob'
S_PREV = 'prev'

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
R08 = 8

# --- sets --------------------------------------------------------------------
setCond = {COND_NORMAL, COND_COLD, COND_DIZZY}

# --- lists -------------------------------------------------------------------
lStates = [STATE_HEALTHY, STATE_FEVER]

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
doViterbi = True

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------

# --- lists -------------------------------------------------------------------
lObs = [COND_NORMAL, COND_COLD, COND_DIZZY]

# --- dictionaries ------------------------------------------------------------
startPr = {STATE_HEALTHY: 0.6,
           STATE_FEVER: 0.4}

transPr = {STATE_HEALTHY: {STATE_HEALTHY: 0.7,
                           STATE_FEVER: 0.3},
           STATE_FEVER: {STATE_HEALTHY: 0.4,
                         STATE_FEVER: 0.6}}

emitPr = {STATE_HEALTHY: {COND_NORMAL: 0.5,
                          COND_COLD: 0.4,
                          COND_DIZZY: 0.1},
          STATE_FEVER: {COND_NORMAL: 0.1,
                        COND_COLD: 0.3,
                        COND_DIZZY: 0.6}}

# === assertions ==============================================================
for cObs in lObs:
    assert cObs in setCond

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

def addToDictL(cD, cK, cE, lUnqEl=False):
    if cK in cD:
        if not lUnqEl or cE not in cD[cK]:
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
    print('Optimal state path:', dR['optStPath'])
    print('Maximal probability:', dR['maxProb'])
    # print('Previous state:', dR['previousSt'])
    for t, dSt in V.items():
        print(S_ST08, 'Time', t)
        for st, dProbPrev in dSt.items():
            print(S_EQ08, 'State', st)
            print(S_DS08, 'Prob.', round(dProbPrev[S_PROB], dI['R08']),
                  S_VLINE, 'Previous state selected:', dProbPrev[S_PREV])

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' ViterbiTest.py ', S_DS34, S_NEWL, sep='')
if doViterbi:
    # dfrInp = readCSV(pF=dInp['pFInpViterbi'], iCol=0)
    dRes, V = ViterbiCore(dI=dInp)
    printRes(dI=dInp, dR=dRes, V=V)
print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
