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

S_PROB = 'Prob'
S_PREV = 'prev'
S_NONE = 'None'

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
DELTA = 1.0E-3

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
lO01 = list('MSGSRRKATPASRTRVGNYEMGRTLGEGSFAKVKYAKNTVTGDQAAIKILDREKVFRHKMVEQ' +
            'LKREISTMKLIKHPNVVEIIEVMASKTKIYIVLELVNGGELFDKIAQQGRLKEDEARRYFQQL' +
            'INAVDYCHSRGVYHRDLKPENLILDANGVLKVSDFGLSAFSRQVREDGLLHTACGTPNYVAPE' +
            'VLSDKGYDGAAADVWSCGVILFVLMAGYLPFDEPNLMTLYKRICKAEFSCPPWFSQGAKRVIK' +
            'RILEPNPITRISIAELLEDEWFKKGYKPPSFDQDDEDITIDDVDAAFSNSKECLVTEKKEKPV' +
            'SMNAFELISSSSEFSLENLFEKQAQLVKKETRFTSQRSASEIMSKMEETAKPLGFNVRKDNYK' +
            'IKMKGDKSGRKGQLSVATEVFEVAPSLHVVELRKTGGDTLEFHKFYKNFSSGLKDVVWNTDAA' +
            'AEEQKQ')
lO02 = list('MTSLLKSSPGRRRGGDVESGKSEHADSDSDTFYIPSKNASIERLQQWRKAALVLNASRRFRYT' +
            'LDLKKEQETREMRQKIRSHAHALLAANRFMDMGRESGVEKTTGPATPAGDFGITPEQLVIMSK' +
            'DHNSGALEQYGGTQGLANLLKTNPEKGISGDDDDLLKRKTIYGSNTYPRKKGKGFLRFLWDAC' +
            'HDLTLIILMVAAVASLALGIKTEGIKEGWYDGGSIAFAVILVIVVTAVSDYKQSLQFQNLNDE' +
            'KRNIHLEVLRGGRRVEISIYDIVVGDVIPLNIGNQVPADGVLISGHSLALDESSMTGESKIVN' +
            'KDANKDPFLMSGCKVADGNGSMLVTGVGVNTEWGLLMASISEDNGEETPLQVRLNGVATFIGS' +
            'IGLAVAAAVLVILLTRYFTGHTKDNNGGPQFVKGKTKVGHVIDDVVKVLTVAVTIVVVAVPEG' +
            'LPLAVTLTLAYSMRKMMADKALVRRLSACETMGSATTICSDKTGTLTLNQMTVVESYAGGKKT' +
            'DTEQLPATITSLVVEGISQNTTGSIFVPEGGGDLEYSGSPTEKAILGWGVKLGMNFETARSQS' +
            'SILHAFPFNSEKKRGGVAVKTADGEVHVHWKGASEIVLASCRSYIDEDGNVAPMTDDKASFFK' +
            'NGINDMAGRTLRCVALAFRTYEAEKVPTGEELSKWVLPEDDLILLAIVGIKDPCRPGVKDSVV' +
            'LCQNAGVKVRMVTGDNVQTARAIALECGILSSDADLSEPTLIEGKSFREMTDAERDKISDKIS' +
            'VMGRSSPNDKLLLVQSLRRQGHVVAVTGDGTNDAPALHEADIGLAMGIAGTEVAKESSDIIIL' +
            'DDNFASVVKVVRWGRSVYANIQKFIQFQLTVNVAALVINVVAAISSGDVPLTAVQLLWVNLIM' +
            'DTLGALALATEPPTDHLMGRPPVGRKEPLITNIMWRNLLIQAIYQVSVLLTLNFRGISILGLE' +
            'HEVHEHATRVKNTIIFNAFVLCQAFNEFNARKPDEKNIFKGVIKNRLFMGIIVITLVLQVIIV' +
            'EFLGKFASTTKLNWKQWLICVGIGVISWPLALVGKFIPVPAAPISNKLKVLKFWGKKKNSSGE' +
            'GSL')
lO03 = list('MAEEQKTSKVDVESPAVLAPAKEPTPAPVEVADEKIHNPPPVESKALAVVEKPIEEHTPKKAS' +
            'SGSADRDVILADLEKEKKTSFIKAWEESEKSKAENRAQKKISDVHAWENSKKAAVEAQLRKIE' +
            'EKLEKKKAQYGEKMKNKVAAIHKLAEEKRAMVEAKKGEELLKAEEMGAKYRATGVVPKATCGCF')

# --- dictionaries ------------------------------------------------------------
# dObs = {1: lO01, 2: lO02, 3: lO03}
dObs = {1: lO01, 2: lO02}

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

# --- Helper functions for the Viterbi algorithm handling ">", exp and ln -----
def X_exp(x=None):
    if x is None:
        return 0.
    else:
        return np.exp(x)

def X_ln(x=None):
    if x is None or x < 0:
        print('ERROR: Value to calculate natural logarithm from is', x, '...')
        # assert False
        return None
    elif x == 0:
        return None
    else:
        return np.log(x)

def X_sum(x=None, y=None):
    if x is None or y is None:
        return None
    else:
        return x + y

def X_lnSum(x=None, y=None):
    if x is None and y is None:
        return None
    elif x is None and y is not None:
        return X_ln(y)
    elif x is not None and y is None:
        return X_ln(x)
    else:
        if x > 0 and y > 0:
            if X_ln(x) > X_ln(y):
                return X_ln(x) + X_ln(1 + np.exp(X_ln(y) - X_ln(x)))
            else:
                return X_ln(y) + X_ln(1 + np.exp(X_ln(x) - X_ln(y)))
        else:
            return None

def X_lnProd(x=None, y=None):
    if x is None or y is None or X_ln(x) is None or X_ln(y) is None:
        return None
    else:
        return X_ln(x) + X_ln(y)

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
        dData = {S_PROB: X_lnProd(dI['startPr'][st],
                                  dI['emitPr'][st][lObs[i]]),
                 S_PREV: None}
        addToDictD(V, cKMain=i, cKSub=st, cVSub=dData)
    return V

def getTransProb(dI, V, i, cSt, prevSt):
    return X_sum(V[i - 1][prevSt][S_PROB], X_ln(dI['transPr'][prevSt][cSt]))

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
            maxProb = X_sum(maxTransProb, X_ln(dI['emitPr'][st][lObs[i]]))
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
            lnProb = S_NONE
            if dProbPrev[S_PROB] is not None:
                lnProb = round(dProbPrev[S_PROB], dI['R08'])
            print(S_DS08, 'ln(prob).:', lnProb,
                  S_VLINE, 'Previous state selected:', dProbPrev[S_PREV])
    print('Optimal state path:', dR['optStPath'])
    print('Maximal probability:', np.exp(dR['maxProb']))
    print('Maximal ln(probability):', dR['maxProb'])

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
