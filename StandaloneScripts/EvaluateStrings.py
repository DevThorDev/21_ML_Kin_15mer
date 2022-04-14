# -*- coding: utf-8 -*-
###############################################################################
# --- EvaluateStrings.py ------------------------------------------------------
###############################################################################
import os, time

import numpy as np
import pandas as pd

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_PROC_I_N_MER = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                              '11_ProcInpData')
P_COMB_RES = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                          '31_ResCombined')
P_PROB_RES = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                          '35_ResProb')

S_F_PROC_I_N_MER = 'Pho15mer_202202'
S_F_RES_COMB_S = 'Combined_S_KinasesPho15mer_202202'
S_F_SNIP_TBL = 'SnipProbTable'
S_F_N_MER_TBL = 'NmerProbTable'

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

S_SP04 = S_SPACE*4
S_PL24 = S_PLUS*24
S_ST24 = S_STAR*24
S_ST25 = S_STAR*25
S_DS30 = S_DASH*30
S_DS80 = S_DASH*80
S_EQ80 = S_EQ*80

S_PROC_I_N_MER = 'ProcINmer'
S_COMB_S = 'Combined_S'
S_CSV = 'csv'

S_MER = 'mer'
S_CODE_SEQ = 'code_seq'
S_N_MER_SEQ = 'c15' + S_MER
S_PROB = 'Prob'
S_PYL = 'Pyl'
S_TOTAL = 'Total'

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
LEN_N_MER_DEF = 15
I_CENT = LEN_N_MER_DEF//2

R02 = 2
R04 = 4
R06 = 6

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
calcSnipTbl = True
calcNmerTbl = True

inpTbl = 'CombS'            # 'ProcI' / 'CombS'

# --- numbers -----------------------------------------------------------------
# iMaxLFullSeqUnique = 11000
# iMaxLNmerSeqUnique = 1000

# modDispLSnips = 500
modDispPyl = 500
modDispOcc = 100000
modDispProb = 100
modDispReArrange = 1000

# --- strings -----------------------------------------------------------------
sFSnipTbl, sFNmerTbl = S_F_SNIP_TBL, S_F_N_MER_TBL
if inpTbl == 'ProcI':
    sFSnipTbl += S_USC + S_PROC_I_N_MER
    sFNmerTbl += S_USC + S_PROC_I_N_MER
elif inpTbl == 'CombS':
    sFSnipTbl += S_USC + S_COMB_S
    sFNmerTbl += S_USC + S_COMB_S

sMer = S_MER
sMerPyl = S_MER + S_USC + S_PYL
sMerTotal = S_MER + S_USC + S_TOTAL
sMerProb = S_MER + S_USC + S_PROB

# --- lists -------------------------------------------------------------------
lLenNmer = [1, 3, 5, 7]

# list of Nmer sequences to test or None (--> test all)
# lNmerSeq2Test = ['AAAALKGSDHRRTTE', 'SDNSGTESSPRTNGG', 'DNSGTESSPRTNGGS']
lNmerSeq2Test = None

lSCProbTbl = ['sSnip', 'lenSnip', 'nOcc', 'nPyl', 'probSnip']
lISrtProbTbl = [1, -1, 0]
lAscProbTbl = [True, False, True]

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
assert (len(lSCProbTbl) >= 3 and len(lISrtProbTbl) >= 3 and
        len(lAscProbTbl) >= 3)

# === derived values and input processing =====================================
lSCLenMer = [str(n) + sMer for n in lLenNmer]
lSCLenMerPyl = [str(n) + sMerPyl for n in lLenNmer]
lSCLenMerTotal = [str(n) + sMerTotal for n in lLenNmer]
lSCLenMerProb = [str(n) + sMerProb for n in lLenNmer]
dSCLenMer = {n: sC for n, sC in zip(lLenNmer, lSCLenMer)}
dSCLenMerPyl = {n: sC for n, sC in zip(lLenNmer, lSCLenMerPyl)}
dSCLenMerTotal = {n: sC for n, sC in zip(lLenNmer, lSCLenMerTotal)}
dSCLenMerProb = {n: sC for n, sC in zip(lLenNmer, lSCLenMerProb)}
dSCLenMerAll = {sMer: dSCLenMer,
                sMerPyl: dSCLenMerPyl,
                sMerTotal: dSCLenMerTotal,
                sMerProb: dSCLenMerProb}

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'calcSnipTbl': calcSnipTbl,
        'calcNmerTbl': calcNmerTbl,
        'inpTbl': inpTbl,
        # --- files, directories and paths ------------------------------------
        'pProcINmer': P_PROC_I_N_MER,
        'pCombRes': P_COMB_RES,
        'pProbRes': P_PROB_RES,
        'sFProcINmer': S_F_PROC_I_N_MER + XT_CSV,
        'sFResCombS': S_F_RES_COMB_S + XT_CSV,
        'sFSnipTbl': sFSnipTbl + XT_CSV,
        'sFNmerTbl': sFNmerTbl + XT_CSV,
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sCSV': S_CSV,
        'sMer': sMer,
        'sMerPyl': sMerPyl,
        'sMerTotal': sMerTotal,
        'sMerProb': sMerProb,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers ---------------------------------------------------------
        'lenNmerDef': LEN_N_MER_DEF,
        'iCent': I_CENT,
        # 'modDispLSnips': modDispLSnips,
        'modDispPyl': modDispPyl,
        'modDispOcc': modDispOcc,
        'modDispProb': modDispProb,
        'modDispReArrange': modDispReArrange,
        # --- lists -----------------------------------------------------------
        'lLenNmer': lLenNmer,
        'lNmerSeq2Test': lNmerSeq2Test,
        'lSCProbTbl': lSCProbTbl,
        'lISrtProbTbl': lISrtProbTbl,
        'lAscProbTbl': lAscProbTbl,
        # --- dictionaries ----------------------------------------------------
        # === derived values and input processing =============================
        'lSCLenMer': lSCLenMer,
        'lSCLenMerPyl': lSCLenMerPyl,
        'lSCLenMerTotal': lSCLenMerTotal,
        'lSCLenMerProb': lSCLenMerProb,
        'dSCLenMer': dSCLenMer,
        'dSCLenMerPyl': dSCLenMerPyl,
        'dSCLenMerTotal': dSCLenMerTotal,
        'dSCLenMerProb': dSCLenMerProb,
        'dSCLenMerAll': dSCLenMerAll}

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

# --- Time printing functions -------------------------------------------------
def startSimu():
    startTime = time.time()
    print(S_PL24 + ' START', time.ctime(startTime), S_PL24)
    print('Process and handle data and generate plots.')
    return startTime

def printElapsedTimeSim(stT=None, cT=None, sPre='Time', nDig=R04):
    if stT is not None and cT is not None:
        # calculate and display elapsed time
        elT = round(cT - stT, nDig)
        print(sPre, 'elapsed:', elT, 'seconds, this is', round(elT/60, nDig),
              'minutes or', round(elT/3600, nDig), 'hours or',
              round(elT/(3600*24), nDig), 'days.')

def showElapsedTime(startTime=None):
    if startTime is not None:
        print(S_DS80)
        printElapsedTimeSim(startTime, time.time(), 'Time')
        print(S_SP04 + 'Current time:', time.ctime(time.time()), S_SP04)
        print(S_DS80)

def endSimu(startTime=None):
    if startTime is not None:
        print(S_DS80)
        printElapsedTimeSim(startTime, time.time(), 'Total time')
        print(S_ST24 + ' DONE', time.ctime(time.time()), S_ST25)

# --- Functions handling strings ----------------------------------------------
def getLSubStrNmer(dI, sNmer):
    assert len(sNmer) == dI['lenNmerDef']
    return [sNmer[(dI['iCent'] - cLen//2):(dI['iCent'] + cLen//2 + 1)]
            for cLen in dI['lLenNmer']]

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

# --- Functions creating and manipulating lists -------------------------------
# def genLSnips(dI, lNmerSeq=[], nDig=R02):
#     lSnips = []
#     for n, sNmer in enumerate(lNmerSeq):
#         for sSub in getLSubStrNmer(dI, sNmer):
#             if sSub not in lSnips:
#                 lSnips.append(sSub)
#         if n%dI['modDispLSnips'] == 0:
#             print('- genLSnips: processed ', n, ' of ', len(lNmerSeq), ' (',
#                   round(n/len(lNmerSeq)*100, nDig), '%)', sep='')
#     return lSnips

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

# def iniDicts(dI, lNmerSeq):
#     dOcc, dPyl, dProb = {}, {}, {}
#     for sNmer in lNmerSeq:
#         for sSubNmer in getLSubStrNmer(dI, sNmer):
#             dOcc[sSubNmer] = 0
#     for sK in dOcc:
#         dPyl[sK] = 0
#         dProb[sK] = 0.

# def fillDictPyl(dI, dPyl, lNmerSeq, nDig=R02):
#     N = len(dPyl)*len(lNmerSeq)
#     for i, sSubNmer in enumerate(dPyl):
#         for j, sNmer in enumerate(lNmerSeq):
#             if sSubNmer in getLSubStrNmer(dI, sNmer):
#                 addToDictCt(dPyl, cK=sSubNmer)
#             n = i*len(lNmerSeq) + j + 1
#             if n%dI['modDispPyl'] == 0:
#                 print('- fillDictPyl: processed ', n, ' of ', N, ' (',
#                       round(n/N*100, nDig), '%)', sep='')

def genDictPyl(dI, lNmerSeq=[], nDig=R02):
    dPyl = {}
    for n, sNmer in enumerate(lNmerSeq):
        for sSubNmer in getLSubStrNmer(dI, sNmer):
            addToDictCt(dPyl, cK=sSubNmer)
        if n%dI['modDispPyl'] == 0:
            print('- genDictPyl: processed ', n, ' of ', len(lNmerSeq), ' (',
                  round(n/len(lNmerSeq)*100, nDig), '%)', sep='')
    return dPyl

def iniDicts(dI, dPyl):
    dOcc, dProb = {}, {}
    for sSubNmer in dPyl:
        dOcc[sSubNmer], dProb[sSubNmer] = 0, 0.
    return dOcc, dProb

def fillDictOcc(dI, dOcc, lFSeq=[], nDig=R02):
    N = len(dOcc)*len(lFSeq)
    for i, sSubNmer in enumerate(dOcc):
        for j, sFSeq in enumerate(lFSeq):
            addToDictCt(dOcc, cK=sSubNmer, cIncr=sFSeq.count(sSubNmer))
            n = i*len(lFSeq) + j + 1
            if n%dI['modDispOcc'] == 0:
                print('- fillDictOcc: processed ', n, ' of ', N, ' (',
                      round(n/N*100, nDig), '%)', sep='')

def fillDictProb(dI, dOcc, dPyl, dProb, nDig=R02):
    assert list(dOcc) == list(dPyl)
    for n, sK in enumerate(dOcc):
        if dOcc[sK] > 0:
            dProb[sK] = dPyl[sK]/dOcc[sK]
        if n%dI['modDispProb'] == 0:
            print('- fillDictProb: processed ', n, ' of ', len(dOcc), ' (',
                  round(n/len(dOcc)*100, nDig), '%)', sep='')

# --- Functions re-arranging and saving DataFrames ----------------------------
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
    dfrResTbl.sort_values(by=[dI['lSCProbTbl'][i] for i in dI['lISrtProbTbl']],
                          ascending=dI['lAscProbTbl'], inplace=True,
                          ignore_index=True)
    saveAsCSV(dfrResTbl, pF=joinToPath(pF=dI['pProbRes'], nmF=dI['sFSnipTbl']))
    return dfrResTbl

def reArrangeToProbTbl(dI, lNmerSeq=[], dfrProbSnip=None, nDig=R02):
    if dfrProbSnip is None:
        dfrProbSnip = readCSV(pF=joinToPath(pF=dInp['pProbRes'],
                                            nmF=dInp['sFSnipTbl']), iCol=0)
    dProbSnip = dfrProbSnip.to_dict(orient='list')
    dProbTbl, dSC = {}, dI['dSCLenMerAll']
    for n, sNmer in enumerate(lNmerSeq):
        for sSubNmer in getLSubStrNmer(dI, sNmer):
            cLen = len(sSubNmer)
            addToDictL(dProbTbl, cK=dSC[dI['sMer']][cLen], cE=sSubNmer)
            i = dProbSnip[dI['lSCProbTbl'][0]].index(sSubNmer)
            nPyl = dProbSnip[dI['lSCProbTbl'][-2]][i]
            nTtl = dProbSnip[dI['lSCProbTbl'][-3]][i]
            cPrb = dProbSnip[dI['lSCProbTbl'][-1]][i]
            addToDictL(dProbTbl, cK=dSC[dI['sMerPyl']][cLen], cE=nPyl)
            addToDictL(dProbTbl, cK=dSC[dI['sMerTotal']][cLen], cE=nTtl)
            addToDictL(dProbTbl, cK=dSC[dI['sMerProb']][cLen], cE=cPrb)
        if n%dI['modDispReArrange'] == 0:
            print('- reArrangeToProbTbl: processed ', n, ' of ', len(lNmerSeq),
                  ' (', round(n/len(lNmerSeq)*100, nDig), '%)', sep='')
    dfrProbTbl = iniPdDfr(dProbTbl, lSNmR=lNmerSeq)
    saveAsCSV(dfrProbTbl, pF=joinToPath(pF=dI['pProbRes'],
                                        nmF=dI['sFNmerTbl']))

# ### MAIN ####################################################################
startTime = startSimu()
print(S_EQ80, S_NEWL, S_DS30, ' EvaluateStrings.py ', S_DS30, S_NEWL, sep='')
pFInp, sFInp = dInp['pProcINmer'], dInp['sFProcINmer']
if dInp['inpTbl'] == 'CombS':
    pFInp, sFInp = dInp['pCombRes'], dInp['sFResCombS']
if calcSnipTbl or calcNmerTbl:
    dfrInp = readCSV(pF=joinToPath(pF=pFInp, nmF=sFInp), iCol=0)
# lFullSeqUnique = list(dfrInp[S_CODE_SEQ].unique())[:iMaxLFullSeqUnique]
if calcSnipTbl:
    lFullSeqUnique = list(dfrInp[S_CODE_SEQ].unique())
if calcSnipTbl or calcNmerTbl:
    lNmerSeqUnique = list(dfrInp[S_N_MER_SEQ].unique())
    if dInp['lNmerSeq2Test'] is None:
        # dInp['lNmerSeq2Test'] = lNmerSeqUnique[:iMaxLNmerSeqUnique]
        dInp['lNmerSeq2Test'] = lNmerSeqUnique
# lAllSnips = genLSnips(dI=dInp, lNmerSeq=dInp['lNmerSeq2Test'])
# print(lAllSnips[:10])
# print(len(lAllSnips))
if calcSnipTbl:
    dictPyl = genDictPyl(dI=dInp, lNmerSeq=dInp['lNmerSeq2Test'])
    showElapsedTime(startTime)
    dictOcc, dictPrb = iniDicts(dI=dInp, dPyl=dictPyl)
    fillDictOcc(dI=dInp, dOcc=dictOcc, lFSeq=lFullSeqUnique)
    showElapsedTime(startTime)
    fillDictProb(dI=dInp, dOcc=dictOcc, dPyl=dictPyl, dProb=dictPrb)
    # print(len(dictOcc), len(dictPyl), len(dictPrb))
    dfrSnipTbl = saveResTbl(dI=dInp, dOcc=dictOcc, dPyl=dictPyl, dProb=dictPrb)
if calcNmerTbl:
    reArrangeToProbTbl(dI=dInp, lNmerSeq=dInp['lNmerSeq2Test'])
endSimu(startTime)

###############################################################################
