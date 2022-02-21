# -*- coding: utf-8 -*-
###############################################################################
# --- F_00__GenFunctions.py ---------------------------------------------------
###############################################################################
import os, time, itertools

import numpy as np
from numpy.random import default_rng as RNG
import pandas as pd

import Core.C_00__GenConstants as GC

# === General functions =======================================================
# --- File system related functions -------------------------------------------
def createDir(pF):
    if not os.path.isdir(pF):
        os.mkdir(pF)

def joinToPath(pF='', nmF='Dummy.txt'):
    if len(pF) > 0:
        createDir(pF)
        return os.path.join(pF, nmF)
    else:
        return nmF

def joinDirToPath(pF='', nmD='Directory'):
    if len(pF) > 0:
        pF = os.path.join(pF, nmD)
        createDir(pF)
        return pF
    else:
        return nmD

def readCSV(pF, iCol=None, dDTp=None, cSep=GC.S_SEMICOL):
    if os.path.isfile(pF):
        return pd.read_csv(pF, sep=cSep, index_col=iCol, dtype=dDTp)

def saveAsCSV(pdDfr, pF, reprNA='', cSep=GC.S_SEMICOL):
    if pdDfr is not None:
        pdDfr.to_csv(pF, sep=cSep, na_rep=reprNA)

# --- String selection and manipulation functions -----------------------------
def joinS(itS, sJoin=GC.S_USC):
    return sJoin.join([str(s) for s in itS]).strip()

def getPartStr(s, iStart=None, iEnd=None, sSpl=GC.S_USC):
    if iStart is None:
        if iEnd is None:
            return joinS(s.split(sSpl), sJoin=sSpl)
        else:
            return joinS(s.split(sSpl)[:iEnd], sJoin=sSpl)
    else:
        if iEnd is None:
            return joinS(s.split(sSpl)[iStart:], sJoin=sSpl)
        else:
            return joinS(s.split(sSpl)[iStart:iEnd], sJoin=sSpl)

def extSID(sID, n, N, preChar=GC.S_0):
    return joinS([sID, preChar*(len(str(N)) - len(str(n))) + str(n)])

def getSF(sMid, iStart=None, iEnd=None, sPre='', sPost='', sSpl=GC.S_USC):
    lS = [sPre, getPartStr(sMid, iStart=iStart, iEnd=iEnd, sSpl=sSpl), sPost]
    if len(sPre) > 0 and len(sPost) == 0:
        lS = [sPre, getPartStr(sMid, iStart=iStart, iEnd=iEnd, sSpl=sSpl)]
    elif len(sPre) == 0 and len(sPost) > 0:
        lS = [getPartStr(sMid, iStart=iStart, iEnd=iEnd, sSpl=sSpl), sPost]
    elif len(sPre) == 0 and len(sPost) == 0:
        lS = [getPartStr(sMid, iStart=iStart, iEnd=iEnd, sSpl=sSpl)]
    return joinS(lS)

# --- Functions performing calculations with scalars --------------------------
def isEq(x, xCmp, maxDlt=GC.MAX_DELTA):
    return abs(x - xCmp) < maxDlt

def isInSeqSet(x, cSqSet, maxDlt=GC.MAX_DELTA):
    for cEl in cSqSet:
        if isEq(x, cEl, maxDlt=maxDlt):
            return True
    return False

# --- Functions handling dictionaries -----------------------------------------
def addIfAbsent(lD, cK, cV=None):
    for cD in lD:
        if cK not in cD:
            cD[cK] = cV

def complDict(cDFull, cDAdd):
    for cK, cV in cDAdd.items():
        if cK in cDFull and type(cDFull[cK]) == dict and type(cV) == dict:
            complDict(cDFull[cK], cV)
        else:
            cDFull[cK] = cV

def setOthValSub(cD, lKMain, cKMain, cKSub, oVSub=None):
    for kM in [k for k in lKMain if k != cKMain]:
        complDict(cD, {kM: {cKSub: oVSub}})

def extractFromDictL(cD, lKeys):
    lRet, lLen = [], [len(l) for l in cD.values()]
    assert min(lLen) == max(lLen)
    for k in range(max(lLen)):
        for cK in lKeys:
            if cK in cD:
                lRet.append(cD[cK][k])
            else:
                lRet.append(GC.S_WAVE)
    return lRet

def addToDictL(cD, cK, cE, lUniqEl=False):
    if cK in cD:
        if not lUniqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def addToDict2L(cD, cKMain, cKSub, cE, lUniqEl=False):
    if cKMain not in cD:
        cD[cKMain] = {}
    addToDictL(cD[cKMain], cKSub, cE, lUniqEl=lUniqEl)

def addToDictDSpc(cDMain, cKMain, cKSub, cVSub=None):
    if cDMain[cKMain] == None:
        cDMain[cKMain] = {cKSub: cVSub}
    elif type(cDMain[cKMain]) == dict:
        cDMain[cKMain][cKSub] = cVSub

def addToDictD(cD, cKMain, cKSub, lV=[]):
    if cKMain in cD:
        if cKSub not in cD[cKMain]:
            cD[cKMain][cKSub] = lV
        else:
            print('ERROR: Key', cKSub, 'already in', cD[cKMain], '\nlV =', lV)
            assert False
    else:
        cD[cKMain] = {cKSub: lV}

def printSizeDDDfr(dDDfr, modeF=False, nDig=GC.R04):
    nKMain, nKSub = len(dDDfr), sum([len(cDSub) for cDSub in dDDfr.values()])
    sumNLDfr = 0
    for cKMain in dDDfr:
        sumNLDfr += sum([cDfr.shape[0] for cDfr in dDDfr[cKMain].values()])
    print(GC.S_DS80)
    print('Number of keys in main dictionary: ', nKMain)
    print('Sum of numbers of keys in sub-dictionaries:', nKSub)
    print('Average number of keys of a sub-dictionary:',
          round(nKSub/nKMain, nDig))
    print('Sum of numbers of lines of DataFrames:', sumNLDfr)
    print('Average number of lines of DataFrames per sub-dictionary:',
          round(sumNLDfr/nKSub, nDig))
    print('Average number of lines of DataFrames per main dictionary:',
          round(sumNLDfr/nKMain, nDig))
    if modeF:
        print(GC.S_DS80)
        for cKMain, cDSub in dDDfr.items():
            dN = {len(cDSub): sum([cDfr.shape[0] for cDfr in cDSub.values()])}
            print(GC.S_DASH, cKMain, GC.S_ARR_LR, dN)
    print(GC.S_DS80)

# --- Functions handling iterables --------------------------------------------
def allTrue(cIt):
    return [cB for cB in cIt] == [True]*len(cIt)

def iterIntersect(it1, it2, notIn2=False):
    lIntersect = [x for x in it1 if x in set(it2)]
    if notIn2:
        lIntersect = [x for x in it1 if x not in set(it2)]
    return lIntersect

def flattenIt(cIterable, retArr=False):
    itFlat = list(itertools.chain.from_iterable(cIterable))
    if retArr:
        itFlat = np.array(itFlat)
    return itFlat

# --- Functions performing numpy array calculation and manipulation -----------
def getArrCartProd(it1, it2):
    return np.array(list(itertools.product(it1, it2)))

def iniNpArr(data=None, shape=(0, 0), fillV=np.nan):
    if data is None:
        return np.full(shape, fillV)
    else:       # ignore shape
        return np.array(data)

# --- Functions performing pandas DataFrame calculation and manipulation ------
def iniPdSer(data=None, lSNmI=[], shape=(0,), fillV=np.nan):
    assert len(shape) == 1
    if len(lSNmI) == 0:
        if data is None:
            return pd.Series(np.full(shape, fillV))
        else:
            return pd.Series(data)
    else:
        if data is None:
            return pd.Series(np.full(len(lSNmI), fillV), index=lSNmI)
        else:
            assert data.size == len(lSNmI)
            return pd.Series(data, index=lSNmI)

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

def iniWShape(tmplDfr, fillV=np.nan):
    iniPdDfr(shape=tmplDfr.shape, fillV=fillV)

def tryRoundX(x, RD=None):
    if RD is not None:
        try:
            x = round(x, RD)
        except:
            print('Cannot round "', x, '" to ', RD, ' digits.', sep='')
    return x

def getSglElWSelC(pdDfr, sCVal, sCSel, xSel, RD=None):
    xSel = tryRoundX(xSel, RD=RD)
    if sCVal in pdDfr and sCSel in pdDfr:
        cSer = pdDfr[sCVal][pdDfr[sCSel].round(RD) == xSel]
        if cSer.size > 1:
            print('"', xSel, '" not unique in column "', sCSel, '"!', sep='')
            return cSer.iloc[0].round(RD).item()
        else:
            if cSer.size == 0:
                print('"', xSel, '" not in column "', sCSel, '"!', sep='')
                return np.nan
            else:   # standard case (size of series == 1 => return this value)
                return cSer.round(RD).item()
    else:
        print('"', sCVal, '" or "', sCSel, '" not in DataFrame columns: ',
              list(pdDfr.columns), '!', sep='')

def concPdDfrS(lPdDfr, concAx=0, verInt=True, srt=False, ignIdx=False,
               dropAx=None):
    d = pd.concat(lPdDfr, axis=concAx, verify_integrity=verInt, sort=srt,
                  ignore_index=ignIdx)
    if dropAx in [0, 1, 'index', 'columns']:
        d.dropna(axis=dropAx, inplace=True)
    return d

def splitDfr(pdDfr, tHd, j=0):
    lSubDfr, setV = [], set(pdDfr[tHd[j]])
    for cV in setV:
        lSubDfr.append(pdDfr[pdDfr[tHd[j]] == cV])
    if j == len(tHd) - 1:
        return lSubDfr
    else:
        j += 1
        return [splitDfr(cSubDfr, tHd, j) for cSubDfr in lSubDfr]

def checkColSums(pdDfr, lHdC, lSumC, rndDig=GC.R04):
    assert len(lHdC) == len(lSumC)
    lIsOK = [0]*len(lSumC)
    for sHdC in pdDfr.columns:
        if sHdC in lHdC:
            k = lHdC.index(sHdC)
            xS, xC = round(pdDfr[sHdC].sum(), rndDig), round(lSumC[k], rndDig)
            if xS == xC:
                lIsOK[k] = 1
            else:
                lIsOK[k] = -1
                print('WARNING: Sum of column "', sHdC, '" is ', xS, ', but ',
                      'should be ', xC, '!', sep='')
    return lIsOK

# --- Functions calculating running mean and standard deviation (numpy array) -
def iniArrMnM2(shape=(0, 0)):
    return np.zeros(shape), np.zeros(shape)

def adjShapeArrMnM2(arrNew, lArrMnM2):
    assert len(lArrMnM2) == 2
    for k, cArr in enumerate(lArrMnM2):
        if cArr.shape != arrNew.shape:
            lArrMnM2[k] = np.zeros(arrNew.shape)

# For a new array arrNew, compute the new count, new arrMn, new arrM2
# arrMn accumulates the mean values of the entire data series
# arrM2 aggregates the squared distances from the mean values
# count aggregates the number of samples seen so far
def updateMeanM2(arrMn, arrM2, arrNew, cCnt):
    if arrMn.shape == arrNew.shape and arrM2.shape == arrNew.shape:
        delta = arrNew - arrMn
        arrMn += delta/cCnt
        delta2 = arrNew - arrMn
        arrM2 += delta*delta2
    else:
        print('Shape of arrMn:', arrMn.shape)
        print('Shape of arrM2:', arrM2.shape)
        print('Shape of arrNew:', arrNew.shape)
        print('cCnt =', cCnt)
        assert False

# Retrieve mean, variance and sample variance from arrMn, arrM2 and cCnt
def finalRunSD(arrM2, cCnt):
    arrVar = iniNpArr(shape=arrM2.shape)
    if cCnt > 1:
        arrVar = arrM2/(cCnt - 1)       # sample standard deviation
    return np.sqrt(arrVar)

def calcArrRunMnSD(lArr, cCt=0):
    arrMn, arrM2 = iniNpArr(), iniNpArr()
    if len(lArr) > 0:
        assert [a.shape == lArr[0].shape for a in lArr] == [True]*len(lArr)
        arrMn, arrM2 = iniArrMnM2(shape=lArr[0].shape)
        for cArr in lArr:
            cCt += 1
            updateMeanM2(arrMn, arrM2, cArr, cCt)
    return arrMn, finalRunSD(arrM2, cCt)

# --- Functions related to reshuffling and drawing of random numbers ----------
def shuffleArr(cArr, cAx=0):
    RNG().shuffle(cArr, axis=cAx)
    return cArr

def fullShuffleArr(cArr):
    shuffledArr = cArr.flatten()
    RNG().shuffle(shuffledArr)
    return shuffledArr.reshape(cArr.shape)

def shuffleFull(cDfr):
    return fullShuffleArr(cDfr.to_numpy(copy=True))

def shufflePerRow(cDfr):
    cArr = iniNpArr(shape=cDfr.shape)
    for k, (_, serRow) in enumerate(cDfr.iterrows()):
        cArr[k, :] = shuffleArr(serRow.to_numpy(copy=True))
    return cArr

def drawFromNorm(cMn=0., cSD=1., tPar=None, arrShape=None):
    # if tPar is None, then use cMn, cSD, else use tPar
    if tPar is not None and len(tPar) == 2:
        cMn, cSD = tPar
    # prevent a SD < 0
    if type(cSD) == np.ndarray:
        cSD = cSD.clip(min=0)
    else:
        cSD = max(cSD, 0)
    return RNG().normal(loc=cMn, scale=cSD, size=arrShape)

# --- General printing functions ----------------------------------------------
def printMode(isTestMode=False):
    if isTestMode:
        print(GC.S_ST34, 'TEST mode', GC.S_ST35)
    else:
        print(GC.S_ST32, 'Standard mode', GC.S_ST33)

# --- Data printing functions -------------------------------------------------
def printStructDict(cD, cLvl=0, sOut=''):
    for cK, cV in cD.items():
        sOut += ('Level' + GC.S_SPACE + str(cLvl) + GC.S_SPACE + GC.S_VBAR +
                 GC.S_SPACE + str(cK) + GC.S_COL + GC.S_SPACE)
        if type(cV) == dict:
            printStructDict(cV, cLvl=cLvl+1, sOut=sOut)
        else:
            sOut += str(type(cV))
            print(sOut)
        sOut = ''

def printIterableVal(cIt):
    if type(cIt) in [list, tuple, set]:
        for k, cEl in enumerate(cIt):
            print(GC.S_SP04, 'Element with index ', k, ':', sep='')
            print(cEl)
    else:
        print(cIt)

def printDataD1(sK, cV):
    print(GC.S_EQ20, sK, GC.S_EQ20)
    printIterableVal(cV)

def printDataD2(sKL0, dL0, maxLvl=1):
    print(GC.S_EQ20, sKL0, GC.S_EQ20)
    for sKL1, cVL1 in dL0.items():
        print(GC.S_DS20, sKL1, GC.S_DS20)
        if maxLvl > 1:
            for sKL2, cVL2 in cVL1.items():
                print(GC.S_DT20, sKL2, GC.S_DT20)
                printIterableVal(cVL2)
        else:
            printIterableVal(cVL1)

# --- Time printing functions -------------------------------------------------
def startSimu():
    startTime = time.time()
    print(GC.S_PL24 + ' START', time.ctime(startTime), GC.S_PL24)
    print('Process and handle data and generate plots.')
    return startTime

def printElapsedTimeSim(stT, cT, sPre='Time', nDig=GC.R04):
    # calculate and display elapsed time
    elT = round(cT - stT, nDig)
    print(sPre, 'elapsed:', elT, 'seconds, this is', round(elT/60, nDig),
          'minutes or', round(elT/3600, nDig), 'hours or',
          round(elT/(3600*24), nDig), 'days.')

def showElapsedTime(startTime):
    print(GC.S_DS80)
    printElapsedTimeSim(startTime, time.time(), 'Time')
    print(GC.S_SP04 + 'Current time:', time.ctime(time.time()), GC.S_SP04)
    print(GC.S_DS80)

def endSimu(startTime):
    print(GC.S_DS80)
    printElapsedTimeSim(startTime, time.time(), 'Total time')
    print(GC.S_ST24 + ' DONE', time.ctime(time.time()), GC.S_ST25)

###############################################################################
