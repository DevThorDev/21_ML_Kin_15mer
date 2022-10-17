# -*- coding: utf-8 -*-
###############################################################################
# --- F_00__GenFunctions.py ---------------------------------------------------
###############################################################################
import os, pickle, itertools, time

import numpy as np
from numpy.random import default_rng as RNG
import pandas as pd
from pandas.api.types import is_numeric_dtype

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

def selFs(lSel, lSSpl, itSCmp=None):
    for k, s in enumerate(lSSpl):
        if s in itSCmp:
            lSel.append((k, s))
            if (len(itSCmp) > 1) and (k < len(lSSpl[:-1])):
                for j, s in enumerate(lSSpl[(k + 1):]):
                    if s in itSCmp:
                        lSel.append((k + j, s))
            break

def fillDSF(dSFTI, lSel, sF, iF=0, addI=True):
    if addI:
        iF += 1
        dSFTI[tuple([iF] + [t[1] for t in lSel])] = sF
    else:
        if len(lSel) == 1:
            dSFTI[lSel[0][1]] = sF
        else:
            dSFTI[tuple([t[1] for t in lSel])] = sF
    return iF

def getIFS(pF='', itSCmp=None, addI=True, sSpl=GC.S_USC):
    if itSCmp is None:                      # trivial case - no filtering
        return os.listdir(pF)
    dSFTI, iF = {}, 0
    for sF in os.listdir(pF):
        lSel, lSSpl = [], sF.split(GC.S_DOT)[0].split(sSpl)
        selFs(lSel, lSSpl, itSCmp=itSCmp)
        if len(lSel) == len(itSCmp):        # boolean AND
            iF = fillDSF(dSFTI, lSel, sF=sF, iF=iF, addI=addI)
    return dSFTI

def addSStartSEnd(sF, sStart='', sEnd='', sJoin=''):
    if len(sStart) > 0:
        sF = sStart + sJoin + sF
    if len(sEnd) > 0:
        sF = sF + sJoin + str(sEnd)
    return sF

def modSF(sF, sStart='', sEnd='', sJoin=''):
    if GC.S_DOT in sF:
        lSSpl = sF.split(GC.S_DOT)
        sFMod = joinS(lSSpl[:-1], cJ=GC.S_DOT)
        sFMod = addSStartSEnd(sFMod, sStart=sStart, sEnd=sEnd, sJoin=sJoin)
        return sFMod + GC.S_DOT + lSSpl[-1]
    else:
        return addSStartSEnd(sF, sStart=sStart, sEnd=sEnd, sJoin=sJoin)

def modPF(pF, sStart='', sEnd='', sJoin='', sJoinP=GC.S_DBLBACKSL):
    lSSpl = pF.split(sJoinP)
    sFMod = modSF(sF=lSSpl[-1], sStart=sStart, sEnd=sEnd, sJoin=sJoin)
    return joinS(lSSpl[:-1] + [sFMod], cJ=sJoinP)

def fileXist(pF):
    return os.path.isfile(pF)

def readCSV(pF, iCol=None, dDTp=None, cSep=GC.S_SEMICOL):
    if fileXist(pF):
        return pd.read_csv(pF, sep=cSep, index_col=iCol, dtype=dDTp)

def saveCSV(pdObj, pF, reprNA='', cSep=GC.S_SEMICOL, saveIdx=True, iLbl=None):
    if pdObj is not None:
        pdObj.to_csv(pF, sep=cSep, na_rep=reprNA, index=saveIdx,
                     index_label=iLbl)

def checkDupSaveCSV(pdDfr, pF, reprNA='', cSep=GC.S_SEMICOL, saveIdx=True,
                    iLbl=None, dropDup=True, igI=True):
    if pdDfr is not None:
        if dropDup:
            pdDfr.drop_duplicates(inplace=True, ignore_index=igI)
        pdDfr.to_csv(pF, sep=cSep, na_rep=reprNA, index=saveIdx,
                     index_label=iLbl)

def pickleSaveDict(cD, pF=('Dict' + GC.XT_BIN)):
    try:
        with open(pF, 'wb') as fDict:
            pickle.dump(cD, fDict)
    except:
        print('ERROR: Dumping dictionary to', pF, 'failed.')

def pickleLoadDict(pF=('Dict' + GC.XT_BIN), reLoad=False):
    cD = None
    if fileXist(pF) and (cD is None or len(cD) == 0 or reLoad):
        with open(pF, 'rb') as fDict:
            cD = pickle.load(fDict)
        print('Loaded dictionary from', pF)
    if cD is None:
        print('ERROR: Loading dictionary from', pF, 'failed.')
    return cD

# --- Functions checking whether a data structure is filled -------------------
def Xist(cDat):
    if cDat is not None:
        if type(cDat) in [str, tuple, list, dict]:
            if len(cDat) > 0:
                return True
        elif type(cDat) == pd.core.series.Series:
            if cDat.shape[0] > 0:
                return True
        elif type(cDat) in [np.ndarray, pd.core.frame.DataFrame]:
            if cDat.shape[0] > 0 and cDat.shape[1] > 0:
                return True
    return False

# --- String selection and manipulation functions -----------------------------
def joinS(itS, cJ=GC.S_USC):
    if type(itS) in [str, int, float]:    # convert to 1-element list
        itS = [str(itS)]
    if len(itS) == 0:
        return ''
    elif len(itS) == 1:
        return str(itS[0])
    lS2J = [str(s) for s in itS if (s is not None and len(str(s)) > 0)]
    if type(cJ) == str:
        return cJ.join(lS2J).strip()
    elif type(cJ) in [tuple, list, range]:
        sRet = str(itS[0])
        if len(cJ) == 0:
            cJ = [cJ]*(len(itS) - 1)
        else:
            cJ = [str(j) for j in cJ]
        if len(cJ) >= len(itS) - 1:
            for k, sJ in zip(range(1, len(itS)), cJ[:(len(itS) - 1)]):
                sRet = joinS([sRet, str(itS[k])], cJ=sJ)
        else:
            # cJ too short --> only use first element of cJ
            sRet = cJ[0].join(lS2J).strip()
        return sRet

def getPartStr(s, iStart=None, iEnd=None, sSpl=GC.S_USC):
    if iStart is None:
        if iEnd is None:
            return joinS(s.split(sSpl), cJ=sSpl)
        else:
            return joinS(s.split(sSpl)[:iEnd], cJ=sSpl)
    else:
        if iEnd is None:
            return joinS(s.split(sSpl)[iStart:], cJ=sSpl)
        else:
            return joinS(s.split(sSpl)[iStart:iEnd], cJ=sSpl)

def getPartSF(sF, iStart=None, iEnd=None, sDot=GC.S_DOT, sSpl=GC.S_USC):
    sFNoXt = sF
    if sDot in sF:
        sFNoXt = joinS(sF.split(sDot)[:-1], cJ=sDot)
    return getPartStr(s=sFNoXt, iStart=iStart, iEnd=iEnd, sSpl=sSpl)

def extS(s, sXt='', sSep=GC.S_USC):
    if len(sXt) > 0:
        return sSep.join([s, sXt])
    else:
        return s

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

def getSFEnd(sF, nCharEnd):
    if GC.S_DOT in sF:
        sF = joinS(sF.split(GC.S_DOT)[:-1], cJ=GC.S_DOT)[-nCharEnd:]
    return sF[-nCharEnd:]

def addSCentNmerToDict(dNum, dENmer, iCNmer):
    for k in range(iCNmer + 1):
        for sEff, lSNmer in dENmer.items():
            for sNmer in lSNmer:
                sCentNmer = sNmer[(iCNmer - k):(iCNmer + k + 1)]
                addToDictDNum(dNum, sEff, sCentNmer)

def findAllSSubInStr(sFull, sSub, overLap=True):
    i = sFull.find(sSub)
    while i >= 0:
        yield i
        i = sFull.find(sSub, i + (1 if overLap else len(sSub)))

def startPosToCentPos(iPSt, sSeq):
    return iPSt + len(sSeq)//2

def getLCentPosSSub(sFull, sSub, overLap=True):
    return [startPosToCentPos(iSt, sSub)
            for iSt in findAllSSubInStr(sFull, sSub, overLap=overLap)]

def getSClFromCHdr(sCHdr, idxSpl=-1, sSep=GC.S_USC):
    return sSep.join(sCHdr.split(sSep)[:idxSpl])

def getDSClFromCHdr(itCol, idxSpl=-1, sSep=GC.S_USC):
    dSCl = {}
    for sCHdr in itCol:
        dSCl[getSClFromCHdr(sCHdr=sCHdr, idxSpl=idxSpl, sSep=sSep)] = sCHdr
    return dSCl

# --- Functions yielding boolean values ---------------------------------------
def isTrain(doSplit):
    return (True if doSplit else None)

# --- Functions performing calculations with scalars --------------------------
def isEq(x, xCmp, maxDlt=GC.MAX_DELTA):
    return abs(x - xCmp) < maxDlt

def isInSeqSet(x, cSqSet, maxDlt=GC.MAX_DELTA):
    for cEl in cSqSet:
        if isEq(x, cEl, maxDlt=maxDlt):
            return True
    return False

def getFract(cEnum=0, cDenom=1, defFract=0.):
    return (defFract if cDenom == 0 else cEnum/cDenom)

def calcPylProb(cSS, dCSS):
    nFSeq, nPyl, nTtl, cPrb = len(dCSS), 0, 0, 0.
    for tV in dCSS.values():
        nPyl += tV[0]
        nTtl += tV[1]
        cPrb += tV[3]/nFSeq
    return [cSS, nPyl, nTtl, getFract(nPyl, nTtl), cPrb]

# --- Functions handling iterables --------------------------------------------
def allTrue(cIt):
    return [cB for cB in cIt] == [True]*len(cIt)

def toStr(cIt, sJoin=GC.S_USC, modStr=False):
    if ((type(cIt) in [list, tuple, range, set, dict]) or
        (type(cIt) == str and modStr)):
        return sJoin.join([str(x) for x in cIt])
    elif type(cIt) == str and not modStr:
        return cIt
    else:
        return str(cIt)

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

def getItStartToEnd(cIt, iStart=None, iEnd=None):
    if iStart is not None:
        iStart = max(iStart, 0)
    else:
        iStart = 0
    if iEnd is not None:
        iEnd = min(iEnd, len(cIt))
    else:
        iEnd = len(cIt)
    return cIt[iStart:iEnd], iStart, iEnd

def restrIt(cIt, lRestrLen=[], useLenS=False):
    if len(lRestrLen) > 0:
        if type(cIt) in [list, tuple]:
            if useLenS:     # use string length (assuming s is a string)
                lRet = [s for s in cIt if len(s) in lRestrLen]
            else:           # use element directly (assuming k is an int)
                lRet = [k for k in cIt if k in lRestrLen]
            if type(cIt) == tuple:
                return tuple(lRet)
            else:
                return lRet
        elif type(cIt) == dict:
            if useLenS:     # use string length (assuming sK is a string)
                return {sK: v for sK, v in cIt.items() if len(sK) in lRestrLen}
            else:           # use element directly (assuming cK is an int)
                return {cK: v for cK, v in cIt.items() if cK in lRestrLen}
    else:
        return cIt

def restrInt(cIt, lRestrLen=[]):
    return restrIt(cIt=cIt, lRestrLen=lRestrLen)

def restrLenS(cIt, lRestrLen=[]):
    return restrIt(cIt=cIt, lRestrLen=lRestrLen, useLenS=True)

def tryRoundX(x, RD=None):
    if RD is not None:
        try:
            x = round(x, RD)
        except:
            print('Cannot round "', x, '" to ', RD, ' digits.', sep='')
    return x

# --- Functions handling lists ------------------------------------------------
def insStartOrEnd(itS, cEl, sPos=GC.S_E):
    if type(itS) in [str, int, float]:    # convert to 1-element list
        itS = [str(itS)]
    if sPos == GC.S_S:
        return [str(cEl)] + [s for s in itS]
    elif sPos == GC.S_E:
        return [s for s in itS] + [str(cEl)]
    else:
        return [s for s in itS]

def convToL(itTest, itRef):
    if type(itTest) in [list, tuple]:
        assert len(itTest) >= len(itRef)
        return itTest[:len(itRef)]
    else:
        return [itTest]*len(itRef)

def fillCondList(elCond, lToFill=[], lLoop=[], lUnqEl=True):
    for elCheckContain in lLoop:
        if elCond in elCheckContain:
            if not lUnqEl or elCond not in lToFill:
                lToFill.append(elCond)
            break

def fillLValSnip(lValSnip, lIdxPos=[], lIdxPyl=[]):
    if len(lIdxPos) > 0:
        lB = [(1 if iPos in lIdxPyl else 0) for iPos in lIdxPos]
        nPyl, nOcc = sum(lB), len(lB)
        lValSnip[0] += nPyl
        lValSnip[1] += nOcc
        lValSnip[3] += nPyl/nOcc
        return 1
    return 0

def addToList(cL, cEl, isUnq=False):
    if isUnq:
        if cEl not in cL:
            cL.append(cEl)
    else:
        cL.append(cEl)

def addToListUnq(cL, cEl):
    addToList(cL=cL, cEl=cEl, isUnq=True)

def fillListUnique(cL=[], cIt=[]):
    for cEl in cIt:
        if cEl not in cL:
            cL.append(cEl)

def toListUnique(cIt=[]):
    cLUnq = []
    for cEl in cIt:
        if cEl not in cLUnq:
            cLUnq.append(cEl)
    return cLUnq

def toListUnqViaSer(cIt=None):
    if cIt is not None:
        return toSerUnique(pd.Series(cIt)).to_list()
    return None

def getListUniqueWD2(d2, srtDir=None):
    lUnq = []
    for cDSub in d2.values():
        fillListUnique(cL=lUnq, cIt=cDSub)
    if srtDir is not None:
        if srtDir == 'asc':
            return sorted(lUnq, reverse=False)
        elif srtDir == 'desc':
            return sorted(lUnq, reverse=True)
    return lUnq

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

def iniD2(itHdL1, itHdL2, fillV=np.nan):
    d2 = {cKL1: None for cKL1 in itHdL1}
    for cKL1 in d2:
        d2[cKL1] = {cKL2: fillV for cKL2 in itHdL2}
    return d2

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

def addToDictCt(cD, cK, nInc=1):
    if cK in cD:
        cD[cK] += nInc
    else:
        cD[cK] = nInc

def addToDictMnV(cD, cK, cEl, nEl):
    assert nEl > 0
    if cK in cD:
        cD[cK] += cEl/nEl
    else:
        cD[cK] = cEl/nEl

def addToDictL(cD, cK, cE, lUnqEl=False):
    if cK in cD:
        if not lUnqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def addToDict2L(cD, cKMain, cKSub, cE, lUnqEl=False):
    if cKMain not in cD:
        cD[cKMain] = {}
    addToDictL(cD[cKMain], cKSub, cE, lUnqEl=lUnqEl)

def addToDictDNum(d2N, cKMain, cKSub, nInc=1, bothDir=False):
    if bothDir and cKSub[::-1] < cKSub:    # reverse cKSub string
        cKSub = cKSub[::-1]
    if cKMain in d2N:
        if cKSub in d2N[cKMain]:
            d2N[cKMain][cKSub] += nInc
        else:
            d2N[cKMain][cKSub] = nInc
    else:
        d2N[cKMain] = {cKSub: nInc}

def addToDictDSpc(cDMain, cKMain, cKSub, cVSub=None):
    if cDMain[cKMain] == None:
        cDMain[cKMain] = {cKSub: cVSub}
    elif type(cDMain[cKMain]) == dict:
        cDMain[cKMain][cKSub] = cVSub

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

def addToD3(cD3, cKL1, cKL2, cKL3, cVL3=[], allowRpl=False):
    if cKL1 in cD3:
        addToDictD(cD=cD3[cKL1], cKMain=cKL2, cKSub=cKL3, cVSub=cVL3,
                   allowRpl=allowRpl)
    else:
        cD3[cKL1] = {cKL2: {cKL3: cVL3}}

def updateDict(cDFull, cDUp, cK=None):
    if cK is None:
        cDFull.update(cDUp)
    else:
        if cK in cDFull:
            cDFull[cK].update(cDUp)
        else:
            cDFull[cK] = cDUp

def getMeansD2Val(d2):
    dMn = {}
    for d in d2.values():
        for cK, cV in d.items():
            addToDictL(dMn, cK=cK, cE=cV)
    return {cK: np.mean(cL) for cK, cL in dMn.items()}

def calcMnSEMOfDfrs(itDfrs, idxDfr=None, colDfr=None):
    d2Mn, d2SEM, N = {}, {}, len(itDfrs)
    for i in idxDfr:
        for j in colDfr:
            for cDfr in itDfrs:
                assert i in cDfr.index and j in cDfr.columns
            cLV = [cDfr.at[i, j] for cDfr in itDfrs]
            addToDictD(d2Mn, cKMain=i, cKSub=j, cVSub=np.mean(cLV))
            cSEM = (0. if N == 0 else np.std(cLV, ddof=1)/np.sqrt(N))
            addToDictD(d2SEM, cKMain=i, cKSub=j, cVSub=cSEM)
    return iniPdDfr(d2Mn), iniPdDfr(d2SEM)

def calcMnSEMFromD3Val(cD3, sJoin=GC.S_USC):
    d2MnSEM, dT, N = {}, {}, len(cD3)
    # rearrange data in temp dict dT
    for cD2 in cD3.values():
        for cKL2, cD in cD2.items():
            for cKL3, cV in cD.items():
                addToDictL(dT, cK=(cKL2, cKL3), cE=cV)
    # fill d2MnSEM
    for cKT, cLV in dT.items():
        addToDictD(d2MnSEM, cKMain=sJoin.join([toStr(cKT[0]), GC.S_MEAN]),
                   cKSub=cKT[1], cVSub=np.mean(cLV))
        cSEM = (0. if N == 0 else np.std(cLV, ddof=1)/np.sqrt(N))
        addToDictD(d2MnSEM, cKMain=sJoin.join([toStr(cKT[0]), GC.S_SEM]),
                   cKSub=cKT[1], cVSub=cSEM)
    return d2MnSEM

def calcMnSEMFromD2Dfr(d2Dfr, sJoin=GC.S_USC):
    dMnSEM, dT, lI = {}, {}, None
    # rearrange data in temp dict dT
    for dDfr in d2Dfr.values():
        for cKL2, cDfr in dDfr.items():
            if lI is None:
                lI = cDfr.index.to_list()
            assert lI == cDfr.index.to_list() and lI == cDfr.columns.to_list()
            addToDictL(dT, cK=cKL2, cE=cDfr)
    # fill dMnSEM
    if lI is not None:
        for cK, cLDfr in dT.items():
            dfrMn, dfrSEM = calcMnSEMOfDfrs(cLDfr, idxDfr=lI, colDfr=lI)
            dMnSEM[sJoin.join([toStr(cK), GC.S_MEAN])] = dfrMn
            dMnSEM[sJoin.join([toStr(cK), GC.S_SEM])] = dfrSEM
    return dMnSEM

def convDDNumToDDProp(dDNum, nDigRnd):
    dDSum, dDProp = {}, {}
    for cKM, cDSub in dDNum.items():
        for cKS, cNum in cDSub.items():
            addToDictDNum(dDSum, cKMain=cKM, cKSub=len(cKS), nInc=cNum)
    for cKM, cDSub in dDNum.items():
        dDProp[cKM] = {}
        for cKS, cNum in cDSub.items():
            cProp = round(dDNum[cKM][cKS]/dDSum[cKM][len(cKS)], nDigRnd)
            dDProp[cKM][cKS] = cProp
    return dDProp

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

def dLV3CToDfr(dLV):
    return pd.DataFrame(dLV)

def d3ValToDfr(d3V, lSCol, minLenL=5):
    assert len(lSCol) >= minLenL
    dLV = {}
    for cKMain, cDSub1 in d3V.items():
        for cKSub1, cDSub2 in cDSub1.items():
            for cKSub2, cV in cDSub2.items():
                lElRow = [cKMain, cKSub1, cKSub2, len(cKSub2), cV]
                for sC, cV in zip(lSCol[:minLenL], lElRow):
                    addToDictL(dLV, cK=sC, cE=cV)
    return pd.DataFrame(dLV)

def dDDfrToDfr(dDDfr, lSColL, lSColR):
    fullDfr = iniPdDfr(lSNmC=lSColL+lSColR)
    for sKMain, cDSub in dDDfr.items():
        for sKSub, rightDfr in cDSub.items():
            leftDfr = iniPdDfr(lSNmR=rightDfr.index, lSNmC=lSColL)
            leftDfr[lSColL[:len(sKMain)]] = sKMain
            leftDfr[lSColL[-len(sKSub):]] = sKSub
            subDfr = pd.concat([leftDfr, rightDfr], axis=1)
            fullDfr = pd.concat([fullDfr, subDfr], axis=0)
    return fullDfr.reset_index(drop=True).convert_dtypes()

def fillDNOcc(dfrNOcc, dNOcc, sNmer=GC.S_N_MER, sLenNmer=GC.S_LEN_N_MER):
    lNmer, lLenNmer = list(dfrNOcc[sNmer]), list(dfrNOcc[sLenNmer])
    assert len(lNmer) == len(lLenNmer)
    for sNmer, cLenNmer in zip(lNmer, lLenNmer):
        addToDictL(dNOcc, cLenNmer, sNmer)

def toDfr(cData, idxDfr=None, colDfr=None):
    cDfr = None
    if type(cData) == pd.core.frame.DataFrame:      # the standard case
        cDfr = cData
    elif type(cData) == pd.core.series.Series:
        cDfr = cData
    elif type(cData) == dict:
        cDfr = iniPdDfr(cData, lSNmC=colDfr, lSNmR=idxDfr)
    elif type(cData) == np.ndarray and len(cData.shape) <= 2:
        cDfr = iniPdDfr(cData, lSNmC=colDfr, lSNmR=idxDfr)
    return cDfr

def dictToDfr(cD, idxDfr=None, dropNA=False, dropHow='any', srtBy=None,
              srtAsc=None):
    if dropNA:
        pdDfr = pd.DataFrame(cD, index=idxDfr).dropna(axis=0, how=dropHow)
    else:
        pdDfr = pd.DataFrame(cD, index=idxDfr)
    if srtBy is not None:
        if srtAsc is not None:
            pdDfr.sort_values(axis=0, by=srtBy, ascending=srtAsc, inplace=True,
                              ignore_index=True)
        else:
            pdDfr.sort_values(axis=0, by=srtBy, inplace=True,
                              ignore_index=True)
    return pdDfr

def fillDProbSnip(dProb, lSeq, sSnip):
    lV, n = [0, 0, 0., 0.], 0
    for cSeq in lSeq:
        lIPos = getLCentPosSSub(sFull=cSeq.sSeq, sSub=sSnip)
        n += fillLValSnip(lV, lIdxPos=lIPos, lIdxPyl=cSeq.lIPyl)
    if n > 0:
        dProb[sSnip] = [lV[0], lV[1], lV[0]/lV[1], lV[-1]/n]
    else:
        dProb[sSnip] = lV

def appendToDictL(cD, itKeys, lVApp):
    for k, cKey in enumerate(itKeys):
        cD[cKey].append(lVApp[k])

def genDictFromD2PI(d2PI, cK, modPI=False):
    cD = d2PI[cK]
    if not modPI:
        cD = {sK: cV for sK, cV in d2PI[cK].items()}
    return cD

# --- Functions performing numpy array calculation and manipulation -----------
def getArrCartProd(it1, it2):
    return np.array(list(itertools.product(it1, it2)))

def iniNpArr(data=None, shape=(0, 0), fillV=np.nan):
    if data is None:
        return np.full(shape, fillV)
    else:       # ignore shape
        return np.array(data)

def getYProba(cClf, dat2Pr=None, lSC=None, lSR=None, i=0):
    if hasattr(cClf, 'predict_proba'):
        arrProba = cClf.predict_proba(dat2Pr)
    else:
        return None
    if type(arrProba) == list:    # e.g. result of a random forest classifier
        return iniPdDfr(np.column_stack([1 - cArr[:, i] for cArr in arrProba]),
                        lSNmC=lSC, lSNmR=lSR)
    elif type(arrProba) == np.ndarray:      # standard case
        return iniPdDfr(arrProba, lSNmC=lSC, lSNmR=lSR)

# --- Functions performing pandas Series/DataFrame operations -----------------
def concLO(lObj, concAx=0, ignIdx=False, verifInt=False, srtDfr=False):
    return pd.concat(lObj, axis=concAx, ignore_index=ignIdx,
                     verify_integrity=verifInt, sort=srtDfr)

def concLOAx0(lObj, ignIdx=False, verifInt=False, srtDfr=False):
    return concLO(lObj, ignIdx=ignIdx, verifInt=verifInt, srtDfr=srtDfr)

def concLOAx1(lObj, ignIdx=False, verifInt=False, srtDfr=False):
    return concLO(lObj, concAx=1, ignIdx=ignIdx, verifInt=verifInt,
                  srtDfr=srtDfr)

def makeLblUnq(itLbl, getDict=False, sJoin=GC.S_USC):
    dCtLbl, lLblUnq, dLbl2Unq = {}, [], {}
    for cLbl in itLbl:
        addToDictCt(dCtLbl, cK=cLbl)
    for cK, cCt in dCtLbl.items():
        dCtLbl[cK] = [cCt, cCt]
    for cLbl in itLbl:
        n, N = dCtLbl[cLbl][0], dCtLbl[cLbl][1]
        sLblUnq = sJoin.join([str(cLbl), str(N - n)])
        if getDict:
            addToDictL(dLbl2Unq, cK=cLbl, cE=sLblUnq)
        else:
            lLblUnq.append(sLblUnq)
        dCtLbl[cLbl][0] -= 1
    if getDict:
        return dLbl2Unq
    else:
        return lLblUnq

def dropLblUnq(itLbl, sJoin=GC.S_USC):
    lLbl, lLblRed = [], [sJoin.join(cLbl.split(sJoin)[:-1]) for cLbl in itLbl]
    fillListUnique(cL=lLbl, cIt=lLblRed)
    return lLbl

def getMaxC(cIt, thrV=None):
    if pd.isna(cIt).all():
        # return pd.Series([np.nan for _ in cIt])
        return [np.nan for _ in cIt]
    lRet, maxIt = [0 for _ in cIt], max(cIt)
    for k, x in enumerate(cIt):
        if x is None or np.isnan(x):
            lRet[k] = np.nan
        if x == maxIt:
            if (thrV is None) or (thrV is not None and x > thrV):
                lRet[k] = 1
    # return pd.Series(lRet)
    return lRet

def getRelVals(cIt):
    if pd.isna(cIt).all():
        # return pd.Series([np.nan for _ in cIt])
        [np.nan for _ in cIt]
    lenIt, lRet = len(cIt), []
    if lenIt > 0:
        lRet = [x/lenIt for x in cIt]
    # return pd.Series(lRet)
    return lRet

def getPredCl(cD, sRetNoCl=GC.S_NONE, sRetMltCl=GC.S_MULTI_CL):
    if pd.isna(pd.Series(cD.values())).all():
        return np.nan
    cIt1 = [n for n in cD.values() if n == 1]
    if len(cIt1) > 1:
        return sRetMltCl
    elif len(cIt1) == 1:
        return [sCl for sCl in cD if cD[sCl] == 1][0]
    else:
        return sRetNoCl

def classifyTP(cIt):        # works only if always exactly one true class
    if pd.isna(cIt).all():
        return np.nan
    sRet, cIt = np.nan, [x for x in cIt if not pd.isna(x)]
    lenIt, minIt, maxIt = len(cIt), min(cIt), max(cIt)
    if maxIt not in {1, 2}:
        return np.nan
        # print('ERROR: cIt:\n', cIt, '\nminIt = ', minIt, '\nmaxIt = ',
        #       maxIt, sep='')
        # assert False
    lVRem = [cV for cV in cIt if (cV < maxIt)]
    if len(lVRem) > 0:
        maxVRem = max(lVRem)
        if maxIt == 2:
            if maxVRem == 0:
                sRet = GC.S_A               # A: pred. cl. == true cl. (only)
            elif maxVRem == 1 and minIt == 1:
                sRet = GC.S_D               # D: all classes pred.
            elif maxVRem == 1 and minIt < 1:
                sRet = GC.S_B               # B: pred. cl. == true cl. (+ oth.)
        elif maxIt == 1:
            # assert maxVRem == 0
            if maxVRem != 0:
                print('ERROR: cIt:\n', cIt, '\nlVRem:\n', lVRem,
                      '\nmaxVRem = ', maxVRem)
                assert False
            if len(lVRem) == lenIt - 1:
                sRet = GC.S_C               # C: no class pred.
            elif len(lVRem) < lenIt - 1:
                sRet = GC.S_E               # E: pred. class(es) != true cl.
    else:
        assert minIt == 1 and maxIt == 1
        sRet = GC.S_E                       # E: pred. class(es) != true cl.
    return sRet

def red2TpStr(cR):
    l = [x for x in cR if type(x) == str]
    if len(l) == 0:
        return np.nan
    else:
        return l[0]

def calcMeanItO(itO, ignIdx=False, srtDfr=False, lKMn=[]):
    lSer, lKDfr = [], []
    if type(itO) == dict:
        itO = itO.values()
    fullObj = concLOAx1(itO, ignIdx=ignIdx, verifInt=False, srtDfr=srtDfr)
    dLbl = makeLblUnq(itLbl=fullObj.columns, getDict=True)
    fullObj.columns = makeLblUnq(itLbl=fullObj.columns)
    for cK in dLbl:
        lKDfr.append(cK)
        if cK in lKMn:
            redObj = fullObj[dLbl[cK]].mean(axis=1)
        else:
            redObj = fullObj[dLbl[cK]].apply(red2TpStr, axis=1)
        lSer.append(redObj)
    retDfr = concLOAx1(lSer, ignIdx=ignIdx, verifInt=True, srtDfr=srtDfr)
    retDfr.columns = lKDfr
    return retDfr

def toSerUnique(pdSer, sName=None):
    nameS = pdSer.name
    if sName is not None:
        nameS = sName
    return pd.Series(pdSer.unique(), name=nameS)

def iniPdSer(data=None, lSNmI=[], shape=(0,), nameS=None, fillV=np.nan):
    assert (((type(data) == np.ndarray and len(data.shape) == 1) or
             (type(data) in [list, tuple]) or (data is None))
            and (len(shape) == 1))
    if lSNmI is None or len(lSNmI) == 0:
        if data is None:
            return pd.Series(np.full(shape, fillV), name=nameS)
        else:
            return pd.Series(data, name=nameS)
    else:
        if data is None:
            return pd.Series(np.full(len(lSNmI), fillV), index=lSNmI,
                             name=nameS)
        else:
            assert ((type(data) == np.ndarray and data.shape[0] == len(lSNmI))
                    or (type(data) == list and len(data) == len(lSNmI)))
            return pd.Series(data, index=lSNmI, name=nameS)

def iniPdDfr(data=None, lSNmC=[], lSNmR=[], shape=(0, 0), fillV=np.nan):
    assert len(shape) == 2
    nR, nC = shape
    if lSNmC is None or len(lSNmC) == 0:
        if lSNmR is None or len(lSNmR) == 0:
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
        if lSNmR is None or len(lSNmR) == 0:
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
    return iniPdDfr(shape=tmplDfr.shape, fillV=fillV)

def dictSglKey2Ser(cD):
    if len(cD) == 1:
        return pd.Series(list(cD.values())[0], name=list(cD.keys())[0])
    return None

def dictLUneqLen2Dfr(cD, itIdx=None, itCol=None, fillV=np.nan, doTrans=False):
    maxLenL = max([len(l) for l in cD.values()])
    cDX = {cK: (cL + [fillV]*(maxLenL - len(cL))) for cK, cL in cD.items()}
    if doTrans:
        dfrOut = pd.DataFrame(cDX).T
    else:
        dfrOut = pd.DataFrame(cDX)
    # if given and of correct size, set index and/or columns
    if (itIdx is not None) and (len(itIdx) == dfrOut.index.size):
        dfrOut.index = itIdx
    if (itCol is not None) and (len(itCol) == dfrOut.columns.size):
        dfrOut.columns = itCol
    return dfrOut

def getIdxDfr(maxLen=0, idxDfr=None):
    if idxDfr is not None and len(idxDfr) >= maxLen:
        idxDfr = idxDfr[:maxLen]
    else:
        idxDfr = None
    return idxDfr

def getColDfr(nLvl=1, colDfr=None):
    nCol = nLvl + 1    # nLvl dict levels plus 1 level for the sub-dict values
    if colDfr is None or len(colDfr) < nCol:
        colDfr = range(nCol)
    return nCol, colDfr

def extractLHdNumColDDfr(dDfr, inAll=True):
    dHdCnt, n = {}, len(dDfr)
    for cDfr in dDfr.values():
        for sCHd in cDfr.columns:
            if is_numeric_dtype(cDfr[sCHd]):
                addToDictCt(dHdCnt, cK=sCHd)
    if inAll:
        return [sHd for sHd in dHdCnt if dHdCnt[sHd] == n]
    else:
        return list(dHdCnt)

def addToDfrSpc(d2, pdDfr=None, concAx=1):
    if pdDfr is None:
        pdDfr = iniPdDfr(d2)
    else:
        pdDfr = concPdDfrS(lPdDfr=[pdDfr, iniPdDfr(d2)], concAx=concAx)
    return pdDfr

def iniDfrFromDictIt(cD, idxDfr=None):
    maxLen = 0
    for cK, cIt in cD.items():
        if len(cIt) > maxLen:
            maxLen = len(cIt)
    for cK, cIt in cD.items():
        cD[cK] = list(cIt) + [np.nan]*(maxLen - len(cIt))
    idxDfr = getIdxDfr(maxLen=maxLen, idxDfr=idxDfr)
    return pd.DataFrame(cD, index=idxDfr)

def iniDfrFromD2(d2, idxDfr=None, colDfr=None):
    nCol, colDfr = getColDfr(nLvl=2, colDfr=colDfr)
    lLenL2 = [len(cD) for cD in d2.values()]
    dLV = {sC: None for sC in colDfr[:nCol]}
    lVC0 = flattenIt([[cKMain]*lLenL2[k] for k, cKMain in enumerate(d2)])
    lVC1 = flattenIt([list(cD) for cD in d2.values()])
    lVC2 = flattenIt([list(cD.values()) for cD in d2.values()])
    for sC, lV in zip(dLV, [lVC0, lVC1, lVC2]):
        dLV[sC] = lV
    return iniDfrFromDictIt(dLV, idxDfr=idxDfr)

def iniDfrFromD3(d3, idxDfr=None, colDfr=None):
    nCol, colDfr = getColDfr(nLvl=3, colDfr=colDfr)
    fullDfr = pd.DataFrame(columns=colDfr)
    for cKL1, cDL1 in d3.items():
        cDfr = iniDfrFromD2(cDL1, colDfr=colDfr[1:])
        cSer = pd.Series([cKL1]*cDfr.shape[0], name=colDfr[0])
        cDfr = pd.concat([cSer, cDfr], axis=1, verify_integrity=True)
        fullDfr = pd.concat([fullDfr, cDfr], axis=0)
    idxDfr = getIdxDfr(maxLen=fullDfr.shape[0], idxDfr=idxDfr)
    if idxDfr is not None:
        fullDfr.index = idxDfr
    return fullDfr.reset_index(drop=True)

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
def drawListInt(minInt=0, maxInt=1, nIntDrawn=1, wRepl=True, sortL=False):
    minInt, maxInt = min(minInt, maxInt), max(minInt, maxInt)
    lI = list(RNG().choice(maxInt - minInt, size=nIntDrawn, replace=wRepl))
    if minInt != 0:
        lI = [k + minInt for k in lI]
    if sortL:
        lI.sort()
    return lI

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

# --- Functions implementing custom imbalanced sampling strategies ------------
def smplStratRealMajo(Y):
    sStrat = {cCl: (Y[Y == cCl].size) for cCl in Y.unique()}
    lVSrtDsc, n1, n2 = sorted(sStrat.values(), reverse=True), 0, 0
    if len(lVSrtDsc) >= 2:
        n1, n2 = lVSrtDsc[0], lVSrtDsc[1]
    elif len(lVSrtDsc) == 1:
        n1, n2 = lVSrtDsc[0], lVSrtDsc[0]
    for cCl, nEl in sStrat.items():
        if nEl == n1:
            sStrat[cCl] = n2
    return sStrat

def smplStratShareMino(Y, dI):
    shMino = (dI[GC.S_STRAT_SHARE_MINO] if GC.S_STRAT_SHARE_MINO in dI else 1)
    sStrat = {cCl: (Y[Y == cCl].size) for cCl in Y.unique()}
    lVSrtAsc, n = sorted(sStrat.values()), 0
    if len(lVSrtAsc) >= 1:
        n = round(lVSrtAsc[0]*shMino)
    for cCl, nEl in sStrat.items():
        sStrat[cCl] = max(1, n)
    return sStrat

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
    return startTime

def printElapsedTimeSim(stT=None, cT=None, sPre='Time', nDig=GC.R04):
    if stT is not None and cT is not None:
        # calculate and display elapsed time
        elT = round(cT - stT, nDig)
        print(sPre, 'elapsed:', elT, 'seconds, this is', round(elT/60, nDig),
              'minutes or', round(elT/3600, nDig), 'hours or',
              round(elT/(3600*24), nDig), 'days.')

def showElapsedTime(startTime=None):
    cTime = time.time()
    if startTime is not None:
        print(GC.S_DS80)
        printElapsedTimeSim(startTime, cTime, 'Time')
        print(GC.S_SP04 + 'Current time:', time.ctime(cTime), GC.S_SP04)
        print(GC.S_DS80)
    return cTime

def showProgress(N, n=0, modeDisp=1, varText='', startTime=None, showT=False):
    if (n + 1)%modeDisp == 0:
        print('Processed ', n + 1, ' of ', N, ' ', varText, ' (',
              round((n + 1)/N*100., GC.R02), '%).', sep='')
        if showT:
            showElapsedTime(startTime=startTime)

def endSimu(startTime=None):
    if startTime is not None:
        print(GC.S_DS80)
        printElapsedTimeSim(startTime, time.time(), 'Total time')
        print(GC.S_ST24 + ' DONE', time.ctime(time.time()), GC.S_ST25)

###############################################################################