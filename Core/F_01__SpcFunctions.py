# -*- coding: utf-8 -*-
###############################################################################
# --- F_01__SpcFunctions.py ---------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

# --- Functions (O_00__BaseClass) ---------------------------------------------

# --- Functions (O_01__ExpData) -----------------------------------------------
def getSerTEff(dITp, dfrK, sMd=GC.S_X_SHORT):
    dEffTarg, serTEff = {}, []
    if sMd in [GC.S_X_SHORT, GC.S_SHORT, GC.S_MED]:
        serKEff = dfrK[dITp['sEffCode']]
        serTEff = serKEff.apply(lambda x: (x,))
    elif sMd == GC.S_LONG:
        dfrKEff = dfrK[[dITp['sEffCode'], dITp['sEffSeq']]]
        serTEff = dfrKEff.apply(lambda x: tuple(x), axis=1)
    return dEffTarg, serTEff

def getSerTTarg(dITp, dfrK, tSE, sMd=GC.S_X_SHORT):
    if sMd in [GC.S_X_SHORT, GC.S_SHORT, GC.S_MED]:
        dfrE = dfrK[dfrK[dITp['sEffCode']] == tSE[0]]
        serTTarg = dfrE[dITp['sTargCode']].apply(lambda x: (x,))
        dT = {t: t for t in serTTarg}
    elif sMd == GC.S_LONG:
        dfrE = dfrK[(dfrK[dITp['sEffCode']] == tSE[0]) &
                    (dfrK[dITp['sEffSeq']] == tSE[1])]
        serTDfrE = dfrE.apply(lambda x: tuple(x), axis=1)
        dfrKTarg = dfrE[[dITp['sTargCode'], dITp['sTargSeq']]]
        serTTarg = dfrKTarg.apply(lambda x: tuple(x), axis=1)
        dT = {t2: t6 for t2, t6 in zip(serTTarg, serTDfrE)}
    return serTTarg, dT

def createDEffTarg(dITp, dfrK, dfrNmer, lCDfrNmer, sMd=GC.S_X_SHORT):
    dEffTarg, serTEff = getSerTEff(dITp, dfrK, sMd=sMd)
    for tSE in serTEff.unique():
        dEffTarg[tSE] = {}
        serTTarg, dT = getSerTTarg(dITp, dfrK, tSE, sMd=sMd)
        for tST in serTTarg.unique():
            dfrT = dfrNmer[dfrNmer[dITp['sCodeTrunc']] == tST[0]]
            dEffTarg[tSE][dT[tST]] = dfrT[lCDfrNmer]
    return dEffTarg

def dDNumToDfrINmer(dITp, dDNum):
    lSCol = dITp['lSCDfrNmer']
    assert len(lSCol) >= 4
    fullDfr = GF.iniPdDfr(lSNmC=lSCol)
    for sKMain, cDSub in dDNum.items():
        cDSubR = GF.restrLenS(cDSub, lRestrLen=dITp['lLenNmer'])
        subDfr = GF.iniPdDfr(lSNmC=lSCol)
        subDfr[lSCol[0]] = [sKMain]*len(cDSubR)
        subDfr[lSCol[1]] = list(cDSubR)
        subDfr[lSCol[2]] = [len(sK) for sK in cDSubR]
        subDfr[lSCol[3]] = list(cDSubR.values())
        fullDfr = pd.concat([fullDfr, subDfr], axis=0)
    return fullDfr.reset_index(drop=True).convert_dtypes()

def dDNumToDfrIEff(dITp, dDNum, wAnyEff=False):
    d4Dfr, sEffCode, sAnyEff = {}, dITp['sEffCode'], dITp['sAnyEff']
    dDPrp = GF.convDDNumToDDProp(dDNum=dDNum, nDigRnd=dITp['rndDigProp'])
    assert sAnyEff in dDPrp
    lSEff, lAllSNmer = [s for s in dDPrp if s != sAnyEff], list(dDPrp[sAnyEff])
    if wAnyEff:
        lSEff = list(dDPrp)
    lAllSNmer = GF.restrLenS(lAllSNmer, lRestrLen=dITp['lLenNmer'])
    lAllSNmer.sort(key=(lambda x: len(x)))
    d4Dfr[sEffCode] = lSEff
    for sNmer in lAllSNmer:
        d4Dfr[sNmer] = [0]*len(lSEff)
    for sNmer in lAllSNmer:
        for k, sEff in enumerate(lSEff):
            if sNmer in dDPrp[sEff]:
                d4Dfr[sNmer][k] = dDPrp[sEff][sNmer]
    return pd.DataFrame(d4Dfr).convert_dtypes()

# --- Functions (O_02__SeqAnalysis) -------------------------------------------
def getLSFullSeq(dITp, dfrInpSeq, iS=None, iE=None, unqS=True):
    if unqS:
        lSFullSeq = GF.toListUnqViaSer(dfrInpSeq[dITp['sCCodeSeq']])
    else:
        lSFullSeq = list(dfrInpSeq[dITp['sCCodeSeq']])
    return GF.getItStartToEnd(lSFullSeq, iS, iE)

def getLSNmer(dITp, dfrInpSeq, lSFull=[], lSNmer=[], red2WF=True, unqS=True):
    if lSNmer is None and dITp['sCNmer'] in dfrInpSeq.columns:
        dfrCSeq = dfrInpSeq
        if (dITp['sCCodeSeq'] in dfrInpSeq.columns and red2WF):
            # reduce to Nmer sequences with corresponding full sequences
            dfrCSeq = dfrInpSeq[dfrInpSeq[dITp['sCCodeSeq']].isin(lSFull)]
        if unqS:
            lSNmer = GF.toListUnqViaSer(dfrCSeq[dITp['sCNmer']])
        else:
            lSNmer = list(dfrCSeq[dITp['sCNmer']])
    return lSNmer

def modLSF(dITp, lSKeyF, sFE):
    for sKeyF in lSKeyF:
        if not GF.getSFEnd(sF=dITp[sKeyF], nCharEnd=len(sFE)) == sFE:
            dITp[sKeyF] = GF.modSF(dITp[sKeyF], sEnd=sFE, sJoin=dITp['sUS02'])

def calcDictLikelihood(dITp, dLV, d3, dSqProfile, serLh, mxLSnip, cSSq, cEff):
    dLh, wtLh, lSCWtLh = {}, 0., dITp['lSCDfrLhV']
    assert len(lSCWtLh) >= 3
    for lenSnip, sSnip in dSqProfile.items():
        if lenSnip <= mxLSnip:
            cLh = (serLh.at[sSnip] if sSnip in serLh.index else 0.)
            dLh[sSnip] = cLh
            if lenSnip in dITp['dWtsLenSeq']:
                wtLh += cLh*dITp['dWtsLenSeq'][lenSnip]
    if dITp['calcRelLh']:
        GF.addToDictD(d3, cKMain=cSSq, cKSub=cEff, cVSub=dLh)
    if dITp['calcWtLh']:
        for sC, cV in zip(lSCWtLh[:3], [cSSq, cEff, wtLh]):
            GF.addToDictL(dLV, cK=sC, cE=cV)

# --- Functions (O_06__ClfDataLoader) -----------------------------------------
def getCentSNmerDefLen(dITp, sSeq):
    iStart = dITp['iCentNmer'] - dITp['maxPosNmer']
    iEnd = dITp['iCentNmer'] + dITp['maxPosNmer'] + 1
    return sSeq[iStart:iEnd]

def getDSqNoCl(dITp, serFullSeqUnq=[], lNmerSeqUnq=None):
    dSnip, iCA, dRst = {}, dITp['maxPosNmer'], dITp['dAAcPosRestr']
    for sFullSeq in serFullSeqUnq:
        lenSeq = len(sFullSeq)
        if lenSeq >= dITp['maxLenNmer']:
            for iCentC in range(iCA, lenSeq - iCA):
                sSnip = sFullSeq[(iCentC - iCA):(iCentC + iCA + 1)]
                if GF.allTrue([sSnip[iPR + iCA] in lAAc for iPR, lAAc in
                               dRst.items()]):
                    sAAc = ''.join([sSnip[iPR + iCA] for iPR in dRst])
                    if lNmerSeqUnq is None or sSnip not in lNmerSeqUnq:
                        GF.addToDictL(dSnip, cK=sAAc, cE=sSnip, lUnqEl=False)
    return {cK: GF.toListUnqViaSer(cL) for cK, cL in dSnip.items()}

def filterNmerSeq(dITp, dSeq={}, serSeq=None):
    dSeqFilt, iCentAdj = dSeq, dITp['maxPosNmer']
    if dITp['dAAcPosRestr'] is not None and serSeq is not None:
        minLen = min([len(sSeq) for sSeq in serSeq])
        for iP, lAAc in dITp['dAAcPosRestr'].items():
            iSeq = iP + iCentAdj
            if iSeq >= 0 and iSeq < minLen:
                lB = [(sSeq[iSeq] in lAAc) for sSeq in serSeq]
                serSeq = serSeq[lB]
        dSeqFilt = {sSeq: dSeq[sSeq] for sSeq in serSeq}
    return dSeqFilt, serSeq

def getClassStrOldCl(dITp, lSCl=[]):
    setSDigC, n, sClOut = set(), max([len(sCl) for sCl in lSCl]), dITp['sC']
    # step 1: fill setSDigC
    for sCl in lSCl:
        for cChar in sCl:
            if cChar in dITp['setSDig'] and cChar not in setSDigC:
                setSDigC.add(cChar)
    # step 2: assemble sClOut
    for i in range(1, n):
        if str(i) in setSDigC:
            sClOut += str(i)
        else:
            sClOut += dITp['sDash']
    return sClOut

def getClassStr(dITp, lSCl=[], sMd=GC.S_CLF):
    if dITp['usedClType' + sMd].startswith(GC.S_OLD):
        return getClassStrOldCl(dITp, lSCl=lSCl)
    elif dITp['usedClType' + sMd].startswith(GC.S_NEW):
        return dITp['sDash'].join(lSCl)

def toUnqNmerSeq(dITp, dfrInp, serNmerSeq, sMd=GC.S_CLF):
    lSer, sCY = [], dITp['sCY' + sMd]
    serNmerSeq = GF.toSerUnique(serNmerSeq, sName=dITp['sCNmer'])
    _, serNmerSeq = filterNmerSeq(dITp, serSeq=serNmerSeq)
    for cSeq in serNmerSeq:
        cDfr = dfrInp[dfrInp[dITp['sCNmer']] == cSeq]
        lSC = GF.toListUnqViaSer(cDfr[sCY].to_list())
        cSer = cDfr.iloc[0, :]
        cSer.at[sCY] = getClassStr(dITp, lSCl=lSC, sMd=sMd)
        lSer.append(cSer)
    return GF.concLOAx1(lObj=lSer, ignIdx=True).T, serNmerSeq

def preProcInp(dITp, dfrInp, dNmerNoCl):
    lINoCl = [dITp['sEffCode'], dITp['sCNmer'], dITp['sEffFam'], dITp['sLoc']]
    if dfrInp.columns.to_list() != lINoCl:
        print('ERROR: Columns of dfrInp:', dfrInp.columns.to_list())
    assert dfrInp.columns.to_list() == lINoCl
    dIC, dNmerEffF, sNoFam = dfrInp.to_dict(orient='list'), {}, dITp['sNoFam']
    for sNmer, sEffFam in zip(dIC[dITp['sCNmer']], dIC[dITp['sEffFam']]):
        GF.addToDictL(dNmerEffF, cK=getCentSNmerDefLen(dITp, sSeq=sNmer),
                      cE=sEffFam, lUnqEl=True)
    if dNmerNoCl is not None:
        for sAAc, lNmer in dNmerNoCl.items():
            for sNmer in lNmer:
                GF.addToDictL(dNmerEffF, cK=sNmer, cE=sNoFam, lUnqEl=True)
    serNmerSeq = GF.iniPdSer(list(dNmerEffF), nameS=dITp['sCNmer'])
    return filterNmerSeq(dITp, dSeq=dNmerEffF, serSeq=serNmerSeq)

def loadInpData(dITp, dfrInp, sMd=GC.S_CLF):
    serNmerSeq, lSCl, X, Y = None, None, None, None
    if dITp['sCNmer'] in dfrInp.columns:
        serNmerSeq = dfrInp[dITp['sCNmer']]
    if dITp['usedNmerSeq' + sMd] == dITp['sUnqList']:
        dfrInp, serNmerSeq = toUnqNmerSeq(dITp, dfrInp, serNmerSeq, sMd=sMd)
    assert dITp['sCY' + sMd] in dfrInp.columns
    for sCX in dITp['lSCX' + sMd]:
        assert sCX in dfrInp.columns
    X = dfrInp[dITp['lSCX' + sMd]]
    Y = dfrInp[dITp['sCY' + sMd]]
    lSCl = sorted(GF.toListUnqViaSer(dfrInp[dITp['sCY' + sMd]]))
    return dfrInp, X, Y, serNmerSeq, lSCl

def getDClMap(dITp):
    dClMap, dITp['lXCl'] = {}, []
    pDCl = GF.joinToPath(dITp['pInpClf'], dITp['sFInpDClMpClf'])
    dfrClMap = GF.readCSV(pF=pDCl, iCol=0)
    for _, cRow in dfrClMap.iterrows():
        [sK, sV] = cRow.to_list()
        if sV not in [dITp['sNoFam'], dITp['sNone']]:
            dClMap[sK] = sV
            GF.addToListUnq(dITp['lXCl'], cEl=sV)
    print('Calculated class dictionary.')
    if dITp['printDClMap']:
        for sK, sV in dClMap.items():
            print(sK, dITp['sColon'], dITp['sTab'], sV, sep='')
        print(len(dITp['lXCl']), 'different X-classes. List of X-classes:')
        print(dITp['lXCl'])
    return dClMap

def getDLocKeyMap(dITp):
    dLocKeyMap, dITp['lLocKeys'] = {}, []
    pDLocKey = GF.joinToPath(dITp['pInpClf'], dITp['sFInpDLocMpClf'])
    dfrLocKeyMap = GF.readCSV(pF=pDLocKey, iCol=0)
    for _, cRow in dfrLocKeyMap.iterrows():
        [sK, sV] = cRow.to_list()
        dLocKeyMap[sK] = sV
        GF.addToListUnq(dITp['lLocKeys'], cEl=sV)
    print('Calculated location key dictionary.')
    if dITp['printDLocKeyMap']:
        for sK, sV in dLocKeyMap.items():
            print(sK, dITp['sColon'], dITp['sTab'], sV, sep='')
        print(len(dITp['lLocKeys']), 'different locations. List of locations:')
        print(dITp['lLocKeys'])
    return dLocKeyMap

def iniDicts(dITp):
    dXS = {'lI': [], 'dDat': {sI: [] for sI in dITp['lSCXClf']}}
    dXM = {'lI': [], 'dDat': {sI: [] for sI in dITp['lSCXClf']}}
    dYS = {'lI': [], 'dDat': {dITp['sEffFam']: []}}
    dYM = {'lI': [], 'dDat': {sXCl: [] for sXCl in dITp['lXCl']}}
    return {'lI': [], 'dDat': {}}, dXS, dXM, dYS, dYM

def getLXCl(dITp, dNEF, dClMap, cSeq):
    # translation of effector families to XClasses
    lXCl = []
    lEFIncl = [sFam for sFam in dNEF[cSeq] if (sFam in dClMap)]
    lEFExcl = [sFam for sFam in dNEF[cSeq] if (sFam not in dClMap)]
    if ((dITp['noExclEffFam'] and len(lEFExcl) == 0 and len(lEFIncl) > 0) or
        not dITp['noExclEffFam']):
        lXCl = [dClMap[sFam] for sFam in lEFIncl]
    return lXCl

def addToDProcDY(cD, cK, cE, cSeq, appI=False):
    GF.addToDictL(cD['dDat'], cK=cK, cE=cE)
    if appI:
        cD['lI'].append(cSeq)

def addToDX(cD, cSeq, appI=False):
    for sKeyX, sAAc in zip(cD['dDat'], cSeq):
        GF.addToDictL(cD['dDat'], cK=sKeyX, cE=sAAc)
    if appI:
        cD['lI'].append(cSeq)

def fillDDat(dITp, dNEF, dProc, dXS, dXM, dYS, dYM, dClMap, cSeq, lF):
    assert len(cSeq) == len(dXS['dDat']) and len(cSeq) == len(dXM['dDat'])
    lXCl = getLXCl(dITp, dNEF=dNEF, dClMap=dClMap, cSeq=cSeq)
    sEffFam, sNoCl, sNone = dITp['sEffFam'], dITp['sNoCl'], dITp['sNone']
    # filling of data dictionaries (dProc and single-label)
    addToDProcDY(cD=dProc, cK=dITp['sCNmer'], cE=cSeq, cSeq=cSeq, appI=True)
    if len(lXCl) == 0:      # no XCl assigned to this Nmer sequence
        if len(lF) == 1 and lF[0] == dITp['sNoFam']:
            addToDProcDY(cD=dProc, cK=sEffFam, cE=sNoCl, cSeq=cSeq)
            addToDProcDY(cD=dYS, cK=sEffFam, cE=sNoCl, cSeq=cSeq, appI=True)
            addToDX(cD=dXS, cSeq=cSeq, appI=True)
        else:
            addToDProcDY(cD=dProc, cK=sEffFam, cE=sNone, cSeq=cSeq)
    elif len(lXCl) == 1:      # exactly 1 XCl assigned to this Nmer sequence
        addToDProcDY(cD=dProc, cK=sEffFam, cE=lXCl[0], cSeq=cSeq)
        addToDProcDY(cD=dYS, cK=sEffFam, cE=lXCl[0], cSeq=cSeq, appI=True)
        addToDX(cD=dXS, cSeq=cSeq, appI=True)
    else:
        addToDProcDY(cD=dProc, cK=sEffFam, cE=dITp['sMultiCl'], cSeq=cSeq)
    # filling of data dictionaries (multi-label)
    for k, cXCl in enumerate(dITp['lXCl']):
        addToDProcDY(cD=dYM, cK=cXCl, cE=(1 if cXCl in lXCl else 0), cSeq=cSeq,
                     appI=(k == 0))
    addToDX(cD=dXM, cSeq=cSeq, appI=True)
    return lXCl

def procInp(dITp, dNmerEffF):
    iCentAdj, lIPosUsed = dITp['maxPosNmer'], dITp['lIPosUsed']
    dClMap, lSXCl = getDClMap(dITp), []
    dProc, dXS, dXM, dYS, dYM = iniDicts(dITp)
    for cSeq, lFam in sorted(dNmerEffF.items()):
        cSeqRed = ''.join([cSeq[i + iCentAdj] for i in lIPosUsed])
        lXCl = fillDDat(dITp, dNEF=dNmerEffF, dProc=dProc, dXS=dXS, dXM=dXM,
                        dYS=dYS, dYM=dYM, dClMap=dClMap, cSeq=cSeqRed, lF=lFam)
        GF.fillListUnique(cL=lSXCl, cIt=lXCl)
    for cD in [dXM, dYM]:
        GF.complDict(cDFull=dProc['dDat'], cDAdd=cD['dDat'])
    YS = GF.iniPdSerFromDict(dYS['dDat'], lSNmR=dYS['lI'])
    lD2Dfr = [dProc, dXS, dXM, dYM]
    lDfr = [GF.iniPdDfr(cD['dDat'], lSNmR=cD['lI']) for cD in lD2Dfr]
    dfrProc, XS, XM, YM = lDfr
    return dfrProc, XS, XM, YS, YM, dClMap, sorted(lSXCl)

def genDictNmerLoc(dITp, dfrNL=None):
    dNL, sComma, sCCodeSeq = {}, dITp['sComma'], dITp['sCCodeSeq']
    for dRec in dfrNL.to_dict(orient='records'):
        for sLoc in dRec[dITp['sLoc']].split(sComma):
            GF.addToDictL(dNL, cK=dRec[sCCodeSeq], cE=sLoc, lUnqEl=True)
    return dNL

def genDictSubStr(dFullStr, itSubStr, lUnqEl=True, mdD=1000):
    dSubStr, N = {}, itSubStr.size
    for k, sSub in enumerate(itSubStr):
        for sKey, lV in dFullStr.items():
            if sSub in sKey:
                for cV in lV:
                    GF.addToDictL(dSubStr, cK=sSub, cE=cV, lUnqEl=lUnqEl)
        if (k + 1)%mdD == 0:
            print('Finished', k + 1, 'of', N, 'substrings.')
    return dSubStr

def getDfrNOccTupUnq(dITp, dUNOcc, idxDfr=None):
    dfrKeys = pd.concat([pd.Series(t, index=idxDfr) for t in dUNOcc], axis=1).T
    serK = dfrKeys.apply(GF.getKeyFromL, axis=1, sSep=dITp['sVBar'])
    dfrKeys[dITp['sLocKey']] = serK
    dfrVals = pd.DataFrame(list(dUNOcc.values()), columns=[dITp['sNumOcc']])
    return pd.concat([dfrKeys, dfrVals], axis=1)

def genYStS(dITp, dMltSt, YSt, iSt=0):
    for sCl, lCl in dMltSt.items():
        YSt.replace(sCl, lCl[iSt - 1], inplace=True)
    return YSt[YSt != dITp['sNone']]

def genYStM(dITp, dInv, YSt):
    lC = [s for s in dInv if s != dITp['sNone']]
    YM = YSt.apply(lambda x: pd.Series([(1 if x.loc[dInv[s]].sum() >= 1 else 0)
                                        for s in lC], index=lC), axis=1)
    return YM

def genYSt(dITp, dMltSt, Y, nSt, sLbl=GC.S_SGL_LBL):
    # dYCSt contains the Y for each step (iSt == 0: initial Y)
    dYCSt = {iSt: None for iSt in range(nSt + 1)}
    dYCSt[0] = Y
    for iSt in range(1, nSt + 1):
        YCSt = Y.copy(deep=True)
        if sLbl == dITp['sSglLbl']:
            YCSt = genYStS(dITp, dMltSt=dMltSt, YSt=YCSt, iSt=iSt)
        else:
            YCSt = genYStM(dITp, dInv=GF.getInvDict(cD=dMltSt, i=(iSt - 1)),
                           YSt=YCSt)
        dYCSt[iSt] = YCSt
    return dYCSt

def getDYMltSt(dITp, dMltSt=None, Y=None, nSt=None, sLbl=GC.S_SGL_LBL):
    if dMltSt is None or Y is None or nSt is None:
        return None
    else:
        return genYSt(dITp, dMltSt=dMltSt, Y=Y, nSt=nSt, sLbl=sLbl)

def getIMltSt(dITp, Y=None, sLbl=GC.S_SGL_LBL):
    pDMS = GF.joinToPath(dITp['pInpClf'], dITp['sFInpDClStClf'])
    dfrMltSt, dMltSt, sSt = GF.readCSV(pF=pDMS, iCol=0), {}, dITp['sStep']
    if dfrMltSt is not None:
        # check DataFrame columns
        assert dITp['sXCl'] in dfrMltSt.columns
        for sC in dfrMltSt.columns:
            assert (sC == dITp['sXCl']) or (sC.startswith(sSt))
        nSt = len([sC for sC in dfrMltSt.columns if sC.startswith(sSt)])
        for _, serR in dfrMltSt.iterrows():
            lCl = [serR.at[cI] for cI in serR.index if cI != dITp['sXCl']]
            dMltSt[serR.at[dITp['sXCl']]] = lCl
        dYSt = getDYMltSt(dITp, dMltSt=dMltSt, Y=Y, nSt=nSt, sLbl=sLbl)
        return {'dXCl': dMltSt, 'dYSt': dYSt, 'nSteps': nSt}

# --- Functions (O_07__Classifier) --------------------------------------------
# --- Functions converting between single- and multi-labels (imbalanced) ------
def toMultiLbl(serY, lXCl):     # faster - for large objects
    if len(serY.shape) > 1:     # already multi-column (DataFrame) format
        return serY
    assert type(serY) == pd.core.series.Series
    dY = {}
    for sLbl in serY:
        for sXCl in lXCl:
            GF.addToDictL(dY, cK=sXCl, cE=(1 if sXCl == sLbl else 0))
    return GF.iniPdDfr(dY, lSNmR=serY.index)

def toMultiLblExt(serY, lXCl, sJoin=GC.S_VBAR):
    # also converts "NoFam" and "MultiFam" labels
    if len(serY.shape) > 1:     # already multi-column (DataFrame) format
        return serY
    assert type(serY) == pd.core.series.Series
    dY = {}
    for sLbl in serY:
        if sJoin in sLbl:
            lSLC = sLbl.split(sJoin)[1:]
            for sXCl in lXCl:
                GF.addToDictL(dY, cK=sXCl, cE=(1 if sXCl in lSLC else 0))
        else:
            for sXCl in lXCl:
                GF.addToDictL(dY, cK=sXCl, cE=(1 if sXCl == sLbl else 0))
    return GF.iniPdDfr(dY, lSNmR=serY.index)

def toMultiLbl2(serY, lXCl):    # slower but more elegant - for small objects
    if len(serY.shape) > 1:     # already multi-column (DataFrame) format
        return serY
    assert type(serY) == pd.core.series.Series
    return serY.apply(lambda x: pd.Series([1 if x == s else 0 for s in lXCl],
                                          index=lXCl))

def toSglLbl(dITp, dfrY):      # faster and more elegant - for large objects
    if len(dfrY.shape) == 1:   # already single-column (Series) format
        return dfrY
    lXCl, s = dfrY.columns, dITp['sNoCl']
    serY = dfrY.apply(lambda x: ([sXCl for k, sXCl in enumerate(lXCl) if
                                  x[k] == 1][0] if sum(x) == 1 else s), axis=1)
    serY.name = dITp['sEffFam']
    return serY

def toSglLblExt(dITp, dfrY, sJoin=GC.S_VBAR):
    # also assigns "NoFam" and "MultiFam" labels
    if len(dfrY.shape) == 1:   # already single-column (Series) format
        return dfrY
    serY = dfrY.apply(GF.getSSglLbl, lXCl=dfrY.columns, sNoFam=dITp['sNoFam'],
                      sMltFam=dITp['sMultiFam'], sJoin=sJoin, axis=1)
    serY.name = dITp['sEffFam']
    return serY

def toSglLblOLD(dITp, dfrY):   # slower and inelegant - rather useless
    if len(dfrY.shape) == 1:   # already single-column (Series) format
        return dfrY
    lY = [dITp['sNoCl']]*dfrY.shape[0]
    for k, (_, serR) in enumerate(dfrY.iterrows()):
        if sum(serR) == 1:
            lY[k] = serR.index[serR == 1].to_list()[0]
    return GF.iniPdSer(lY, lSNmI=dfrY.index, nameS=dITp['sEffFam'])

def formatDfrCVRes(dIG, dITp, cClf):
    nL, R04 = dITp['sNewl'], dIG['R04']
    print(GC.S_DS80, nL, 'Grid search results:', nL,
          'Best estimator:', nL, cClf.best_estimator_, nL,
          'Best parameters:', nL, cClf.best_params_, nL,
          'Best score: ', round(cClf.best_score_, R04), sep='')
    dfrCVRes = GF.iniPdDfr(cClf.cv_results_)
    dfrCVRes = dfrCVRes.sort_values(by=['rank_test_score'])
    dfrCVRes = dfrCVRes[[s for s in dfrCVRes.columns if not
                         (s.startswith('split') or s == 'params')]]
    dfrPrt = dfrCVRes[[s for s in dfrCVRes.columns if not s.endswith('time')]]
    print('CV results:', nL, dfrPrt, sep='')
    return dfrCVRes

# --- Function generating a list of strings for DataFrame column names --------
def getLSC(dITp, YTest, YPred, YProba):
    sTCl, sPCl, sPrb = dITp['sTrueCl'], dITp['sPredCl'], dITp['sProba']
    if len(YTest.shape) > 1:
        lSC = [GF.joinS([s, sTCl], cJ=dITp['sUSC']) for s in YTest.columns]
    else:
        lSC = [GF.joinS([YTest.name, sTCl], cJ=dITp['sUSC'])]
    for cYP, sP in zip([YPred, YProba], [sPCl, sPrb]):
        if len(cYP.shape) > 1:
            lSC += [GF.joinS([s, sP], cJ=dITp['sUSC']) for s in cYP.columns]
        else:
            lSC += [GF.joinS([cYP.name, sP], cJ=dITp['sUSC'])]
    return lSC

# --- Functions (O_80__Looper) ------------------------------------------------
def getLSE(dITp, sMth, lIFE):
    lSEPar = [sMth] + list(dITp['d3Par'][sMth])
    lSESum = [sMth] + list(dITp['d3Par'][sMth])
    lSEDet = lIFE + [sMth]
    l2Add = []
    if sMth in dITp['lSMthPartFit'] and dITp['nItPtFit'] is not None:
        l2Add += [dITp['sPartFitS'] + str(round(dITp['nItPtFit']))]
    else:
        l2Add += [dITp['sFullFitS']]
    if dITp['doImbSampling'] and dITp['sSmplS'] is not None:
        l2Add += [dITp['sSmplS'] + dITp['sStratS'].replace(' ', '')]
    else:
        l2Add += [dITp['sSmplNoS']]
    lSESum += l2Add
    lSEDet += l2Add
    return lSEPar, lSESum, lSEDet

# --- Functions (O_80__Evaluator) ---------------------------------------------
def getRep(dITp, itS=[]):
    cRep = None
    for s in itS:
        if s.startswith(dITp['sRepS']):
            try:
                cRep = int(s[len(dITp['sRepS']):])
                break
            except:
                pass
    return cRep

def selFs(lSel, lSSpl, itSCmp=None):
    for k, s in enumerate(lSSpl):
        if s in itSCmp:
            lSel.append((k, s))
            if (len(itSCmp) > 1) and (k < len(lSSpl[:-1])):
                for j, s in enumerate(lSSpl[(k + 1):]):
                    if s in itSCmp:
                        lSel.append((k + j, s))
            break

def fillDSF(dSFTI, lSel, sF, iRp=None, addI=True):
    if addI:
        dSFTI[tuple([iRp] + [t[1] for t in lSel])] = sF
    else:
        if len(lSel) == 1:
            dSFTI[lSel[0][1]] = sF
        else:
            dSFTI[tuple([t[1] for t in lSel])] = sF

def getIFS(dITp, pF='', itSCmp=None, addI=True, sSpl=GC.S_USC):
    if itSCmp is None:                      # trivial case - no filtering
        return os.listdir(pF)
    dSFTI = {}
    for sF in os.listdir(pF):
        lSel, lSSpl = [], sF.split(GC.S_DOT)[0].split(sSpl)
        selFs(lSel, lSSpl, itSCmp=itSCmp)
        if len(lSel) == len(itSCmp):        # boolean AND
            fillDSF(dSFTI, lSel, sF=sF, iRp=getRep(dITp, itS=lSSpl), addI=addI)
    return dSFTI

def getDMapCl(dITp, dDfr, sMth=None):
    dSCl, sUSC, sAvProba = {}, dITp['sUSC'], dITp['sAvProba']
    for cDfr in dDfr.values():
        for sC in cDfr.columns:
            sCl = GF.getSClFromCHdr(sCHdr=sC, sSep=sUSC)
            if sCl not in dSCl:
                dSCl[sCl] = None
    for sCl in dSCl:
        if sMth is not None:
            dSCl[sCl] = [sUSC.join([sCl, sMth, sAvProba])]
        else:
            dSCl[sCl] = [sUSC.join([sCl, sAvProba])]
    return dSCl

def fillDictDP(dITp, dDP, tK, cDfr):
    if len(tK) > 1:
        for sKM in [dITp['sPredProba']]:
            if sKM in tK:
                if tK[0] is None:
                    dDP[sKM][tK[1:]] = None
                else:
                    try:
                        cRep = int(tK[0])
                        if ((tK[1:] not in dDP[sKM]) or
                            (dDP[sKM][tK[1:]] is not None)):
                            GF.addToDictD(dDP[sKM], cKMain=tK[1:], cKSub=cRep,
                                          cVSub=cDfr)
                    except:
                        pass

def fillDPrimaryRes(dRes, d2CD, cDfr, lCPb, sMth):
    for sCHd in lCPb:
        [sNewHd] = d2CD[sMth][GF.getSClFromCHdr(sCHdr=sCHd)]
        if (sNewHd in dRes and dRes[sNewHd] is None) or (sNewHd not in dRes):
            dRes[sNewHd] = GF.iniPdSer(lSNmI=cDfr.index, nameS=sCHd, fillV=0)
        dRes[sNewHd] = dRes[sNewHd].add(cDfr[sCHd])

def getL01(lNOcc, sCmp='PDiff'):
    if sCmp == 'RelMax':
        return GF.getMaxC(cIt=lNOcc)
    elif sCmp == 'AbsMax':
        return GF.getMaxC(cIt=lNOcc, thrV=len(lNOcc)/2)
    elif sCmp == 'PDiff':
        return GF.getRelVals(cIt=lNOcc)

def compTP(cIt, d2CD, dMapCT, lSM=[], sCmp='PDiff', doCompTP=False):
    # only for single-class predictions
    lRet = []
    for sMth in lSM:
        assert sorted(list(d2CD[sMth])) == sorted(list(dMapCT))
        if doCompTP:
            lT = [cIt.at[dMapCT[sCl]] for sCl in sorted(dMapCT)]
            lP = [cIt.at[lSCHd[0]] for lSCHd in sorted(d2CD[sMth].values())]
            lP = getL01(lNOcc=lP, sCmp=sCmp)
            assert len(lT) == len(lP)
            lSumTP = [np.sum([cT, cP]) for cT, cP in zip(lT, lP)]
            lRet.append(GF.classifyTP(cIt=lSumTP))
        else:
            lSCl, lV = ['']*len(d2CD[sMth]), [np.nan]*len(d2CD[sMth])
            for k, (sCl, lSCHd) in enumerate(sorted(d2CD[sMth].items())):
                lSCl[k] = sCl
                lV[k] = cIt.at[lSCHd[0]]
            lV = getL01(lNOcc=lV, sCmp=sCmp)
            lRet.append(GF.getPredCl(cD={sCl: n for sCl, n in zip(lSCl, lV)}))
    return pd.Series(lRet, name=sCmp, index=lSM)

###############################################################################