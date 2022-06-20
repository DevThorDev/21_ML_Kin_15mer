# -*- coding: utf-8 -*-
###############################################################################
# --- F_01__SpcFunctions.py ---------------------------------------------------
###############################################################################
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
        lSFullSeq = list(dfrInpSeq[dITp['sCCodeSeq']].unique())
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
            lSNmer = list(dfrCSeq[dITp['sCNmer']].unique())
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
def getClassStrOldCl(dITp, setSDC, lSCl=[]):
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

def getClassStr(dITp, lSCl=[], sMd='Clf'):
    if dITp['usedClType' + sMd].startswith(GC.S_OLD):
        return getClassStrOldCl(dITp, lSCl=lSCl)
    elif dITp['usedClType' + sMd].startswith(GC.S_NEW):
        return dITp['sDash'].join(lSCl)

def toUnqNmerSeq(dITp, dfrInp, serNmerSeq, sMd='Clf'):
    lSer, sCY = [], dITp['sCY' + sMd]
    serNmerSeq = GF.iniPdSer(serNmerSeq.unique(), nameS=dITp['sCNmer'])
    for cSeq in serNmerSeq:
        cDfr = dfrInp[dfrInp[dITp['sCNmer']] == cSeq]
        lSC = GF.toListUnique(cDfr[sCY].to_list())
        cSer = cDfr.iloc[0, :]
        cSer.at[sCY] = getClassStr(dITp, lSCl=lSC, sMd=sMd)
        lSer.append(cSer)
    return GF.concLObjAx1(lObj=lSer, ignIdx=True).T, serNmerSeq

def loadInpData(dITp, dfrInp, sMd='Clf', iC=0):
    serNmerSeq, lSCl, X, Y = None, None, None, None
    if dITp['sCNmer'] in dfrInp.columns:
        serNmerSeq = dfrInp[dITp['sCNmer']]
    if dITp['usedNmerSeq' + sMd] == dITp['sUnqList']:
        dfrInp, serNmerSeq = toUnqNmerSeq(dITp, dfrInp, serNmerSeq, sMd=sMd)
    assert dITp['sCY' + sMd] in dfrInp.columns
    for sCX in dITp['lSCX' + sMd]:
        assert sCX in dfrInp.columns
    lSCl = sorted(list(dfrInp[dITp['sCY' + sMd]].unique()))
    X = dfrInp[dITp['lSCX' + sMd]]
    Y = dfrInp[dITp['sCY' + sMd]]
    return serNmerSeq, dfrInp, lSCl, X, Y

def procInpData(dITp, dfrInp, sMd='Clf', iC=0):
    sCNmer, sEffFam, dProc, lEffFam = dITp['sCNmer'], dITp['sEffFam'], {}, []
    dfrProc, serNmerSeq, lSCl, X, Y = None, None, None, None, None
    if sCNmer in dfrInp.columns:
        serNmerSeq = GF.iniPdSer(dfrInp[sCNmer].unique(), nameS=sCNmer)
        for cSeq in serNmerSeq:
            GF.addToDictL(dProc, cK=sCNmer, cE=cSeq)
            dfrT = dfrInp[dfrInp[sCNmer] == cSeq]
            lEffFamSeq = dfrT[sEffFam].to_list()
            GF.addToDictL(dProc, cK=sEffFam, cE=lEffFamSeq)
            GF.fillListUnique(lEffFam, cIt=lEffFamSeq)
        for cSeq in serNmerSeq:
            lEffFamSeq = dProc[sEffFam]
            for sEffFam in lEffFam:
                GF.addToDictL(dProc, cK=dITp['sXCl'],
                              cE=(1 if sEffFam in lEffFamSeq else 0))

# --- Functions (O_07__Classifier) --------------------------------------------

###############################################################################
