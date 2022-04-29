# -*- coding: utf-8 -*-
###############################################################################
# --- F_01__SpcFunctions.py ---------------------------------------------------
###############################################################################
# import numpy as np
import pandas as pd
# import scipy.stats as stats

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
def getLSFullSeq(dITp, dfrInpSeq, iS=None, iE=None):
    lSFullSeq = list(dfrInpSeq[dITp['sCCodeSeq']].unique())
    return GF.getItStartToEnd(lSFullSeq, iS, iE)

def getLSNmerSeq(dITp, dfrInpSeq, lSFull=[], lSNmer=[]):
    if lSNmer is None and dITp['sCNmer'] in dfrInpSeq.columns:
        if dITp['sCCodeSeq'] in dfrInpSeq.columns:
            if dITp['reduceFullToNmerSeq']:
                dfrCSeq = dfrInpSeq
            else:
                # reduce to Nmer sequences with corresponding full sequences
                dfrCSeq = dfrInpSeq[dfrInpSeq[dITp['sCCodeSeq']].isin(lSFull)]
            lSNmer = list(dfrCSeq[dITp['sCNmer']].unique())
        else:
            lSNmer = list(dfrInpSeq[dITp['sCNmer']].unique())
    return lSNmer

def modLSF(dITp, lSKeyF, sFE):
    for sKeyF in lSKeyF:
        dITp[sKeyF] = GF.modSF(dITp[sKeyF], sEnd=sFE, sJoin=dITp['sUS02'])

def calcDictLikelihood(dITp, dLV, d3, dSqProfile, serLh, mxLSnip, cSSq, cEff):
    dLh, wtLh, lSCWtLh = {}, 0., dITp['lSCDfrLhV']
    assert len(lSCWtLh) >= 3
    for lenSnip, sSnip in dSqProfile.items():
        if lenSnip <= mxLSnip:
            cLh = 0.
            if sSnip in serLh.index:
                cLh = serLh.at[sSnip]
            dLh[sSnip] = cLh
            if lenSnip in dITp['dWtsLenSeq']:
                wtLh += cLh*dITp['dWtsLenSeq'][lenSnip]
    if dITp['calcRelLh']:
        GF.addToDictD(d3, cKMain=cSSq, cKSub=cEff, cVSub=dLh)
    if dITp['calcWtLh']:
        for sC, cV in zip(lSCWtLh[:3], [cSSq, cEff, wtLh]):
            GF.addToDictL(dLV, cK=sC, cE=cV)

###############################################################################
