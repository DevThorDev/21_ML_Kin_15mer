# -*- coding: utf-8 -*-
###############################################################################
# --- F_03__OTpFunctions.py ---------------------------------------------------
###############################################################################
# import numpy as np
import pandas as pd
# import scipy.stats as stats

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

# --- Functions (O_00__BaseClass) ---------------------------------------------

# --- Functions (O_01__ExpData) -----------------------------------------------
def getSerTEff(dITp, dfrK, sMd=GC.S_SHORT):
    dEffTarg, serTEff = {}, []
    if sMd in [GC.S_SHORT, GC.S_MED]:
        serKEff = dfrK[dITp['sEffCode']]
        serTEff = serKEff.apply(lambda x: (x,))
    elif sMd == GC.S_LONG:
        dfrKEff = dfrK[[dITp['sEffCode'], dITp['sEffSeq']]]
        serTEff = dfrKEff.apply(lambda x: tuple(x), axis=1)
    return dEffTarg, serTEff

def getSerTTarg(dITp, dfrK, tSE, sMd=GC.S_SHORT):
    if sMd in [GC.S_SHORT, GC.S_MED]:
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

def createDEffTarg(dITp, dfrK, dfr15m, lCDfr15m, sMd=GC.S_SHORT):
    dEffTarg, serTEff = getSerTEff(dITp, dfrK, sMd=sMd)
    for tSE in serTEff.unique():
        dEffTarg[tSE] = {}
        serTTarg, dT = getSerTTarg(dITp, dfrK, tSE, sMd=sMd)
        for tST in serTTarg.unique():
            dfrT = dfr15m[dfr15m[dITp['sCodeTrunc']] == tST[0]]
            dEffTarg[tSE][dT[tST]] = dfrT[lCDfr15m]
    return dEffTarg

def dDNumToDfrINmer(dITp, dDNum):
    lSCol = dITp['lSCDfrNMer']
    assert len(lSCol) >= 4
    fullDfr = GF.iniPdDfr(lSNmC=lSCol)
    for sKMain, cDSub in dDNum.items():
        subDfr = GF.iniPdDfr(lSNmC=lSCol)
        subDfr[lSCol[0]] = [sKMain]*len(cDSub)
        subDfr[lSCol[1]] = list(cDSub)
        subDfr[lSCol[2]] = [len(sK) for sK in cDSub]
        subDfr[lSCol[3]] = list(cDSub.values())
        fullDfr = pd.concat([fullDfr, subDfr], axis=0)
    return fullDfr.reset_index(drop=True).convert_dtypes()

def dDNumToDfrIEff(dITp, dDNum):
    d4Dfr, sEffCode, sAnyEff = {}, dITp['sEffCode'], dITp['sAnyEff']
    dDPrp = GF.convDDNumToDDProp(dDNum=dDNum, nDigRnd=dITp['rndDigProp'])
    assert sAnyEff in dDPrp
    lSEff, lAllS15m = [s for s in dDPrp if s != sAnyEff], list(dDPrp[sAnyEff])
    if len(dITp['lLenNMer']) > 0:
        lAllS15m = [s for s in dDPrp[sAnyEff] if len(s) in dITp['lLenNMer']]
    lAllS15m.sort(key=(lambda x: len(x)))
    d4Dfr[sEffCode] = lSEff
    for s15m in lAllS15m:
        d4Dfr[s15m] = [0]*len(lSEff)
    for s15m in lAllS15m:
        for k, sEff in enumerate(lSEff):
            if s15m in dDPrp[sEff]:
                d4Dfr[s15m][k] = dDPrp[sEff][s15m]
    return pd.DataFrame(d4Dfr).convert_dtypes()

###############################################################################
