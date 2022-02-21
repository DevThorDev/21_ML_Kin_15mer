# -*- coding: utf-8 -*-
###############################################################################
# --- F_03__OTpFunctions.py ---------------------------------------------------
###############################################################################
# import numpy as np
import pandas as pd
# import scipy.stats as stats

# import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

# --- Functions (O_00__BaseClass) ---------------------------------------------

# --- Functions (O_01__ExpData) -----------------------------------------------
def createDEffTarg(dITp, dfrK, dfr15m):
    dEffTarg, dfrKEff = {}, dfrK[[dITp['sEffCode'], dITp['sEffSeq']]]
    serTEff = dfrKEff.apply(lambda x: tuple(x), axis=1)
    for tSE in serTEff.unique():
        dEffTarg[tSE] = {}
        dfrE = dfrK[(dfrK[dITp['sEffCode']] == tSE[0]) &
                    (dfrK[dITp['sEffSeq']] == tSE[1])]
        serTDfrE = dfrE.apply(lambda x: tuple(x), axis=1)
        dfrKTarg = dfrE[[dITp['sTargCode'], dITp['sTargSeq']]]
        serTTarg = dfrKTarg.apply(lambda x: tuple(x), axis=1)
        dT = {t2: t4 for t2, t4 in zip(serTTarg, serTDfrE)}
        for tST in serTTarg.unique():
            dfrT = dfr15m[dfr15m[dITp['sCodeTrunc']] == tST[0]]
            dEffTarg[tSE][dT[tST]] = dfrT
    return dEffTarg

def dDDfrToDfr(dDDfr, lSColL, lSColR):
    fullDfr = GF.iniPdDfr(lSNmC=lSColL+lSColR)
    for sKMain, cDSub in dDDfr.items():
        for sKSub, rightDfr in cDSub.items():
            print('(sKMain, sKSub) =', (sKMain, sKSub))
            leftDfr = GF.iniPdDfr(lSNmR=rightDfr.index, lSNmC=lSColL)
            print('leftDfr =', leftDfr)
            assert False
            leftDfr[lSColL[:len(sKMain)]] = sKMain
            leftDfr[lSColL[len(sKMain):]] = sKSub
            subDfr = pd.concat([leftDfr, rightDfr], axis=1)
            fullDfr = pd.concat([fullDfr, subDfr], axis=0)
    return fullDfr.reset_index(drop=True)

###############################################################################
