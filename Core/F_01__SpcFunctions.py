# -*- coding: utf-8 -*-
###############################################################################
# --- F_01__SpcFunctions.py ---------------------------------------------------
###############################################################################
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
def filterNmerSeq(dITp, dSeq={}, serSeq=None):
    dSeqFilt = dSeq
    if dITp['dAAcPosRestr'] is not None and serSeq is not None:
        assert min([len(sSeq) for sSeq in serSeq]) >= dITp['lenNmerDef']
        for iP, lAAc in dITp['dAAcPosRestr'].items():
            iSeq = iP + dITp['iCentNmer']
            if iSeq >= 0 and iSeq < dITp['lenNmerDef']:
                lB = [(sSeq[iSeq] in lAAc) for sSeq in serSeq]
                serSeq = serSeq[lB]
        dSeqFilt = {sSeq: dSeq[sSeq] for sSeq in serSeq}
    return dSeqFilt, serSeq

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
    lInpNoCl = [dITp['sEffCode'], dITp['sCNmer'], dITp['sEffFam']]
    if dfrInp.columns.to_list() != lInpNoCl:
        print('ERROR: Columns of dfrInp:', dfrInp.columns.to_list())
    assert dfrInp.columns.to_list() == lInpNoCl
    dIC, dNmerEffF = dfrInp.to_dict(orient='list'), {}
    for sNmer, sEffFam in zip(dIC[dITp['sCNmer']], dIC[dITp['sEffFam']]):
        GF.addToDictL(dNmerEffF, cK=sNmer, cE=sEffFam, lUnqEl=True)
    if dNmerNoCl is None:
        return dfrInp
    for sAAc, lNmer in dNmerNoCl.items():
        for sNmer in lNmer:
            GF.addToDictL(dNmerEffF, cK=sNmer, cE=dITp['sNoFam'], lUnqEl=True)
    serNmerSeq = GF.iniPdSer(list(dNmerEffF), nameS=dITp['sCNmer'])
    return filterNmerSeq(dITp, dSeq=dNmerEffF, serSeq=serNmerSeq)

def loadInpData(dITp, dfrInp, sMd=GC.S_CLF, iC=0):
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

def getDClMap(dIG, dITp):
    dClMap, dITp['lXCl'] = {}, []
    pDCl = GF.joinToPath(dITp['pInpClf'], dITp['sFInpDClMpClf'] + dIG['xtCSV'])
    dfrClMap = GF.readCSV(pF=pDCl, iCol=0)
    for _, cRow in dfrClMap.iterrows():
        [sK, sV] = cRow.to_list()
        if sV != dITp['sNone']:
            dClMap[sK] = sV
            GF.addToListUnq(dITp['lXCl'], cEl=sV)
    print('Calculated class dictionary.')
    if dITp['printDClMap']:
        for sK, sV in dClMap.items():
            print(sK, dITp['sColon'], dITp['sTab'], sV, sep='')
        print(len(dITp['lXCl']), 'different X-classes. List of X-classes:')
        print(dITp['lXCl'])
    return dClMap

def iniObj(dIG, dITp, dNmerEffF):
    dX, dY = {sI: [] for sI in dITp['lSCXClf']}, {dITp['sEffFam']: []}
    if not dITp['onlySglLbl']:
        dY = {sXCl: [] for sXCl in dITp['lXCl']}
    return {}, dX, dY

def getLXCl(dIG, dITp, dNEF, dClMap, cSeq):
    # translation of effector families to XClasses
    lXCl = []
    lEFIncl = [sFam for sFam in dNEF[cSeq] if (sFam in dClMap)]
    lEFExcl = [sFam for sFam in dNEF[cSeq] if (sFam not in dClMap)]
    if ((dITp['noExclEffFam'] and len(lEFExcl) == 0 and len(lEFIncl) > 0) or
        not dITp['noExclEffFam']):
        lXCl = [dClMap[sFam] for sFam in lEFIncl]
    return lXCl

def fill_DProc_DX(dITp, dProc, dX, cSeq):
    GF.addToDictL(dProc, cK=dITp['sCNmer'], cE=cSeq)
    for sKeyX, sAAc in zip(dX, cSeq):
        GF.addToDictL(dX, cK=sKeyX, cE=sAAc)

def fill_DX_DY_Sgl(dITp, dProc, dX, dY, cSeq, lXCl=[]):
    if len(lXCl) == 1:      # exactly 1 XCl assigned to this Nmer sequence
        GF.addToDictL(dY, cK=dITp['sEffFam'], cE=lXCl[0])
        fill_DProc_DX(dITp, dProc=dProc, dX=dX, cSeq=cSeq)

def fill_DX_DY_Mlt(dITp, dProc, dX, dY, cSeq, lXCl=[]):
    for cXCl in dITp['lXCl']:
        GF.addToDictL(dY, cK=cXCl, cE=(1 if cXCl in lXCl else 0))
    fill_DProc_DX(dITp, dProc=dProc, dX=dX, cSeq=cSeq)

def fill_DProc_DX_DY(dIG, dITp, dNmerEffF, dProc, dX, dY, dClMap, cSeq):
    assert len(cSeq) == len(dX)
    lXCl = getLXCl(dIG, dITp, dNEF=dNmerEffF, dClMap=dClMap, cSeq=cSeq)
    # filling of data dictionaries
    if dITp['onlySglLbl']:
        fill_DX_DY_Sgl(dITp, dProc=dProc, dX=dX, dY=dY, cSeq=cSeq, lXCl=lXCl)
    else:
        fill_DX_DY_Mlt(dITp, dProc=dProc, dX=dX, dY=dY, cSeq=cSeq, lXCl=lXCl)
    return lXCl

def procInp(dIG, dITp, dNmerEffF):
    iCent, lIPosUsed = dITp['iCentNmer'], dITp['lIPosUsed']
    dfrProc, X, Y, dClMap, lSXCl = None, None, None, getDClMap(dIG, dITp), []
    dProc, dX, dY = iniObj(dIG, dITp, dNmerEffF)
    for cSeq in dNmerEffF:
        cSeqRed = ''.join([cSeq[i + iCent] for i in lIPosUsed])
        lXCl = fill_DProc_DX_DY(dIG, dITp, dNmerEffF, dProc, dX, dY, dClMap,
                                cSeq=cSeqRed)
        GF.fillListUnique(cL=lSXCl, cIt=lXCl)
    for cD in [dX, dY]:
        GF.complDict(cDFull=dProc, cDAdd=cD)
    dfrProc, X, Y = GF.iniPdDfr(dProc), GF.iniPdDfr(dX), GF.iniPdDfr(dY)
    if dITp['onlySglLbl']:
        Y = GF.dictSglKey2Ser(dY)
    return dfrProc, X, Y, dClMap, sorted(lSXCl)

def getIMltSt(dIG, dITp, Y):
    pDMS = GF.joinToPath(dITp['pInpClf'], dITp['sFInpDClStClf'] + dIG['xtCSV'])
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
        # dYSt contains the Y for each step (iSt == 0: initial Y)
        dYSt = {iSt: None for iSt in range(nSt + 1)}
        dYSt[0] = Y
        for iSt in range(1, nSt + 1):
            YSt = Y.copy(deep=True)
            for sCl, lCl in dMltSt.items():
                YSt.replace(sCl, lCl[iSt - 1], inplace=True)
                YSt = YSt[YSt != dITp['sNone']]
            dYSt[iSt] = YSt
        return {'dXCl': dMltSt, 'dYSt': dYSt, 'nSteps': nSt}

def getDSqNoCl(dITp, serFullSeqUnq=[], lNmerSeqUnq=None, iPCent=0):
    dSnip, iCentNmer = {}, dITp['iCentNmer']
    for sFullSeq in serFullSeqUnq:
        lenSeq = len(sFullSeq)
        if lenSeq >= dITp['lenNmerDef']:
            for iCentC in range(iCentNmer, lenSeq - iCentNmer):
                sSnip = sFullSeq[(iCentC - iCentNmer):(iCentC + iCentNmer + 1)]
                sAAc = sSnip[iCentNmer]
                if sAAc in dITp['dAAcPosRestr'][iPCent]:
                    if lNmerSeqUnq is None or sSnip not in lNmerSeqUnq:
                        GF.addToDictL(dSnip, cK=sAAc, cE=sSnip, lUnqEl=False)
    return {cK: GF.toListUnqViaSer(cL) for cK, cL in dSnip.items()}

# --- Functions (O_07__Classifier) --------------------------------------------
# --- Functions converting between single- and multi-labels (imbalanced) ------
def toMultiLbl(dITp, serY, lXCl):
    assert type(serY) == pd.core.series.Series
    dY = {}
    for sLbl in serY:
        for sXCl in lXCl:
            GF.addToDictL(dY, cK=sXCl, cE=(1 if sXCl == sLbl else 0))
    return GF.iniPdDfr(dY, lSNmR=serY.index)

def toSglLbl(dITp, dfrY):
    lY = [dITp['sNone']]*dfrY.shape[0]
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

# --- Functions (O_80__Looper) ------------------------------------------------
def getLSE(dITp, sMth, lIFE):
    lSEPar = [sMth] + list(dITp['d3Par'][sMth])
    lSESum = [sMth] + list(dITp['d3Par'][sMth])
    lSEDet = lIFE + [sMth]
    l2Add = []
    if sMth in dITp['lSMthPartFit']:
        if dITp['nItPtFit'] is None:
            l2Add += [dITp['sPartFitS'] + dITp['sNone']]
        else:
            sNIt = str(round(dITp['nItPtFit']))
            l2Add += [dITp['sPartFitS'] + sNIt]
    if dITp['doImbSampling'] and dITp['sSmplS'] is not None:
        l2Add += [dITp['sSmplS']]
    lSESum += l2Add
    lSEDet += l2Add
    return lSEPar, lSESum, lSEDet

# --- Functions (O_80__Evaluator) ---------------------------------------------
def getDMapCl(dITp, dDfr, sMth=None):
    dSCl, sUSC, lSPred = {}, dITp['sUSC'], dITp['lSPred']
    for cDfr in dDfr.values():
        for sC in cDfr.columns:
            sCl = GF.getSClFromCHdr(sCHdr=sC, sSep=sUSC)
            if sCl not in dSCl:
                dSCl[sCl] = None
    for sCl in dSCl:
        if sMth is not None:
            dSCl[sCl] = [sUSC.join([sCl, sMth, sPred]) for sPred in lSPred]
        else:
            dSCl[sCl] = [sUSC.join([sCl, sPred]) for sPred in lSPred]
    return dSCl

# def fillDPrimaryRes(dRes, d2CD, cDfr, nCl, tFlt, sMth):
#     for sCHd in cDfr.columns[-nCl:]:
#         k0, k1 = d2CD[(tFlt, sMth)][GF.getSClFromCHdr(sCHdr=sCHd)]
#         for cK in [k0, k1]:
#             if (cK in dRes and dRes[cK] is None) or (cK not in dRes):
#                 dRes[cK] = GF.iniPdSer(lSNmI=cDfr.index, nameS=sCHd, fillV=0)
#         dRes[k0] = dRes[k0].add(cDfr[sCHd].apply(lambda k: 1 - k))
#         dRes[k1] = dRes[k1].add(cDfr[sCHd])

def fillDPrimaryRes(dRes, d2CD, cDfr, nCl, sMth):
    for sCHd in cDfr.columns[-nCl:]:
        [sNewHd] = d2CD[sMth][GF.getSClFromCHdr(sCHdr=sCHd)]
        if (sNewHd in dRes and dRes[sNewHd] is None) or (sNewHd not in dRes):
            dRes[sNewHd] = GF.iniPdSer(lSNmI=cDfr.index, nameS=sCHd, fillV=0)
        dRes[sNewHd] = dRes[sNewHd].add(cDfr[sCHd])

def modDfrP(dfrP, sCmp='PDiff'):
    if sCmp == 'RelMax':
        return dfrP.apply(GF.getMaxC, axis=1).convert_dtypes()
    elif sCmp == 'AbsMax':
        halfNCl = dfrP.shape[1]/2
        return dfrP.apply(GF.getMaxC, axis=1, thrV=halfNCl).convert_dtypes()
    elif sCmp == 'PDiff':
        return dfrP.apply(GF.getRelVals, axis=1).convert_dtypes()

def compTP(d2CD, dMapCT, lDfr, lSM=[], sCmp='PDiff'):   # for single-class
    dDfrAllM, (dfrT, dfrP) = {}, lDfr
    for sMth in lSM:
        dfrPMd = dfrP[GF.flattenIt(cIterable=d2CD[sMth].values())]
        itC = dfrPMd.columns
        dDfrSglCl, dfrPMd = {}, modDfrP(dfrPMd, sCmp=sCmp)
        dfrPMd.columns=itC
        for sCl, lSCHd in d2CD[sMth].items():
            if sCl in dMapCT:
                dfr2C = GF.concLOAx1([dfrT[dMapCT[sCl]], dfrPMd[lSCHd[0]]])
                dDfrSglCl[sCl] = dfr2C.apply(np.sum, axis=1)
        dfrMth = GF.concLOAx1([dDfrSglCl[sCl] for sCl in d2CD[sMth]])
        dDfrAllM[sMth] = dfrMth.apply(GF.classifyTP, axis=1)
    return GF.concLOAx1([dDfrAllM[sMth] for sMth in dDfrAllM])

###############################################################################