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
def filterNmerSeq(dITp, serSeq):
    if dITp['dAAcPosRestr'] is not None:
        assert min([len(sSeq) for sSeq in serSeq]) >= dITp['lenNmerDef']
        for iP, lAAc in dITp['dAAcPosRestr'].items():
            iSeq = iP + dITp['iCentNmer']
            if iSeq >= 0 and iSeq < dITp['lenNmerDef']:
                lB = [(sSeq[iSeq] in lAAc) for sSeq in serSeq]
                serSeq = serSeq[lB]
    return serSeq

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
    serNmerSeq = GF.toSerUnique(serNmerSeq, sName=dITp['sCNmer'])
    serNmerSeq = filterNmerSeq(dITp, serSeq=serNmerSeq)
    for cSeq in serNmerSeq:
        cDfr = dfrInp[dfrInp[dITp['sCNmer']] == cSeq]
        lSC = GF.toListUnqViaSer(cDfr[sCY].to_list())
        cSer = cDfr.iloc[0, :]
        cSer.at[sCY] = getClassStr(dITp, lSCl=lSC, sMd=sMd)
        lSer.append(cSer)
    return GF.concLObjAx1(lObj=lSer, ignIdx=True).T, serNmerSeq

def complDfrInpNoCl(dITp, dfrInp, dNmerNoCl):
    if dNmerNoCl is None:
        return dfrInp
    lInpNoCl = [dITp['sEffCode'], dITp['sCNmer'], dITp['sEffFam']]
    assert dfrInp.columns.to_list() == lInpNoCl
    dCompl = dfrInp.to_dict(orient='list')
    for sAAc, lNmer in dNmerNoCl.items():
        for sNmer in lNmer:
            lVA = [dITp['sNoEff'], sNmer, dITp['sNoEff']]
            if not dITp['onlySglLbl']:
                GF.appendToDictL(dCompl, itKeys=dfrInp.columns, lVApp=lVA)
            else:
                if sNmer not in dCompl[dITp['sCNmer']]:
                    GF.appendToDictL(dCompl, itKeys=dfrInp.columns, lVApp=lVA)
    return GF.iniPdDfr(dCompl)

def loadInpData(dITp, dfrInp, sMd='Clf', iC=0):
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

def getDClasses(dITp):
    dITp['dClasses'], dITp['lXCl'] = {}, []
    pDCl = GF.joinToPath(dITp['pInpClf'], dITp['sFInpDClClf'] + dITp['xtCSV'])
    dfrClMap = GF.readCSV(pF=pDCl, iCol=0)
    for _, cRow in dfrClMap.iterrows():
        [sK, sV] = cRow.to_list()
        if sV != dITp['sStar']:
            dITp['dClasses'][sK] = sV
            GF.addToListUnq(dITp['lXCl'], cEl=sV)
    print('Calculated class dictionary.')
    if dITp['printDClasses']:
        for sK, sV in dITp['dClasses'].items():
            print(sK, dITp['sColon'], dITp['sTab'], sV, sep='')
        print(len(dITp['lXCl']), 'different X classes. List of X classes:')
        print(dITp['lXCl'])

def iniObj(dITp, dfrInp, pFDTmp):
    sCNmer, sFam = dITp['sCNmer'], dITp['sEffFam']
    getDClasses(dITp)
    dX = {sI: [] for sI in dITp['lSCXClf']}
    dY, dT, dProc = {sXCl: [] for sXCl in dITp['lXCl']}, {}, {}
    serNmerSeq = GF.toSerUnique(dfrInp[sCNmer], sName=sCNmer)
    serNmerSeq = filterNmerSeq(dITp, serSeq=serNmerSeq)
    if GF.fileXist(pF=pFDTmp):
        dT = GF.pickleLoadDict(pF=pFDTmp)
    else:
        for k, cSeq in enumerate(serNmerSeq):
            lFam = dfrInp[dfrInp[sCNmer] == cSeq][sFam].to_list()
            dT[cSeq] = GF.toListUnqViaSer(lFam)
            if k % dITp['modDispIni'] == 0:
                print('Processed', k, 'of', serNmerSeq.size)
        GF.pickleSaveDict(cD=dT, pF=pFDTmp)
    return dT, dProc, dX, dY, serNmerSeq

def fill_DProc_DX(dITp, dProc, dX, cSeq):
    assert len(cSeq) == len(dX)
    GF.addToDictL(dProc, cK=dITp['sCNmer'], cE=cSeq)
    for sKeyX, sAAc in zip(dX, cSeq):
        GF.addToDictL(dX, cK=sKeyX, cE=sAAc)

def fill_DY(dITp, dT, dY, cSeq):
    lXCl = []
    for sCFam in dT[cSeq]:
        if sCFam in dITp['dClasses']:
            GF.fillListUnique(cL=lXCl, cIt=[dITp['dClasses'][sCFam]])
    for cXCl in dITp['lXCl']:
        GF.addToDictL(dY, cK=cXCl, cE=(1 if cXCl in lXCl else 0))
    return lXCl

def procClfInp(dITp, dfrInp, pFDTmp):
    iCent, lIPosUsed = dITp['iCentNmer'], dITp['lIPosUsed']
    dfrProc, X, Y, serNmerSeq, lSXCl = None, None, None, None, []
    if dITp['sCNmer'] in dfrInp.columns:
        dT, dProc, dX, dY, serNmerSeq = iniObj(dITp, dfrInp, pFDTmp=pFDTmp)
        for cSeq in serNmerSeq:
            cSeqRed = ''.join([cSeq[i + iCent] for i in lIPosUsed])
            fill_DProc_DX(dITp, dProc, dX, cSeq=cSeqRed)
            GF.fillListUnique(cL=lSXCl, cIt=fill_DY(dITp, dT, dY, cSeq))
        for cD in [dX, dY]:
            GF.complDict(cDFull=dProc, cDAdd=cD)
        dfrProc, X, Y = GF.iniPdDfr(dProc), GF.iniPdDfr(dX), GF.iniPdDfr(dY)
    if dITp['onlySglLbl']:
        Y = toSglLbl(dITp, dfrY=Y)
    return dfrProc, X, Y, serNmerSeq, sorted(lSXCl)

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
    dY = {}
    for sLbl in serY:
        for sXCl in lXCl:
            GF.addToDictL(dY, cK=sXCl, cE=(1 if sXCl == sLbl else 0))
    return GF.iniPdDfr(dY, lSNmR=serY.index)

# def toSglLbl(dITp, dfrY):
#     serY = None
#     # check sanity
#     if GF.iniNpArr([(sum(serR) <= 1) for _, serR in dfrY.iterrows()]).all():
#         lY = [dITp['sStar']]*dfrY.shape[0]
#         lSer = [serR.index[serR == 1] for _, serR in dfrY.iterrows()]
#         for k, cI in enumerate(lSer):
#             if cI.size >= 1:
#                 lY[k] = cI.to_list()[0]
#         serY = GF.iniPdSer(lY, lSNmI=dfrY.index, nameS=dITp['sEffFam'])
#     return serY
def toSglLbl(dITp, dfrY):
    serY = None
    # check sanity
    if (GF.iniNpArr([(sum(serR) <= 1) for _, serR in dfrY.iterrows()]).all()
        or dITp['onlySglLbl']):
        lY = [dITp['sStar']]*dfrY.shape[0]
        lSer = [serR.index[serR == 1] for _, serR in dfrY.iterrows()]
        for k, cI in enumerate(lSer):
            if cI.size == 1:
                lY[k] = cI.to_list()[0]
        serY = GF.iniPdSer(lY, lSNmI=dfrY.index, nameS=dITp['sEffFam'])
    return serY

###############################################################################