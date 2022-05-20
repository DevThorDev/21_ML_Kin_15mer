# -*- coding: utf-8 -*-
###############################################################################
# --- O_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
import os, time

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass, NmerSeq, FullSeq

# -----------------------------------------------------------------------------
class SeqAnalysis(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=2, lITpUpd=[1]):
        super().__init__(inpDat)
        self.idO = 'O_02'
        self.descO = 'Analysis of Nmer-sequences'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.iniObj()
        self.getPDir()
        self.fillDPFVal()
        self.loadInpDfrs()
        print('Initiated "SeqAnalysis" base object.')

    def iniObj(self):
        self.nNmer = 0
        self.lSFull, self.lSNmer, self.lFullSeq, self.lNmerSeq = [], [], [], []
        self.d2TotalProbSnip, self.d2CondProbSnip = {}, {}
        self.d2NOccSglPos, self.d2RFreqSglPos = {}, {}

    # --- print methods -------------------------------------------------------
    def printDictPaths(self):
        print(GC.S_EQ20, 'Dictionary of file paths:')
        for sK, sP in self.dPF.items():
            print(sK, ':\t', sP, sep='')

    def printLInpSeq(self, sLSeqPrnt='Nmer', prntFullI=False, maxNPrnt=10):
        lSeq = self.lFullSeq
        if sLSeqPrnt == 'Nmer':
            lSeq = self.lNmerSeq
        print(GC.S_EQ20, 'List of', sLSeqPrnt, 'sequences:')
        for k, cSeq in enumerate(lSeq[:maxNPrnt]):
            if prntFullI:
                print(GC.S_DS24, 'Sequence ', k + 1, ':', sep='')
                print(cSeq)
            else:
                print('Sequence ', k + 1, ': ', cSeq.sSeq, sep='')
        print(GC.S_DS80, GC.S_NEWL, 'Length of list of ', sLSeqPrnt,
              ' sequences: ', len(lSeq), sep='')

    def printLFullSeq(self, prntFullI=False):
        self.printLInpSeq(sLSeqPrnt='Full', prntFullI=prntFullI)

    def printLNmerSeq(self, prntFullI=False):
        self.printLInpSeq(sLSeqPrnt='Nmer', prntFullI=prntFullI)

    def printDictLenSeq(self):
        print(GC.S_DS80, '\nLength-sequence dictionary:\n', GC.S_DS80, sep='')
        for cLen, lSeq in self.dLenSeq.items():
            print('Length-', cLen, '-sequence list (list length = ', len(lSeq),
                  '):\n\t', lSeq, sep='')

    def printDictSnipX(self, lSnipLen=None, cSnip=None, maxLenSeqF=GC.R24):
        print(GC.S_DS80, '\nNmer snippet probability extended dictionary:\n',
              GC.S_DS80, sep='')
        for sSnip, dSub in self.dSnipX.items():
            if lSnipLen is None or len(sSnip) in lSnipLen:
                if sSnip is None or sSnip == cSnip:
                    print(GC.S_DS04, ' Nmer snippet ', sSnip, GC.S_COL, sep='')
                    for sSeqF, tV in dSub.items():
                        print(sSeqF[:maxLenSeqF], GC.S_DT03, GC.S_COL,
                              GC.S_SPACE, tV, sep='')

    def printDSglPosSub(self, dSub, iP=0):
        print(GC.S_PL20, 'Position index:', iP)
        for chAAc, cV in sorted(dSub.items()):
            print(GC.S_DS04, chAAc, GC.S_COL, round(cV, GC.R08))
        print(GC.S_PL04, GC.S_QMK, GC.S_COL, round(sum(dSub.values()), GC.R08))

    def printD2SglPos(self, sDPrnt='RelFreq', lIPPrnt=None):
        cD = self.d2NOccSglPos
        if sDPrnt == 'RelFreq':
            cD = self.d2RFreqSglPos
        print(GC.S_EQ20, 'Dictionary of', sDPrnt, 'of single positions:')
        if lIPPrnt is None:
            for iP, dSub in cD.items():
                self.printDSglPosSub(dSub, iP=iP)
        else:
            for iP in lIPPrnt:
                if iP in cD:
                    self.printDSglPosSub(cD[iP], iP=iP)

    def printD2NOccSglPos(self, lIPPrnt=None):
        self.printD2SglPos(sDPrnt='NOcc', lIPPrnt=lIPPrnt)

    def printD2RFreqSglPos(self, lIPPrnt=None):
        self.printD2SglPos(sDPrnt='RelFreq', lIPPrnt=lIPPrnt)

    # --- methods for filling the result paths dictionary ---------------------
    def getPDirRes(self, pDirR=None):
        sCombS = GF.joinS([self.dITp['sCombined'], self.dITp['sCapS']])
        setSTT = {self.dITp['sTest'], self.dITp['sTrain']}
        if self.dITp['sFSeqCheck'] == self.dITp['sFProcInpNmer']:
            pDirR = self.pDirProcInp
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iEnd=2) == sCombS:
            pDirR = self.pDirResComb
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iStart=-1) in setSTT:
            pDirR = self.pDirResComb
        return pDirR

    def fillDPFVal(self):
        pI, pC, pP = self.pDirResInfo, self.pDirResComb, self.pDirResProb
        self.dPF = {'IEffInp': GF.joinToPath(pI, self.dITp['sFIEffInp']),
                    'ProcInp': GF.joinToPath(pC, self.dITp['sFProcInp']),
                    'CombInp': GF.joinToPath(pC, self.dITp['sFCombInp'])}
        self.dPF['SeqCheck'] = GF.joinToPath(self.getPDirRes(pDirR=pC),
                                             self.dITp['sFSeqCheck'])
        self.dPF['lNmerSeq'] = GF.joinToPath(pP, self.dITp['sFLNmerSeq'])
        self.dPF['lFullSeq'] = GF.joinToPath(pP, self.dITp['sFLFullSeq'])
        self.dPF['ResWtLh'] = GF.joinToPath(pP, self.dITp['sFResWtLh'])
        self.dPF['ResRelLh'] = GF.joinToPath(pP, self.dITp['sFResRelLh'])
        self.dPF['SnipDictS'] = GF.joinToPath(pP, self.dITp['sFSnipDictS'])
        self.dPF['SnipDictX'] = GF.joinToPath(pP, self.dITp['sFSnipDictX'])
        self.dPF['SnipDfrS'] = GF.joinToPath(pP, self.dITp['sFSnipDfrS'])
        self.dPF['SnipDfrX'] = GF.joinToPath(pP, self.dITp['sFSnipDfrX'])
        self.dPF['ProbTblFS'] = GF.joinToPath(pP, self.dITp['sFProbTblFS'])
        self.dPF['ProbTblTP'] = GF.joinToPath(pP, self.dITp['sFProbTblTP'])
        self.dPF['ProbTblCP'] = GF.joinToPath(pP, self.dITp['sFProbTblCP'])
        self.dPF['ResWtProb'] = GF.joinToPath(pP, self.dITp['sFResWtProb'])
        self.dPF['ResRelProb'] = GF.joinToPath(pP, self.dITp['sFResRelProb'])
        self.dPF['DictNOccSP'] = GF.joinToPath(pP, self.dITp['sFDictNOccSP'])
        self.dPF['DictRFreqSP'] = GF.joinToPath(pP, self.dITp['sFDictRFreqSP'])
        self.dPF['ResNOccSP'] = GF.joinToPath(pP, self.dITp['sFResNOccSP'])
        self.dPF['ResRFreqSP'] = GF.joinToPath(pP, self.dITp['sFResRFreqSP'])

    def updateDPFVal(self):
        pP = self.pDirResProb
        self.dPF['SnipDictS'] = GF.joinToPath(pP, self.dITp['sFSnipDictS'])
        self.dPF['SnipDictX'] = GF.joinToPath(pP, self.dITp['sFSnipDictX'])
        self.dPF['SnipDfrS'] = GF.joinToPath(pP, self.dITp['sFSnipDfrS'])
        self.dPF['SnipDfrX'] = GF.joinToPath(pP, self.dITp['sFSnipDfrX'])

    # --- methods for modifying file names ------------------------------------
    def getSFResEnd(self):
        sFResEnd, sTest, sTrain = '', self.dITp['sTest'], self.dITp['sTrain']
        sCombS = GF.joinS([self.dITp['sCombined'], self.dITp['sCapS']])
        if self.dITp['sFSeqCheck'] == self.dITp['sFProcInp']:
            sFResEnd = self.dITp['sAllSeqNmer']
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iEnd=2) == sCombS:
            sFResEnd = sCombS
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iStart=-1) == sTest:
            sFResEnd = self.dITp['sTestData']
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iStart=-1) == sTrain:
            sFResEnd = self.dITp['sTrainData']
        return sFResEnd

    # --- methods for loading data --------------------------------------------
    def loadInpDfrs(self):
        self.lCombSeq, self.dLenSeq = [], {}
        if self.dITp['useNmerSeqFrom'] == self.dITp['sIEffInp']:
            self.dfrIEff = self.loadData(pF=self.dPF['IEffInp'], iC=0)
        if self.dITp['calcWtLh'] or self.dITp['calcRelLh']:
            self.dfrComb = self.loadData(pF=self.dPF['CombInp'], iC=0)
        if GF.Xist(self.dfrComb) and self.dITp['sCNmer'] in self.dfrComb:
            self.lCombSeq = list(self.dfrComb[self.dITp['sCNmer']].unique())

    # --- methods for obtaining the list of input Nmer-sequences --------------
    def getLIPosPyl(self, sFullSeq):
        lIPosPyl, cDfr = [], self.dfrInpSeq
        if self.dITp['sPepPIP'] in cDfr.columns:
            dfrCSeq = cDfr[cDfr[self.dITp['sCCodeSeq']] == sFullSeq]
            lIPosPyl = dfrCSeq[self.dITp['sPepPIP']].unique()
        return [i - 1 for i in lIPosPyl]

    def fillLFullSeq(self, sTxt='', stT=None):
        for n, sSq in enumerate(self.lSFull):
            lI = self.getLIPosPyl(sFullSeq=sSq)
            self.lFullSeq.append(FullSeq(self.dITp, sSq=sSq, lIPosPyl=lI))
            GF.showProgress(N=len(self.lSFull), n=n, varText=sTxt,
                            modeDisp=self.dITp['mDsp'], startTime=stT)

    def getLFullSeq(self, iS=None, iE=None, unqS=True, stT=None):
        self.lSFull = None
        if self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns:
            t = SF.getLSFullSeq(self.dITp, self.dfrInpSeq, iS, iE, unqS=unqS)
            self.lSFull, iS, iE = t
            if not GF.Xist(self.lFullSeq):
                print('Creating list of full input sequences...')
                sFS = 'full sequences [getLFullSeq]'
                self.fillLFullSeq(sTxt=sFS, stT=stT)
                print('Created list of full input sequences.')
        return iS, iE

    def getLNmerSeq(self, red2WF=True, unqS=True, stT=None):
        if not GF.Xist(self.lSNmer):
            self.lSNmer = SF.getLSNmer(self.dITp, self.dfrInpSeq, self.lSFull,
                                       self.lSNmer, red2WF=red2WF, unqS=unqS)
        if not GF.Xist(self.lNmerSeq):
            print('Creating list of Nmer input sequences...')
            sNS = 'Nmer sequences [getLNmerSeq]'
            for n, sSq in enumerate(self.lSNmer):
                self.lNmerSeq.append(NmerSeq(self.dITp, sSq=sSq))
                GF.showProgress(N=len(self.lSNmer), n=n, varText=sNS,
                                modeDisp=self.dITp['mDsp'], startTime=stT)
            print('Created list of Nmer input sequences.')

    def getLInpSeq(self, cTim, lSSeq=None, getFullS=True, red2WFullS=True,
                   uniqueS=True, stT=None):
        cStT = time.time()
        pFNS, pFFS = self.dPF['lNmerSeq'], self.dPF['lFullSeq']
        [iS, iE], self.lSNmer = self.dITp['lIStartEnd'], lSSeq
        self.dfrInpSeq = self.loadData(pF=self.dPF['SeqCheck'], iC=0)
        if getFullS:
            iS, iE = self.getLFullSeq(iS, iE, unqS=uniqueS, stT=stT)
        self.getLNmerSeq(red2WF=red2WFullS, unqS=uniqueS, stT=stT)
        SF.modLSF(self.dITp, lSKeyF=self.dITp['lSKeyFRes'],
                  sFE=GF.joinS([self.getSFResEnd(), iS, iE]))
        self.updateDPFVal()
        if getFullS:
            sNmSer = self.dITp['sTargSeq']
            self.saveListAsSer(self.lSFull, pF=pFFS, sName=sNmSer)
        self.saveListAsSer(self.lSNmer, pF=pFNS, sName=self.dITp['sNmer'])
        cTim.updateTimes(iMth=1, stTMth=cStT, endTMth=time.time())

    # --- methods for generating the {seq. length: seq. list} dictionary ------
    def genDLenSeq(self, cTim, stT=None):
        cStT = time.time()
        print('Creating Nmer {seq. length: seq. list} dictionary...')
        mDsp, varTxt = self.dITp['mDsp'], 'Nmer sequences [genDLenSeq]'
        if self.dITp['useNmerSeqFrom'] in [self.dITp['sProcInp'],
                                           self.dITp['sCombInp']]:
            for n, cNmerSeq in enumerate(self.lNmerSeq):
                for cLen, sSeq in cNmerSeq.dPrf.items():
                    GF.addToDictL(self.dLenSeq, cK=cLen, cE=sSeq, lUniqEl=True)
                GF.showProgress(N=len(self.lNmerSeq), n=n, modeDisp=mDsp,
                                varText=varTxt, startTime=stT)
        elif self.dITp['useNmerSeqFrom'] == self.dITp['sIEffInp']:
            lSEffCode = [self.dITp['sEffCode']]
            lSq = [s for s in self.dfrIEff.columns if s not in lSEffCode]
            for n, sSq in enumerate(lSq):
                GF.addToDictL(self.dLenSeq, cK=len(sSq), cE=sSq)
                GF.showProgress(N=len(lSq), n=n, modeDisp=mDsp, varText=varTxt,
                                startTime=stT)
        print('Created Nmer {seq. length: seq. list} dictionary.')
        cTim.updateTimes(iMth=2, stTMth=cStT, endTMth=time.time())

    # --- methods performing the Nmer-sequence snippet likelihood analysis ----
    def getRelLikelihoods(self, cEff=None):
        serEff, maxLenSnip = None, max([len(s) for s in self.dfrIEff.columns])
        if cEff is None:
            cEff = self.dITp['sAnyEff']
        if cEff in self.dfrIEff.index:
            serEff = self.dfrIEff.loc[cEff, :]
            for k in self.dLenSeq:
                maxV = max(serEff.loc[self.dLenSeq[k]])
                if maxV > 0:
                    serEff.loc[self.dLenSeq[k]] /= maxV
        return serEff, cEff, maxLenSnip

    def complementDLV(self, dLV):
        dLV[self.dITp['sInCombRes']] = [0]*len(dLV[self.dITp['sCNmer']])
        for k, sNmer in enumerate(dLV[self.dITp['sCNmer']]):
            if sNmer in self.lCombSeq:
                dLV[self.dITp['sInCombRes']][k] = 1

    def lhAnalysisLoop(self, lEff=[None], lSSeq=None, stT=None):
        dLV, d3, k, N = {}, {}, 0, len(lSSeq)*len(lEff)
        sNS = 'Nmer sequences [lhAnalysisLoop]'
        for cNmerSeq in self.lNmerSeq:
            for cEff in lEff:
                serLh, cEff, maxLSnip = self.getRelLikelihoods(cEff)
                SF.calcDictLikelihood(self.dITp, dLV, d3, cNmerSeq.dPrf,
                                      serLh, maxLSnip, cNmerSeq.sSeq, cEff)
                self.complementDLV(dLV)
                k += 1
                GF.showProgress(N=N, n=k, modeDisp=self.dITp['mDsp'],
                                varText=sNS, startTime=stT)
        return dLV, d3

    def performLhAnalysis(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if self.dITp['calcWtLh'] or self.dITp['calcRelLh']:
            self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
            self.genDLenSeq(cTim=cTim, stT=stT)
            print('Performing calculation of Nmer sequence likelihoods...')
            cStT = time.time()
            dLV, d3 = self.lhAnalysisLoop(lEff=lEff, lSSeq=lSSeq, stT=stT)
            self.saveDfrRelLikelihood(dLV, d3, lSCD3=self.dITp['lSCDfrLhD'])
            print('Performed calculation of Nmer sequence likelihoods.')
            cTim.updateTimes(iMth=3, stTMth=cStT, endTMth=time.time())

    # --- methods performing the Nmer-sequence snippet probability analysis ---
    def addToDictSnip(self, cSeqF, dIPosSeq):
        for sSS, (_, _, nPyl, nTtl, cPrb) in dIPosSeq.items():
            if (self.dITp['calcSnipS'] and nPyl > 0):
                nFS = len([cSq for cSq in self.lFullSeq if sSS in cSq.sSeq])
                GF.addToDictMnV(self.dSnipS, cK=sSS, cEl=cPrb, nEl=nFS)
            if self.dITp['calcSnipX']:
                tV = (nPyl, nTtl, GF.getFract(nPyl, nTtl), cPrb)
                GF.addToDictD(self.dSnipX, sSS, cSeqF.sSeq, tV)

    def probAnalysisLoop(self, cTim, N, stT=None):
        cStT = time.time()
        cDLenSeq = GF.restrInt(self.dLenSeq, lRestrLen=self.dITp['lLenNmer'])
        cTim.updateTimes(iMth=4, stTMth=cStT, endTMth=time.time())
        sFS = 'full sequences [probAnalysisLoop]'
        for n, cSeqF in enumerate(self.lFullSeq):
            for lSSeqNmer in cDLenSeq.values():
                cStT = time.time()
                dIPosSeq = cSeqF.getDictPosSeq(lSSeq2F=lSSeqNmer)
                cTim.updateTimes(iMth=5, stTMth=cStT, endTMth=time.time())
                cStT = time.time()
                self.addToDictSnip(cSeqF, dIPosSeq)
                cTim.updateTimes(iMth=6, stTMth=cStT, endTMth=time.time())
            GF.showProgress(N=N, n=n, modeDisp=self.dITp['mDsp'],
                            varText=sFS, startTime=stT)
        print('Performed calculation of Nmer sequence probabilities.')

    def performProbAnalysis(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if (self.dITp['calcSnipS'] or self.dITp['calcSnipX'] or
            self.dITp['calcWtProb'] or self.dITp['calcRelProb']):
            self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
            self.genDLenSeq(cTim=cTim, stT=stT)
            print('Performing calculation of Nmer sequence probabilities...')
            self.dSnipS, self.dSnipX, N = {}, {}, len(self.lFullSeq)
            self.probAnalysisLoop(cTim=cTim, N=N, stT=stT)
            cStT = time.time()
            self.saveDfrSnipS()
            self.saveDfrSnipX()
            cTim.updateTimes(iMth=7, stTMth=cStT, endTMth=time.time())
            print('Saved Nmer sequence probability DataFrame.')
            # self.printDictSnipX(lSnipLen=[7], cSnip='AQRTLHG')
        if ((self.dITp['calcWtProb'] or self.dITp['calcRelProb']) and
            self.dITp['calcSnipX']):
            self.calcProbTable(cTim=cTim, lSSeq=lSSeq, stT=stT)

    def probTableLoop(self, lSerD, nXt, nSeq=0, stT=None):
        sNS = 'Nmer sequences [calcProbTable]'
        for i, cNmerS in enumerate(self.lNmerSeq):
            dProf = cNmerS.getProfileDict(maxLenSeq=max(self.dITp['lLenNmer']))
            for j, sSS in enumerate(dProf.values()):
                if sSS in self.dSnipX:
                    lDat = GF.calcPylProb(cSS=sSS, dCSS=self.dSnipX[sSS])
                    for k, sCX in enumerate(self.dITp['lSCDfrProbX']):
                        lSerD[j*nXt + k].at[i] = lDat[k]
            GF.showProgress(N=nSeq, n=i, modeDisp=self.dITp['mDsp'],
                            varText=sNS, startTime=stT)

    def calcProbTable(self, cTim, lSSeq=None, stT=None):
        if not self.dITp['convSnipXToProbTbl']:
            return None
        self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
        cStT = time.time()
        self.dSnipX = GF.pickleLoadDict(pF=self.dPF['SnipDictX'], reLoad=True)
        lSSeq, nSeq = [cSeq.sSeq for cSeq in self.lNmerSeq], len(self.lNmerSeq)
        lSC, nXt =  self.dITp['lSCMerAll'], len(self.dITp['lSCXt'])
        lSerD = [GF.iniPdSer(shape=(nSeq,), nameS=sC) for sC in lSC]
        self.probTableLoop(lSerD, nXt=nXt, nSeq=nSeq, stT=stT)
        self.dfrProbTbl = GF.concLSerAx1(lSerD)
        self.dfrProbTbl.index = lSSeq
        self.saveDfr(self.dfrProbTbl, pF=self.dPF['ProbTblFS'],
                     idxLbl=self.dITp['sCNmer'])
        cTim.updateTimes(iMth=8, stTMth=cStT, endTMth=time.time())

    # --- methods checking the Nmer-sequence snippet probability analysis -----
    def getD2TotalProbSnip(self, stT=None):
        nSeq, sNS = len(self.lNmerSeq), 'Nmer sequences [getD2TotalProbSnip]'
        for i, cNmerS in enumerate(self.lNmerSeq):
            dSub = cNmerS.getTotalProbSnip(self.dITp, lInpSeq=self.lFullSeq)
            self.d2TotalProbSnip[cNmerS.sSeq] = dSub
            GF.showProgress(N=nSeq, n=i, modeDisp=self.dITp['mDsp'],
                            varText=sNS, startTime=stT)

    def getD2CondProbSnip(self, stT=None):
        nSeq, sNS = len(self.lNmerSeq), 'Nmer sequences [getD2CondProbSnip]'
        for i, cNmerS in enumerate(self.lNmerSeq):
            dSub = cNmerS.getCondProbSnip(self.dITp, lInpSeq=self.lNmerSeq,
                                          lFullSeq=self.lFullSeq)
            self.d2CondProbSnip[cNmerS.sSeq] = dSub
            GF.showProgress(N=nSeq, n=i, modeDisp=self.dITp['mDsp'],
                            varText=sNS, startTime=stT)

    def performTCProbAnalysis(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if (self.dITp['calcTotalProb'] or self.dITp['calcCondProb']):
            self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
            if self.dITp['calcTotalProb']:
                cStT = time.time()
                if not GF.Xist(self.d2TotalProbSnip):
                    self.getD2TotalProbSnip(stT=stT)
                cTim.updateTimes(iMth=9, stTMth=cStT, endTMth=time.time())
                cStT = time.time()
                self.saveD2TCProbSnipAsDfr(typeProb=self.dITp['sTtlProb'])
                cTim.updateTimes(iMth=11, stTMth=cStT, endTMth=time.time())
            if self.dITp['calcCondProb']:
                cStT = time.time()
                if not GF.Xist(self.d2CondProbSnip):
                    self.getD2CondProbSnip(stT=stT)
                cTim.updateTimes(iMth=10, stTMth=cStT, endTMth=time.time())
                cStT = time.time()
                self.saveD2TCProbSnipAsDfr(typeProb=self.dITp['sCndProb'])
                cTim.updateTimes(iMth=11, stTMth=cStT, endTMth=time.time())

    # --- methods performing the Nmer-sequence single pos. prob. analysis -----
    def getInfoLSFull(self, unqS=True):
        [iS, iE] = self.dITp['lIStartEnd']
        self.dfrInpSeq = self.loadData(pF=self.dPF['SeqCheck'], iC=0)
        if not GF.Xist(self.lSFull):
            assert self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns
            t = SF.getLSFullSeq(self.dITp, self.dfrInpSeq, iS, iE, unqS=unqS)
            self.lSFull, iS, iE = t
        SF.modLSF(self.dITp, lSKeyF=['sFResNOccSP', 'sFResRFreqSP'],
                  sFE=GF.joinS([self.getSFResEnd(), iS, iE]))

    def getD2Full2NmerSRem(self, iCt=0):
        d2Full2Nmer, sSeqRem = {sFull: {} for sFull in self.lSFull}, ''
        for sFull in self.lSFull:
            lI, sMod = sorted(self.getLIPosPyl(sFullSeq=sFull)), sFull
            for k, cI in enumerate(lI):
                iL, iR = max(0, cI - iCt), min(len(sFull), cI + iCt + 1)
                sNmer = sFull[iL:iR]
                GF.addToDictCt(d2Full2Nmer[sFull], cK=sNmer)
                sMod = sMod[:iL] + GC.S_0*(iR - iL) + sMod[iR:]
            sSeqRem += ''.join([chAAc for chAAc in sMod if chAAc != GC.S_0])
        self.nNmer = sum([sum(dSub.values()) for dSub in d2Full2Nmer.values()])
        return d2Full2Nmer, sSeqRem

    def fillD2ResSglPos(self, d2Full2Nmer, sR='', iCt=0, stT=None):
        # fill Nmer position dictionaries
        if self.nNmer > 0:
            nNmer, sNS = len(d2Full2Nmer), 'Nmer sequences [fillD2ResSglPos]'
            for m, dNmer in enumerate(d2Full2Nmer.values()):
                for sNmer, n in dNmer.items():
                    for iP in self.d2NOccSglPos:
                        chAAc = sNmer[iP + iCt]
                        GF.addToDictCt(self.d2NOccSglPos[iP], cK=chAAc, nInc=n)
                        GF.addToDictCt(self.d2RFreqSglPos[iP], cK=chAAc,
                                       nInc=n/self.nNmer)
                GF.showProgress(N=nNmer, n=m, modeDisp=self.dITp['mDsp'],
                                varText=sNS, startTime=stT)
        # fill outside-Nmer amino acid dictionaries
        self.d2NOccSglPos[None], self.d2RFreqSglPos[None] = {}, {}
        for chAAc in sR:
            GF.addToDictCt(self.d2NOccSglPos[None], cK=chAAc)
            GF.addToDictCt(self.d2RFreqSglPos[None], cK=chAAc, nInc=1/len(sR))

    def processLFullSeq(self, unqS=True, stT=None):
        iCt, lenNmer = self.dITp['iCentNmer'], self.dITp['lenNmerDef']
        self.getInfoLSFull(unqS=unqS)
        d2Full2Nmer, sR = self.getD2Full2NmerSRem(iCt=iCt)
        self.d2NOccSglPos = {iP: {} for iP in list(range(-iCt, lenNmer - iCt))}
        self.d2RFreqSglPos = {iP: {} for iP in self.d2NOccSglPos}
        self.fillD2ResSglPos(d2Full2Nmer, sR=sR, iCt=iCt, stT=stT)
        GF.pickleSaveDict(cD=self.d2NOccSglPos, pF=self.dPF['DictNOccSP'])
        GF.pickleSaveDict(cD=self.d2RFreqSglPos, pF=self.dPF['DictRFreqSP'])

    def getProbSglPos(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if self.dITp['calcSglPos']:
            cStT = time.time()
            if not os.path.isfile(self.dPF['DictRFreqSP']):
                self.processLFullSeq(unqS=True, stT=stT)
            else:
                self.d2RFreqSglPos = GF.pickleLoadDict(self.dPF['DictRFreqSP'])
            if (not GF.Xist(self.d2NOccSglPos) and
                os.path.isfile(self.dPF['DictNOccSP'])):
                self.d2NOccSglPos = GF.pickleLoadDict(self.dPF['DictNOccSP'])
            self.saveDfrResNOccSglPos()
            self.saveDfrResRFreqSglPos()
            cTim.updateTimes(iMth=12, stTMth=cStT, endTMth=time.time())

    # --- methods for saving and loading data ---------------------------------
    def saveDfrRelLikelihood(self, dLV, d3, lSCD3):
        if self.dITp['calcWtLh']:
            dfrVal = GF.dLV3CToDfr(dLV)
            self.saveDfr(dfrVal, pF=self.dPF['ResWtLh'])
        if self.dITp['calcRelLh']:
            dfrDict = GF.d3ValToDfr(d3, lSCD3)
            self.saveDfr(dfrDict, pF=self.dPF['ResRelLh'])

    def convDictToDfrS(self):
        dLV, lSCDfr = {}, self.dITp['lSCDfrProbS']
        for sSnip, cProb in self.dSnipS.items():
            lElRow = [sSnip, len(sSnip), round(cProb, GC.R08)]
            assert len(lElRow) == len(lSCDfr)
            for sC, cV in zip(lSCDfr, lElRow):
                GF.addToDictL(dLV, cK=sC, cE=cV)
        return GF.dictToDfr(dLV, srtBy=self.dITp['lSrtByDfrProbS'],
                            srtAsc=self.dITp['lSrtAscDfrProbS'], dropNA=True)

    def saveDfrSnipS(self):
        if GF.Xist(self.dSnipS):
            GF.pickleSaveDict(cD=self.dSnipS, pF=self.dPF['SnipDictS'])
            if self.dITp['saveAsDfrS']:
                self.dfrSnipS = self.convDictToDfrS()
                self.saveDfr(self.dfrSnipS, pF=self.dPF['SnipDfrS'])

    def convDictToDfrX(self):
        dLV, lSCDfr = {}, self.dITp['lSCDfrWFSProbX']
        for sSnip, cDSub in self.dSnipX.items():
            for sFSeq, t in cDSub.items():
                lElRow = [sSnip, sFSeq] + list(t)
                assert len(lElRow) == len(lSCDfr)
                for sC, cV in zip(lSCDfr, lElRow):
                    GF.addToDictL(dLV, cK=sC, cE=cV)
        return GF.dictToDfr(dLV, srtBy=self.dITp['lSrtByDfrProbX'],
                            srtAsc=self.dITp['lSrtAscDfrProbX'], dropNA=True)

    def saveDfrSnipX(self):
        if GF.Xist(self.dSnipX):
            GF.pickleSaveDict(cD=self.dSnipX, pF=self.dPF['SnipDictX'])
            if self.dITp['saveAsDfrX']:
                self.dfrSnipX = self.convDictToDfrX()
                lIdx = self.dfrSnipX[self.dITp['sSnippet']].to_list()
                self.dfrSnipX.index = lIdx
                self.saveDfr(self.dfrSnipX, pF=self.dPF['SnipDfrX'],
                             saveIdx=True, idxLbl=False)

    def loadDfrSnipX(self, reLoad=False):
        pFDfr = self.dPF['SnipDfrX']
        if ((reLoad or not GF.Xist(self.dSnipX)) and os.path.isfile(pFDfr)):
            self.dfrSnipX = self.loadData(pFDfr, iC=0)
        elif not (self.dSnipX is None or os.path.isfile(pFDfr)):
            self.dfrSnipX = self.convDictToDfrX()
        elif (self.dSnipX is None and not os.path.isfile(pFDfr)):
            print('ERROR: dSnipX is None and file', pFDfr, 'not found.')
            assert False
        if self.dfrSnipX is None:
            print('ERROR: dfrSnipX is None.')
            assert False
        else:
            return self.dfrSnipX.to_dict(orient='dict')

    def saveD2TCProbSnipAsDfr(self, typeProb=GC.S_TOTAL_PROB):
        cD2, pFC = self.d2TotalProbSnip, self.dPF['ProbTblTP']
        if typeProb == self.dITp['sCndProb']:
            cD2, pFC = self.d2CondProbSnip, self.dPF['ProbTblCP']
        if GF.Xist(cD2):
            dLV, lSCDfr = {}, self.dITp['lSCMerAll']
            for cDSub in cD2.values():
                lElRow = []
                for sSnip, lV in cDSub.items():
                    lElRow += ([sSnip] + lV)
                assert len(lElRow) == len(lSCDfr)
                for sC, cV in zip(lSCDfr, lElRow):
                    GF.addToDictL(dLV, cK=sC, cE=cV)
            dfrVal = GF.dictToDfr(dLV, idxDfr=list(cD2))
            self.saveDfr(dfrVal, pF=pFC, idxLbl=self.dITp['sCNmer'])

    def saveDfrResSglPos(self, sDSave='RelFreq'):
        cD2, pFC = self.d2NOccSglPos, self.dPF['ResNOccSP']
        if sDSave == 'RelFreq':
            cD2, pFC = self.d2RFreqSglPos, self.dPF['ResRFreqSP']
        assert list(cD2).count(None) == 1
        lIP = sorted([iP for iP in cD2 if iP is not None]) + [None]
        dSCol = {str(iP): iP for iP in lIP if iP is not None}
        dSCol[self.dITp['sNotInNmer']] = None
        lSRow = GF.getListUniqueWD2(d2=cD2, srtDir='asc')
        dfrRes = GF.iniPdDfr(lSNmC=list(dSCol), lSNmR=lSRow)
        for chAAc in dfrRes.index:
            for sIP in dfrRes.columns:
                if chAAc in cD2[dSCol[sIP]]:
                    dfrRes.at[chAAc, sIP] = cD2[dSCol[sIP]][chAAc]
        self.saveDfr(dfrRes, pF=pFC, idxLbl=self.dITp['sAminoAcid'])

    def saveDfrResNOccSglPos(self):
        self.saveDfrResSglPos(sDSave='NOcc')

    def saveDfrResRFreqSglPos(self):
        self.saveDfrResSglPos(sDSave='RelFreq')

###############################################################################
