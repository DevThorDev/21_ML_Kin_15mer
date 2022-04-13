# -*- coding: utf-8 -*-
###############################################################################
# --- O_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
import time

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
        self.lFullSeq, self.lNmerSeq = [], []
        self.getPDir()
        self.loadInpDfrs()
        print('Initiated "SeqAnalysis" base object.')

    # --- print methods -------------------------------------------------------
    def printDictLenSeq(self):
        print(GC.S_DS80, '\nLength-sequence dictionary:\n', GC.S_DS80, sep='')
        for cLen, lSeq in self.dLenSeq.items():
            print('Length-', cLen, '-sequence list (list length = ', len(lSeq),
                  '):\n\t', lSeq, sep='')

    def printDictSnipProbX(self, lSnipLen=None, maxLenSeqF=GC.R08):
        print(GC.S_DS80, '\nNmer snippet probability extended dictionary:\n',
              GC.S_DS80, sep='')
        for sSnip, dSub in self.dSnipProbX.items():
            if lSnipLen is None or len(sSnip) in lSnipLen:
                print(GC.S_DS04, ' Nmer snippet ', sSnip, GC.S_COL, sep='')
                for sSeqF, cProb in dSub.items():
                    if cProb > 0:
                        print(sSeqF[:maxLenSeqF], GC.S_DT03, GC.S_COL,
                              GC.S_SPACE, round(cProb, GC.R06), sep='')

    # --- methods for loading data --------------------------------------------
    def getPDirSFResEnd(self, pDirR=None):
        sFResEnd, sTest, sTrain = '', self.dITp['sTest'], self.dITp['sTrain']
        sCombS = GF.joinS([self.dITp['sCombined'], self.dITp['sCapS']])
        if self.dITp['sFSeqCheck'] == self.dITp['sFProcInpNmer']:
            pDirR, sFResEnd = self.pDirProcInp, self.dITp['sAllSeqNmer']
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iEnd=2) == sCombS:
            pDirR, sFResEnd = self.pDirResComb, sCombS
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iStart=-1) == sTest:
            pDirR, sFResEnd = self.pDirResComb, self.dITp['sTestData']
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iStart=-1) == sTrain:
            pDirR, sFResEnd = self.pDirResComb, self.dITp['sTrainData']
        return pDirR, sFResEnd

    # def loadInpNmer(self):
    #     self.dfrInpNmer, pFInp, iC = None, None, None
    #     if self.dITp['useNmerSeqFrom'] == self.dITp['sSeqCheck']:
    #         pDir, _ = self.getPDirSFResEnd(pDirR=self.pDirResComb)
    #         pFInp, iC = GF.joinToPath(pDir, self.dITp['sFSeqCheck']), 0
    #     elif self.dITp['useNmerSeqFrom'] == self.dITp['sIEffInp']:
    #         pFInp = GF.joinToPath(self.pDirResComb, self.dITp['sFIEffInp'])
    #     elif self.dITp['useNmerSeqFrom'] == self.dITp['sCombInp']:
    #         pFInp, iC = GF.joinToPath(self.pDirResComb, self.dITp['sFCombInp']), 0
    #     if pFInp is not None:
    #         self.dfrInpNmer = self.loadDfr(pF=pFInp, iC=iC)
    #     return iC

    def loadInpDfrs(self):
        self.lCombSeq, self.dLenSeq = [], {}
        pFIEffInp = GF.joinToPath(self.pDirResInfo, self.dITp['sFIEffInp'])
        pFCombInp = GF.joinToPath(self.pDirResComb, self.dITp['sFCombInp'])
        self.dfrIEff = self.loadDfr(pF=pFIEffInp, iC=0)
        self.dfrComb = self.loadDfr(pF=pFCombInp, iC=0)
        if self.dfrComb is not None:
            self.lCombSeq = list(self.dfrComb[self.dITp['sCNmer']].unique())

    # --- methods for obtaining the list of input Nmer-sequences --------------
    def getLIPosPyl(self, sFullSeq):
        # print('Obtaining list of Pyl position indices...')
        lIPosPyl, cDfr = [], self.dfrInpSeq
        if self.dITp['sPepPIP'] in cDfr.columns:
            dfrCSeq = cDfr[cDfr[self.dITp['sCCodeSeq']] == sFullSeq]
            lIPosPyl = dfrCSeq[self.dITp['sPepPIP']].unique()
        # print('Obtained list of Pyl position indices.')
        return [i - 1 for i in lIPosPyl]

    def getLFullSeq(self, iS=None, iE=None, stT=None):
        if self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns:
            print('Creating list of full input sequences...')
            mDsp, sFS = self.dITp['mDsp'], 'full sequences'
            lSFullSeq = list(self.dfrInpSeq[self.dITp['sCCodeSeq']].unique())
            lSFullSeq, iS, iE = GF.getItStartToEnd(lSFullSeq, iS, iE)
            for n, sSq in enumerate(lSFullSeq):
                lI = self.getLIPosPyl(sFullSeq=sSq)
                self.lFullSeq.append(FullSeq(self.dITp, sSq=sSq, lIPosPyl=lI))
                GF.showProgress(N=len(lSFullSeq), n=n, modeDisp=mDsp,
                                varText=sFS, startTime=stT)
            print('Created list of full input sequences...')
        return lSFullSeq, iS, iE

    def getLNmerSeq(self, lSFullSeq=[], lSNmerSeq=[], stT=None):
        print('Creating list of Nmer input sequences...')
        mDsp, sNS = self.dITp['mDsp'], 'Nmer sequences'
        if lSNmerSeq is None and self.dITp['sCNmer'] in self.dfrInpSeq.columns:
            if self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns:
                # reduce to Nmer sequences with corresponding full sequences
                cDfr = self.dfrInpSeq
                dfrCSeq = cDfr[cDfr[self.dITp['sCCodeSeq']].isin(lSFullSeq)]
                lSNmerSeq = list(dfrCSeq[self.dITp['sCNmer']].unique())
            else:
                lSNmerSeq = list(self.dfrInpSeq[self.dITp['sCNmer']].unique())
        for n, sSq in enumerate(lSNmerSeq):
            self.lNmerSeq.append(NmerSeq(self.dITp, sSq=sSq))
            GF.showProgress(N=len(lSNmerSeq), n=n, modeDisp=mDsp, varText=sNS,
                            startTime=stT)
        print('Created list of Nmer input sequences.')

    def getLInpSeq(self, cTim, lSSeq=None, stT=None):
        cStT = time.time()
        [iS, iE], lSNmerSeq = self.dITp['lIStartEnd'], lSSeq
        pDir, e = self.getPDirSFResEnd(pDirR=self.pDirResComb)
        pFInpSeq = GF.joinToPath(pDir, self.dITp['sFSeqCheck'])
        self.dfrInpSeq = self.loadDfr(pF=pFInpSeq, iC=0)
        lSFullSeq, iS, iE = self.getLFullSeq()
        self.getLNmerSeq(lSFullSeq, lSNmerSeq)
        e = GF.joinS([e, iS, iE])
        SF.modLSF(self.dITp, lSKeyF=self.dITp['lSKeyFRes'], sFE=e)
        cTim.updateTimes(iMth=1, stTMth=cStT, endTMth=time.time())

    # --- methods for generating the {seq. length: seq. list} dictionary ------
    def genDLenSeq(self, cTim, stT=None):
        cStT = time.time()
        print('Creating Nmer {seq. length: seq. list} dictionary...')
        mDsp, varTxt = self.dITp['mDsp'], 'Nmer sequences'
        if self.dITp['useNmerSeqFrom'] in [self.dITp['sSeqCheck'],
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

    # --- methods for performing the Nmer-sequence analysis -------------------
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

    def performLhAnalysis(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if self.dITp['calcWtLh'] or self.dITp['calcRelLh']:
            self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
            self.genDLenSeq(cTim=cTim, stT=stT)
            print('Performing calculation of Nmer sequence likelihoods...')
            cStT = time.time()
            dLV, d3, k, N = {}, {}, 0, len(lSSeq)*len(lEff)
            for cNmerSeq in self.lNmerSeq:
                for cEff in lEff:
                    serLh, cEff, maxLSnip = self.getRelLikelihoods(cEff)
                    SF.calcDictLikelihood(self.dITp, dLV, d3, cNmerSeq.dPrf,
                                          serLh, maxLSnip, cNmerSeq.sSeq, cEff)
                    self.complementDLV(dLV)
                    k += 1
                    GF.showProgress(N=N, n=k, modeDisp=self.dITp['mDsp'],
                                    varText='Nmer sequences', startTime=stT)
            self.saveDfrRelLikelihood(dLV, d3, lSCD3=self.dITp['lSCDfrLhD'])
            print('Performed calculation of Nmer sequence likelihoods.')
            cTim.updateTimes(iMth=3, stTMth=cStT, endTMth=time.time())

    def addCProbToDict(self, cSeqF, dIPosSeq):
        for sSS, (_, _, cProb) in dIPosSeq.items():
            if cProb > 0:
                nFS = len([cSq for cSq in self.lFullSeq if sSS in cSq.sSeq])
                GF.addToDictMnV(self.dSnipProbS, cK=sSS, cEl=cProb, nEl=nFS)
            GF.addToDictD(self.dSnipProbX, sSS, cSeqF.sSeq, cProb)

    def calcSnipProbTable(self):
        lSSeq = [cSeq.sSeq for cSeq in self.lNmerSeq]
        arrProb = GF.iniNpArr(shape=(len(lSSeq), len(self.dITp['lLenNmer'])))
        for i, cNmerS in enumerate(self.lNmerSeq):
            dProf = cNmerS.getProfileDict(maxLenSeq=max(self.dITp['lLenNmer']))
            for j, sSS in enumerate(dProf.values()):
                if sSS in self.dSnipProbS:
                    arrProb[i, j] = self.dSnipProbS[sSS]
        self.dfrProbTbl = GF.iniPdDfr(arrProb, lSNmR=lSSeq,
                                      lSNmC=self.dITp['lSCLenMerProb'])
        pFDfrProbTbl = GF.joinToPath(self.pDirResProb, self.dITp['sFProbTbl'])
        self.saveDfr(self.dfrProbTbl, pF=pFDfrProbTbl, saveAnyway=True)

    def performProbAnalysis(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if (self.dITp['calcSnipProb'] or self.dITp['calcWtProb'] or
            self.dITp['calcRelProb']):
            self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
            self.genDLenSeq(cTim=cTim, stT=stT)
            print('Performing calculation of Nmer sequence probabilities...')
            self.dSnipProbS, self.dSnipProbX, N = {}, {}, len(self.lFullSeq)
            for n, cSeqF in enumerate(self.lFullSeq):
                cStT = time.time()
                cD = GF.restrInt(self.dLenSeq, lRestrLen=self.dITp['lLenNmer'])
                cTim.updateTimes(iMth=4, stTMth=cStT, endTMth=time.time())
                for cLen, lSSeqNmer in cD.items():
                    cStT = time.time()
                    dIPosSeq = cSeqF.getDictPosSeq(lSSeq2F=lSSeqNmer)
                    cTim.updateTimes(iMth=5, stTMth=cStT, endTMth=time.time())
                    cStT = time.time()
                    self.addCProbToDict(cSeqF, dIPosSeq)
                    cTim.updateTimes(iMth=6, stTMth=cStT, endTMth=time.time())
                GF.showProgress(N=N, n=n, modeDisp=self.dITp['mDsp'],
                                varText='full sequences', startTime=stT)
            # self.printDictIPossProbPyl()
            print('Performed calculation of Nmer sequence probabilities.')
            cStT = time.time()
            self.saveDfrSnipProbS()
            self.saveDfrSnipProbX()
            cTim.updateTimes(iMth=7, stTMth=cStT, endTMth=time.time())
            print('Saved Nmer sequence probability DataFrame.')
            # self.printDictSnipProbX(lSnipLen=[1])
            self.calcSnipProbTable()

    # --- methods for printing results ----------------------------------------

    # --- methods for saving data ---------------------------------------------
    def saveDfrRelLikelihood(self, dLV, d3, lSCD3):
        pFDfrVal = GF.joinToPath(self.pDirResProb, self.dITp['sFResWtLh'])
        pFDfrDict = GF.joinToPath(self.pDirResProb, self.dITp['sFResRelLh'])
        if self.dITp['calcWtLh']:
            dfrVal = GF.dLV3CToDfr(dLV)
            self.saveDfr(dfrVal, pF=pFDfrVal, saveAnyway=True)
        if self.dITp['calcRelLh']:
            dfrDict = GF.d3ValToDfr(d3, lSCD3)
            self.saveDfr(dfrDict, pF=pFDfrDict, saveAnyway=True)

    def saveDfrSnipProbS(self):
        if self.dITp['calcSnipProb']:
            pRes, sFRes = self.pDirResProb, self.dITp['sFResSnipProbS']
            pFDfrSnipProb = GF.joinToPath(pRes, sFRes)
            dLV, lSCDfr = {}, self.dITp['lSCDfrProbS']
            for sSnip, cProb in self.dSnipProbS.items():
                lElRow = [sSnip, len(sSnip), round(cProb, GC.R06)]
                for sC, cV in zip(lSCDfr, lElRow):
                    GF.addToDictL(dLV, cK=sC, cE=cV)
            self.saveDfr(GF.dictToDfr(dLV, srtBy=self.dITp['lSrtByDfrProbS'],
                                      srtAsc=self.dITp['lSrtAscDfrProbS']),
                         pF=pFDfrSnipProb, saveAnyway=True)

    def saveDfrSnipProbX(self):
        if self.dITp['calcSnipProb']:
            pRes, sFRes = self.pDirResProb, self.dITp['sFResSnipProbX']
            pFDfrSnipProb = GF.joinToPath(pRes, sFRes)
            dLV, lSCDfr = {}, self.dITp['lSCDfrProbS']
            for sSnip, cDSub in self.dSnipProbX.items():
                cProb = round(sum(cDSub.values())/len(cDSub), GC.R06)
                lElRow = [sSnip, len(sSnip), cProb]
                for sC, cV in zip(lSCDfr, lElRow):
                    GF.addToDictL(dLV, cK=sC, cE=cV)
            self.saveDfr(GF.dictToDfr(dLV, srtBy=self.dITp['lSrtByDfrProbS'],
                                      srtAsc=self.dITp['lSrtAscDfrProbS']),
                         pF=pFDfrSnipProb, saveAnyway=True)

###############################################################################
