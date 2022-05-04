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
        self.lSFull, self.lSNmer, self.lFullSeq, self.lNmerSeq = [], [], [], []
        self.d2CondProbSnip = {}
        self.getPDir()
        self.fillDPFVal()
        self.loadInpDfrs()
        print('Initiated "SeqAnalysis" base object.')

    # --- print methods -------------------------------------------------------
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
                    'CombInp': GF.joinToPath(pC, self.dITp['sFCombInp'])}
        self.dPF['SeqCheck'] = GF.joinToPath(self.getPDirRes(pDirR=pC),
                                             self.dITp['sFSeqCheck'])
        self.dPF['LInpSeq'] = GF.joinToPath(pP, self.dITp['sFLInpSeq'])
        self.dPF['ResWtLh'] = GF.joinToPath(pP, self.dITp['sFResWtLh'])
        self.dPF['ResRelLh'] = GF.joinToPath(pP, self.dITp['sFResRelLh'])
        self.dPF['SnipDictS'] = GF.joinToPath(pP, self.dITp['sFSnipDictS'])
        self.dPF['SnipDictX'] = GF.joinToPath(pP, self.dITp['sFSnipDictX'])
        self.dPF['SnipDfrS'] = GF.joinToPath(pP, self.dITp['sFSnipDfrS'])
        self.dPF['SnipDfrX'] = GF.joinToPath(pP, self.dITp['sFSnipDfrX'])
        self.dPF['ProbTblFS'] = GF.joinToPath(pP, self.dITp['sFProbTblFS'])
        self.dPF['ProbTblCP'] = GF.joinToPath(pP, self.dITp['sFProbTblCP'])

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
        if self.dITp['sFSeqCheck'] == self.dITp['sFProcInpNmer']:
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
        self.dfrIEff = self.loadData(pF=self.dPF['IEffInp'], iC=0)
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
        print('Created list of full input sequences.')

    def getLFullSeq(self, iS=None, iE=None, stT=None):
        self.lSFull = None
        if self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns:
            print('Creating list of full input sequences...')
            sFS = 'full sequences [getLFullSeq]'
            t = SF.getLSFullSeq(self.dITp, self.dfrInpSeq, iS, iE)
            self.lSFull, iS, iE = t
            self.fillLFullSeq(sTxt=sFS, stT=stT)
        return iS, iE

    def getLNmerSeq(self, stT=None):
        print('Creating list of Nmer input sequences...')
        sNS = 'Nmer sequences [getLNmerSeq]'
        self.lSNmer = SF.getLSNmer(self.dITp, self.dfrInpSeq, self.lSFull,
                                   self.lSNmer)
        for n, sSq in enumerate(self.lSNmer):
            self.lNmerSeq.append(NmerSeq(self.dITp, sSq=sSq))
            GF.showProgress(N=len(self.lSNmer), n=n, varText=sNS,
                            modeDisp=self.dITp['mDsp'], startTime=stT)
        print('Created list of Nmer input sequences.')

    def getLInpSeq(self, cTim, lSSeq=None, readF=True, stT=None):
        cStT = time.time()
        pFIS, self.lSNmer, saveLS = self.dPF['LInpSeq'], lSSeq, True
        [iS, iE] = self.dITp['lIStartEnd']
        if os.path.isfile(pFIS) and self.lSNmer is None and readF:
            self.lSNmer = self.loadSerOrList(pF=pFIS, iC=0, toList=True)
            saveLS = False
        self.dfrInpSeq = self.loadData(pF=self.dPF['SeqCheck'], iC=0)
        iS, iE = self.getLFullSeq(iS, iE, stT=stT)
        self.getLNmerSeq(stT=stT)
        SF.modLSF(self.dITp, lSKeyF=self.dITp['lSKeyFRes'],
                  sFE=GF.joinS([self.getSFResEnd(), iS, iE]))
        self.updateDPFVal()
        if saveLS:
            self.saveListAsSer(self.lSNmer, pF=pFIS, sName=self.dITp['sNmer'])
        cTim.updateTimes(iMth=1, stTMth=cStT, endTMth=time.time())

    # --- methods for generating the {seq. length: seq. list} dictionary ------
    def genDLenSeq(self, cTim, stT=None):
        cStT = time.time()
        print('Creating Nmer {seq. length: seq. list} dictionary...')
        mDsp, varTxt = self.dITp['mDsp'], 'Nmer sequences [genDLenSeq]'
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

    def addToDictSnip(self, cSeqF, dIPosSeq):
        for sSS, (_, _, nPyl, nTtl, cPrb) in dIPosSeq.items():
            if (self.dITp['calcSnipS'] and nPyl > 0):
                nFS = len([cSq for cSq in self.lFullSeq if sSS in cSq.sSeq])
                GF.addToDictMnV(self.dSnipS, cK=sSS, cEl=cPrb, nEl=nFS)
            if self.dITp['calcSnipX']:
                tV = (nPyl, nTtl, cPrb)
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
            self.printDictSnipX(lSnipLen=[7], cSnip='AQRTLHG')
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
        if not GF.Xist(self.lNmerSeq):
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
                     saveAnyway=True)
        cTim.updateTimes(iMth=8, stTMth=cStT, endTMth=time.time())

    def getD2CondProbSnip(self, stT=None):
        nSeq, sNS = len(self.lNmerSeq), 'Nmer sequences [getD2CondProbSnip]'
        for i, cNmerS in enumerate(self.lNmerSeq):
            dSub = cNmerS.getCondProbSubSnip(self.dITp, lFullSeq=self.lFullSeq)
            self.d2CondProbSnip[cNmerS.sSeq] = dSub
            GF.showProgress(N=nSeq, n=i, modeDisp=self.dITp['mDsp'],
                            varText=sNS, startTime=stT)

    def performCondProbAnalysis(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if self.dITp['calcCondProb']:
            self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
            cStT = time.time()
            if not GF.Xist(self.d2CondProbSnip):
                self.getD2CondProbSnip(stT=stT)
            cTim.updateTimes(iMth=9, stTMth=cStT, endTMth=time.time())
            cStT = time.time()
            self.saveD2CondProbSnipAsDfr()
            cTim.updateTimes(iMth=10, stTMth=cStT, endTMth=time.time())

    # --- methods for printing results ----------------------------------------

    # --- methods for saving and loading data ---------------------------------
    def saveDfrRelLikelihood(self, dLV, d3, lSCD3):
        if self.dITp['calcWtLh']:
            dfrVal = GF.dLV3CToDfr(dLV)
            self.saveDfr(dfrVal, pF=self.dPF['ResWtLh'], saveAnyway=True)
        if self.dITp['calcRelLh']:
            dfrDict = GF.d3ValToDfr(d3, lSCD3)
            self.saveDfr(dfrDict, pF=self.dPF['ResRelLh'], saveAnyway=True)

    def convDictToDfrS(self):
        dLV, lSCDfr = {}, self.dITp['lSCDfrProbS']
        for sSnip, cProb in self.dSnipS.items():
            lElRow = [sSnip, len(sSnip), round(cProb, GC.R06)]
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
                self.saveDfr(self.dfrSnipS, pF=self.dPF['SnipDfrS'],
                             saveAnyway=True)

    def convDictToDfrX(self):
        dLV, lSCDfr = {}, self.dITp['lSCDfrWFullSProbX']
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
                             saveIdx=True, idxLbl=False, saveAnyway=True)

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

    def saveD2CondProbSnipAsDfr(self):
        if GF.Xist(self.d2CondProbSnip):
            dLV, lSCDfr = {}, self.dITp['lSCMerAll']
            for cDSub in self.d2CondProbSnip.values():
                lElRow = []
                for sSnip, lV in cDSub.items():
                    lElRow += ([sSnip] + lV)
                assert len(lElRow) == len(lSCDfr)
                for sC, cV in zip(lSCDfr, lElRow):
                    GF.addToDictL(dLV, cK=sC, cE=cV)
            dfrVal = GF.dictToDfr(dLV, idxDfr=list(self.d2CondProbSnip))
            self.saveDfr(dfrVal, pF=self.dPF['ProbTblCP'], saveIdx=True,
                         idxLbl=self.dITp['sCNmer'], saveAnyway=True)

###############################################################################
