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

    def printDictIPosProbPyl(self):
        print(GC.S_DS80, '\nPosition index dictionary:\n', GC.S_DS80, sep='')
        for sFullSeq, dSub in self.dIPosProbPyl.items():
            print(GC.S_ST04, ' Full sequence ', GC.S_STAR*61)
            print(sFullSeq)
            for cLen, dIPosSeq in dSub.items():
                print(GC.S_PL04, ' Nmers with length ', cLen, ' have Pyl ',
                      'index dictionary:', GC.S_NEWL, dIPosSeq, sep='')

    # --- methods for loading data --------------------------------------------
    def getPDirSFResEnd(self, pDirR=None):
        sFResEnd, sTest, sTrain = '', self.dITp['sTest'], self.dITp['sTrain']
        sCombS = GF.joinS([self.dITp['sCombined'], self.dITp['sCapS']])
        if self.dITp['sFSeqCheck'] == self.dITp['sFProcInpNmer']:
            pDirR, sFResEnd = self.pDirProcInp, self.dITp['sAllSeqNmer']
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iEnd=2) == sCombS:
            pDirR, sFResEnd = self.pDirRes, sCombS
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iStart=-1) == sTest:
            pDirR, sFResEnd = self.pDirRes, self.dITp['sTestData']
        elif GF.getPartSF(sF=self.dITp['sFSeqCheck'], iStart=-1) == sTrain:
            pDirR, sFResEnd = self.pDirRes, self.dITp['sTrainData']
        return pDirR, sFResEnd

    # def loadInpNmer(self):
    #     self.dfrInpNmer, pFInp, iC = None, None, None
    #     if self.dITp['useNmerSeqFrom'] == self.dITp['sSeqCheck']:
    #         pDir, _ = self.getPDirSFResEnd(pDirR=self.pDirRes)
    #         pFInp, iC = GF.joinToPath(pDir, self.dITp['sFSeqCheck']), 0
    #     elif self.dITp['useNmerSeqFrom'] == self.dITp['sIEffInp']:
    #         pFInp = GF.joinToPath(self.pDirRes, self.dITp['sFIEffInp'])
    #     elif self.dITp['useNmerSeqFrom'] == self.dITp['sCombInp']:
    #         pFInp, iC = GF.joinToPath(self.pDirRes, self.dITp['sFCombInp']), 0
    #     if pFInp is not None:
    #         self.dfrInpNmer = self.loadDfr(pF=pFInp, iC=iC)
    #     return iC

    def loadInpDfrs(self):
        self.lCombSeq, self.dLenSeq = [], {}
        pFIEffInp = GF.joinToPath(self.pDirRes, self.dITp['sFIEffInp'])
        pFCombInp = GF.joinToPath(self.pDirRes, self.dITp['sFCombInp'])
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

    def getLInpSeq(self, cTim, lSSeq=None, stT=None):
        cStT = time.time()
        [iS, iE], lSNmerSeq = self.dITp['lIStartEnd'], lSSeq
        pDir, e = self.getPDirSFResEnd(pDirR=self.pDirRes)
        pFInpSeq = GF.joinToPath(pDir, self.dITp['sFSeqCheck'])
        mDsp, sFS, sNS = self.dITp['mDsp'], 'full sequences', 'Nmer sequences'
        self.dfrInpSeq = self.loadDfr(pF=pFInpSeq, iC=0)
        if self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns:
            print('Creating list of full input sequences...')
            lSFullSeq = list(self.dfrInpSeq[self.dITp['sCCodeSeq']].unique())
            lSFullSeq, iS, iE = GF.getItStartToEnd(lSFullSeq, iS, iE)
            for n, sSq in enumerate(lSFullSeq):
                lI = self.getLIPosPyl(sFullSeq=sSq)
                self.lFullSeq.append(FullSeq(self.dITp, sSq=sSq, lIPosPyl=lI))
                GF.showProgress(N=len(lSFullSeq), n=n, modeDisp=mDsp,
                                varText=sFS, startTime=stT)
            print('Created list of full input sequences...')
        print('Creating list of Nmer input sequences...')
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
        e = GF.joinS([e, iS, iE])
        SF.modLSF(self.dITp, lSKeyF=self.dITp['lSKeyFRes'], sFE=e)
        print('Created list of Nmer input sequences.')
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
            GF.addToDictD(self.dIPosProbPyl, sSS, cSeqF.sSeq, cProb)

    def performProbAnalysis(self, cTim, lEff=[None], lSSeq=None, stT=None):
        if self.dITp['calcWtProb'] or self.dITp['calcRelProb']:
            self.getLInpSeq(cTim=cTim, lSSeq=lSSeq, stT=stT)
            self.genDLenSeq(cTim=cTim, stT=stT)
            print('Performing calculation of Nmer sequence probabilities...')
            self.dIPosProbPyl, N = {}, len(self.lFullSeq)
            for n, cSeqF in enumerate(self.lFullSeq):
                cStT = time.time()
                cD = GF.restrInt(self.dLenSeq, lRestrLen=self.dITp['lLenNMer'])
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
            self.saveDfrRelProb()
            cTim.updateTimes(iMth=7, stTMth=cStT, endTMth=time.time())
            print('Saved Nmer sequence probability DataFrame.')
            # dLV, d3, k, N = {}, {}, 0, len(lSSeq)*len(lEff)
            # print('-'*80, '\nself.lNmerSeq:', sep='')
            # for cSeq in self.lNmerSeq:
            #     print(cSeq)
            # print('-'*80, '\nself.lFullSeq:', sep='')
                # print(cSeq)
                # cSeq.printNmerDict()
                # l = ['GLSPK', 'IVGSAYY', 'LSD', 'XYZ', 'T']
            # self.printDictLenSeq()
            # dLV, d3, k, N = {}, {}, 0, len(lSSeq)*len(lEff)
            # for cSSeq in lSSeq:
            #     cNmerSeq = NmerSeq(self.dITp, sSq=cSSeq)
            #     for cEff in lEff:
            #         serLh, cEff, maxLSnip = self.getRelLikelihoods(cEff)
            #         SF.calcDictLikelihood(self.dITp, dLV, d3, cNmerSeq.dPrf,
            #                               serLh, maxLSnip, cSSeq, cEff)
            #         self.complementDLV(dLV)
            #         k += 1
            #         if k%self.dITp['mDsp'] == 0:
            #             print('Processed', k, 'of', N, '...')
            # self.saveDfrRelLikelihood(dLV, d3, lSCD3=self.dITp['lSCDfrLhD'])

    # --- methods for printing results ----------------------------------------

    # --- methods for saving data ---------------------------------------------
    def saveDfrRelLikelihood(self, dLV, d3, lSCD3):
        pFDfrVal = GF.joinToPath(self.pDirRes, self.dITp['sFResWtLh'])
        pFDfrDict = GF.joinToPath(self.pDirRes, self.dITp['sFResRelLh'])
        if self.dITp['calcWtLh']:
            dfrVal = GF.dLV3CToDfr(dLV)
            self.saveDfr(dfrVal, pF=pFDfrVal, saveAnyway=True)
        if self.dITp['calcRelLh']:
            dfrDict = GF.d3ValToDfr(d3, lSCD3)
            self.saveDfr(dfrDict, pF=pFDfrDict, saveAnyway=True)

    # def saveDfrRelProb(self, lSCol=['FullSeq', 'lenNmer', 'Nmer', 'Prob']):
    #     pFDfrVal = GF.joinToPath(self.pDirRes, self.dITp['sFResWtProb'])
    #     # pFDfrDict = GF.joinToPath(self.pDirRes, self.dITp['sFResRelProb'])
    #     if self.dITp['calcWtProb']:
    #         dLV = {}
    #         for cKMain, cDSub1 in self.dIPosProbPyl.items():
    #             for cKSub1, cDSub2 in cDSub1.items():
    #                 for cKSub2, (lIPos, lB) in cDSub2.items():
    #                     cProb = None
    #                     if len(lB) > 0:
    #                         cProb = sum(lB)/len(lB)
    #                     lElRow = [cKMain, cKSub1, cKSub2, cProb]
    #                     # for sC, cV in zip(lSCol[:minLenL], lElRow):
    #                     for sC, cV in zip(lSCol, lElRow):
    #                         GF.addToDictL(dLV, cK=sC, cE=cV)
    #         self.saveDfr(GF.dictToDfr(dLV), pF=pFDfrVal, saveAnyway=True)

    # def saveDfrRelProb(self, lSCol=['Nmer', 'lenNmer', 'FullSeq', 'Prob']):
    #     pFDfrVal = GF.joinToPath(self.pDirRes, self.dITp['sFResWtProb'])
    #     # pFDfrDict = GF.joinToPath(self.pDirRes, self.dITp['sFResRelProb'])
    #     if self.dITp['calcWtProb']:
    #         dLV = {}
    #         for sSS, cDSub in self.dIPosProbPyl.items():
    #             for sLS, cProb in cDSub.items():
    #                 lElRow = [sSS, len(sSS), sLS, cProb]
    #                 # for sC, cV in zip(lSCol[:minLenL], lElRow):
    #                 for sC, cV in zip(lSCol, lElRow):
    #                     GF.addToDictL(dLV, cK=sC, cE=cV)
    #         self.saveDfr(GF.dictToDfr(dLV), pF=pFDfrVal, saveAnyway=True)
    def saveDfrRelProb(self):
        pFDfrVal = GF.joinToPath(self.pDirRes, self.dITp['sFResWtProb'])
        # pFDfrDict = GF.joinToPath(self.pDirRes, self.dITp['sFResRelProb'])
        if self.dITp['calcWtProb']:
            dLV = {}
            for sSnip, cDSub in self.dIPosProbPyl.items():
                # sSProb, n = 0., len({cK: cV for cK, cV in cDSub.items()
                #                      if cV is not None})
                # for sLS, cProb in cDSub.items():
                #     if cProb is not None:
                #         sSProb += cProb/len(cDSub)
                # sSProb = sum(cDSub.values())/len(cDSub)
                lElRow = [sSnip, len(sSnip), sum(cDSub.values())/len(cDSub)]
                # for sC, cV in zip(lSCol[:minLenL], lElRow):
                for sC, cV in zip(self.dITp['lSCDfrProbS'], lElRow):
                    GF.addToDictL(dLV, cK=sC, cE=cV)
            self.saveDfr(GF.dictToDfr(dLV, srtBy=self.dITp['lSrtByDfrProbS'],
                                      srtAsc=self.dITp['lSrtAscDfrProbS']),
                         pF=pFDfrVal, saveAnyway=True)

###############################################################################
