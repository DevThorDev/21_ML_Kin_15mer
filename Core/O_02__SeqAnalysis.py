# -*- coding: utf-8 -*-
###############################################################################
# --- O_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
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

    def printDictIPos(self):
        print(GC.S_DS80, '\nPosition index dictionary:\n', GC.S_DS80, sep='')
        for sFullSeq, dSub in self.dIPos.items():
            print('*** Sequence', sFullSeq)
            for cLen, cV in dSub.items():
                print('+ Length', cLen)
                print('\t', cV, sep='')

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
        lIPosPyl, cDfr = [], self.dfrInpSeq
        if self.dITp['sPepPIP'] in cDfr.columns:
            dfrCSeq = cDfr[cDfr[self.dITp['sCCodeSeq']] == sFullSeq]
            lIPosPyl = dfrCSeq[self.dITp['sPepPIP']].unique()
        return [i - 1 for i in lIPosPyl]

    def getLInpSeq(self, lSSeq=None):
        [iS, iE], lSNmerSeq = self.dITp['lIStartEnd'], lSSeq
        pDir, e = self.getPDirSFResEnd(pDirR=self.pDirRes)
        pFInpSeq = GF.joinToPath(pDir, self.dITp['sFSeqCheck'])
        self.dfrInpSeq = self.loadDfr(pF=pFInpSeq, iC=0)
        if lSNmerSeq is None and self.dITp['sCNmer'] in self.dfrInpSeq.columns:
            lSNmerSeq = list(self.dfrInpSeq[self.dITp['sCNmer']].unique())
            lSNmerSeq, iS, iE = GF.getItStartToEnd(lSNmerSeq, iS, iE)
        for sSq in lSNmerSeq:
            self.lNmerSeq.append(NmerSeq(self.dITp, sSq=sSq))
        if self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns:
            lSFullSeq = list(self.dfrInpSeq[self.dITp['sCCodeSeq']].unique())
            lSFullSeq, iS, iE = GF.getItStartToEnd(lSFullSeq, iS, iE)
            for sSq in lSFullSeq:
                lI = self.getLIPosPyl(sFullSeq=sSq)
                self.lFullSeq.append(FullSeq(self.dITp, sSq=sSq, lIPosPyl=lI))
        e = GF.joinS([e, iS, iE])
        SF.modLSF(self.dITp, lSKeyF=self.dITp['lSKeyFRes'], sFE=e)
        
    # --- methods for generating the {seq. length: seq. list} dictionary ------
    def genDLenSeq(self):
        # iCol = self.loadInpNmer()
        if self.dITp['useNmerSeqFrom'] in [self.dITp['sSeqCheck'],
                                           self.dITp['sCombInp']]:
            for cNmerSeq in self.lNmerSeq:
                for cLen, sSeq in cNmerSeq.dPrf.items(): 
                    GF.addToDictL(self.dLenSeq, cK=cLen, cE=sSeq, lUniqEl=True)
        elif self.dITp['useNmerSeqFrom'] == self.dITp['sIEffInp']:
            lSEffCode = [self.dITp['sEffCode']]
            lSq = [s for s in self.dfrIEff.columns if s not in lSEffCode]
            for sSq in lSq:
                GF.addToDictL(self.dLenSeq, cK=len(sSq), cE=sSq)

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

    def performLhAnalysis(self, lEff=[None], lSSeq=None):
        if self.dITp['calcWtLh'] or self.dITp['calcRelLh']:
            self.getLInpSeq(lSSeq=lSSeq)
            self.genDLenSeq()
            dLV, d3, k, N = {}, {}, 0, len(lSSeq)*len(lEff)
            for cNmerSeq in self.lNmerSeq:
                for cEff in lEff:
                    serLh, cEff, maxLSnip = self.getRelLikelihoods(cEff)
                    SF.calcDictLikelihood(self.dITp, dLV, d3, cNmerSeq.dPrf,
                                          serLh, maxLSnip, cNmerSeq.sSeq, cEff)
                    self.complementDLV(dLV)
                    k += 1
                    if k%self.dITp['mDsp'] == 0:
                        print('Processed', k, 'of', N, '...')
            self.saveDfrRelLikelihood(dLV, d3, lSCD3=self.dITp['lSCDfrLhD'])

    def performProbAnalysis(self, lEff=[None], lSSeq=None):
        if self.dITp['calcWtProb'] or self.dITp['calcRelProb']:
            self.dIPos = {}
            self.getLInpSeq(lSSeq=lSSeq)
            self.genDLenSeq()
            for cSeq in self.lFullSeq:
                self.dIPos[cSeq.sSeq] = {}
                cD = GF.restrInt(self.dLenSeq, lRestrLen=self.dITp['lLenNMer'])
                for cLen, lSSeqNmer in cD.items():
                    # dIPosCSeq = cSeq.getDictPosSeq(lSSeq2F=lSSeqNmer)
                    # print('dIPosCSeq:')
                    # print(dIPosCSeq)
                    self.dIPos[cSeq.sSeq][cLen] = cSeq.getDictPosSeq(lSSeq2F=lSSeqNmer)
            self.printDictIPos()
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

###############################################################################
