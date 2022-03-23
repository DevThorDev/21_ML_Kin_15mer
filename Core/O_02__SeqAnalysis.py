# -*- coding: utf-8 -*-
###############################################################################
# --- O_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
# import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass, NmerSeq

# -----------------------------------------------------------------------------
class SeqAnalysis(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=2, lITpUpd=[1]):
        super().__init__(inpDat)
        self.idO = 'O_02'
        self.descO = 'Analysis of Nmer-sequences'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.getPDir()
        self.loadInpDfrs()
        print('Initiated "SeqAnalysis" base object.')

    # --- methods for reading the input DataFrame -----------------------------
    def genDLenSeq(self):
        lSEffCode = [self.dITp['sEffCode']]
        lSq = [s for s in self.dfrIEff.columns if s not in lSEffCode]
        for sSq in lSq:
            GF.addToDictL(self.dLenSeq, cK=len(sSq), cE=sSq)

    def loadInpDfrs(self):
        self.lCombSeq, self.dLenSeq = [], {}
        pFIEffInp = GF.joinToPath(self.pDirRes, self.dITp['sFIEffInp'])
        pFCombInp = GF.joinToPath(self.pDirRes, self.dITp['sFCombInp'])
        self.dfrIEff = self.loadDfr(pF=pFIEffInp, iC=0)
        self.dfrComb = self.loadDfr(pF=pFCombInp, iC=0)
        if self.dfrComb is not None:
            self.lCombSeq = list(self.dfrComb[self.dITp['sCNmer']].unique())
        if self.dfrIEff is not None:
            self.genDLenSeq()

    # --- methods for obtaining the list of input Nmer-sequences --------------
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

    def getlInpSeq(self):
        self.lNmerSeq, self.lFullSeq, pDir = [], [], self.pDirRes
        [iS, iE] = self.dITp['lIStartEnd']
        pDir, e = self.getPDirSFResEnd(pDirR=pDir)
        pFInpSeq = GF.joinToPath(pDir, self.dITp['sFSeqCheck'])
        self.dfrInpSeq = self.loadDfr(pF=pFInpSeq, iC=0)
        if self.dITp['sCNmer'] in self.dfrInpSeq.columns:
            lSNmerFull = list(self.dfrInpSeq[self.dITp['sCNmer']].unique())
            self.lNmerSeq, iS, iE = GF.getItStartToEnd(lSNmerFull, iS, iE)
        if self.dITp['sCCodeSeq'] in self.dfrInpSeq.columns:
            lSFullSeqs = list(self.dfrInpSeq[self.dITp['sCCodeSeq']].unique())
            self.lFullSeq, iS, iE = GF.getItStartToEnd(lSFullSeqs, iS, iE)
        e = GF.joinS([e, iS, iE])
        SF.modLSF(self.dITp, lSKeyF=self.dITp['lSKeyFRes'], sFE=e)
        return self.lNmerSeq

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

    def performAnalysis(self, lEff=[None], lSSeq=None):
        if self.dITp['calcWtLh'] or self.dITp['calcRelLh']:
            if lSSeq is None:
                lSSeq = self.getlInpSeq()
            dLV, d3, k, N = {}, {}, 0, len(lSSeq)*len(lEff)
            for cSSeq in lSSeq:
                cNmerSeq = NmerSeq(self.dITp, sSq=cSSeq)
                for cEff in lEff:
                    serLh, cEff, maxLSnip = self.getRelLikelihoods(cEff)
                    SF.calcDictLikelihood(self.dITp, dLV, d3, cNmerSeq.dPrf,
                                          serLh, maxLSnip, cSSeq, cEff)
                    self.complementDLV(dLV)
                    k += 1
                    if k%self.dITp['mDsp'] == 0:
                        print('Processed', k, 'of', N, '...')
            self.saveDfrRelLikelihood(dLV, d3, lSCD3=self.dITp['lSCDfrLhD'])

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
