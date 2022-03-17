# -*- coding: utf-8 -*-
###############################################################################
# --- O_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
# import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

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
        self.dLenSeq, lSq = {}, [s for s in self.dfrIEff.columns
                                 if s not in [self.dITp['sEffCode']]]
        for sSq in lSq:
            GF.addToDictL(self.dLenSeq, cK=len(sSq), cE=sSq)

    def loadInpDfrs(self):
        pFIEffInp = GF.joinToPath(self.pDirRes, self.dITp['sFIEffInp'])
        pFCombInp = GF.joinToPath(self.pDirRes, self.dITp['sFCombInp'])
        self.dfrIEff = self.loadDfr(pF=pFIEffInp, iC=0)
        self.dfrComb = self.loadDfr(pF=pFCombInp, iC=0)
        self.lCombNmerSeq = list(self.dfrComb[self.dITp['sCNmer']].unique())
        self.genDLenSeq()

    # --- methods for obtaining the list of input Nmer-sequences --------------
    def getLInpNmerSeq(self):
        sCNmer, self.lInpNmerSeq = self.dITp['sCNmer'], []
        [iS, iE] = self.dITp['lIStartEnd']
        pFInpSeq = GF.joinToPath(self.pDirProcInp, self.dITp['sFProcInpNmer'])
        self.dfrInpSeq = self.loadDfr(pF=pFInpSeq, iC=0)
        if sCNmer in self.dfrInpSeq.columns:
            lSNmerFull = list(self.dfrInpSeq[sCNmer].unique())
            self.lInpNmerSeq, iS, iE = GF.getItStartToEnd(lSNmerFull, iS, iE)
            e = self.dITp['sUSC'].join([str(iS), str(iE)])
            self.dITp['sFResWtLh'] = GF.modSF(self.dITp['sFResWtLh'], sEnd=e,
                                              sJoin=self.dITp['sUS02'])
            self.dITp['sFResRelLh'] = GF.modSF(self.dITp['sFResRelLh'], sEnd=e,
                                               sJoin=self.dITp['sUS02'])
        return self.lInpNmerSeq

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
            if sNmer in self.lCombNmerSeq:
                dLV[self.dITp['sInCombRes']][k] = 1

    def performAnalysis(self, lEff=[None], lSSeq=None):
        if self.dITp['calcWtLh'] or self.dITp['calcRelLh']:
            if lSSeq is None:
                lSSeq = self.getLInpNmerSeq()
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

# -----------------------------------------------------------------------------
class NmerSeq():
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dITp, sSq=''):
        self.sSeq = sSq
        self.createProfileDict(dITp)

    # --- methods for creating the profile dictionary -------------------------
    def createProfileDict(self, dITp):
        self.dPrf, iCNmer = {}, dITp['iCentNmer']
        if type(self.sSeq) == str and len(self.sSeq) == dITp['lenSDef']:
            for k in range(iCNmer + 1):
                self.dPrf[2*k + 1] = self.sSeq[(iCNmer - k):(iCNmer + k + 1)]

###############################################################################
