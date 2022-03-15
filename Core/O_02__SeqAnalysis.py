# -*- coding: utf-8 -*-
###############################################################################
# --- O_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
import copy

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
        self.dITp = copy.deepcopy(self.dIG[0])  # type of base class = 0
        for iTpU in lITpUpd + [iTp]:            # updated with types in list
            self.dITp.update(self.dIG[iTpU])
        self.loadDfrIEff()
        print('Initiated "SeqAnalysis" base object.')

    # --- methods for reading the input DataFrame -----------------------------
    def genDLenSeq(self):
        self.dLenSeq, lSq = {}, [s for s in self.dfrIEff.columns
                                 if s not in [self.dITp['sEffCode']]]
        for sSq in lSq:
            GF.addToDictL(self.dLenSeq, cK=len(sSq), cE=sSq)

    def loadDfrIEff(self):
        if self.dIG['isTest']:
            self.pDirProcInp = self.dITp['pDirProcInp_T']
            self.pDirRes = self.dITp['pDirRes_T']
        else:
            self.pDirProcInp = self.dITp['pDirProcInp']
            self.pDirRes = self.dITp['pDirRes']
        pFIEffInp = GF.joinToPath(self.pDirRes, self.dITp['sFIEffInp'])
        self.dfrIEff = self.loadDfr(pF=pFIEffInp, iC=0)
        self.genDLenSeq()

    # --- methods for obtaining the list of input Nmer-sequences --------------
    def getLInpNmerSeq(self):
        sCNmer, self.lInpNmerSeq = self.dITp['sCNmer'], []
        [iS, iE] = self.dITp['lIStartEnd']
        pFInpSeq = GF.joinToPath(self.pDirProcInp, self.dITp['sFProcInpNmer'])
        self.dfrInpSeq = self.loadDfr(pF=pFInpSeq, iC=0)
        if sCNmer in self.dfrInpSeq.columns:
            lSNmerFull = list(self.dfrInpSeq[sCNmer].unique())
            self.lInpNmerSeq = GF.getItStartToEnd(lSNmerFull, iS, iE)
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

    def performAnalysis(self, lEff=[None], lSSeq=None):
        if self.dITp['analyseSeq']:
            if lSSeq is None:
                lSSeq = self.getLInpNmerSeq()
            dLV, d3, k = {}, {}, 0
            for cSSeq in lSSeq:
                cNmerSeq = NmerSeq(self.dITp, sSq=cSSeq)
                for cEff in lEff:
                    serLlh, cEff, maxLSnip = self.getRelLikelihoods(cEff)
                    SF.calcDictLikelihood(self.dITp, dLV, d3, cNmerSeq.dPrf,
                                          serLlh, maxLSnip, cSSeq, cEff)
                    k += 1
                    print('Processed', k, 'of', len(lSSeq)*len(lEff), '...')
            self.saveDfrRelLikelihood(dLV, d3, lSCD3=self.dITp['lSCDfrLhD'])

    # --- methods for printing results ----------------------------------------

    # --- methods for saving data ---------------------------------------------
    def saveDfrRelLikelihood(self, dLV, d3, lSCD3):
        dfrVal = GF.dLV3CToDfr(dLV)
        pFDfrVal = GF.joinToPath(self.pDirRes, self.dITp['sFResWtLh'])
        self.saveDfr(dfrVal, pF=pFDfrVal, dropDup=False, saveAnyway=True)
        dfrDict = GF.d3ValToDfr(d3, lSCD3)
        pFDfrDict = GF.joinToPath(self.pDirRes, self.dITp['sFResRelLh'])
        self.saveDfr(dfrDict, pF=pFDfrDict, dropDup=False, saveAnyway=True)

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
