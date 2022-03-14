# -*- coding: utf-8 -*-
###############################################################################
# --- O_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
import copy

# import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
# import Core.F_01__SpcFunctions as SF

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
    def loadDfrIEff(self):
        if self.dIG['isTest']:
            self.pDirRes = self.dITp['pDirRes_T']
        else:
            self.pDirRes = self.dITp['pDirRes']
        pFIEffInp = GF.joinToPath(self.pDirRes, self.dITp['sFIEffInp'])
        self.dfrIEff = self.loadDfr(pF=pFIEffInp)
        self.dLenSeq, lSq = {}, [s for s in self.dfrIEff.columns
                                 if s not in [self.dITp['sEffCode']]]
        for sSq in lSq:
            GF.addToDictL(self.dLenSeq, cK=len(sSq), cE=sSq)
        if self.dITp['sAnyEff'] in self.dfrIEff:
            serAnyEff = self.dfrIEff.loc[self.dITp['sAnyEff'], :]
            for k in self.dLenSeq:
                maxV = max(serAnyEff.loc[self.dLenSeq[k]])
                if maxV > 0:
                    serAnyEff.loc[self.dLenSeq[k]] /= maxV

    # --- methods for performing the Nmer-sequence analysis -------------------
    def performAnalysis(self, lSSeq):
        for cSSeq in lSSeq:
            cNmerSeq = NmerSeq(self.dITp, sSq=cSSeq)
    
    # --- methods for filling the result paths dictionary ---------------------

    # --- methods for printing results ----------------------------------------

    # --- methods for saving data ---------------------------------------------
    # def saveResultSeqAnalysis(self):
    #     self.saveDfr(cDfr, pF=GF.joinToPath(self.pDirProcInp, sFPI),
    #                  dropDup=True, saveAnyway=False)

# -----------------------------------------------------------------------------
class NmerSeq():
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dITp, sSq=''):
        self.sSeq = sSq
        self.createProfileDict(dITp)
        print('Initiated "NmerSeq" base object.')

    # --- methods for creating the profile dictionary -------------------------
    def createProfileDict(self, dITp):
        self.dPrf, iCNmer = {}, dITp['iCentNmer']
        if type(self.sSeq) == str and len(self.sSeq) == dITp['lenSDef']:
            for k in range(iCNmer + 1):
                self.dPrf[2*k + 1] = self.sSeq[(iCNmer - k):(iCNmer + k + 1)]
        
###############################################################################
