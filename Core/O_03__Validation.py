# -*- coding: utf-8 -*-
###############################################################################
# --- O_03__Validation.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
# import Core.F_01__SpcFunctions as SF

from Core.O_01__ExpData import ExpData

# -----------------------------------------------------------------------------
class Validation(ExpData):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=3, lITpUpd=[1]):
        super().__init__(inpDat)
        self.idO = 'O_03'
        self.descO = 'Validation of analysis of Nmer-sequences'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.getPDir()
        self.fillDPF()
        self.loadInpDfr()
        print('Initiated "Validation" base object.')

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPF(self):
        self.fillDPFRes()
        sJ, sFlFE, pRes = self.dITp['sUS02'], self.dITp['sFull'], self.pDirRes
        sFE = self.dITp['sUSC'].join(self.dITp['lSLenNMer'])
        if len(sFE) > 0:
            sFlFE = sJ.join([sFlFE, sFE])
        sFIEffTrain = GF.modSF(self.dITp['sFResIEff'], sJoin=sJ,
                               sEnd=sJ.join([sFE, self.dITp['sTrain']]))
        sFIEffFTrain = GF.modSF(self.dITp['sFResIEff'], sJoin=sJ,
                                sEnd=sJ.join([sFlFE, self.dITp['sTrain']]))
        sFIEffTest = GF.modSF(self.dITp['sFResIEff'], sJoin=sJ,
                              sEnd=sJ.join([sFE, self.dITp['sTest']]))
        sFIEffFTest = GF.modSF(self.dITp['sFResIEff'], sJoin=sJ,
                               sEnd=sJ.join([sFlFE, self.dITp['sTest']]))
        pFCombInp = GF.joinToPath(pRes, self.dITp['sFCombInp'])
        pFCombTrain = GF.joinToPath(pRes, self.dITp['sFCombTrain'])
        pFCombTest = GF.joinToPath(pRes, self.dITp['sFCombTest'])
        self.dPFRes[GC.S_COMBINED] = pFCombInp
        self.dPFRes[self.dITp['sTrain']] = pFCombTrain
        self.dPFRes[self.dITp['sTest']] = pFCombTest
        self.dPFRes[GC.S_I_EFF_TRAIN] = GF.joinToPath(pRes, sFIEffTrain)
        self.dPFRes[GC.S_I_EFF_F_TRAIN] = GF.joinToPath(pRes, sFIEffFTrain)
        self.dPFRes[GC.S_I_EFF_TEST] = GF.joinToPath(pRes, sFIEffTest)
        self.dPFRes[GC.S_I_EFF_F_TEST] = GF.joinToPath(pRes, sFIEffFTest)

    # --- methods for reading the input DataFrame -----------------------------
    def loadInpDfr(self):
        self.dfrComb = self.loadDfr(pF=self.dPFRes[GC.S_COMBINED], iC=0)
        assert self.dfrComb is not None
        self.nRecTrain = round(self.dfrComb.shape[0]*self.dITp['share4Train'])
        self.nRecTest = self.dfrComb.shape[0] - self.nRecTrain

    # --- methods for splitting the input data set ----------------------------
    def splitInpData(self):
        k, N, bF, bT = self.nRecTest, self.dfrComb.shape[0], False, True
        self.lITest = GF.drawListInt(maxInt=N, nIntDrawn=k, wRepl=bF, sortL=bT)
        self.lITrain = [n for n in range(N) if n not in self.lITest]
        self.dfrTrain = self.dfrComb.iloc[self.lITrain, :].reset_index(drop=bT)
        self.dfrTest = self.dfrComb.iloc[self.lITest, :].reset_index(drop=bT)
        self.saveTrainTestData()

    # --- methods for printing objects ----------------------------------------
    def printTestObj(self, printDfrComb=False):
        if printDfrComb:
            print(GC.S_DS80, GC.S_NEWL, 'Full combined DataFrame:', sep='')
            print(self.dfrComb)
        print(GC.S_DS80, GC.S_NEWL, 'Number of records train DataFrame:',
              GC.S_NEWL, self.nRecTrain, sep='')
        print(GC.S_DS80, GC.S_NEWL, 'First 5 indices train DataFrame:', sep='')
        print(self.lITrain[:5], ' (length = ', len(self.lITrain), ')', sep='')
        print(GC.S_DS80, GC.S_NEWL, 'Train DataFrame:', sep='')
        print(self.dfrTrain)
        print(GC.S_DS80, GC.S_NEWL, 'Number of records test DataFrame:',
              GC.S_NEWL, self.nRecTest, sep='')
        print(GC.S_DS80, GC.S_NEWL, 'First 5 indices test DataFrame:', sep='')
        print(self.lITest[:5], ' (length = ', len(self.lITest), ')', sep='')
        print(GC.S_DS80, GC.S_NEWL, 'Test DataFrame:', sep='')
        print(self.dfrTest, GC.S_NEWL, GC.S_DS80, sep='')

    # --- methods for producing result DataFrames -----------------------------
    def createResultsTrain(self, stT):
        if self.dITp['predictWTrain']:
            self.splitInpData()
            self.getInfoKinNmer(stT=stT, sIEff=GC.S_I_EFF_TRAIN,
                                sIEffF=GC.S_I_EFF_F_TRAIN,
                                tpDfr=self.dITp['sTrain'])

    # --- methods for saving DataFrames ---------------------------------------
    def saveTrainTestData(self):
        if self.dITp['predictWTrain']:
            self.saveDfr(self.dfrTrain, pF=self.dPFRes[self.dITp['sTrain']],
                         dropDup=False, saveAnyway=True)
            self.saveDfr(self.dfrTest, pF=self.dPFRes[self.dITp['sTest']],
                         dropDup=False, saveAnyway=True)

###############################################################################
