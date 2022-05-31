# -*- coding: utf-8 -*-
###############################################################################
# --- O_03__Validation.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

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
        self.fillDPFVal()
        self.fillDTpDfrVal()
        self.loadInpDfr()
        print('Initiated "Validation" base object.')

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPFVal(self):
        sJ, sFlFE = self.dITp['sUS02'], self.dITp['sFull']
        pR, sFE = self.pDirResComb, GF.joinS(self.dITp['lSLenNmer'])
        if len(sFE) > 0:
            sFlFE = GF.joinS([sFlFE, sFE], sJoin=sJ)
        pFCombInp = GF.joinToPath(pR, self.dITp['sFCombInp'])
        dPFTrain = {self.dITp['sCombinedInp']: pFCombInp}
        dPFTest = {self.dITp['sCombinedInp']: pFCombInp}
        for sTp, sFComb, cD in zip(['sTrain', 'sTest'],
                                   ['sFCombTrain', 'sFCombTest'],
                                   [dPFTrain, dPFTest]):
            sE = GF.joinS([sFE, self.dITp[sTp]], sJoin=sJ)
            sEFl = GF.joinS([sFlFE, self.dITp[sTp]], sJoin=sJ)
            sFINmer = GF.modSF(self.dITp['sFResINmer'], sJoin=sJ, sEnd=sE)
            sFIEff = GF.modSF(self.dITp['sFResIEff'], sJoin=sJ, sEnd=sE)
            sFIEffF = GF.modSF(self.dITp['sFResIEff'], sJoin=sJ, sEnd=sEFl)
            pFComb = GF.joinToPath(pR, self.dITp[sFComb])
            cD[self.dITp['sCombinedOut']] = pFComb
            cD[self.dITp['sImer']] = GF.joinToPath(pR, sFINmer)
            cD[self.dITp['sIEff']] = GF.joinToPath(pR, sFIEff)
            cD[self.dITp['sIEffF']] = GF.joinToPath(pR, sFIEffF)
            self.dPF[self.dITp[sTp]] = cD

    # --- methods for filling the DataFrame type dictionary -------------------
    def fillDTpDfrVal(self):
        sBGIN, sBGIE = self.dITp['sBGenInfoNmer'], self.dITp['sBGenInfoEff']
        self.dTpDfr[self.dITp['sTrain']][sBGIN] = self.dITp['genInfoNmerTrain']
        self.dTpDfr[self.dITp['sTrain']][sBGIE] = self.dITp['genInfoEffTrain']
        self.dTpDfr[self.dITp['sTest']][sBGIN] = self.dITp['genInfoNmerTest']
        self.dTpDfr[self.dITp['sTest']][sBGIE] = self.dITp['genInfoEffTest']

    # --- methods for loading data --------------------------------------------
    def loadInpDfr(self):
        pFCombInp = self.dPF[self.dITp['sTrain']][self.dITp['sCombinedInp']]
        self.dfrComb = self.loadData(pF=pFCombInp, iC=0)
        assert self.dfrComb is not None
        self.dTpDfr[self.dITp['sBase']][self.dITp['sCDfrComb']] = self.dfrComb
        self.nRecTrain = round(self.dfrComb.shape[0]*self.dITp['share4Train'])
        self.nRecTest = self.dfrComb.shape[0] - self.nRecTrain

    # --- methods for splitting the input data set ----------------------------
    def splitInpData(self):
        k, N, bF, bT = self.nRecTest, self.dfrComb.shape[0], False, True
        sDfrC = self.dITp['sCDfrComb']
        self.lITest = GF.drawListInt(maxInt=N, nIntDrawn=k, wRepl=bF, sortL=bT)
        self.lITrain = [n for n in range(N) if n not in self.lITest]
        self.dfrTrain = self.dfrComb.iloc[self.lITrain, :].reset_index(drop=bT)
        self.dfrTest = self.dfrComb.iloc[self.lITest, :].reset_index(drop=bT)
        self.dTpDfr[self.dITp['sTrain']][sDfrC] = self.dfrTrain
        self.dTpDfr[self.dITp['sTest']][sDfrC] = self.dfrTest
        self.saveTrainTestData()

    # --- methods for printing objects ----------------------------------------
    def printTestObj(self, printDfrComb=False, mL=GC.MAX_LEN_L_DSP):
        if printDfrComb:
            print(GC.S_DS80, GC.S_NEWL, 'Full combined DataFrame:', sep='')
            print(self.dfrComb)
        print(GC.S_DS80, GC.S_NEWL, 'Number of records train DataFrame:',
              GC.S_NEWL, self.nRecTrain, sep='')
        print(GC.S_DS80, GC.S_NEWL, 'First ', mL, ' indices train DataFrame:',
              sep='')
        print(self.lITrain[:mL], ' (length = ', len(self.lITrain), ')', sep='')
        print(GC.S_DS80, GC.S_NEWL, 'Train DataFrame:', sep='')
        print(self.dfrTrain)
        print(GC.S_DS80, GC.S_NEWL, 'Number of records test DataFrame:',
              GC.S_NEWL, self.nRecTest, sep='')
        print(GC.S_DS80, GC.S_NEWL, 'First ', mL, ' indices test DataFrame:',
              sep='')
        print(self.lITest[:mL], ' (length = ', len(self.lITest), ')', sep='')
        print(GC.S_DS80, GC.S_NEWL, 'Test DataFrame:', sep='')
        print(self.dfrTest, GC.S_NEWL, GC.S_DS80, sep='')

    # --- methods for producing result DataFrames -----------------------------
    def createResultsTrain(self, stT):
        if self.dITp['predictWTrain']:
            self.splitInpData()
            if self.dITp['genInfoEffTrain']:
                self.getInfoKinNmer(stT=stT, tpDfr=self.dITp['sTrain'])

    # --- methods for saving DataFrames ---------------------------------------
    def saveTrainTestData(self):
        if self.dITp['saveCombTrain']:
            pFOut = self.dPF[self.dITp['sTrain']][self.dITp['sCombinedOut']]
            self.saveData(self.dfrTrain, pF=pFOut)
        if self.dITp['saveCombTest']:
            pFOut = self.dPF[self.dITp['sTest']][self.dITp['sCombinedOut']]
            self.saveData(self.dfrTest, pF=pFOut)

###############################################################################
