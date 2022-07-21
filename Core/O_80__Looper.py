# -*- coding: utf-8 -*-
###############################################################################
# --- O_80__Looper.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Core.O_00__BaseClass import BaseClass
from Core.O_07__Classifier import RndForestClf, NNMLPClf

# -----------------------------------------------------------------------------
class Looper(BaseClass):
# --- initialisation of the class ---------------------------------------------
    def __init__(self, inpDat, D, iTp=80, lITpUpd=[6]):
        super().__init__(inpDat)
        self.idO = 'O_80'
        self.descO = 'Looper'
        self.inpD = inpDat
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.D = D
        self.iniDicts()
        print('Initiated "Looper" base object.')

    def iniDicts(self):
        self.d3ResClf, self.d2MnSEMResClf = {}, {}
        self.d2CnfMat, self.dMnSEMCnfMat = {}, {}
        self.dPF = {'OutParClf': None, 'OutSumClf': None, 'ConfMat': None,
                    'OutDetClf': None, 'OutProbaClf': None}

# --- print methods -----------------------------------------------------------
    def printD2MnSEMResClf(self, sMth):
        if GF.Xist(self.d2MnSEMResClf):
            print(GC.S_DS04, ' Dictionary of results (means and SEMs) for ',
                  'method "', sMth, '":', sep='')
            dfrMnSEM = GF.iniPdDfr(self.d2MnSEMResClf)
            for k in range(len(self.d2MnSEMResClf)//2):
                print(GC.S_NEWL, dfrMnSEM.iloc[:, (k*2):(k*2 + 2)], sep='')

# --- loop methods ------------------------------------------------------------
    def adapMthDPF(self, cClf, sMth):
        sJ1, sJ2 = self.dITp['sUSC'], self.dITp['sUS02']
        xtCSV, sSum = self.dITp['xtCSV'], self.dITp['sSummary']
        sFE = GF.joinS([self.dITp['sMaxLenNmer'], self.dITp['sRestr'],
                        cClf.dITp['sCLblsTrain'], cClf.dITp['sSglMltLbl'],
                        cClf.dITp['sXclEffFam']], sJoin=sJ1)
        sLKP = GF.joinS(list(self.dITp['d3Par'][sMth]), sJoin=sJ1)
        sFCPar = GF.joinS([cClf.dITp['sParClf'], sMth], sJoin=sJ1)
        sFCore = GF.joinS([cClf.dITp['sOutClf'], sMth, sFE], sJoin=sJ1)
        sFPar = GF.joinS([self.dITp['sPar'], sLKP, sFCPar], sJoin=sJ2) + xtCSV
        sFSum = GF.joinS([sSum, sLKP, sFCore], sJoin=sJ2) + xtCSV
        sFConfM = GF.joinS([self.dITp['sConfMat'], sFCore], sJoin=sJ2) + xtCSV
        sFDet = GF.joinS([self.dITp['sDetailed'], sFCore], sJoin=sJ2) + xtCSV
        sFProba = GF.joinS([self.dITp['sProba'], sFCore], sJoin=sJ2) + xtCSV
        self.dPF['OutParClf'] = GF.joinToPath(cClf.dITp['pOutPar'], sFPar)
        self.dPF['OutSumClf'] = GF.joinToPath(cClf.dITp['pOutSum'], sFSum)
        self.dPF['ConfMat'] = GF.joinToPath(cClf.dITp['pConfMat'], sFConfM)
        self.dPF['OutDetClfB'] = GF.joinToPath(cClf.dITp['pOutDet'], sFDet)
        self.dPF['OutProbaClfB'] = GF.joinToPath(cClf.dITp['pOutDet'], sFProba)

    def adaptKParRpDPF(self, cClf, sMth, sKP, cRp):
        sKPR = GF.joinS([sKP, str(cRp + 1)], sJoin=self.dITp['sUSC'])
        self.dPF['OutDetClf'] = GF.modPF(self.dPF['OutDetClfB'], sEnd=sKPR,
                                         sJoin=self.dITp['sUSC'])
        self.dPF['OutProbaClf'] = GF.modPF(self.dPF['OutProbaClfB'], sEnd=sKPR,
                                           sJoin=self.dITp['sUSC'])

    def doCRep(self, sMth, k, sKPar, cRp, cTim, stT=None):
        if sMth in self.dITp['lSMth']:
            cStT, iM = GF.showElapsedTime(startTime=stT), 0
            if sMth == self.dITp['sMthRF']:     # random forest classifier
                d2Par, iM = self.dITp['d2Par_RF'], 15
                cClf = RndForestClf(self.inpD, self.D, d2Par, sKPar=sKPar)
            elif sMth == self.dITp['sMthMLP']:  # NN MLP classifier
                d2Par, iM = self.dITp['d2Par_NNMLP'], 17
                cClf = NNMLPClf(self.inpD, self.D, d2Par, sKPar=sKPar)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM, stTMth=cStT, endTMth=cEndT)
            cStT = GF.showElapsedTime(startTime=stT)
            cClf.ClfPred()
            cClf.printFitQuality()
            GF.updateDict(self.d3ResClf, cDUp=cClf.d2ResClf, cK=cRp)
            GF.updateDict(self.d2CnfMat, cDUp=cClf.dConfMat, cK=cRp)
            if k == 0 and cRp == 0:
                self.adapMthDPF(cClf=cClf, sMth=sMth)
            self.adaptKParRpDPF(cClf=cClf, sMth=sMth, sKP=sKPar, cRp=cRp)
            if cClf.dITp['saveDetailedClfRes']:
                self.saveData(cClf.dfrPred, pF=self.dPF['OutDetClf'])
                self.saveData(cClf.dfrProba, pF=self.dPF['OutProbaClf'])
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM + 1, stTMth=cStT, endTMth=cEndT)

    def saveCombRes(self, sMth, d2Par, nRep=0):
        sJ = self.dITp['sUSC']
        if nRep > 0:
            self.saveData(GF.iniPdDfr(d2Par), pF=self.dPF['OutParClf'],
                          saveAnyway=False)
            self.d2MnSEMResClf = GF.calcMnSEMFromD3Val(self.d3ResClf)
            self.dMnSEMCnfMat = GF.calcMnSEMFromD2Dfr(self.d2CnfMat)
            self.saveData(self.d2MnSEMResClf, pF=self.dPF['OutSumClf'])
            for sK in self.dMnSEMCnfMat:
                pFMod = GF.modPF(self.dPF['ConfMat'], sEnd=sK, sJoin=sJ)
                self.saveData(self.dMnSEMCnfMat[sK], pF=pFMod)
            self.printD2MnSEMResClf(sMth=sMth)

    def doDoubleLoop(self, cTim, stT=None):
        for sMth in self.dITp['lSMth']:
            self.d3ResClf, self.d2CnfMat = {}, {}
            d2Par, nRep = self.dITp['d3Par'][sMth], self.dITp['dNumRep'][sMth]
            for k, sKPar in enumerate(d2Par):
                for cRep in range(nRep):
                    print(GC.S_EQ20, 'Method:', sMth, GC.S_VBAR, 'Parameter',
                          'set:', sKPar, GC.S_VBAR, 'Repetition:', cRep + 1)
                    self.doCRep(sMth, k, sKPar, cRp=cRep, cTim=cTim, stT=stT)
            self.saveCombRes(sMth=sMth, d2Par=d2Par, nRep=nRep)

###############################################################################