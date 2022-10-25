# -*- coding: utf-8 -*-
###############################################################################
# --- O_80__Looper.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass
from Core.O_07__Classifier import (DummyClf, AdaClf, RFClf, XTrClf, GrBClf,
                                   HGrBClf, GPClf, PaAggClf, PctClf, SGDClf,
                                   CtNBClf, CpNBClf, GsNBClf, MLPClf, LinSVClf,
                                   NuSVClf)

# -----------------------------------------------------------------------------
class Looper(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, iTp=80, lITpUpd=[6, 7]):
        super().__init__(inpDat)
        self.idO = 'O_80'
        self.descO = 'Looper'
        self.inpD = inpDat
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.D = D
        self.iniDicts()
        # define the iterator of the (MultiSteps) step index
        self.itISt = (range(1, self.D.dMltStClf['nSteps'] + 1) if
                      (self.dITp['doMultiSteps']) else [None])
        print('Initiated "Looper" base object.')

    def iniDicts(self):
        self.d3ResClf, self.d2MnSEMResClf = {}, {}
        self.d2CnfMat, self.dMnSEMCnfMat = {}, {}

    # --- print methods -------------------------------------------------------
    def printD2MnSEMResClf(self):
        if GF.Xist(self.d2MnSEMResClf):
            print(GC.S_DS04, ' Dictionary of results (means and SEMs) for ',
                  'method "', self.sMth, '":', sep='')
            dfrMnSEM = GF.iniPdDfr(self.d2MnSEMResClf)
            for k in range(len(self.d2MnSEMResClf)//2):
                print(GC.S_NEWL, dfrMnSEM.iloc[:, (k*2):(k*2 + 2)], sep='')

    def printSettingCRep(self, sKPar, cRep):
        print(GC.S_EQ20, 'Method:', self.sMth, GC.S_VBAR, 'Parameter',
              'set:', sKPar, GC.S_VBAR, 'Step index:', self.iSt, GC.S_VBAR,
              'Repetition:', cRep + 1)

    def printSettingParGrid(self, sKPar):
        print(GC.S_EQ20, ' Method: ', self.sMth, GC.S_NEWL,
              'Parameter grid optimisation using non-optimised values from ',
              'parameter set ', sKPar, GC.S_NEWL, 'Step index: ', self.iSt,
              sep='')

    # --- loop methods --------------------------------------------------------
    # --- method for updating the type dictionary -----------------------------
    def updateAttr(self):
        self.CLblsTrain = self.cClf.CLblsTrain
        self.lIFE = self.D.lIFE + [self.CLblsTrain]

    # --- methods for filling and changing the file paths ---------------------
    def fillFPsMth(self):
        self.FPs = self.D.yieldFPs()
        d2PI, dIG, dITp, sSz = {}, self.dIG, self.dITp, self.dITp['sFInpSzClf']
        lSEPar, lSESum, lSEDet = SF.getLSE(dITp, self.sMth, self.lIFE)
        d2PI['OutParClf'] = {dIG['sPath']: dITp['pOutPar'],
                             dIG['sLFS']: [dITp['sPar'], sSz],
                             dIG['sLFC']: lSEPar,
                             dIG['sLFJS']: dITp['sUS02'],
                             dIG['sLFJSC']: dITp['sUS02'],
                             dIG['sFXt']: dIG['xtCSV']}
        d2PI['SumryClf'] = {dIG['sPath']: dITp['pOutSum'],
                            dIG['sLFS']: [dITp['sSummary'], sSz],
                            dIG['sLFC']: self.lIFE,
                            dIG['sLFE']: lSESum,
                            dIG['sLFJS']: dITp['sUS02'],
                            dIG['sLFJSC']: dITp['sUS02'],
                            dIG['sFXt']: dIG['xtCSV']}
        d2PI['CnfMat'] = {dIG['sPath']: dITp['pCnfMat'],
                          dIG['sLFC']: [dITp['sCnfMat'], sSz],
                          dIG['sLFE']: lSEDet,
                          dIG['sLFJC']: dITp['sUS02'],
                          dIG['sLFJCE']: dITp['sUS02'],
                          dIG['sFXt']: dIG['xtCSV']}
        d2PI['DetldClf'] = {dIG['sPath']: dITp['pOutDet'],
                            dIG['sLFC']: [dITp['sDetailed'], sSz],
                            dIG['sLFE']: lSEDet,
                            dIG['sLFJC']: dITp['sUS02'],
                            dIG['sLFJCE']: dITp['sUS02'],
                            dIG['sFXt']: dIG['xtCSV']}
        d2PI['ProbaClf'] = {dIG['sPath']: dITp['pOutDet'],
                            dIG['sLFC']: [dITp['sProba'], sSz],
                            dIG['sLFE']: lSEDet,
                            dIG['sLFJC']: dITp['sUS02'],
                            dIG['sLFJCE']: dITp['sUS02'],
                            dIG['sFXt']: dIG['xtCSV']}
        self.FPs.addFPs(d2PI)
        self.d2PInf = d2PI

    def adaptFPs(self, sKP, cRep):
        sRepS, sUSC = self.dITp['sRepS'], self.dITp['sUSC']
        sKPR = GF.joinS([sKP, sRepS + str(cRep + 1), self.sSt], cJ=sUSC)
        if self.dITp['doParGrid']:
            sKPR = GF.joinS([sKP, self.sSt], cJ=sUSC)
        for s in ['DetldClf', 'ProbaClf']:
            self.FPs.modFP(d2PI=self.d2PInf, kMn=s, kPos='sLFE', cS=sKPR)

    # --- method for performing the calculations of the current repetition ----
    def getClfCRep(self, sKPar=GC.S_0, cRep=0):
        sMth, d2Par, iSt, cClf, tM = self.sMth, self.d2Par, self.iSt, None, 0
        if sMth == self.dITp['sMthDummy']:  # Dummy Classifier
            tM, lG = (7, 1), self.dITp['lParGrid_Dummy']
            cClf = DummyClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthAda']:  # AdaBoost Classifier
            tM, lG = (7, 11), self.dITp['lParGrid_Ada']
            cClf = AdaClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthRF']:   # Random Forest Classifier
            tM, lG = (7, 21), self.dITp['lParGrid_RF']
            cClf = RFClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthXTr']:  # Extra Trees Classifier
            tM, lG = (7, 31), self.dITp['lParGrid_XTr']
            cClf = XTrClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthGrB']:  # Gradient Boosting Classifier
            tM, lG = (7, 41), self.dITp['lParGrid_GrB']
            cClf = GrBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthHGrB']: # Hist. Gradient Boosting Clf.
            tM, lG = (7, 51), self.dITp['lParGrid_HGrB']
            cClf = HGrBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthGP']:   # Gaussian Process Classifier
            tM, lG = (7, 61), self.dITp['lParGrid_GP']
            cClf = GPClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthPaA']:  # Passive Aggressive Classifier
            tM, lG = (7, 71), self.dITp['lParGrid_PaA']
            cClf = PaAggClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthPct']:  # Perceptron Classifier
            tM, lG = (7, 81), self.dITp['lParGrid_Pct']
            cClf = PctClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthSGD']:  # SGD Classifier
            tM, lG = (7, 91), self.dITp['lParGrid_SGD']
            cClf = SGDClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthCtNB']: # Categorical NB Classifier
            tM, lG = (7, 101), self.dITp['lParGrid_CtNB']
            cClf = CtNBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthCpNB']: # Complement NB Classifier
            tM, lG = (7, 111), self.dITp['lParGrid_CpNB']
            cClf = CpNBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthGsNB']: # Gaussian NB Classifier
            tM, lG = (7, 121), self.dITp['lParGrid_GsNB']
            cClf = GsNBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthMLP']:  # NN MLP Classifier
            tM, lG = (7, 131), self.dITp['lParGrid_MLP']
            cClf = MLPClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthLSV']:  # Linear SV Classifier
            tM, lG = (7, 141), self.dITp['lParGrid_LSV']
            cClf = LinSVClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthNSV']:  # Nu-Support SV Classifier
            tM, lG = (7, 151), self.dITp['lParGrid_NSV']
            cClf = NuSVClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        self.cTM, self.lParGrid, self.cClf = tM, lG, cClf

    def doRep(self, cTim, k=0, sKPar=GC.S_0, cRep=0, stT=None):
        if self.sMth in self.dITp['lSMth']:
            cT = GF.showElapsedTime(startTime=stT)
            self.getClfCRep(sKPar=sKPar, cRep=cRep)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(tMth=self.cTM, stTMth=cT, endTMth=cEndT)
            cT = GF.showElapsedTime(startTime=stT)
            self.updateAttr()
            self.cClf.ClfPred()
            self.cClf.printFitQuality()
            GF.updateDict(self.d3ResClf, cDUp=self.cClf.d2ResClf, cK=cRep)
            GF.updateDict(self.d2CnfMat, cDUp=self.cClf.dDfrCnfMat, cK=cRep)
            if k == 0 and cRep == 0:
                self.fillFPsMth()
            self.adaptFPs(sKP=sKPar, cRep=cRep)
            if self.dITp['saveDetailedClfRes']:
                lSrtBy = [self.dITp['sCNmer']]
                dfrPred = self.cClf.dfrPred.sort_values(by=lSrtBy)
                dfrProba = self.cClf.dfrProba.sort_values(by=lSrtBy)
                self.saveData(dfrPred, pF=self.FPs.dPF['DetldClf'])
                self.saveData(dfrProba, pF=self.FPs.dPF['ProbaClf'])
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(tMth=(self.cTM[0], self.cTM[1] + 1), stTMth=cT,
                             endTMth=cEndT)

    # --- method for saving the results ---------------------------------------
    def saveCombRes(self, nRep=0):
        if nRep > 0:
            self.saveData(GF.iniPdDfr(self.d2Par),
                          pF=self.FPs.dPF['OutParClf'], saveAnyway=False)
            self.d2MnSEMResClf = GF.calcMnSEMFromD3Val(self.d3ResClf)
            self.dMnSEMCnfMat = GF.calcMnSEMFromD2Dfr(self.d2CnfMat)
            sKSC, sKCM, sSt = 'SumryClf', 'CnfMat', self.sSt
            self.FPs.modFP(d2PI=self.d2PInf, kMn=sKSC, kPos='sLFE', cS=sSt)
            self.saveData(self.d2MnSEMResClf, pF=self.FPs.dPF[sKSC])
            for sK in self.dMnSEMCnfMat:
                sKS = GF.joinS([sK, sSt])
                self.FPs.modFP(d2PI=self.d2PInf, kMn=sKCM, kPos='sLFE', cS=sKS)
                self.saveData(self.dMnSEMCnfMat[sK], pF=self.FPs.dPF[sKCM])
            self.printD2MnSEMResClf()

    # --- method for performing the outer loops -------------------------------
    def loopParRepKFold(self, cTim, stT=None):
        if not self.dITp['useKey0'] and self.dITp['s0'] in self.d2Par:
            del self.d2Par[self.dITp['s0']]
        for k, sKPar in enumerate(self.d2Par):
            for cRep in range(self.nRep):
                self.printSettingCRep(sKPar, cRep)
                self.doRep(cTim, k=k, sKPar=sKPar, cRep=cRep, stT=stT)
        self.saveCombRes(nRep=self.nRep)

    def doQuadLoop(self, cTim, stT=None):
        sFInpStIClf, sStep = self.dITp['sFInpStIClf'], self.dITp['sStep']
        for self.sMth in self.dITp['lSMth']:
            self.d3ResClf, self.d2CnfMat = {}, {}
            self.d2Par = self.dITp['d3Par'][self.sMth]
            self.nRep = self.dITp['dNumRep'][self.sMth]
            for self.iSt in self.itISt:
                sStI = GF.joinS([sFInpStIClf, sStep + str(self.iSt)])
                self.sSt = ('' if (self.iSt is None) else sStI)
                if self.dITp['doParGrid'] and self.nRep > 0:
                    assert self.dITp['s0'] in self.d2Par
                    self.printSettingParGrid(sKPar=self.dITp['s0'])
                    self.doRep(cTim, sKPar=self.dITp['s0'], stT=stT)
                elif not self.dITp['doParGrid']:
                    self.loopParRepKFold(cTim, stT=stT)

###############################################################################