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

from sklearn.model_selection import StratifiedKFold

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
    def printD2MnSEMResClf(self, sMth):
        if GF.Xist(self.d2MnSEMResClf):
            print(GC.S_DS04, ' Dictionary of results (means and SEMs) for ',
                  'method "', sMth, '":', sep='')
            dfrMnSEM = GF.iniPdDfr(self.d2MnSEMResClf)
            for k in range(len(self.d2MnSEMResClf)//2):
                print(GC.S_NEWL, dfrMnSEM.iloc[:, (k*2):(k*2 + 2)], sep='')

    def printSettingCRep(self, sMth, iSt, sKPar, cRep):
        print(GC.S_EQ20, 'Method:', sMth, GC.S_VBAR, 'Parameter',
              'set:', sKPar, GC.S_VBAR, 'Step index:', iSt, GC.S_VBAR,
              'Repetition:', cRep + 1)

    def printSettingParGrid(self, sMth, iSt, sKPar):
        print(GC.S_EQ20, ' Method: ', sMth, GC.S_NEWL,
              'Parameter grid optimisation using non-optimised values from ',
              'parameter set ', sKPar, GC.S_NEWL, 'Step index: ', iSt, sep='')

    # --- loop methods --------------------------------------------------------
    # --- method for updating the type dictionary -----------------------------
    def updateAttr(self):
        self.CLblsTrain = self.cClf.CLblsTrain
        self.lIFE = self.D.lIFE + [self.CLblsTrain]

    # --- methods for filling and changing the file paths ---------------------
    def fillFPsMth(self, sMth):
        self.FPs = self.D.yieldFPs()
        d2PI, dIG, dITp, sSz = {}, self.dIG, self.dITp, self.dITp['sFInpSzClf']
        lSEPar, lSESum, lSEDet = SF.getLSE(dITp, sMth, self.lIFE)
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

    def adaptFPs(self, sMth, iSt, sKP, cRep):
        sKPR = GF.joinS([sKP, str(cRep + 1), self.sSt], cJ=self.dITp['sUSC'])
        if self.dITp['doParGrid']:
            sKPR = GF.joinS([sKP, self.sSt], cJ=self.dITp['sUSC'])
        for s in ['DetldClf', 'ProbaClf']:
            self.FPs.modFP(d2PI=self.d2PInf, kMn=s, kPos='sLFE', cS=sKPR)

    # --- method for performing the calculations of the current repetition ----
    def getClfCRep(self, sMth, iSt=None, sKPar=GC.S_0, cRep=0):
        cClf, tM = None, 0
        if sMth == self.dITp['sMthDummy']:  # Dummy Classifier
            tM = (7, 1)
            lG, d2Par = self.dITp['lParGrid_Dummy'], self.dITp['d2Par_Dummy']
            cClf = DummyClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthAda']:  # AdaBoost Classifier
            tM = (7, 11)
            lG, d2Par = self.dITp['lParGrid_Ada'], self.dITp['d2Par_Ada']
            cClf = AdaClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthRF']:   # Random Forest Classifier
            tM = (7, 21)
            lG, d2Par = self.dITp['lParGrid_RF'], self.dITp['d2Par_RF']
            cClf = RFClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthXTr']:  # Extra Trees Classifier
            tM = (7, 31)
            lG, d2Par = self.dITp['lParGrid_XTr'], self.dITp['d2Par_XTr']
            cClf = XTrClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthGrB']:  # Gradient Boosting Classifier
            tM = (7, 41)
            lG, d2Par = self.dITp['lParGrid_GrB'], self.dITp['d2Par_GrB']
            cClf = GrBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthHGrB']: # Hist. Gradient Boosting Clf.
            tM = (7, 51)
            lG, d2Par = self.dITp['lParGrid_HGrB'], self.dITp['d2Par_HGrB']
            cClf = HGrBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthGP']:   # Gaussian Process Classifier
            tM = (7, 61)
            lG, d2Par = self.dITp['lParGrid_GP'], self.dITp['d2Par_GP']
            cClf = GPClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthPaA']:  # Passive Aggressive Classifier
            tM = (7, 71)
            lG, d2Par = self.dITp['lParGrid_PaA'], self.dITp['d2Par_PaA']
            cClf = PaAggClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthPct']:  # Perceptron Classifier
            tM = (7, 81)
            lG, d2Par = self.dITp['lParGrid_Pct'], self.dITp['d2Par_Pct']
            cClf = PctClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthSGD']:  # SGD Classifier
            tM = (7, 91)
            lG, d2Par = self.dITp['lParGrid_SGD'], self.dITp['d2Par_SGD']
            cClf = SGDClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthCtNB']: # Categorical NB Classifier
            tM = (7, 101)
            lG, d2Par = self.dITp['lParGrid_CtNB'], self.dITp['d2Par_CtNB']
            cClf = CtNBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthCpNB']: # Complement NB Classifier
            tM = (7, 111)
            lG, d2Par = self.dITp['lParGrid_CpNB'], self.dITp['d2Par_CpNB']
            cClf = CpNBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthGsNB']: # Gaussian NB Classifier
            tM = (7, 121)
            lG, d2Par = self.dITp['lParGrid_GsNB'], self.dITp['d2Par_GsNB']
            cClf = GsNBClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthMLP']:  # NN MLP Classifier
            tM = (7, 131)
            lG, d2Par = self.dITp['lParGrid_MLP'], self.dITp['d2Par_MLP']
            cClf = MLPClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthLSV']:  # Linear SV Classifier
            tM = (7, 141)
            lG, d2Par = self.dITp['lParGrid_LSV'], self.dITp['d2Par_LSV']
            cClf = LinSVClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        elif sMth == self.dITp['sMthNSV']:  # Nu-Support SV Classifier
            tM = (7, 151)
            lG, d2Par = self.dITp['lParGrid_NSV'], self.dITp['d2Par_NSV']
            cClf = NuSVClf(self.inpD, self.D, lG, d2Par, iSt, sKPar, cRep)
        self.cTM, self.cLParGrid, self.cD2Par, self.cClf = tM, lG, d2Par, cClf

    def doRep(self, cTim, sMth, iSt=None, k=0, sKPar=GC.S_0, cRep=0, stT=None):
        if sMth in self.dITp['lSMth']:
            cT = GF.showElapsedTime(startTime=stT)
            self.getClfCRep(sMth=sMth, iSt=iSt, sKPar=sKPar, cRep=cRep)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(tMth=self.cTM, stTMth=cT, endTMth=cEndT)
            cT = GF.showElapsedTime(startTime=stT)
            self.updateAttr()
            self.cClf.ClfPred()
            self.cClf.printFitQuality()
            GF.updateDict(self.d3ResClf, cDUp=self.cClf.d2ResClf, cK=cRep)
            GF.updateDict(self.d2CnfMat, cDUp=self.cClf.dCnfMat, cK=cRep)
            if k == 0 and cRep == 0:
                self.fillFPsMth(sMth=sMth)
            self.adaptFPs(sMth=sMth, iSt=iSt, sKP=sKPar, cRep=cRep)
            if self.dITp['saveDetailedClfRes']:
                self.saveData(self.cClf.dfrPred, pF=self.FPs.dPF['DetldClf'])
                self.saveData(self.cClf.dfrProba, pF=self.FPs.dPF['ProbaClf'])
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(tMth=(self.cTM[0], self.cTM[1] + 1), stTMth=cT,
                             endTMth=cEndT)

    # --- method for saving the results ---------------------------------------
    def saveCombRes(self, sMth, d2Par, nRep=0):
        if nRep > 0:
            # TEMP - BEGIN:
            for sK, sF in self.FPs.dPF.items():
                print(sK, ':', sF)
            # TEMP - END.
            self.saveData(GF.iniPdDfr(d2Par), pF=self.FPs.dPF['OutParClf'],
                          saveAnyway=False)
            self.d2MnSEMResClf = GF.calcMnSEMFromD3Val(self.d3ResClf)
            self.dMnSEMCnfMat = GF.calcMnSEMFromD2Dfr(self.d2CnfMat)
            sKSC, sKCM, sSt = 'SumryClf', 'CnfMat', self.sSt
            self.FPs.modFP(d2PI=self.d2PInf, kMn=sKSC, kPos='sLFE', cS=sSt)
            self.saveData(self.d2MnSEMResClf, pF=self.FPs.dPF[sKSC])
            for sK in self.dMnSEMCnfMat:
                sKS = GF.joinS([sK, sSt])
                self.FPs.modFP(d2PI=self.d2PInf, kMn=sKCM, kPos='sLFE', cS=sKS)
                self.saveData(self.dMnSEMCnfMat[sK], pF=self.FPs.dPF[sKCM])
            self.printD2MnSEMResClf(sMth=sMth)

    # --- method for performing the outer loops -------------------------------
    def doQuadLoop(self, cTim, stT=None):
        for sMth in self.dITp['lSMth']:
            self.d3ResClf, self.d2CnfMat, sStep = {}, {}, self.dITp['sStep']
            d2Par, nRep = self.dITp['d3Par'][sMth], self.dITp['dNumRep'][sMth]
            for iSt in self.itISt:
                sStI = GF.joinS([self.dITp['sFInpStIClf'], sStep + str(iSt)])
                self.sSt = ('' if (iSt is None) else sStI)
                if self.dITp['doParGrid'] and nRep > 0:
                    assert self.dITp['s0'] in d2Par
                    self.printSettingParGrid(sMth, iSt, sKPar=self.dITp['s0'])
                    self.doRep(cTim, sMth, iSt, sKPar=self.dITp['s0'], stT=stT)
                elif not self.dITp['doParGrid']:
                    if not self.dITp['useKey0'] and self.dITp['s0'] in d2Par:
                        del d2Par[self.dITp['s0']]
                    for k, sKPar in enumerate(d2Par):
                        for cRep in range(nRep):
                            self.printSettingCRep(sMth, iSt, sKPar, cRep)
                            self.doRep(cTim, sMth, iSt, k, sKPar, cRep, stT)
                    self.saveCombRes(sMth, d2Par, nRep)

###############################################################################