# -*- coding: utf-8 -*-
###############################################################################
# --- O_80__Looper.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Core.O_00__BaseClass import BaseClass
from Core.O_07__Classifier import DummyClf, AdaClf, RFClf, GPClf, MLPClf

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

    def printSettingCRep(self, sMth, sKPar, iSt, cRep):
        print(GC.S_EQ20, 'Method:', sMth, GC.S_VBAR, 'Parameter',
              'set:', sKPar, GC.S_VBAR, 'Step index:', iSt, GC.S_VBAR,
              'Repetition:', cRep + 1)

    # --- loop methods --------------------------------------------------------
    # --- method for updating the type dictionary -----------------------------
    def updateAttr(self, cClf):
        self.CLblsTrain = cClf.CLblsTrain
        self.lIFE = self.D.lIFE + [self.CLblsTrain]

    # --- methods for filling and changing the file paths ---------------------
    def fillFPsMth(self, sMth):
        self.FPs = self.D.yieldFPs()
        d2PI, dIG, dITp, sFIB = {}, self.dIG, self.dITp, self.dITp['sFInpBClf']
        d2PI['OutParClf'] = {dIG['sPath']: dITp['pOutPar'],
                             dIG['sLFS']: [dITp['sPar'], sFIB],
                             dIG['sLFC']: [sMth] + list(dITp['d3Par'][sMth]),
                             dIG['sLFJS']: dITp['sUS02'],
                             dIG['sLFJSC']: dITp['sUS02'],
                             dIG['sFXt']: dIG['xtCSV']}
        d2PI['OutSumClf'] = {dIG['sPath']: dITp['pOutSum'],
                             dIG['sLFS']: [dITp['sSummary'], sFIB],
                             dIG['sLFC']: self.lIFE,
                             dIG['sLFE']: [sMth] + list(dITp['d3Par'][sMth]),
                             dIG['sLFJS']: dITp['sUS02'],
                             dIG['sLFJSC']: dITp['sUS02'],
                             dIG['sFXt']: dIG['xtCSV']}
        d2PI['CnfMat'] = {dIG['sPath']: dITp['pCnfMat'],
                          dIG['sLFC']: [dITp['sCnfMat'], sFIB],
                          dIG['sLFE']: self.lIFE + [sMth],
                          dIG['sLFJC']: dITp['sUS02'],
                          dIG['sLFJCE']: dITp['sUS02'],
                          dIG['sFXt']: dIG['xtCSV']}
        d2PI['OutDetClf'] = {dIG['sPath']: dITp['pOutDet'],
                             dIG['sLFC']: [dITp['sDetailed'], sFIB],
                             dIG['sLFE']: self.lIFE + [sMth],
                             dIG['sLFJC']: dITp['sUS02'],
                             dIG['sLFJCE']: dITp['sUS02'],
                             dIG['sFXt']: dIG['xtCSV']}
        d2PI['OutProbaClf'] = {dIG['sPath']: dITp['pOutDet'],
                               dIG['sLFC']: [dITp['sProba'], sFIB],
                               dIG['sLFE']: self.lIFE + [sMth],
                               dIG['sLFJC']: dITp['sUS02'],
                               dIG['sLFJCE']: dITp['sUS02'],
                               dIG['sFXt']: dIG['xtCSV']}
        self.FPs.addFPs(d2PI)
        self.d2PInf = d2PI

    def adaptFPs(self, sMth, sKP, cRep, iSt):
        sSt = ('' if (iSt is None) else (self.dITp['sStep'] + str(iSt)))
        sKPR = GF.joinS([sKP, str(cRep + 1), sSt], cJ=self.dITp['sUSC'])
        for s in ['OutDetClf', 'OutProbaClf']:
            self.FPs.modFP(d2PI=self.d2PInf, kMn=s, kPos='sLFE', cS=sKPR)

    # --- method for performing the calculations of the current repetition ----
    def getClfCRep(self, sMth, iSt, sKPar):
        cClf, iM = None, 0
        if sMth == self.dITp['sMthDummy']:  # Dummy Classifier
            iM = 15
            lG, d2Par = self.dITp['lParGrid_Dummy'], self.dITp['d2Par_Dummy']
            cClf = DummyClf(self.inpD, self.D, lG, d2Par, sKPar=sKPar, iSt=iSt)
        elif sMth == self.dITp['sMthAda']:  # AdaBoost Classifier
            iM = 17
            lG, d2Par = self.dITp['lParGrid_Ada'], self.dITp['d2Par_Ada']
            cClf = AdaClf(self.inpD, self.D, lG, d2Par, sKPar=sKPar, iSt=iSt)
        elif sMth == self.dITp['sMthRF']:   # random forest Classifier
            iM = 19
            lG, d2Par = self.dITp['lParGrid_RF'], self.dITp['d2Par_RF']
            cClf = RFClf(self.inpD, self.D, lG, d2Par, sKPar=sKPar, iSt=iSt)
        elif sMth == self.dITp['sMthGP']:   # Gaussian Process Classifier
            iM = 21
            lG, d2Par = self.dITp['lParGrid_GP'], self.dITp['d2Par_GP']
            cClf = GPClf(self.inpD, self.D, lG, d2Par, sKPar=sKPar, iSt=iSt)
        elif sMth == self.dITp['sMthMLP']:  # NN MLP Classifier
            iM = 23
            lG, d2Par = self.dITp['lParGrid_MLP'], self.dITp['d2Par_MLP']
            cClf = MLPClf(self.inpD, self.D, lG, d2Par, sKPar=sKPar, iSt=iSt)
        return cClf, iM

    def doCRep(self, sMth, k, sKPar, iSt, cRep, cTim, stT=None):
        if sMth in self.dITp['lSMth']:
            cStT = GF.showElapsedTime(startTime=stT)
            cClf, iM = self.getClfCRep(sMth=sMth, iSt=iSt, sKPar=sKPar)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM, stTMth=cStT, endTMth=cEndT)
            cStT = GF.showElapsedTime(startTime=stT)
            self.updateAttr(cClf)
            cClf.ClfPred()
            cClf.printFitQuality()
            GF.updateDict(self.d3ResClf, cDUp=cClf.d2ResClf, cK=cRep)
            GF.updateDict(self.d2CnfMat, cDUp=cClf.dCnfMat, cK=cRep)
            if k == 0 and cRep == 0:
                self.fillFPsMth(sMth=sMth)
            self.adaptFPs(sMth=sMth, sKP=sKPar, cRep=cRep, iSt=iSt)
            if self.dITp['saveDetailedClfRes']:
                self.saveData(cClf.dfrPred, pF=self.FPs.dPF['OutDetClf'])
                self.saveData(cClf.dfrProba, pF=self.FPs.dPF['OutProbaClf'])
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM + 1, stTMth=cStT, endTMth=cEndT)

    # --- method for saving the results ---------------------------------------
    def saveCombRes(self, sMth, iSt, d2Par, nRep=0):
        if nRep > 0:
            self.saveData(GF.iniPdDfr(d2Par), pF=self.FPs.dPF['OutParClf'],
                          saveAnyway=False)
            self.d2MnSEMResClf = GF.calcMnSEMFromD3Val(self.d3ResClf)
            self.dMnSEMCnfMat = GF.calcMnSEMFromD2Dfr(self.d2CnfMat)
            sSt, sKSC, sKCM = None, 'OutSumClf', 'CnfMat'
            if iSt is not None:
                sSt = self.dITp['sStep'] + str(iSt)
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
            self.d3ResClf, self.d2CnfMat = {}, {}
            d2Par, nRep = self.dITp['d3Par'][sMth], self.dITp['dNumRep'][sMth]
            for iSt in self.itISt:
                for k, sKPar in enumerate(d2Par):
                    for cRep in range(nRep):
                        self.printSettingCRep(sMth, sKPar, iSt, cRep)
                        self.doCRep(sMth, k, sKPar, iSt, cRep, cTim, stT)
                self.saveCombRes(sMth, iSt, d2Par, nRep)

###############################################################################