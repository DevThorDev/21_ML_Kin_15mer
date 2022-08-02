# -*- coding: utf-8 -*-
###############################################################################
# --- O_80__Looper.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Core.O_00__BaseClass import BaseClass
from Core.O_07__Classifier import RFClf, MLPClf

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

    # --- loop methods --------------------------------------------------------
    # --- method for updating the type dictionary -----------------------------
    def updateDITpAttr(self, cClf):
        self.dITp['sCLblsTrain'] = cClf.dITp['sCLblsTrain']
        self.lIFE = self.D.lIFE + [self.dITp['sCLblsTrain']]
    
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

    def changePFKParRp(self, sMth, sKP, cRp):
        sKPR = GF.joinS([sKP, str(cRp + 1)], cJ=self.dITp['sUSC'])
        for s in ['OutDetClf', 'OutProbaClf']:
            self.FPs.modFP(d2PI=self.d2PInf, kMn=s, kPos='sLFE', cS=sKPR)

    # --- method for performing the calculations of the current repetition ----
    def doCRep(self, sMth, k, sKPar, cRp, cTim, stT=None):
        if sMth in self.dITp['lSMth']:
            cStT, iM = GF.showElapsedTime(startTime=stT), 0
            if sMth == self.dITp['sMthRF']:     # random forest classifier
                iM = 15
                lG, d2Par = self.dITp['lParGrid_RF'], self.dITp['d2Par_RF']
                cClf = RFClf(self.inpD, self.D, lG, d2Par, sKPar=sKPar)
            elif sMth == self.dITp['sMthMLP']:  # NN MLP classifier
                iM = 17
                lG, d2Par = self.dITp['lParGrid_MLP'], self.dITp['d2Par_MLP']
                cClf = MLPClf(self.inpD, self.D, lG, d2Par, sKPar=sKPar)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM, stTMth=cStT, endTMth=cEndT)
            cStT = GF.showElapsedTime(startTime=stT)
            self.updateDITpAttr(cClf)
            cClf.ClfPred()
            cClf.printFitQuality()
            GF.updateDict(self.d3ResClf, cDUp=cClf.d2ResClf, cK=cRp)
            GF.updateDict(self.d2CnfMat, cDUp=cClf.dCnfMat, cK=cRp)
            if k == 0 and cRp == 0:
                self.fillFPsMth(sMth=sMth)
            self.changePFKParRp(sMth=sMth, sKP=sKPar, cRp=cRp)
            if self.dITp['saveDetailedClfRes']:
                self.saveData(cClf.dfrPred, pF=self.FPs.dPF['OutDetClf'])
                self.saveData(cClf.dfrProba, pF=self.FPs.dPF['OutProbaClf'])
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM + 1, stTMth=cStT, endTMth=cEndT)

    # --- method for saving the results ---------------------------------------
    def saveCombRes(self, sMth, d2Par, nRep=0):
        if nRep > 0:
            self.saveData(GF.iniPdDfr(d2Par), pF=self.FPs.dPF['OutParClf'],
                          saveAnyway=False)
            self.d2MnSEMResClf = GF.calcMnSEMFromD3Val(self.d3ResClf)
            self.dMnSEMCnfMat = GF.calcMnSEMFromD2Dfr(self.d2CnfMat)
            self.saveData(self.d2MnSEMResClf, pF=self.FPs.dPF['OutSumClf'])
            for sK in self.dMnSEMCnfMat:
                self.FPs.modFP(d2PI=self.d2PInf, kMn='CnfMat', kPos='sLFE',
                               cS=sK)
                self.saveData(self.dMnSEMCnfMat[sK], pF=self.FPs.dPF['CnfMat'])
            self.printD2MnSEMResClf(sMth=sMth)

    # --- method for performing the outer loop --------------------------------
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