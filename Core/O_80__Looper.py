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
    def __init__(self, inpDat, D, iTp=80, lITpUpd=[]):
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
        self.dPF = {'OutParClf': None, 'OutDataClf': None, 'ConfMat': None}

# --- print methods -----------------------------------------------------------
    def printD2MnSEMResClf(self, sMth):
        if GF.Xist(self.d2MnSEMResClf):
            print(GC.S_DS04, ' Dictionary of results (means and SEMs) for ',
                  'method "', sMth, '":', sep='')
            dfrMnSEM = GF.iniPdDfr(self.d2MnSEMResClf)
            for k in range(len(self.d2MnSEMResClf)//2):
                print(GC.S_NEWL, dfrMnSEM.iloc[:, (k*2):(k*2 + 2)], sep='')

# --- loop methods ------------------------------------------------------------
    def adaptDPF(self, cClf, sMth):
        sJ, xtCSV = cClf.dITp['sUSC'], self.dITp['xtCSV']
        sFCore = GF.joinS([cClf.dITp['sFOutClf'], sMth], sJoin=sJ)
        sFPar = GF.joinS([sFCore, self.dITp['sPar']], sJoin=sJ) + xtCSV
        sFData = sFCore + xtCSV
        sFConfMat = GF.joinS([cClf.dITp['sFConfMat'], sMth], sJoin=sJ) + xtCSV
        self.dPF['OutParClf'] = GF.joinToPath(cClf.dITp['pOutClf'], sFPar)
        self.dPF['OutDataClf'] = GF.joinToPath(cClf.dITp['pOutClf'], sFData)
        self.dPF['ConfMat'] = GF.joinToPath(cClf.dITp['pConfMat'], sFConfMat)

    def doCRep(self, sMth, k, sKPar, cRp, cTim, stT=None):
        if sMth in self.dITp['lSMth']:
            cStT, iM = GF.showElapsedTime(startTime=stT), 0
            if sMth == self.dITp['sMthRF']:     # random forest classifier
                d2Par, iM = self.dITp['d2Par_RF'], 15
                cClf = RndForestClf(self.inpD, self.D, d2Par, sKPar=sKPar)
            elif sMth == self.dITp['sMthMLP']:  # NN MLP classifier
                d2Par, iM = self.dITp['d2Par_NNMLP'], 16
                cClf = NNMLPClf(self.inpD, self.D, d2Par, sKPar=sKPar)
            cClf.ClfPred()
            cClf.printFitQuality()
            GF.updateDict(self.d3ResClf, cDUp=cClf.d2ResClf, cK=cRp)
            GF.updateDict(self.d2CnfMat, cDUp=cClf.dConfMat, cK=cRp)
            if k == 0 and cRp == 0:
                self.adaptDPF(cClf=cClf, sMth=sMth)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM, stTMth=cStT, endTMth=cEndT)

    def doDoubleLoop(self, cTim, stT=None):
        for sMth in self.dITp['lSMth']:
            self.d3ResClf, self.d2CnfMat = {}, {}
            d2Par, nRep = self.dITp['d3Par'][sMth], self.dITp['dNumRep'][sMth]
            for k, sKPar in enumerate(d2Par):
                for cRep in range(nRep):
                    print(GC.S_EQ20, 'Method:', sMth, GC.S_VBAR, 'Parameter',
                          'set:', sKPar, GC.S_VBAR, 'Repetition:', cRep + 1)
                    self.doCRep(sMth, k, sKPar, cRp=cRep, cTim=cTim, stT=stT)
            if nRep > 0:
                self.saveData(GF.iniPdDfr(d2Par), pF=self.dPF['OutParClf'])
                self.d2MnSEMResClf = GF.calcMnSEMFromD3Val(self.d3ResClf)
                self.dMnSEMCnfMat = GF.calcMnSEMFromD2Dfr(self.d2CnfMat)
                self.saveData(self.d2MnSEMResClf, pF=self.dPF['OutDataClf'])
                for sK in self.dMnSEMCnfMat:
                    pFMod = GF.modPF(self.dPF['ConfMat'], sEnd=sK,
                                     sJoin=self.dITp['sUSC'])
                    self.saveData(self.dMnSEMCnfMat[sK], pF=pFMod)
                self.printD2MnSEMResClf(sMth=sMth)

###############################################################################
