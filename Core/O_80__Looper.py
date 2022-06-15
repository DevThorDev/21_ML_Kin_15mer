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
        self.d3ResClf, self.d2MnSEM, self.dPF = {}, {}, {}
        self.dPF['OutParClf'] = None
        self.dPF['OutDataClf'] = None

# --- print methods -----------------------------------------------------------
    def printD2MnSEM(self, sMth):
        if GF.Xist(self.d2MnSEM):
            print(GC.S_DS04, ' Dictionary of results (means and SEMs) for ',
                  'method "', sMth, '":', sep='')
            dfrMnSEM = GF.iniPdDfr(self.d2MnSEM)
            for k in range(len(self.d2MnSEM)//2):
                print(GC.S_NEWL, dfrMnSEM.iloc[:, (k*2):(k*2 + 2)], sep='')

# --- loop methods ------------------------------------------------------------
    def adaptDPF(self, cClf, sMth):
        sFCore = cClf.dITp['sUSC'].join([cClf.dITp['sFOutClf'], sMth])
        sFPar = (cClf.dITp['sUSC'].join([sFCore, self.dITp['sPar']]) +
                 self.dITp['xtCSV'])
        sFData = sFCore + self.dITp['xtCSV']
        sFConfMat = (cClf.dITp['sUSC'].join([cClf.dITp['sFConfMat'], sMth]) +
                     self.dITp['xtCSV'])
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
            if k == 0 and cRp == 0:
                self.adaptDPF(cClf=cClf, sMth=sMth)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM, stTMth=cStT, endTMth=cEndT)

    def doDoubleLoop(self, cTim, stT=None):
        for sMth in self.dITp['lSMth']:
            self.d3ResClf, d2Par = {}, self.dITp['d3Par'][sMth]
            for k, sKPar in enumerate(d2Par):
                for cRep in range(self.dITp['dNumRep'][sMth]):
                    print(GC.S_EQ20, 'Method:', sMth, GC.S_VBAR, 'Parameter',
                          'set:', sKPar, GC.S_VBAR, 'Repetition:', cRep + 1)
                    self.doCRep(sMth, k, sKPar, cRp=cRep, cTim=cTim, stT=stT)
            if self.dITp['dNumRep'][sMth] > 0:
                self.saveData(GF.iniPdDfr(d2Par), pF=self.dPF['OutParClf'])
            self.d2MnSEM = GF.calcMnSEMFromD3Val(self.d3ResClf)
            self.saveData(self.d2MnSEM, pF=self.dPF['OutDataClf'])
            self.printD2MnSEM(sMth=sMth)

###############################################################################
