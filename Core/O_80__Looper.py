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
    def __init__(self, inpDat, iTp=80, lITpUpd=[]):
        super().__init__(inpDat)
        self.idO = 'O_80'
        self.descO = 'Looper'
        self.inpD = inpDat
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.d3ResClf = {}
        print('Initiated "Looper" base object.')

# --- loop methods ------------------------------------------------------------
    def getDPF(self, cClf, sMth):
        self.dPF, pOut = {}, cClf.dITp['pOutClf']
        for sTp in self.dITp['lSTp']:
            sF = (cClf.dITp['sUSC'].join([cClf.dITp['sFOutClf'], sMth, sTp,
                                          cClf.sKPar]) + self.dITp['xtCSV'])
            self.dPF['OutDataClf' + sTp] = GF.joinToPath(pOut, sF)

    def doCRep(self, sMth, sKPar, cRp, cTim, stT=None):
        if sMth in self.dITp['lSMth']:
            cStT, iM = GF.showElapsedTime(startTime=stT), 0
            print(GC.S_EQ20, 'Parameter set', sKPar)
            if sMth == self.dITp['sMthRF']:     # random forest classifier
                d2Par, iM = self.dITp['d2Par_RF'], 14
                cClf = RndForestClf(self.inpD, d2Par, sKPar=sKPar)
            elif sMth == self.dITp['sMthMLP']:  # NN MLP classifier
                d2Par, iM = self.dITp['d2Par_NNMLP'], 15
                cClf = NNMLPClf(self.inpD, d2Par, sKPar=sKPar)
            cClf.ClfPred()
            cClf.printFitQuality()
            self.d3ResClf[cRp] = cClf.d2ResClf
            self.getDPF(cClf=cClf, sMth=sMth)
            cEndT = GF.showElapsedTime(startTime=stT)
            cTim.updateTimes(iMth=iM, stTMth=cStT, endTMth=cEndT)

    def doDoubleLoop(self, cTim, stT=None):
        for sMth in self.dITp['lSMth']:
            for sKPar in self.dITp['d3Par'][sMth]:
                for cRep in range(self.dITp['dNumRep'][sMth]):
                    self.doCRep(sMth, sKPar, cRp=cRep, cTim=cTim, stT=stT)
                d3MnSD = GF.calcMnSDFromD3Val(self.d3ResClf)
                for sTp in self.dITp['lSTp']:
                    if sTp in d3MnSD:
                        pFCTp = self.dPF['OutDataClf' + sTp]
                        self.saveData(d3MnSD[sTp], pF=pFCTp)

###############################################################################
