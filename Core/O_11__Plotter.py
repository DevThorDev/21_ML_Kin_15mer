# -*- coding: utf-8 -*-
###############################################################################
# --- O_11__Plotter.py --------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_02__PltFunctions as PF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

# -----------------------------------------------------------------------------
class Plotter(BaseClass):
# --- initialisation of the class ---------------------------------------------
    def __init__(self, inpDat, iTp=11, lITpUpd=[]):
        super().__init__(inpDat)
        self.idO = 'O_11'
        self.descO = 'Plotter'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        print('Initiated "Plotter" base object.')

# --- path modification methods -----------------------------------------------
    def defPFDatPlt(self):
        self.defPFDat()
        if self.dIG['isTest']:
            self.dITp['pRelPltF'] = self.dITp['pRelPltF_T']
            self.dPlt2['dHistPlt'] = self.dPlt2['dHistPlt_T']

# --- methods for plotting experimental and "mixed" data histograms -----------
    def plotCHistDt(self, vDt, sGT, sMDPD, sHist=GC.S_PLOT_HIST):
        sMid = self.d2FI[sGT][sMDPD]
        pFPlt = SF.addToStructDict(self.dITp, self.d4PFO, sMid, sL0=sHist,
                                   sL1=self.sIDF, sL2=sGT, sL3=sMDPD, iEnd=-2,
                                   sDir=self.dITp['dir51'],
                                   sXt=self.dITp['sFXtPDF'])
        sT = GF.joinS([sHist, self.sIDF, sGT, sMDPD], sJoin=GC.S_SPACE)
        PF.plotHistVXDt(self.dITp, self.dPlt1, vDt, pF=pFPlt, sTtl=sT)

    def plotHistAllX(self, sGT, sMDPD):
        print(GC.S_DT4, ' Plotting histograms of experimental  data (', sGT,
              ', ', sMDPD, ') ', GC.S_DT4, sep='')
        vAllXD = self.d3Dfr[sGT][sMDPD][self.dITp['sXD']].stack().to_numpy()
        self.plotCHistDt(vAllXD, sGT, sMDPD, self.dPlt1['sHistXDt'])
        if self.dPlt1['doHistMnSDDt']:
            for sMnSD, sHistMnSD in zip(self.dITp['lSMnSD'],
                                        self.dPlt1['lSHistXDtMnSD']):
                dfrMnSD = self.getDfrMnSD(sGT, sMDPD, sMnSD=sMnSD)
                self.plotCHistDt(dfrMnSD.stack().to_numpy(), sGT, sMDPD,
                                 sHistMnSD)

    def plotHistAllXAllGT(self):
        if self.dPlt1['doHist']:
            for sGT, dL0 in self.d3Dfr.items():
                for sMDPD, _ in dL0.items():
                    self.calcDfrMnSD(self.getDfrDt(sGT, sMDPD)[0], sGT, sMDPD)
                    self.plotHistAllX(sGT, sMDPD)

    def plotHistMix(self, sGT, sMDPD):
        if self.dPlt1['doHistMix']:
            print(GC.S_DT4, ' Plotting histograms of mixed data (', sGT, ', ',
                  sMDPD, ', ', self.sIDF, ') ', GC.S_DT4, sep='')
            vMixD = self.dDfrMix[sMDPD].stack().to_numpy()
            self.plotCHistDt(vMixD, sGT, sMDPD, self.dPlt1['sHistMix'])
            if self.dPlt1['doHistMixMnSDDt']:
                for sMnSD, sHistMnSD in zip(self.dITp['lSMnSD'],
                                            self.dPlt1['lSHistMixDtMnSD']):
                    vMnSDD = self.d2DfrMixMnSD[sMnSD][sMDPD].stack().to_numpy()
                    self.plotCHistDt(vMnSDD, sGT, sMDPD, sHistMnSD)

# --- methods for plotting binary operations histograms -----------------------
    def barPlot4BO(self, dPltH, sGT, sBO, k=0):
        print(GC.S_DT4, 'Plotting histograms of', sBO, 'values for', sGT,
              GC.S_DT4)
        sMid, sL0 = self.d2FI[sGT][self.dITp['sMetD']], self.dPlt2['sHist']
        pFPlt = SF.addToStructDict(self.dITp, self.d4PFO, sMid, sL0=sL0,
                                   sL1=sBO, sL2=sGT, sL3=self.dITp['sExpD'],
                                   iStart=1, iEnd=-2, sDir=self.dITp['dir61'],
                                   sXt=self.dITp['sFXtPDF'])
        serX = self.d2DfrBH[sGT][sBO][k][self.dITp['sMidBnd']]
        dfrY = self.d2DfrBH[sGT][sBO][k][self.lIDsAll]
        sT = GF.joinS([sBO, self.dITp['dSNmGTPlt'][sGT]], sJoin=GC.S_SPACE)
        PF.barPlotVBOs(self.dPlt2, dPltH, self.dITp['dMix4Plt']['dSLegBOs'],
                       serX, dfrY, pF=pFPlt, sTtl=sT)

    def plotHistBOs(self):
        if (self.dPlt2['doHist'] or self.dPlt3['doPlot'] or
            self.dPlt4['doPlot']):
            for sGT in self.dITp['lSGT']:
                self.doBinOps(sGT)
                self.compHist(sGT)
                self.procMnSDDfrBH(sGT)
                if self.dPlt2['doHist']:
                    for sBO in self.dITp['lBOsUsed']:
                        self.barPlot4BO(self.dPlt2['dHistPlt'][sBO], sGT, sBO)

# --- methods for plotting comparisons between BOs ----------------------------
    def plotCompBOs4ID(self, sID, sGT, sZL, sCum, sCX, sCY):
        print(GC.S_DT4, ' Plotting comparisons between binary operations for ',
              sGT, ', ', sCum, ' and ', sID, ' (', sZL, ') ', GC.S_DT4, sep='')
        sMid, sL0 = self.d2FI[sGT][self.dITp['sMetD']], self.dPlt3['sPlot']
        sL1, sT, sGTT, sIDL = GF.joinS([sZL, sID, sCum]), None, '', ''
        pFPlt = SF.addToStructDict(self.dITp, self.d4PFO, sMid, sL0=sL0,
                                   sL1=sL1, sL2=sGT, sL3=self.dITp['sExpD'],
                                   iStart=1, iEnd=-2, sDir=self.dITp['dir71'],
                                   sXt=self.dITp['sFXtPDF'])
        if self.dPlt3['tPltTtl'][0]:
            sGTT = self.dITp['dSNmGTPlt'][sGT]
            if self.dPlt3['tPltTtl'][1]:
                sIDL = '(' + self.dITp['dMix4Plt']['dSLegBH'][sID] + ')'
        else:
            if self.dPlt3['tPltTtl'][1]:
                sIDL = self.dITp['dMix4Plt']['dSLegBH'][sID]
        if self.dPlt3['tPltTtl'][0] or self.dPlt3['tPltTtl'][1]:
            sT = GF.joinS([sGTT, sIDL], sJoin=GC.S_SPACE)
        PF.plotXYCmpBOs(self.dITp, self.dPlt3, self.d2DfrBH, sID, sGT, sZL,
                        sCX, sCY, sCA=self.dITp['sMidBnd'], pF=pFPlt, sTtl=sT)

    def barPlotCompBOs4ID(self, sID, sGT, sCum, sCIDMx, sCID0):
        print(GC.S_DT4, ' Bar plot comparison between binary operations for ',
              sGT, ', ', sCum, ' and ', sID, GC.S_DT4, sep='')
        sMid, sL0 = self.d2FI[sGT][self.dITp['sMetD']], self.dPlt4['sPlot']
        sL1, sT = GF.joinS([sID, sCum]), None
        pFPlt = SF.addToStructDict(self.dITp, self.d4PFO, sMid, sL0=sL0,
                                   sL1=sL1, sL2=sGT, sL3=self.dITp['sExpD'],
                                   iStart=1, iEnd=-2, sDir=self.dITp['dir72'],
                                   sXt=self.dITp['sFXtPDF'])
        if self.dPlt4['pltTtl']:
            sT = self.dITp['dSNmGTPlt'][sGT]
        PF.plotBarCompBOs(self.dITp, self.dPlt4, self.d2DfrBH, sID, sGT, sCID0,
                          sCIDMx, sCSl=self.dITp['sMidBnd'], pF=pFPlt, sTtl=sT)

    def plotCompBOs(self):
        if self.dPlt3['doPlot'] or self.dPlt4['doPlot']:
            for sGT in self.dITp['lSGT']:
                for sCum in self.dITp['lSCum']:
                    sCY = GF.joinS([self.dITp['s0'], self.dITp['sRel'], sCum])
                    for sID in self.dITp['dMix']:
                        sCX = GF.joinS([sID, self.dITp['sRel'], sCum])
                        if self.dPlt3['doPlot']:
                            for sZL in self.dPlt3['dXLim']:
                                self.plotCompBOs4ID(sID, sGT, sZL, sCum, sCX,
                                                    sCY)
                        if self.dPlt4['doPlot']:
                            self.barPlotCompBOs4ID(sID, sGT, sCum, sCX, sCY)


###############################################################################
