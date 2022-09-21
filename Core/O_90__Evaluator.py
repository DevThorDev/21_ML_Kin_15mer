# -*- coding: utf-8 -*-
###############################################################################
# --- O_90__Evaluator.py ------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

# -----------------------------------------------------------------------------
class Evaluator(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=90):
        super().__init__(inpDat)
        self.idO = 'O_90'
        self.descO = 'Evaluator'
        self.inpD = inpDat
        self.getDITp(iTp=iTp)
        self.fillFPs()
        self.loadInpData()
        self.iniDDfrRes()
        print('Initiated "Evaluator" base object.')

    # --- method for filling the file paths -----------------------------------
    def fillFPs(self):
        # add the file with unique Nmer sequences
        pFNmer, sNmer = self.dITp['pInpUnqNmer'], self.dITp['sUnqNmer']
        dFI = GF.getIFS(pF=pFNmer, itSCmp={sNmer}, addI=False)
        self.FPs.dPF[sNmer] = GF.joinToPath(pFNmer, dFI[sNmer])
        # add the files with detailed and probability info for the classes
        for sKT in [self.dITp['sDetailed'], self.dITp['sProba']]:
            for cSet in self.dITp['lSetFIDet']:
                cSetF = {sKT} | cSet
                dFI = GF.getIFS(pF=self.dITp['pInpDet'], itSCmp=cSetF)
                for tK, sF in dFI.items():
                    self.FPs.dPF[tK] = GF.joinToPath(self.dITp['pInpDet'], sF)

    # --- method for loading the input data -----------------------------------
    def loadInpData(self, iCS=0, iCG=1):
        self.dDfrCmb, self.serSUnqNmer, sNmer = {}, None, self.dITp['sUnqNmer']
        for tK, pF in self.FPs.dPF.items():
            if tK == sNmer:
                self.serSUnqNmer = self.loadData(pF=self.FPs.dPF[tK], iC=iCS)
                self.serSUnqNmer = self.serSUnqNmer[self.dITp['sCNmer']]
            else:
                cDfr = self.loadData(pF=self.FPs.dPF[tK], iC=iCG)
                self.dDfrCmb[tK] = cDfr.iloc[:, 1:]

    # --- method for initialising the dictionary of result DataFrames ---------
    def iniDDfrRes(self):
        self.d2AllCl = {}
        self.dDfrPredCl = {}

    # --- print methods -------------------------------------------------------
    def printCDfrInp(self, tK):
        print(GC.S_DS04, tK, GC.S_DS04)
        print(self.dDfrCmb[tK], GC.S_NEWL, GC.S_DS80, sep='')

    def printdDfrCmb(self, tK=None):
        if tK is not None:
            self.printCDfrInp(tK=tK)
        else:
            # print all input DataFrames
            for tK in self.dDfrCmb:
                print(GC.S_EQ20, 'All input DataFrames', GC.S_EQ20)
                self.printCDfrInp(tK=tK)

    def printKeyAndDfr(self, cKey, cDfr):
        print(GC.S_DS04, 'Key', cKey, GC.S_DS04)
        print(cDfr, GC.S_NEWL, GC.S_DS80, sep='')

    def printDDfrPredCl(self, tFlt=None):
        if tFlt is None:
            for tFlt, dfrPredCl in self.dDfrPredCl.items():
                self.printKeyAndDfr(cKey=tFlt, cDfr=dfrPredCl)
        else:
            self.printKeyAndDfr(cKey=tFlt, cDfr=self.dDfrPredCl[tFlt])

    # def printDDfrPredCl(self, sMth=None, tFlt=None):
    #     if sMth is None and tFlt is None:
    #         for tK, dfrPredCl in self.dDfrPredCl.items():
    #             self.printKeyAndDfr(cKey=tK, cDfr=dfrPredCl)
    #     elif sMth is None and tFlt is not None:
    #         for tK, dfrPredCl in self.dDfrPredCl.items():
    #             if tFlt == tK[1]:
    #                 self.printKeyAndDfr(cKey=tK, cDfr=dfrPredCl)
    #     elif sMth is not None and tFlt is None:
    #         for tK, dfrPredCl in self.dDfrPredCl.items():
    #             if sMth == tK[0]:
    #                 self.printKeyAndDfr(cKey=tK, cDfr=dfrPredCl)
    #     else:
    #         for tK, dfrPredCl in self.dDfrPredCl.items():
    #             if sMth == tK[0] and tFlt == tK[1]:
    #                 self.printKeyAndDfr(cKey=tK, cDfr=dfrPredCl)

    # --- method selecting subsets of the input Dataframe dictionary ----------
    def selSubSetDDfr(self, sMth, itSFlt=None):
        lKSel = list(self.FPs.dPF)
        for tK in self.FPs.dPF:
            # step 1: filter DataFrames that correspond to sMth
            if sMth not in tK:
                lKSel.remove(tK)
            if itSFlt is not None and sMth in tK:
                # step 2: filter DataFrames via list of add. filters (itSFlt)
                for sFlt in itSFlt:
                    if sFlt not in tK:
                        lKSel.remove(tK)
        # step 3: create and return dictionary with DataFrames of subset
        return {tK: self.dDfrCmb[tK] for tK in lKSel if tK in self.dDfrCmb}

    # --- calculation methods -------------------------------------------------
    def calcResSglClf(self, d2, cDfr, sMth, itSFlt=None):
        nCl = cDfr.shape[1]//2
        for sCHd in cDfr.columns[-nCl:]:
            sCl = GF.getSClFromCHdr(sCHdr=sCHd)
            ser0 = cDfr[sCHd].apply(lambda k: 1 - k)
            for sRHd in ser0.index:
                d2[self.d2AllCl[sMth][sCl][0]][sRHd] += ser0.at[sRHd]
            ser1 = cDfr[sCHd]
            for sRHd in ser1.index:
                d2[self.d2AllCl[sMth][sCl][1]][sRHd] += ser1.at[sRHd]

    def calcPredClassRes(self, dMthFlt=None):
        lSHdC = []
        if dMthFlt is not None:
            for sMth, tFlt in dMthFlt.items():
                self.d2AllCl[sMth] = SF.getD2Cl(self.dITp, self.dDfrCmb, sMth)
                lSHdC += list(self.d2AllCl[sMth].values())
        lSHdC = GF.flattenIt(lSHdC)
        d2 = GF.iniD2(itHdL1=lSHdC, itHdL2=self.serSUnqNmer, fillV=0)
        if dMthFlt is not None:
            for sMth, tFlt in dMthFlt.items():
                for tK, cDfr in self.dDfrCmb.items():
                    if (set([sMth]) | set(tFlt)) <= set(tK):
                        self.calcResSglClf(d2, cDfr, sMth=sMth, itSFlt=tFlt)
                        self.dDfrPredCl[tFlt] = GF.iniPdDfr(d2)
            sJ, sXt = self.dITp['sUSC'], self.dIG['xtCSV']
                # sFTmp = sJ.join([sMth, sJ.join(tFlt)]) + sXt
            sFTmp = sJ.join(tFlt) + sXt
            self.saveData(self.dDfrPredCl[tFlt], pF=sFTmp)

###############################################################################