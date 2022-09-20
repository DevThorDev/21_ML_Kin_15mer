# -*- coding: utf-8 -*-
###############################################################################
# --- O_90__Evaluator.py ------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

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
        print('Initiated "Evaluator" base object.')

    # --- methods for filling the file paths ----------------------------------
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

    def loadInpData(self, iC=0):
        self.dDfrInp = {}
        for tK, pF in self.FPs.dPF.items():
            self.dDfrInp[tK] = self.loadData(pF=self.FPs.dPF[tK], iC=iC)

    # --- print methods -------------------------------------------------------
    def printCDfrInp(self, tK):
        print(GC.S_DS04, tK, GC.S_DS04)
        print(self.dDfrInp[tK], GC.S_NEWL, GC.S_DS80, sep='')
    
    def printDDfrInp(self, tK=None):
        if tK is not None:
            self.printCDfrInp(tK=tK)
        else:
            # print all input DataFrames
            for tK in self.dDfrInp:
                print(GC.S_EQ20, 'All input DataFrames', GC.S_EQ20)
                self.printCDfrInp(tK=tK)

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
        return {tK: self.dDfrInp[tK] for tK in lKSel if tK in self.dDfrInp}
    
    # --- method extracting list of classes from input Dataframes -------------
    def extrLCl(self, dDfr=None):
        lCl = []
        if dDfr is None:
            dDfr = self.dDfrInp
    
    # --- calculation methods -------------------------------------------------
    def calcResSglClf(self, sMth, itSFlt=None):
        pass

###############################################################################