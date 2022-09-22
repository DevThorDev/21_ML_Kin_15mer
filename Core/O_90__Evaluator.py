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
        # add the path to the evaluation of class predictions
        d2PI, dIG, dITp = {}, self.dIG, self.dITp
        d2PI['OutEval'] = {dIG['sPath']: dITp['pOutEval'],
                           dIG['sLFS']: dITp['sEvalClPred'],
                           dIG['sLFC']: '',
                           dIG['sLFJSC']: dITp['sUS02'],
                           dIG['sFXt']: dIG['xtCSV']}
        self.FPs.addFPs(d2PI)
        self.d2PInf = d2PI

    # --- method for loading the input data -----------------------------------
    def loadInpData(self, iCS=0, iCG=1):
        self.dDfrCmb, self.serSUnqNmer, sNmer = {}, None, self.dITp['sUnqNmer']
        for tK, pF in self.FPs.dPF.items():
            if tK == sNmer:
                self.serSUnqNmer = self.loadData(pF=self.FPs.dPF[tK], iC=iCS)
                self.serSUnqNmer = self.serSUnqNmer[self.dITp['sCNmer']]
            elif tK in ['OutEval']:     # result file --> do nothing
                pass
            else:
                cDfr = self.loadData(pF=self.FPs.dPF[tK], iC=iCG)
                self.dDfrCmb[tK] = cDfr.iloc[:, 1:]

    # --- method for initialising the dictionary of result DataFrames ---------
    def iniDDfrRes(self):
        self.d2ClDet = {}
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
    def selSubSetDDfr(self, sMth, itFlt=None):
        lKSel = list(self.FPs.dPF)
        for tK in self.FPs.dPF:
            # step 1: filter DataFrames that correspond to sMth
            if sMth not in tK:
                lKSel.remove(tK)
            if itFlt is not None and sMth in tK:
                # step 2: filter DataFrames via list of add. filters (itFlt)
                for sFlt in itFlt:
                    if sFlt not in tK:
                        lKSel.remove(tK)
        # step 3: create and return dictionary with DataFrames of subset
        return {tK: self.dDfrCmb[tK] for tK in lKSel if tK in self.dDfrCmb}

    # --- calculation methods -------------------------------------------------
    def iniLSHdCol(self, dMthFlt=None):
        lSHdC = []
        if dMthFlt is not None:
            for tFlt, lSMth in dMthFlt.items():
                for sMth in lSMth:
                    dMp = SF.getDMapCl(self.dITp, dDfr=self.dDfrCmb, sMth=sMth)
                    self.d2ClDet[(tFlt, sMth)] = dMp
                    lSHdC += list(self.d2ClDet[(tFlt, sMth)].values())
        return GF.flattenIt(lSHdC)

    def calcResSglClf(self, d1, cDfr, sMth, itFlt=None):
        nCl = cDfr.shape[1]//2
        for sCHd in cDfr.columns[-nCl:]:
            sCl = GF.getSClFromCHdr(sCHdr=sCHd)
            lSKD1 = self.d2ClDet[(itFlt, sMth)][sCl]
            d1[lSKD1[0]] = cDfr[sCHd].apply(lambda k: 1 - k)
            d1[lSKD1[1]] = cDfr[sCHd]

    # def calcResSglClf(self, d2, cDfr, sMth, itFlt=None):
    #     nCl = cDfr.shape[1]//2
    #     for sCHd in cDfr.columns[-nCl:]:
    #         sCl = GF.getSClFromCHdr(sCHdr=sCHd)
    #         ser0 = cDfr[sCHd].apply(lambda k: 1 - k)
    #         for sRHd in ser0.index:
    #             d2[self.d2ClDet[(itFlt, sMth)][sCl][0]][sRHd] += ser0.at[sRHd]
    #         ser1 = cDfr[sCHd]
    #         for sRHd in ser1.index:
    #             d2[self.d2ClDet[(itFlt, sMth)][sCl][1]][sRHd] += ser1.at[sRHd]

    def calcCFlt(self, d1, d2, tF, lSM=[]):
        sKMn, sKPos, sJ = 'OutEval', 'sLFC', self.dITp['sUSC']
        print(GC.S_EQ04, 'Handling filter', tF, GC.S_EQ04)
        tFXt = tuple([self.dITp['sDetailed']] + list(tF))
        for sM in lSM:
            print(GC.S_DS04, 'Handling method', sM, GC.S_DS04)
            for tK, cDfr in self.dDfrCmb.items():
                if (set([sM]) | set(tFXt)) <= set(tK):
                    self.calcResSglClf(d1, cDfr, sMth=sM, itFlt=tF)
                    # self.calcResSglClf(d2, cDfr, sMth=sM, itFlt=tF)
                    dfrPredCl = GF.iniPdDfr(d1, lSNmR=self.serSUnqNmer)
                    self.dDfrPredCl[tF] = dfrPredCl
                    # self.dDfrPredCl[tF] = GF.iniPdDfr(d2)
        self.FPs.modFP(d2PI=self.d2PInf, kMn=sKMn, kPos=sKPos, cS=sJ.join(tF))
        self.saveData(self.dDfrPredCl[tF], pF=self.FPs.dPF[sKMn])

    def calcPredClassRes(self, dMthFlt=None):
        lSHdC = self.iniLSHdCol(dMthFlt=dMthFlt)
        d1 = {sC: None for sC in lSHdC}
        d2 = GF.iniD2(itHdL1=lSHdC, itHdL2=self.serSUnqNmer, fillV=0)
        if dMthFlt is not None:
            for tFlt, lSMth in dMthFlt.items():
                self.calcCFlt(d1, d2, tF=tFlt, lSM=lSMth)

###############################################################################