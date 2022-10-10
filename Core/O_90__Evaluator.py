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
        self.iniResObj()
        print('Initiated "Evaluator" base object.')

    # --- method for filling the file paths -----------------------------------
    def fillFPs(self):
        # add the file with unique Nmer sequences
        pFNmer, sNmer = self.dITp['pInpUnqNmer'], self.dITp['sUnqNmer']
        dFI = GF.getIFS(pF=pFNmer, itSCmp={sNmer}, addI=False)
        self.FPs.dPF[sNmer] = GF.joinToPath(pFNmer, dFI[sNmer])
        # add the file with input data (X_Classes)
        pFInpD, sInpD = self.dITp['pInpData'], self.dITp['sInpData']
        dFI = GF.getIFS(pF=pFInpD, itSCmp={sInpD}, addI=False)
        self.FPs.dPF[sInpD] = GF.joinToPath(pFInpD, dFI[sInpD])
        # add the files with detailed and probability info for the classes
        for sKT in [self.dITp['sDetailed'], self.dITp['sProba']]:
            for cSet in self.dITp['lSetFIDet']:
                cSetF = {sKT} | cSet
                dFI = GF.getIFS(pF=self.dITp['pInpDet'], itSCmp=cSetF)
                for tK, sF in dFI.items():
                    self.FPs.dPF[tK] = GF.joinToPath(self.dITp['pInpDet'], sF)
        # initialise filter and classifier methods string
        self.sFltMth, self.lAllMth = '', []
        # add the path to the evaluation of class predictions
        d2PI, dIG, dITp = {}, self.dIG, self.dITp
        d2PI['OutEval'] = {dIG['sPath']: dITp['pOutEval'],
                           dIG['sLFS']: dITp['sEvalClPred'],
                           dIG['sLFC']: self.sFltMth,
                           dIG['sLFJSC']: dITp['sUS02'],
                           dIG['sFXt']: dIG['xtCSV']}
        self.FPs.addFPs(d2PI)
        self.d2PInf = d2PI

    # --- method for loading the input data -----------------------------------
    def loadInpData(self, iCS=0, iCG=1):
        self.dDfrCmb, self.serSUnqNmer = {}, None
        for tK, pF in self.FPs.dPF.items():
            if tK == self.dITp['sUnqNmer']:
                self.serSUnqNmer = self.loadData(pF=self.FPs.dPF[tK], iC=iCS)
                self.serSUnqNmer = self.serSUnqNmer[self.dITp['sCNmer']]
            elif tK == self.dITp['sInpData']:
                self.serXCl = self.loadData(pF=self.FPs.dPF[tK], iC=iCG)
                self.serXCl = self.serXCl[self.dITp['sEffFam']]
            elif tK in ['OutEval']:     # result file --> do nothing
                pass
            else:
                cDfr = self.loadData(pF=self.FPs.dPF[tK], iC=iCG)
                self.dDfrCmb[tK] = cDfr.iloc[:, 1:]

    # --- method for initialising the dictionary of result DataFrames ---------
    def iniResObj(self):
        self.d2ClDet = {}
        self.dfrTrueCl, self.dfrPredCl = None, None
        self.dCmpTP = {'RelMax': None, 'AbsMax': None, 'PDiff': None}

    # --- print methods -------------------------------------------------------
    def printDfrInpFlt(self, tKSub):
        for tK in self.dDfrCmb:
            if set(tKSub) <= set(tK):
                print(GC.S_DS04, 'Input DataFrame with key', tK, GC.S_DS04)
                print(self.dDfrCmb[tK], GC.S_NEWL, GC.S_DS80, sep='')

    def printD2ClDet(self):
        for sMth, dMap in self.d2ClDet.items():
            print(GC.S_DS08, 'Method:', sMth, GC.S_DS08)
            for sCl, lSCHd in dMap.items():
                print(GC.S_DS04, sCl, '| Header list:', lSCHd, GC.S_DS04)

    def printDDfrCmb(self, tK=None):
        if tK is not None:
            self.printDfrInpFlt(tKSub=tK)
        else:
            # print all input DataFrames
            for tK in self.dDfrCmb:
                print(GC.S_EQ20, 'All input DataFrames', GC.S_EQ20)
                self.printDfrInpFlt(tKSub=tK)

    def printDfrCl(self, tpCl=GC.S_PRED_CL):
        dfrCl = self.dfrPredCl
        if tpCl == self.dITp['sTrueCl']:
            dfrCl = self.dfrTrueCl
        print(dfrCl, GC.S_NEWL, GC.S_DS80, sep='')

    def printDfrTrueCl(self):
        self.printDfrCl(tpCl=self.dITp['sTrueCl'])

    def printDfrPredCl(self):
        self.printDfrCl(tpCl=self.dITp['sPredCl'])

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
        if self.dITp['doEvaluation'] and dMthFlt is not None:
            for lSMth in dMthFlt.values():
                for sMth in lSMth:
                    dMp = SF.getDMapCl(self.dITp, dDfr=self.dDfrCmb, sMth=sMth)
                    self.d2ClDet[sMth] = dMp
                    lSHdC += list(self.d2ClDet[sMth].values())
        return GF.flattenIt(lSHdC)

    def calcResSglClf(self, d1, cDfr, sMth):
        nCl = cDfr.shape[1]//2
        dfrTrueCl = cDfr.iloc[:, (-2*nCl):(-nCl)].sort_index(axis=0)
        if self.dfrTrueCl is None:
            self.dfrTrueCl = dfrTrueCl
        else:
            assert self.dfrTrueCl.equals(dfrTrueCl)
        SF.fillDPrimaryRes(d1, self.d2ClDet, cDfr=cDfr, nCl=nCl, sMth=sMth)

    def loopOverMethods(self, d1, tFXt, lSM=[]):
        for sM in lSM:
            print(GC.S_DS04, 'Checking for results of method', sM, GC.S_DS04)
            for tK, cDfr in self.dDfrCmb.items():
                if (len(tK) > 0) and ((set([sM]) | set(tFXt)) <= set(tK)):
                    if tK[0] == 1 and tK[1] == self.dITp['sDetailed']:
                        print(GC.S_DS04, 'Handling method', sM, GC.S_DS04)
                        self.lAllMth.append(sM)
                    self.calcResSglClf(d1, cDfr, sMth=sM)

    def calcSummaryVals(self):
        pass

    def compareTruePred(self, d1, lSM=[]):
        self.dMapClT = GF.getDSClFromCHdr(itCol=self.dfrTrueCl.columns)
        lDfr = [self.dfrTrueCl, self.dfrPredCl]
        self.dCmpTP = SF.compTruePredLoop(self.d2ClDet, self.dMapClT,
                                          lDfr=lDfr, lSM=lSM)

    def calcCFlt(self, d1, tF, lSM=[]):
        sKMn, sKPos, sJ = 'OutEval', 'sLFC', self.dITp['sUSC']
        print(GC.S_EQ04, 'Handling filter', tF, GC.S_EQ04)
        tFXt = tuple([self.dITp['sDetailed']] + list(tF))
        self.loopOverMethods(d1, tFXt=tFXt, lSM=lSM)
        self.dfrPredCl = GF.iniPdDfr(d1, lSNmR=self.serSUnqNmer)
        lO = [self.serXCl, self.dfrTrueCl, self.dfrPredCl]
        self.dfrPredCl = GF.concLOAx1(lObj=lO, verifInt=True, srtDfr=True)
        self.sFltMth = self.dITp['sUS02'].join([sJ.join(tF),
                                                sJ.join(self.lAllMth)])
        self.FPs.modFP(d2PI=self.d2PInf, kMn=sKMn, kPos=sKPos, cS=self.sFltMth)
        self.saveData(self.dfrPredCl.convert_dtypes(), pF=self.FPs.dPF[sKMn])
        self.compareTruePred(d1, lSM=lSM)

    def calcPredClassRes(self, dMthFlt=None):
        lSHdPred = self.iniLSHdCol(dMthFlt=dMthFlt)
        # self.printD2ClDet()
        d1 = {sC: None for sC in lSHdPred}
        if self.dITp['doEvaluation'] and dMthFlt is not None:
            for tFlt, lSMth in dMthFlt.items():
                self.calcCFlt(d1, tF=tFlt, lSM=lSMth)

###############################################################################