# -*- coding: utf-8 -*-
###############################################################################
# --- O_01__ExpData.py --------------------------------------------------------
###############################################################################
import copy
#
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

# -----------------------------------------------------------------------------
class ExpData(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=1, lITpUpd=[]):
        super().__init__(inpDat)
        self.idO = 'O_01'
        self.descO = 'Raw input data'
        self.dITp = copy.deepcopy(self.dIG[0])  # type of base class = 0
        for iTpU in lITpUpd + [iTp]:            # updated with types in list
            self.dITp.update(self.dIG[iTpU])
        self.dPFRes = {GC.S_COMBINED: {GC.S_SHORT: None,
                                       GC.S_MED: None,
                                       GC.S_LONG: None}}
        self.readExpData()
        print('Initiated "ExpData" base object.')

    # --- methods for reading experimental data -------------------------------
    def readExpData(self):
        if self.dIG['isTest']:
            self.dfrKin = self.loadDfr(pF=self.dITp['pFRawInpKin_T'])
            self.dfr15mer = self.loadDfr(pF=self.dITp['pFRawInp15mer_T'])
            self.pDirProcInp = self.dITp['pDirProcInp_T']
            self.pDirRes = self.dITp['pDirRes_T']
        else:
            self.dfrKin = self.loadDfr(pF=self.dITp['pFRawInpKin'])
            self.dfr15mer = self.loadDfr(pF=self.dITp['pFRawInp15mer'])
            self.pDirProcInp = self.dITp['pDirProcInp']
            self.pDirRes = self.dITp['pDirRes']
        self.lDfrInp = [self.dfrKin, self.dfr15mer]

    # --- methods for saving processed input data -----------------------------
    def saveProcInpDfrs(self):
        self.lDfrInp = [self.dfrKin, self.dfr15mer]
        for cDfr, sFPI in zip(self.lDfrInp, self.dITp['lSFProcInp']):
            self.saveDfr(cDfr, pF=GF.joinToPath(self.pDirProcInp, sFPI),
                         dropDup=True, saveAnyway=False)

    # --- methods for processing experimental data ----------------------------
    def filterRawDataKin(self, nDig=GC.R04):
        cDfr, lColF = self.dfrKin, self.dITp['lSCKinF']
        # remove lines with "NaN" or "NULL" in all columns (excl. cColExcl)
        nR0 = cDfr.shape[0]
        cDfr.dropna(axis=0, subset=lColF, inplace=True)
        nR1 = cDfr.shape[0]
        for sC in lColF:
            self.dfrKin = cDfr[cDfr[sC] != self.dITp['sNULL']]
        self.dfrKin.reset_index(drop=True, inplace=True)
        nR2 = self.dfrKin.shape[0]
        print(GC.S_DS80, GC.S_NEWL, 'Rows with "NaN" in columns ', lColF, ': ',
              nR0-nR1, ' of ', nR0, '\t(', round((nR0-nR1)/nR0*100., nDig),
              '%)', GC.S_NEWL, 'Rows with "', self.dITp['sNULL'],
              '" in columns ', lColF, ': ', nR1-nR2, ' of ', nR0, '\t(',
              round((nR1-nR2)/nR0*100., nDig), '%)', GC.S_NEWL, GC.S_ARR_LR,
              ' Remaining rows: ', nR2, GC.S_NEWL, GC.S_DS80, sep='')

    def procColKin(self):
        l = [s for s in self.dfrKin[self.dITp['sPSite']]]
        for k, s in enumerate(l):
            if type(s) == str and len(s) > 0:
                try:
                    l[k] = int(s[1:])
                except:
                    l[k] = s[1:]
        self.dfrKin[self.dITp['sPosPSite']] = l

    def filterRawData15mer(self, nDig=GC.R04):
        cDfr, cCol = self.dfr15mer, self.dITp['sC15mer']
        nR0, sLenS = cDfr.shape[0], self.dITp['sLenSnip']
        # remove lines with "NaN" in the 15mer-column
        cDfr.dropna(axis=0, subset=[cCol], inplace=True)
        cDfr.reset_index(drop=True, inplace=True)
        nR1 = cDfr.shape[0]
        # remove lines with the wrong length of the 15mer
        cDfr[sLenS] = [len(cDfr.at[k, cCol]) for k in range(cDfr.shape[0])]
        self.dfr15mer = cDfr[cDfr[sLenS] == self.dITp['lenSDef']]
        self.dfr15mer.reset_index(drop=True, inplace=True)
        self.dfr15mer = self.dfr15mer.drop(sLenS, axis=1)
        nR2 = self.dfr15mer.shape[0]
        print(GC.S_DS80, GC.S_NEWL, 'Rows with "NaN" in column "', cCol, '": ',
              nR0-nR1, ' of ', nR0, '\t\t(', round((nR0-nR1)/nR0*100., nDig),
              '%)', GC.S_NEWL, 'Rows with length of snippet not ',
              self.dITp['lenSDef'], ': ', nR1-nR2, ' of ', nR0, '\t(',
              round((nR1-nR2)/nR0*100., nDig), '%)', GC.S_NEWL, GC.S_ARR_LR,
              ' Remaining rows: ', nR2, GC.S_NEWL, GC.S_DS80, sep='')

    def procCol15mer(self):
        l = [s for s in self.dfr15mer[self.dITp['sCCode']]]
        for k, s in enumerate(l):
            if type(s) == str and len(s) > 0 and GC.S_DOT in s:
                l[k] = s.split(GC.S_DOT)[0]
        self.dfr15mer[self.dITp['sCodeTrunc']] = l

    def combine(self, lSCK, lSC15m, sFResComb, sMd=GC.S_SHORT):
        dEffTarg = SF.createDEffTarg(self.dITp, self.dfrKin, self.dfr15mer,
                                     lCDfr15m=lSC15m, sMd=sMd)
        GF.printSizeDDDfr(dDDfr=dEffTarg, modeF=True)
        dfrComb = GF.dDDfrToDfr(dDDfr=dEffTarg, lSColL=lSCK, lSColR=lSC15m)
        pFResComb = GF.joinToPath(self.pDirRes, sFResComb)
        self.dPFRes[GC.S_COMBINED][sMd] = pFResComb
        self.saveDfr(dfrComb, pF=pFResComb, dropDup=True, saveAnyway=True)

    def combineS(self):
        lSCK = [self.dITp['sEffCode'], self.dITp['sTargCode']]
        lSC15m = [s for s in self.dfr15mer.columns
                  if s not in self.dITp['lCXclDfr15merS']]
        self.combine(lSCK, lSC15m, self.dITp['sFResCombS'], sMd=GC.S_SHORT)

    def combineM(self):
        lSCK = [self.dITp['sEffCode'], self.dITp['sTargCode']]
        lSC15m = [s for s in self.dfr15mer.columns
                  if s not in self.dITp['lCXclDfr15merM']]
        self.combine(lSCK, lSC15m, self.dITp['sFResCombM'], sMd=GC.S_MED)

    def combineL(self):
        lSCK, lSC15m = list(self.dfrKin.columns), list(self.dfr15mer.columns)
        self.combine(lSCK, lSC15m, self.dITp['sFResCombL'], sMd=GC.S_LONG)

    def combineInpDfr(self):
        if self.dITp['dBDoDfrRes'][GC.S_SHORT]:
            print('Creating short combined result...')
            self.combineS()
            print('Created short combined result.')
        if self.dITp['dBDoDfrRes'][GC.S_MED]:
            print('Creating medium combined result...')
            self.combineM()
            print('Created medium combined result.')
        if self.dITp['dBDoDfrRes'][GC.S_LONG]:
            print('Creating long combined result...')
            self.combineL()
            print('Created long combined result.')

    def getInfoKin15mer(self, sMd=GC.S_SHORT):
        dfrCombS = self.loadDfr(pF=self.dPFRes[GC.S_COMBINED][sMd])

    def procExpData(self, nDigDsp=GC.R04):
        self.filterRawDataKin(nDig=nDigDsp)
        self.procColKin()
        self.filterRawData15mer(nDig=nDigDsp)
        self.procCol15mer()
        self.saveProcInpDfrs()
        self.combineInpDfr()

###############################################################################
