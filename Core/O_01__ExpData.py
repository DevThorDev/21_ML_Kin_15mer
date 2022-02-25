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
        self.readExpData()
        self.fillDPFRes()
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

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPFRes(self):
        sPFCmbS = GF.joinToPath(self.pDirRes, self.dITp['sFResCombS'])
        sPFCmbM = GF.joinToPath(self.pDirRes, self.dITp['sFResCombM'])
        sPFCmbL = GF.joinToPath(self.pDirRes, self.dITp['sFResCombL'])
        dPFCmb = {GC.S_SHORT: sPFCmbS, GC.S_MED: sPFCmbM, GC.S_LONG: sPFCmbL}
        sPFI15mer = GF.joinToPath(self.pDirRes, self.dITp['sFResI15mer'])
        self.dPFRes = {GC.S_COMBINED: dPFCmb,
                       GC.S_INFO_MER: sPFI15mer}

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

    # --- methods for combining experimental data -----------------------------
    def combine(self, lSCK, lSC15m, sMd=GC.S_SHORT):
        dEffTarg = SF.createDEffTarg(self.dITp, self.dfrKin, self.dfr15mer,
                                     lCDfr15m=lSC15m, sMd=sMd)
        GF.printSizeDDDfr(dDDfr=dEffTarg, modeF=True)
        dfrComb = GF.dDDfrToDfr(dDDfr=dEffTarg, lSColL=lSCK, lSColR=lSC15m)
        self.saveDfr(dfrComb, pF=self.dPFRes[GC.S_COMBINED][sMd], dropDup=True,
                     saveAnyway=True)

    def combineS(self):
        lSCK = [self.dITp['sEffCode'], self.dITp['sTargCode']]
        lSC15m = [s for s in self.dfr15mer.columns
                  if s not in self.dITp['lCXclDfr15merS']]
        self.combine(lSCK, lSC15m, sMd=GC.S_SHORT)

    def combineM(self):
        lSCK = [self.dITp['sEffCode'], self.dITp['sTargCode']]
        lSC15m = [s for s in self.dfr15mer.columns
                  if s not in self.dITp['lCXclDfr15merM']]
        self.combine(lSCK, lSC15m, sMd=GC.S_MED)

    def combineL(self):
        lSCK, lSC15m = list(self.dfrKin.columns), list(self.dfr15mer.columns)
        self.combine(lSCK, lSC15m, sMd=GC.S_LONG)

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

    # --- method calling sub-methods that process and combine exp. data -------
    def procExpData(self, nDigDsp=GC.R04):
        self.filterRawDataKin(nDig=nDigDsp)
        self.procColKin()
        self.filterRawData15mer(nDig=nDigDsp)
        self.procCol15mer()
        self.saveProcInpDfrs()
        self.combineInpDfr()

    # --- methods extracting info from processed/combined data ----------------
    def getInfoKin15mer(self, sMd=GC.S_SHORT):
        dE15m, dfrCombS = {}, self.loadDfr(pF=self.dPFRes[GC.S_COMBINED][sMd])
        if dfrCombS is not None:
            for dRec in dfrCombS.to_dict('records'):
                # considering all effectors pooled
                GF.addToDictL(dE15m, self.dITp['sAnyEff'],
                              dRec[self.dITp['sC15mer']], lUniqEl=False)
                # considering the different effectors separately
                GF.addToDictL(dE15m, dRec[self.dITp['sEffCode']],
                              dRec[self.dITp['sC15mer']], lUniqEl=False)
        print('Length of dE15m:', len(dE15m))
        dNMer, iC15m = {}, self.dITp['iCent15mer']
        for k in range(iC15m + 1):
            for sEff, lS15mer in dE15m.items():
                for s15mer in lS15mer:
                    sCent15mer = s15mer[(iC15m - k):(iC15m + k + 1)]
                    GF.addToDictDNum(dNMer, sEff, sCent15mer)
        dfrINmer = GF.dDNumToDfr(dNMer, lSCol=self.dITp['lSCDfrNMer'])
        dfrINmer.sort_values(by=self.dITp['lSortCDfrNMer'],
                             ascending=self.dITp['lSortDirAscDfrNMer'],
                             inplace=True, ignore_index=True)
        self.saveDfr(dfrINmer, pF=self.dPFRes[GC.S_INFO_MER], dropDup=False,
                     saveAnyway=True)

###############################################################################
