# -*- coding: utf-8 -*-
###############################################################################
# --- O_01__ExpData.py --------------------------------------------------------
###############################################################################
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
        self.descO = 'Experimental data'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.getPDir()
        self.readExpData()
        self.fillFPsExp()
        self.fillDTpDfrExp()
        print('Initiated "ExpData" base object.')

    # --- methods for loading experimental data -------------------------------
    def readExpData(self):
        if self.dIG['isTest']:
            self.dfrKin = self.loadData(pF=self.dITp['pFRawInpKin_T'])
            self.dfrNmer = self.loadData(pF=self.dITp['pFRawInpNmer_T'])
        else:
            self.dfrKin = self.loadData(pF=self.dITp['pFRawInpKin'])
            self.dfrNmer = self.loadData(pF=self.dITp['pFRawInpNmer'])
        self.lDfrInp = [self.dfrKin, self.dfrNmer]
        lC = list(self.dfrKin.columns) + list(self.dfrNmer.columns)
        if self.dITp['genInfoGen']:
            self.dfrResIGen = GF.iniPdDfr(lSNmC=lC, lSNmR=self.dITp['lRResIG'])

    # --- methods for filling the file paths ----------------------------------
    def fillFPsExp(self):
        d2PI, dIG, dITp, sFlFE = {}, self.dIG, self.dITp, self.dITp['sFull']
        for sTp in [GC.S_CAP_XS, GC.S_CAP_S, GC.S_CAP_M, GC.S_CAP_L]:
            d2PI['sCombined' + sTp] = {dIG['sPath']: self.pDirResComb,
                                       dIG['sLFC']: dITp['sFResComb' + sTp],
                                       dIG['sFXt']: dIG['xtCSV']}
        for sTp in ['sINmer', 'sIEff', 'sIEffF']:
            d2PI[dITp[sTp]] = {dIG['sPath']: self.pDirResInfo,
                               dIG['sLFC']: dITp['sFResIEff'],
                               dIG['sLFE']: dITp['lSLenNmer'],
                               dIG['sLFJE']: dITp['sUSC'],
                               dIG['sLFJCE']: dITp['sUS02'],
                               dIG['sFXt']: dIG['xtCSV']}
        d2PI[dITp['sINmer']][dIG['sLFC']] = dITp['sFResINmer']
        d2PI[dITp['sIEffF']][dIG['sLFE']] = [sFlFE] + dITp['lSLenNmer']
        d2PI[dITp['sIGen']] = {dIG['sPath']: self.pDirResInfo,
                               dIG['sLFC']: dITp['sFResIGen'],
                               dIG['sFXt']: dIG['xtCSV']}
        self.FPs.addFPs(d2PI)

    # --- methods for filling the DataFrame type dictionary -------------------
    def fillDTpDfrExp(self):
        sBGIN, sBGIE = self.dITp['sBGenInfoNmer'], self.dITp['sBGenInfoEff']
        self.dTpDfr[self.dITp['sBase']][sBGIN] = self.dITp['genInfoNmer']
        self.dTpDfr[self.dITp['sBase']][sBGIE] = self.dITp['genInfoEff']

    # --- methods for printing results ----------------------------------------
    def printFilteredRawDataKin(self, lC, nR0, nR1, nR2, nDig=GC.R04):
        print(GC.S_DS80, GC.S_NEWL, 'Rows with "NaN" in columns ', lC, ': ',
              nR0-nR1, ' of ', nR0, '\t(', round((nR0-nR1)/nR0*100., nDig),
              '%)', GC.S_NEWL, 'Rows with "', self.dITp['sNULL'],
              '" in columns ', lC, ': ', nR1-nR2, ' of ', nR0, '\t(',
              round((nR1-nR2)/nR0*100., nDig), '%)', GC.S_NEWL, GC.S_ARR_LR,
              ' Remaining rows: ', nR2, GC.S_NEWL, GC.S_DS80, sep='')

    def printFilteredRawDataNmer(self, cC, nR0, nR1, nR2, nDig=GC.R04):
        print(GC.S_DS80, GC.S_NEWL, 'Rows with "NaN" in column "', cC, '": ',
              nR0-nR1, ' of ', nR0, '\t\t(', round((nR0-nR1)/nR0*100., nDig),
              '%)', GC.S_NEWL, 'Rows with length of snippet not ',
              self.dITp['lenNmerDef'], ': ', nR1-nR2, ' of ', nR0, '\t(',
              round((nR1-nR2)/nR0*100., nDig), '%)', GC.S_NEWL, GC.S_ARR_LR,
              ' Remaining rows: ', nR2, GC.S_NEWL, GC.S_DS80, sep='')

    # --- methods for loading data --------------------------------------------
    def loadProcInpDfrs(self):
        pFProcInp = GF.joinToPath(self.pDirProcInp, self.dITp['sFProcInpKin'])
        self.dfrKin = self.loadData(pF=pFProcInp, iC=0)
        pFProcInp = GF.joinToPath(self.pDirProcInp, self.dITp['sFProcInpNmer'])
        self.dfrNmer = self.loadData(pF=pFProcInp, iC=0)
        self.lDfrInp = [self.dfrKin, self.dfrNmer]

    # --- methods for saving data ---------------------------------------------
    def saveProcInpDfrs(self):
        self.lDfrInp, lF = [self.dfrKin, self.dfrNmer], self.dITp['lSFProcInp']
        for cDfr, sFPI in zip(self.lDfrInp, lF):
            self.saveData(cDfr, pF=GF.joinToPath(self.pDirProcInp, sFPI),
                          dropDup=True, saveAnyway=False)

    # --- methods for processing experimental data ----------------------------
    def fillDfrResIGen(self, cDfr, lR, iSt=0):
        if self.dITp['genInfoGen']:
            self.dfrResIGen.loc[lR[iSt], cDfr.columns] = cDfr.shape[0]
            for sC in cDfr.columns:
                nUnique = cDfr[sC].dropna().unique().size
                nNaN = cDfr[sC].size - cDfr[sC].dropna().size
                self.dfrResIGen.at[lR[iSt + 1], sC] = nUnique
                self.dfrResIGen.at[lR[iSt + 2], sC] = nNaN

    def filterRawDataKin(self, nDig=GC.R04):
        cDfr, lColF = self.dfrKin, self.dITp['lSCKinF']
        # add info regarding number of unique elements to self.dfrResIGen
        self.fillDfrResIGen(cDfr, lR=self.dITp['lRResIG'])
        # remove lines with "NaN" or "NULL" in all columns (excl. cColExcl)
        nR0 = cDfr.shape[0]
        cDfr.dropna(axis=0, subset=lColF, inplace=True)
        nR1 = cDfr.shape[0]
        for sC in lColF:
            self.dfrKin = cDfr[cDfr[sC] != self.dITp['sNULL']]
        self.dfrKin.reset_index(drop=True, inplace=True)
        self.fillDfrResIGen(self.dfrKin, lR=self.dITp['lRResIG'], iSt=3)
        nR2 = self.dfrKin.shape[0]
        self.printFilteredRawDataKin(lColF, nR0, nR1, nR2, nDig=nDig)

    def procColKin(self):
        l = [s for s in self.dfrKin[self.dITp['sPSite']]]
        for k, s in enumerate(l):
            if type(s) == str and len(s) > 0:
                try:
                    l[k] = int(s[1:])
                except:
                    l[k] = s[1:]
        self.dfrKin[self.dITp['sPosPSite']] = l

    def filterRawDataNmer(self, nDig=GC.R04):
        cDfr, cCol = self.dfrNmer, self.dITp['sCNmer']
        nR0, sLenS = cDfr.shape[0], self.dITp['sLenSnip']
        # add info regarding number of unique elements to self.dfrResIGen
        self.fillDfrResIGen(cDfr, lR=self.dITp['lRResIG'])
        # remove lines with "NaN" in the Nmer-column
        cDfr.dropna(axis=0, subset=[cCol], inplace=True)
        cDfr.reset_index(drop=True, inplace=True)
        nR1 = cDfr.shape[0]
        # remove lines with the wrong length of the Nmer
        cDfr[sLenS] = [len(cDfr.at[k, cCol]) for k in range(cDfr.shape[0])]
        self.dfrNmer = cDfr[cDfr[sLenS] == self.dITp['lenNmerDef']]
        self.dfrNmer.reset_index(drop=True, inplace=True)
        self.dfrNmer = self.dfrNmer.drop(sLenS, axis=1)
        self.fillDfrResIGen(self.dfrNmer, lR=self.dITp['lRResIG'], iSt=3)
        nR2 = self.dfrNmer.shape[0]
        self.printFilteredRawDataNmer(cCol, nR0, nR1, nR2, nDig=nDig)

    def procColNmer(self):
        l = [s for s in self.dfrNmer[self.dITp['sCCode']]]
        for k, s in enumerate(l):
            if type(s) == str and len(s) > 0 and GC.S_DOT in s:
                l[k] = s.split(GC.S_DOT)[0]
        self.dfrNmer[self.dITp['sCodeTrunc']] = l

    # --- methods for combining experimental data -----------------------------
    def combine(self, lSCK, lSCNmer, iSt=0, sMd=GC.S_X_SHORT):
        dEffTg = SF.createDEffTarg(self.dITp, self.dfrKin, self.dfrNmer,
                                   lCDfrNmer=lSCNmer, sMd=sMd)
        GF.printSizeDDDfr(dDDfr=dEffTg, modeF=False)
        self.dfrComb = GF.dDDfrToDfr(dDDfr=dEffTg, lSColL=lSCK, lSColR=lSCNmer)
        self.dTpDfr[self.dITp['sBase']][self.dITp['sCDfrComb']] = self.dfrComb
        pFCmb = self.FPs.dPF[self.dITp['sCombined'] + self.dITp['dCmb'][sMd]]
        self.saveData(self.dfrComb, pF=pFCmb, dropDup=True)
        self.fillDfrResIGen(self.dfrComb, lR=self.dITp['lRResIG'], iSt=iSt)

    def combineXS(self, iSt=0):
        lSCK = [self.dITp['sEffCode'], self.dITp['sTargCode']]
        lSCNmer = [s for s in self.dfrNmer.columns
                  if s not in self.dITp['lCXclDfrNmerXS']]
        self.combine(lSCK, lSCNmer, iSt=iSt, sMd=GC.S_X_SHORT)

    def combineS(self, iSt=0):
        lSCK = [self.dITp['sEffCode'], self.dITp['sTargCode']]
        lSCNmer = [s for s in self.dfrNmer.columns
                  if s not in self.dITp['lCXclDfrNmerS']]
        self.combine(lSCK, lSCNmer, iSt=iSt, sMd=GC.S_SHORT)

    def combineM(self, iSt=0):
        lSCK = [self.dITp['sEffCode'], self.dITp['sTargCode']]
        lSCNmer = [s for s in self.dfrNmer.columns
                  if s not in self.dITp['lCXclDfrNmerM']]
        self.combine(lSCK, lSCNmer, iSt=iSt, sMd=GC.S_MED)

    def combineL(self, iSt=0):
        lSCK, lSCNmer = list(self.dfrKin.columns), list(self.dfrNmer.columns)
        self.combine(lSCK, lSCNmer, iSt=iSt, sMd=GC.S_LONG)

    def combineInpDfr(self):
        if self.dITp['dBDoDfrRes'][GC.S_X_SHORT]:
            print('Creating extra short combined result...')
            self.combineXS(iSt=self.dITp['iStXS'])
            print('Created extra short combined result.')
        if self.dITp['dBDoDfrRes'][GC.S_SHORT]:
            print('Creating short combined result...')
            self.combineS(iSt=self.dITp['iStS'])
            print('Created short combined result.')
        if self.dITp['dBDoDfrRes'][GC.S_MED]:
            print('Creating medium combined result...')
            self.combineM(iSt=self.dITp['iStM'])
            print('Created medium combined result.')
        if self.dITp['dBDoDfrRes'][GC.S_LONG]:
            print('Creating long combined result...')
            self.combineL(iSt=self.dITp['iStL'])
            print('Created long combined result.')
        if self.dITp['genInfoGen']:
            self.dfrResIGen = self.dfrResIGen.convert_dtypes()
            self.saveData(self.dfrResIGen, pF=self.FPs.dPF[self.dITp['sIGen']])

    # --- method calling sub-methods that process and combine exp. data -------
    def procExpData(self, nDigDsp=GC.R04):
        if self.dITp['procInput']:
            self.filterRawDataKin(nDig=nDigDsp)
            self.procColKin()
            self.filterRawDataNmer(nDig=nDigDsp)
            self.procColNmer()
            self.saveProcInpDfrs()
        else:
            self.loadProcInpDfrs()
        self.combineInpDfr()

    # --- methods extracting info from processed/combined/training/test data --
    def getCombDfr(self, tpDfr, sMd=GC.S_X_SHORT):
        cDfr, sTrain, sTest = None, self.dITp['sTrain'], self.dITp['sTrain']
        cDfr = self.dTpDfr[tpDfr][self.dITp['sCDfrComb']]
        if cDfr is None and tpDfr == self.dITp['sBase']:
            pFC = self.FPs.dPF[self.dITp['sCombined'] + self.dITp['dCmb'][sMd]]
            cDfr = self.loadData(pF=pFC)
        elif cDfr is None and tpDfr in [sTrain, sTest]:
            cDfr = self.loadData(self.dPF[tpDfr][self.dITp['sCombinedInp']])
        return cDfr

    def convDNmerToDfr(self, dNmer, stT, tpDfr):
        if self.dTpDfr[tpDfr][self.dITp['sBGenInfoNmer']]:
            dfrINmer = SF.dDNumToDfrINmer(self.dITp, dNmer)
            dfrINmer.sort_values(by=self.dITp['lSortCDfrNmer'],
                                 ascending=self.dITp['lSortDirAscDfrNmer'],
                                 inplace=True, ignore_index=True)
            self.saveData(dfrINmer, pF=self.FPs.dPF[self.dITp['sINmer']],
                          saveIdx=True)
            GF.showElapsedTime(stT)
        if self.dTpDfr[tpDfr][self.dITp['sBGenInfoEff']]:
            dfrIEff = SF.dDNumToDfrIEff(self.dITp, dNmer)
            self.saveData(dfrIEff, pF=self.FPs.dPF[self.dITp['sIEff']],
                          saveIdx=False)
            dfrIEffF = SF.dDNumToDfrIEff(self.dITp, dNmer, wAnyEff=True)
            self.saveData(dfrIEffF, pF=self.FPs.dPF[self.dITp['sIEffF']],
                          saveIdx=False)
            GF.showElapsedTime(stT)

    def getInfoKinNmer(self, stT, tpDfr, sMd=GC.S_X_SHORT):
        dENmer, dNmer, iCNmer = {}, {}, self.dITp['iCentNmer']
        cDfr = self.getCombDfr(tpDfr=tpDfr, sMd=sMd)
        if cDfr is not None:
            for dRec in cDfr.to_dict('records'):
                # considering all effectors pooled
                GF.addToDictL(dENmer, self.dITp['sAnyEff'],
                              dRec[self.dITp['sCNmer']], lUnqEl=False)
                # considering the different effectors separately
                GF.addToDictL(dENmer, dRec[self.dITp['sEffCode']],
                              dRec[self.dITp['sCNmer']], lUnqEl=False)
        GF.addSCentNmerToDict(dNmer, dENmer, iCNmer)
        GF.showElapsedTime(stT)
        self.convDNmerToDfr(dNmer, stT, tpDfr=tpDfr)

###############################################################################