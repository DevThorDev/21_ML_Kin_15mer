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
        self.fillDPFExp()
        self.fillDTpDfrExp()
        print('Initiated "ExpData" base object.')

    # --- methods for loading experimental data -------------------------------
    def readExpData(self):
        if self.dIG['isTest']:
            self.dfrKin = self.loadDfr(pF=self.dITp['pFRawInpKin_T'])
            self.dfrNmer = self.loadDfr(pF=self.dITp['pFRawInpNmer_T'])
        else:
            self.dfrKin = self.loadDfr(pF=self.dITp['pFRawInpKin'])
            self.dfrNmer = self.loadDfr(pF=self.dITp['pFRawInpNmer'])
        self.lDfrInp = [self.dfrKin, self.dfrNmer]
        lC = list(self.dfrKin.columns) + list(self.dfrNmer.columns)
        if self.dITp['genInfoGen']:
            self.dfrResIGen = GF.iniPdDfr(lSNmC=lC, lSNmR=self.dITp['lRResIG'])

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPFExp(self):
        sJ, sFlFE = self.dITp['sUS02'], self.dITp['sFull']
        pRCmb, pRInf = self.pDirResComb, self.pDirResInfo
        pFCombXS = GF.joinToPath(pRCmb, self.dITp['sFResCombXS'])
        pFCombS = GF.joinToPath(pRCmb, self.dITp['sFResCombS'])
        pFCombM = GF.joinToPath(pRCmb, self.dITp['sFResCombM'])
        pFCombL = GF.joinToPath(pRCmb, self.dITp['sFResCombL'])
        dPFComb = {GC.S_X_SHORT: pFCombXS, GC.S_SHORT: pFCombS,
                   GC.S_MED: pFCombM, GC.S_LONG: pFCombL}
        sFE = GF.joinS(self.dITp['lSLenNMer'])
        sFResINmer = GF.modSF(self.dITp['sFResINmer'], sEnd=sFE, sJoin=sJ)
        sFResIEff = GF.modSF(self.dITp['sFResIEff'], sEnd=sFE, sJoin=sJ)
        if len(sFE) > 0:
            sFlFE = GF.joinS([sFlFE, sFE], sJoin=sJ)
        sFResIEffF = GF.modSF(self.dITp['sFResIEff'], sEnd=sFlFE, sJoin=sJ)
        pFResIGen = GF.joinToPath(pRInf, self.dITp['sFResIGen'])
        dPFBase = {self.dITp['sCombined']: dPFComb,
                   self.dITp['sImer']: GF.joinToPath(pRInf, sFResINmer),
                   self.dITp['sIEff']: GF.joinToPath(pRInf, sFResIEff),
                   self.dITp['sIEffF']: GF.joinToPath(pRInf, sFResIEffF),
                   self.dITp['sIGen']: pFResIGen}
        self.dPF = {self.dITp['sBase']: dPFBase}

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
              self.dITp['lenSDef'], ': ', nR1-nR2, ' of ', nR0, '\t(',
              round((nR1-nR2)/nR0*100., nDig), '%)', GC.S_NEWL, GC.S_ARR_LR,
              ' Remaining rows: ', nR2, GC.S_NEWL, GC.S_DS80, sep='')

    # --- methods for loading data --------------------------------------------
    def loadProcInpDfrs(self):
        pFProcInp = GF.joinToPath(self.pDirProcInp, self.dITp['sFProcInpKin'])
        self.dfrKin = self.loadDfr(pF=pFProcInp, iC=0)
        pFProcInp = GF.joinToPath(self.pDirProcInp, self.dITp['sFProcInpNmer'])
        self.dfrNmer = self.loadDfr(pF=pFProcInp, iC=0)
        self.lDfrInp = [self.dfrKin, self.dfrNmer]

    # --- methods for saving data ---------------------------------------------
    def saveProcInpDfrs(self):
        self.lDfrInp, lF = [self.dfrKin, self.dfrNmer], self.dITp['lSFProcInp']
        for cDfr, sFPI in zip(self.lDfrInp, lF):
            self.saveDfr(cDfr, pF=GF.joinToPath(self.pDirProcInp, sFPI),
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
        self.dfrNmer = cDfr[cDfr[sLenS] == self.dITp['lenSDef']]
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
        sBase, sComb = self.dITp['sBase'], self.dITp['sCombined']
        self.saveDfr(self.dfrComb, pF=self.dPF[sBase][sComb][sMd],
                     dropDup=True, saveAnyway=True)
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

    def combineInpDfr(self, tpDfr):
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
            self.saveDfr(self.dfrResIGen, self.dPF[tpDfr][self.dITp['sIGen']])

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
        self.combineInpDfr(tpDfr=self.dITp['sBase'])

    # --- methods extracting info from processed/combined/training/test data --
    def getCombDfr(self, tpDfr, sMd=GC.S_X_SHORT):
        cDfr, sTrain, sTest = None, self.dITp['sTrain'], self.dITp['sTrain']
        cDfr = self.dTpDfr[tpDfr][self.dITp['sCDfrComb']]
        if cDfr is None and tpDfr == self.dITp['sBase']:
            cDfr = self.loadDfr(self.dPF[tpDfr][self.dITp['sCombined']][sMd])
        elif cDfr is None and tpDfr in [sTrain, sTest]:
            cDfr = self.loadDfr(self.dPF[tpDfr][self.dITp['sCombinedInp']])
        return cDfr

    def convDNmerToDfr(self, dNMer, stT, tpDfr):
        if self.dTpDfr[tpDfr][self.dITp['sBGenInfoNmer']]:
            dfrINmer = SF.dDNumToDfrINmer(self.dITp, dNMer)
            dfrINmer.sort_values(by=self.dITp['lSortCDfrNMer'],
                                 ascending=self.dITp['lSortDirAscDfrNMer'],
                                 inplace=True, ignore_index=True)
            self.saveDfr(dfrINmer, pF=self.dPF[tpDfr][self.dITp['sImer']],
                         saveIdx=True, saveAnyway=True)
            GF.showElapsedTime(stT)
        if self.dTpDfr[tpDfr][self.dITp['sBGenInfoEff']]:
            dfrIEff = SF.dDNumToDfrIEff(self.dITp, dNMer)
            self.saveDfr(dfrIEff, pF=self.dPF[tpDfr][self.dITp['sIEff']],
                         saveIdx=False, saveAnyway=True)
            dfrIEffF = SF.dDNumToDfrIEff(self.dITp, dNMer, wAnyEff=True)
            self.saveDfr(dfrIEffF, pF=self.dPF[tpDfr][self.dITp['sIEffF']],
                         saveIdx=False, saveAnyway=True)
            GF.showElapsedTime(stT)

    def getInfoKinNmer(self, stT, tpDfr, sMd=GC.S_X_SHORT):
        dENmer, dNMer, iCNmer = {}, {}, self.dITp['iCentNmer']
        cDfr = self.getCombDfr(tpDfr=tpDfr, sMd=sMd)
        if cDfr is not None:
            for dRec in cDfr.to_dict('records'):
                # considering all effectors pooled
                GF.addToDictL(dENmer, self.dITp['sAnyEff'],
                              dRec[self.dITp['sCNmer']], lUniqEl=False)
                # considering the different effectors separately
                GF.addToDictL(dENmer, dRec[self.dITp['sEffCode']],
                              dRec[self.dITp['sCNmer']], lUniqEl=False)
        GF.addSCentNmerToDict(dNMer, dENmer, iCNmer)
        GF.showElapsedTime(stT)
        self.convDNmerToDfr(dNMer, stT, tpDfr=tpDfr)

###############################################################################
