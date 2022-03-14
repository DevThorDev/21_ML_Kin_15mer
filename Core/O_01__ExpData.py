# -*- coding: utf-8 -*-
###############################################################################
# --- O_01__ExpData.py --------------------------------------------------------
###############################################################################
import copy

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
            self.dfrNmer = self.loadDfr(pF=self.dITp['pFRawInpNmer_T'])
            self.pDirProcInp = self.dITp['pDirProcInp_T']
            self.pDirRes = self.dITp['pDirRes_T']
        else:
            self.dfrKin = self.loadDfr(pF=self.dITp['pFRawInpKin'])
            self.dfrNmer = self.loadDfr(pF=self.dITp['pFRawInpNmer'])
            self.pDirProcInp = self.dITp['pDirProcInp']
            self.pDirRes = self.dITp['pDirRes']
        self.lDfrInp = [self.dfrKin, self.dfrNmer]
        lC = list(self.dfrKin.columns) + list(self.dfrNmer.columns)
        if self.dITp['createInfoGen']:
            self.dfrResIGen = GF.iniPdDfr(lSNmC=lC, lSNmR=self.dITp['lRResIG'])

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPFRes(self):
        sJ, sFl, pDirRes = self.dITp['sUS02'], self.dITp['sFull'], self.pDirRes
        sPFCmbS = GF.joinToPath(pDirRes, self.dITp['sFResCombS'])
        sPFCmbM = GF.joinToPath(pDirRes, self.dITp['sFResCombM'])
        sPFCmbL = GF.joinToPath(pDirRes, self.dITp['sFResCombL'])
        dPFCmb = {GC.S_SHORT: sPFCmbS, GC.S_MED: sPFCmbM, GC.S_LONG: sPFCmbL}
        sPFINmer = GF.joinToPath(pDirRes, self.dITp['sFResINmer'])
        sFE = self.dITp['sUSC'].join(self.dITp['lSLenNMer'])
        sFResIEff = GF.modSF(self.dITp['sFResIEff'], sEnd=sFE, sJoin=sJ)
        if len(sFE) > 0:
            sFl = sJ.join([self.dITp['sFull'], sFE])
        sFResIEffF = GF.modSF(self.dITp['sFResIEff'], sEnd=sFl, sJoin=sJ)
        sPFResIGen = GF.joinToPath(pDirRes, self.dITp['sFResIGen'])
        self.dPFRes = {GC.S_COMBINED: dPFCmb,
                       GC.S_INFO_MER: sPFINmer,
                       GC.S_INFO_EFF: GF.joinToPath(pDirRes, sFResIEff),
                       GC.S_INFO_EFF_F: GF.joinToPath(pDirRes, sFResIEffF),
                       GC.S_INFO_GEN: sPFResIGen}

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

    # --- methods for saving data ---------------------------------------------
    def saveProcInpDfrs(self):
        self.lDfrInp = [self.dfrKin, self.dfrNmer]
        for cDfr, sFPI in zip(self.lDfrInp, self.dITp['lSFProcInp']):
            self.saveDfr(cDfr, pF=GF.joinToPath(self.pDirProcInp, sFPI),
                         dropDup=True, saveAnyway=False)

    # --- methods for processing experimental data ----------------------------
    def fillDfrResIGen(self, cDfr, lR, iSt=0):
        if self.dITp['createInfoGen']:
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
    def combine(self, lSCK, lSCNmer, iSt=0, sMd=GC.S_SHORT):
        dEffTarg = SF.createDEffTarg(self.dITp, self.dfrKin, self.dfrNmer,
                                     lCDfrNmer=lSCNmer, sMd=sMd)
        GF.printSizeDDDfr(dDDfr=dEffTarg, modeF=True)
        dfrComb = GF.dDDfrToDfr(dDDfr=dEffTarg, lSColL=lSCK, lSColR=lSCNmer)
        self.saveDfr(dfrComb, pF=self.dPFRes[GC.S_COMBINED][sMd], dropDup=True,
                     saveAnyway=True)
        self.fillDfrResIGen(dfrComb, lR=self.dITp['lRResIG'], iSt=iSt)

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
        if self.dITp['dBDoDfrRes'][GC.S_SHORT]:
            print('Creating short combined result...')
            self.combineS(iSt=12)
            print('Created short combined result.')
        if self.dITp['dBDoDfrRes'][GC.S_MED]:
            print('Creating medium combined result...')
            self.combineM(iSt=9)
            print('Created medium combined result.')
        if self.dITp['dBDoDfrRes'][GC.S_LONG]:
            print('Creating long combined result...')
            self.combineL(iSt=6)
            print('Created long combined result.')
        if self.dITp['createInfoGen']:
            self.saveDfr(self.dfrResIGen, pF=self.dPFRes[GC.S_INFO_GEN])

    # --- method calling sub-methods that process and combine exp. data -------
    def procExpData(self, nDigDsp=GC.R04):
        if self.dITp['procInput']:
            self.filterRawDataKin(nDig=nDigDsp)
            self.procColKin()
            self.filterRawDataNmer(nDig=nDigDsp)
            self.procColNmer()
            self.saveProcInpDfrs()
        self.combineInpDfr()

    # --- methods extracting info from processed/combined data ----------------
    def convDNmerToDfr(self, dNMer, stT):
        if self.dITp['createInfoNmer']:
            dfrINmer = SF.dDNumToDfrINmer(self.dITp, dNMer)
            dfrINmer.sort_values(by=self.dITp['lSortCDfrNMer'],
                                 ascending=self.dITp['lSortDirAscDfrNMer'],
                                 inplace=True, ignore_index=True)
            self.saveDfr(dfrINmer, pF=self.dPFRes[GC.S_INFO_MER], saveIdx=True,
                         dropDup=False, saveAnyway=True)
            GF.showElapsedTime(stT)
        if self.dITp['createInfoEff']:
            dfrIEff = SF.dDNumToDfrIEff(self.dITp, dNMer)
            self.saveDfr(dfrIEff, pF=self.dPFRes[GC.S_INFO_EFF], saveIdx=False,
                         dropDup=False, saveAnyway=True)
            dfrIEffF = SF.dDNumToDfrIEff(self.dITp, dNMer, wAnyEff=True)
            self.saveDfr(dfrIEffF, pF=self.dPFRes[GC.S_INFO_EFF_F],
                         saveIdx=False, dropDup=False, saveAnyway=True)
            GF.showElapsedTime(stT)

    def getInfoKinNmer(self, stT, sMd=GC.S_SHORT):
        dENmer, dfrCombS = {}, self.loadDfr(pF=self.dPFRes[GC.S_COMBINED][sMd])
        if dfrCombS is not None:
            for dRec in dfrCombS.to_dict('records'):
                # considering all effectors pooled
                GF.addToDictL(dENmer, self.dITp['sAnyEff'],
                              dRec[self.dITp['sCNmer']], lUniqEl=False)
                # considering the different effectors separately
                GF.addToDictL(dENmer, dRec[self.dITp['sEffCode']],
                              dRec[self.dITp['sCNmer']], lUniqEl=False)
        dNMer, iCNmer = {}, self.dITp['iCentNmer']
        for k in range(iCNmer + 1):
            for sEff, lSNmer in dENmer.items():
                for sNmer in lSNmer:
                    sCentNmer = sNmer[(iCNmer - k):(iCNmer + k + 1)]
                    GF.addToDictDNum(dNMer, sEff, sCentNmer)
        GF.showElapsedTime(stT)
        self.convDNmerToDfr(dNMer, stT=stT)

###############################################################################
