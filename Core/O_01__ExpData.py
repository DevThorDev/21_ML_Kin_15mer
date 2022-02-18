# -*- coding: utf-8 -*-
###############################################################################
# --- O_01__ExpData.py --------------------------------------------------------
###############################################################################
import copy
#
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
# import Core.F_01__SpcFunctions as SF

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
        print('Initiated "ExpData" base object.')

    # --- methods for reading experimental data -------------------------------
    def readExpData(self):
        if self.dIG['isTest']:
            self.dfrKin = self.loadDfr(pF=self.dITp['pFRawInpKin_T'])
            self.dfr15mer = self.loadDfr(pF=self.dITp['pFRawInp15mer_T'])
            self.pDirProcInp = self.dITp['pDirProcInp_T']
        else:
            self.dfrKin = self.loadDfr(pF=self.dITp['pFRawInpKin'])
            self.dfr15mer = self.loadDfr(pF=self.dITp['pFRawInp15mer'])
            self.pDirProcInp = self.dITp['pDirProcInp']
        self.lDfrInp = [self.dfrKin, self.dfr15mer]

    # --- methods for saving processed input data -----------------------------
    def saveProcInpDfrs(self):
        self.lDfrInp = [self.dfrKin, self.dfr15mer]
        for cDfr, sFPI in zip(self.lDfrInp, self.dITp['lSFProcInp']):
            self.saveDfr(cDfr, pF=GF.joinToPath(self.pDirProcInp, sFPI))


    # --- methods for processing experimental data ----------------------------
    def filterRawDataKin(self, nDig=GC.R04):
        cDfr, lColF = self.dfrKin, self.dITp['lSCKinF']
        # remove lines with "NaN" or "NULL" in all columns (excl. cColExcl)
        nR0 = cDfr.shape[0]
        cDfr.dropna(axis=0, subset=lColF, inplace=True)
        nR1 = cDfr.shape[0]
        for sC in lColF:
            cDfr = cDfr[cDfr[sC] != self.dITp['sNULL']]
        cDfr.reset_index(drop=True, inplace=True)
        nR2 = cDfr.shape[0]
        print(GC.S_DS80, GC.S_NEWL, 'Rows with "NaN" in columns ', lColF, ': ',
              nR0-nR1, ' of ', nR0, '\t(', round((nR0-nR1)/nR0*100., nDig),
              '%)', sep='')
        print('Rows with "', self.dITp['sNULL'], '" in columns ', lColF, ': ',
              nR1-nR2, ' of ', nR0, '\t(', round((nR1-nR2)/nR0*100., nDig),
              '%)', GC.S_NEWL, GC.S_DS80, sep='')

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
        cDfr = cDfr[cDfr[sLenS] == self.dITp['lenSDef']]
        cDfr.reset_index(drop=True, inplace=True)
        self.dfr15mer = cDfr.drop(sLenS, axis=1)
        nR2 = self.dfr15mer.shape[0]
        print(GC.S_DS80, GC.S_NEWL, 'Rows with "NaN" in column "', cCol, '": ',
              nR0-nR1, ' of ', nR0, '\t\t(', round((nR0-nR1)/nR0*100., nDig),
              '%)', sep='')
        print('Rows with length of snippet not ', self.dITp['lenSDef'], ': ',
              nR1-nR2, ' of ', nR0, '\t(', round((nR1-nR2)/nR0*100., nDig),
              '%)', GC.S_NEWL, GC.S_DS80, sep='')

    def procCol15mer(self):
        l = [s for s in self.dfr15mer[self.dITp['sCCode']]]
        for k, s in enumerate(l):
            if type(s) == str and len(s) > 0 and GC.S_DOT in s:
                l[k] = s.split(GC.S_DOT)[0]
        self.dfr15mer[self.dITp['sCodeTrunc']] = l

    def matchKinTargetCode(self):
        pass

    def combineInpDfr(self):
        dEffTarg, cDfr = {}, self.dfrKin
        for s in cDfr['sEffCode'].unique():
            dfrRed = cDfr[cDfr[self.dITp['sEffCode']] == s]
            dEffTarg[s] = list(dfrRed[self.dITp['sTargCode']].unique())

    def procExpData(self, nDigDsp=GC.R04):
        self.filterRawDataKin(nDig=nDigDsp)
        self.procColKin()
        self.filterRawData15mer(nDig=nDigDsp)
        self.procCol15mer()
        self.saveProcInpDfrs()

# --- methods initialising and updating dictionaries --------------------------
    def iniDfrs(self):
        pass

# --- methods saving DataFrames -----------------------------------------------

###############################################################################
