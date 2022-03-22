# -*- coding: utf-8 -*-
###############################################################################
# --- O_00__BaseClass.py ------------------------------------------------------
###############################################################################
import os, copy, pprint

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

# -----------------------------------------------------------------------------
class BaseClass:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat):
        self.idO = 'O_00'
        self.descO = 'Base class'
        self.dIG = inpDat.dI
        self.getDITp()
        self.iniDfrs()
        self.fillDTpDfrBase()
        print('Initiated "BaseClass" base object.')

    def getDITp(self, iTpBase=0, iTp=0, lITpUpd=[]):
        self.dITp = copy.deepcopy(self.dIG[iTpBase])    # type of base class: 0
        for iTpUpd in lITpUpd + [iTp]:             # updated with types in list
            self.dITp.update(self.dIG[iTpUpd])

    # --- print methods -------------------------------------------------------
    def __str__(self):
        sIn = (GC.S_WV80 + GC.S_NEWL + GC.S_SP04 + self.descO + ' with ID ' +
               str(self.idO) + GC.S_NEWL + GC.S_WV80)
        return sIn

    def printDIG(self):
        print(GC.S_DS29, 'General dictionary:', GC.S_DS30)
        pprint.pprint(self.dIG)

    def printDITp(self):
        print(GC.S_DS31, 'Type dictionary:', GC.S_DS31)
        pprint.pprint(self.dITp)

    def printInpDfrs(self):
        print(GC.S_DS80)
        print(GC.S_DS24, 'Raw input Kinase targets data:', GC.S_DS24)
        print(self.dfrKin)
        print(GC.S_DS80)
        print(GC.S_DS31, '15-polymer data:', GC.S_DS31)
        print(self.dfrNmer)
        print(GC.S_DS80)

    # --- methods for retrieving values and modifying the input dictionary ----
    def getValDITp(self, cK):
        if cK in self.dITp:
            return self.dITp[cK]
        else:
            print('ERROR: Key', cK, 'not in type dictionary of', self.descO)

    def addToDITp(self, cK, cV):
        if cK in self.dITp:
            print('Assigning key of type dictionary', cK, 'new value', cV)
        else:
            print('Adding entry (', cK, GC.S_COL, cV, ') to type dictionary')
        self.dITp[cK] = cV

    # --- methods for initialising the DataFrames -----------------------------
    def iniDfrs(self):
        self.dfrKin, self.dfrNmer = None, None
        self.lDfrInp = [self.dfrKin, self.dfrNmer]
        self.dfrComb, self.dfrTrain, self.dfrTest = None, None, None

    # --- methods for filling the DataFrame type dictionary -------------------
    def fillDTpDfrBase(self):
        sDfrC, sNMer = self.dITp['sCDfrComb'], self.dITp['sNmer']
        sEff, sEffF = self.dITp['sEff'], self.dITp['sEffF']
        self.dTpDfr = {self.dITp['sBase']: {},
                       self.dITp['sTrain']: {},
                       self.dITp['sTest']: {}}
        self.dTpDfr[self.dITp['sBase']][sDfrC] = self.dfrComb
        self.dTpDfr[self.dITp['sTrain']][sDfrC] = self.dfrTrain
        self.dTpDfr[self.dITp['sTest']][sDfrC] = self.dfrTest
        self.dTpDfr[self.dITp['sBase']][sNMer] = self.dITp['sImer']
        self.dTpDfr[self.dITp['sTrain']][sNMer] = self.dITp['sImerTrain']
        self.dTpDfr[self.dITp['sTest']][sNMer] = self.dITp['sImerTest']
        self.dTpDfr[self.dITp['sBase']][sEff] = self.dITp['sIEff']
        self.dTpDfr[self.dITp['sTrain']][sEff] = self.dITp['sIEffTrain']
        self.dTpDfr[self.dITp['sTest']][sEff] = self.dITp['sIEffTest']
        self.dTpDfr[self.dITp['sBase']][sEffF] = self.dITp['sIEffF']
        self.dTpDfr[self.dITp['sTrain']][sEffF] = self.dITp['sIEffFTrain']
        self.dTpDfr[self.dITp['sTest']][sEffF] = self.dITp['sIEffFTest']

        # --- methods for loading and saving DataFrames ---------------------------
    def getPDir(self):
        if self.dIG['isTest']:
            self.pDirProcInp = self.dITp['pDirProcInp_T']
            self.pDirRes = self.dITp['pDirRes_T']
        else:
            self.pDirProcInp = self.dITp['pDirProcInp']
            self.pDirRes = self.dITp['pDirRes']

    def loadDfr(self, pF, iC=None, dDTp=None, cSep=None):
        cDfr, sPrt = None, 'Path ' + pF + ' does not exist! Returning "None".'
        if cSep is None:
            cSep = self.dITp['cSep']
        if os.path.isfile(pF):
            cDfr = GF.readCSV(pF, iCol=iC, dDTp=dDTp, cSep=cSep)
            sPrt = 'Loading DataFrame from path ' + pF
        print(sPrt)
        return cDfr

    def saveDfr(self, cDfr, pF, saveIdx=True, dropDup=False, saveAnyway=False):
        if (not os.path.isfile(pF) or saveAnyway) and cDfr is not None:
            print('Saving DataFrame as *.csv file to path ' + pF)
            GF.saveAsCSV(cDfr, pF, cSep=self.dITp['cSep'], saveIdx=saveIdx,
                         dropDup=dropDup)

# -----------------------------------------------------------------------------
class Seq():
    # --- initialisation of the class -----------------------------------------
    def __init__(self, sSq=''):
        assert type(sSq) == str
        self.sSeq = sSq
        self.sPCent = None

# -----------------------------------------------------------------------------
class NmerSeq(Seq):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dITp, sSq='', iPCt=None):
        super().__init__(sSq=sSq)
        assert len(self.sSeq) == dITp['lenSDef']
        if iPCt is None:        # no index of centre given --> standalone Nmer
            iPCt = dITp['iCentNmer']
        self.iPCent = iPCt
        self.sPCent = sSq[dITp['iCentNmer']]
        self.createProfileDict(dITp)

    # --- methods for creating and getting the profile dictionary -------------
    def createProfileDict(self, dITp):
        self.dPrf, iCNmer = {}, dITp['iCentNmer']
        for k in range(iCNmer + 1):
            self.dPrf[2*k + 1] = self.sSeq[(iCNmer - k):(iCNmer + k + 1)]
    
    def getProfileDict(self, maxLenSeq=None):
        dPrf = self.dPrf
        if maxLenSeq is not None:
            dPrf = {lenSeq: sSeq for lenSeq, sSeq in self.dPrf.items()
                    if lenSeq <= maxLenSeq}
        return dPrf

# -----------------------------------------------------------------------------
class FullSeq(Seq):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dITp, sSq='', lIPosPyl=[]):
        super().__init__(sSq=sSq)
        iCNmer = dITp['iCentNmer']
        self.lIPyl = sorted([i for i in lIPosPyl
                             if (i >= iCNmer and i < len(self.sSeq) - iCNmer)])
        self.createNmerDict(dITp, iCNmer)

    # --- methods for creating the Nmer dictionary ----------------------------
    def createNmerDict(self, dITp, iCNmer):
        self.dNmer = {}
        for iPyl in self.lIPyl:
            sNmerSeq = self.sSeq[(iPyl - iCNmer):(iPyl + iCNmer + 1)]
            self.dNmer[iPyl] = NmerSeq(dITp, sSq=sNmerSeq, iPCt=iPyl)

###############################################################################
