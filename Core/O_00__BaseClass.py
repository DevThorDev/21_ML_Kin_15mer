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
        self.dITp = copy.deepcopy(self.dIG[0])      # type of base class = 0
        self.dfrKin, self.dfr15mer = None, None
        print('Initiated "BaseClass" base object.')

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

    def printRawInpDfrs(self):
        print(GC.S_DS80)
        print(GC.S_DS24, 'Raw input Kinase targets data:', GC.S_DS24)
        print(self.dfrKin)
        print(GC.S_DS80)
        print(GC.S_DS31, '15-polymer data:', GC.S_DS31)
        print(self.dfr15mer)
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

    # --- methods for loading and saving DataFrames ---------------------------
    def loadDfr(self, pF, iC=None):
        cDfr, sPrt = None, 'Path ' + pF + ' does not exist! Returning "None"'
        if os.path.isfile(pF):
            cDfr = GF.readCSV(pF, iCol=iC, cSep=self.dITp['cSep'])
            sPrt = 'Loading DataFrame from path ' + pF
        print(sPrt)
        return cDfr

    def getFromCSV(self, pF, iCol=0, dDTp=None, cSep=None):
        if cSep is None:
            cSep = self.dITp['cSep']
        if os.path.isfile(pF):
            return GF.readCSV(pF=pF, iCol=iCol, dDTp=dDTp, cSep=cSep)

    def saveDfr(self, cDfr, pF, saveIt=True):
        if (not os.path.isfile(pF) or saveIt) and cDfr is not None:
            print('Saving DataFrame as *.csv file to path ' + pF)
            GF.saveAsCSV(cDfr, pF, cSep=self.dITp['cSep'])

###############################################################################
