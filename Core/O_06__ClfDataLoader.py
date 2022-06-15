# -*- coding: utf-8 -*-
###############################################################################
# --- O_06__ClfDataLoader.py --------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

# -----------------------------------------------------------------------------
class DataLoader(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=6):
        super().__init__(inpDat)
        self.idO = 'O_06'
        self.descO = 'Loader of the classification input data'
        self.getDITp(iTp=iTp)
        self.fillDPF()
        self.iniAttr()
        self.loadInpDataClf(iC=self.dITp['iCInpDataClf'])
        self.loadInpDataPrC(iC=self.dITp['iCInpDataPrC'])
        print('Initiated "DataLoader" base object.')

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPF(self):
        sFInpClf = self.dITp['sFInpClf'] + self.dITp['xtCSV']
        sFInpPrC = self.dITp['sFInpPrC'] + self.dITp['xtCSV']
        self.dPF = {}
        self.dPF['InpDataClf'] = GF.joinToPath(self.dITp['pInpClf'], sFInpClf)
        self.dPF['InpDataPrC'] = GF.joinToPath(self.dITp['pInpPrC'], sFInpPrC)

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self):
        lBase, lAttr2None = ['serNmerSeq', 'dfrInp', 'lSCl', 'X', 'y'], []
        for sTp in ['Clf', 'PrC']:
            lAttr2None += [s + sTp for s in lBase]
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)

    # --- methods for loading input data --------------------------------------
    def loadInpDataClf(self, iC=0):
        if self.dfrInpClf is None:
            self.dfrInpClf = self.loadData(self.dPF['InpDataClf'], iC=iC)
        tInpData = SF.loadInpData(self.dITp, self.dfrInpClf, sMd='Clf', iC=iC)
        (self.serNmerSeqClf, self.dfrInpClf, self.lSClClf,
         self.XClf, self.yClf) = tInpData

    def loadInpDataPrC(self, iC=0):
        if (self.dfrInpPrC is None and self.dfrInpClf is not None and
            self.dPF['InpDataPrC'] == self.dPF['InpDataClf']):
            t = (self.serNmerSeqClf, self.dfrInpClf, self.lSClClf,
                 self.XClf, self.yClf)
            (self.serNmerSeqPrC, self.dfrInpPrC, self.lSClPrC,
             self.XPrC, self.yPrC) = t
        elif ((self.dfrInpPrC is None and self.dfrInpClf is not None and
               self.dPF['InpDataPrC'] != self.dPF['InpDataClf']) or
              (self.dfrInpPrC is None and self.dfrInpClf is None)):
            self.dfrInpPrC = self.loadData(self.dPF['InpDataPrC'], iC=iC)
            t = SF.loadInpData(self.dITp, self.dfrInpPrC, sMd='PrC', iC=iC)
            (self.serNmerSeqPrC, self.dfrInpPrC, self.lSClPrC,
             self.XPrC, self.yPrC) = t

    # --- print methods -------------------------------------------------------
    def printSerNmerSeqClf(self):
        print(GC.S_DS80, GC.S_NEWL, 'Classifier Nmer sequence:', sep='')
        print(self.serNmerSeqClf)
        print(GC.S_DS80)

    def printSerNmerSeqPrC(self):
        print(GC.S_DS80, GC.S_NEWL, 'Proportion calculator Nmer sequence:',
              sep='')
        print(self.serNmerSeqPrC)
        print(GC.S_DS80)

    def printDfrInpClf(self):
        print(GC.S_DS80, GC.S_NEWL, 'Classifier input data:', sep='')
        print(self.dfrInpClf)
        print(GC.S_DS80)

    def printDfrInpPrC(self):
        print(GC.S_DS80, GC.S_NEWL, 'Input data for proportion calculation:',
              sep='')
        dfrClf, dfrPrC = self.dfrInpClf, self.dfrInpPrC
        print(dfrPrC)
        if dfrClf is None and dfrPrC is None:
            print('Input data for proportion calculation and classifier input',
                  ' data are both "', dfrPrC, '"!', sep='')
        elif dfrClf is not None and dfrPrC is None:
            print('Input data for proportion calculation is "', dfrPrC, '", ',
                  'while classifier input data is', GC.S_NEWL, dfrClf, '!',
                  sep='')
        elif dfrClf is None and dfrPrC is not None:
            print('Classifier input is "', dfrClf, '", therefore unequal to ',
                  'input data for proportion calculation!', sep='')
        else:
            if dfrPrC.equals(dfrClf):
                print('Input data for proportion calculation is EQUAL to ',
                      'classifier input data!', sep='')
            else:
                print('Classifier input data is', GC.S_NEWL, dfrClf, ', there',
                      'fore unequal to input data for proportion calculation!',
                      sep='')
        print(GC.S_DS80)

    def printLSClClf(self):
        print(GC.S_DS80, GC.S_NEWL, 'List of class strings for mode "Clf": ',
              self.lSClClf, GC.S_NEWL, '(length = ', len(self.lSClClf), ')',
              GC.S_NEWL, GC.S_DS80, sep='')

    def printLSClPrC(self):
        print(GC.S_DS80, GC.S_NEWL, 'List of class strings for mode "PrC": ',
              self.lSClPrC, GC.S_NEWL, '(length = ', len(self.lSClPrC), ')',
              GC.S_NEWL, GC.S_DS80, sep='')

    def getXorY(self, XorY=GC.S_CAP_X, sMd='Clf'):
        cDat = None
        if XorY == GC.S_CAP_X:
            if sMd == 'Clf':
                cDat = self.XClf
            elif sMd == 'PrC':
                cDat = self.XPrC
        elif XorY == GC.S_CAP_Y:
            if sMd == 'Clf':
                cDat = self.yClf
            elif sMd == 'PrC':
                cDat = self.yPrC
        return cDat

    def printX(self, sMd='Clf'):
        print(GC.S_DS80, GC.S_NEWL, 'Training input samples for mode "', sMd,
              '":', sep='')
        X = self.getXorY(XorY=GC.S_CAP_X, sMd=sMd)
        print(X)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', X.index.to_list())
                print('Columns:', X.columns.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printY(self, sMd='Clf'):
        print(GC.S_DS80, GC.S_NEWL, 'Class labels for mode "', sMd, '":',
              sep='')
        y = self.getXorY(XorY=GC.S_CAP_Y, sMd=sMd)
        print(y)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', y.index.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printXY(self, sMd='Clf'):
        self.printX(sMd=sMd)
        self.printY(sMd=sMd)

    # --- methods for yielding data -------------------------------------------
    def yieldSerNmerSeqClf(self):
        return self.serNmerSeqClf

    def yieldSerNmerSeqPrC(self):
        return self.serNmerSeqPrC

    def yieldDfrInpClf(self):
        return self.dfrInpClf

    def yieldDfrInpPrC(self):
        return self.dfrInpPrC

    def yieldData(self, sMd='Clf'):
        if sMd == 'Clf':
            return (self.serNmerSeqClf, self.dfrInpClf, self.lSClClf,
                    self.XClf, self.yClf)
        elif sMd == 'PrC':
            return (self.serNmerSeqPrC, self.dfrInpPrC, self.lSClPrC,
                    self.XPrC, self.yPrC)
        else: return tuple([None]*5)

###############################################################################
