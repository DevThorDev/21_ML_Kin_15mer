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
        self.loadInpData(iC=self.dITp['iCInpData'])
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
        lAttr2None = ['serNmerSeq', 'dfrInpClf', 'dfrInpPrC', 'lSCl', 'X', 'y']
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)

    # --- methods for loading input data --------------------------------------
    def loadInpData(self, iC=0):
        dfrInp = self.loadData(pF=self.dPF['InpDataClf'], iC=iC)
        self.dfrInpClf, self.dfrInpPrC = dfrInp, dfrInp
        if self.dPF['InpDataPrC'] != self.dPF['InpDataClf']:
            self.dfrInpPrC = self.loadData(pF=self.dPF['InpDataPrC'], iC=iC)
        if self.dITp['sCNmer'] in self.dfrInpClf.columns:
            self.serNmerSeq = self.dfrInpClf[self.dITp['sCNmer']]
        if self.dITp['usedNmerSeq'] == self.dITp['sUnqList']:
            self.dfrInpClf = self.toUnqNmerSeq()
        assert self.dITp['sCY'] in self.dfrInpClf.columns
        for sCX in self.dITp['lSCX']:
            assert sCX in self.dfrInpClf.columns
        self.lSCl = sorted(list(self.dfrInpClf[self.dITp['sCY']].unique()))
        self.X = self.dfrInpClf[self.dITp['lSCX']]
        self.y = self.dfrInpClf[self.dITp['sCY']]

    # --- methods for modifying the input data to contain unique Nmer seq. ----
    def toUnqNmerSeq(self):
        lSer, sCNmer, sCY = [], self.dITp['sCNmer'], self.dITp['sCY']
        self.serNmerSeq = GF.iniPdSer(self.serNmerSeq.unique(), nameS=sCNmer)
        for cSeq in self.serNmerSeq:
            cDfr = self.dfrInpClf[self.dfrInpClf[sCNmer] == cSeq]
            lSC = GF.toListUnique(cDfr[sCY].to_list())
            cSer = cDfr.iloc[0, :]
            cSer.at[sCY] = SF.getClassStr(self.dITp, lSCl=lSC)
            lSer.append(cSer)
        return GF.concLSerAx1(lSer=lSer, ignIdx=True).T

    # --- print methods -------------------------------------------------------
    def printX(self):
        print(GC.S_DS20, GC.S_NEWL, 'Training input samples:', sep='')
        print(self.X)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', self.X.index.to_list())
                print('Columns:', self.X.columns.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printY(self):
        print(GC.S_DS20, GC.S_NEWL, 'Class labels:', sep='')
        print(self.y)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', self.y.index.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printXY(self):
        self.printX()
        self.printY()

###############################################################################
