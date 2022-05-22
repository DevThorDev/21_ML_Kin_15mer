# -*- coding: utf-8 -*-
###############################################################################
# --- O_07__Classifier.py ----------------------------------------------------
###############################################################################
import time

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Core.O_02__SeqAnalysis import SeqAnalysis

# -----------------------------------------------------------------------------
class Classifier(SeqAnalysis):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=7, lITpUpd=[1, 2]):
        super().__init__(inpDat)
        self.idO = 'O_07'
        self.descO = 'Classifier for data classification'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.addValsToDPF()
        self.loadInpData()
        print('Initiated "Classifier" base object.')

    # --- methods for loading input data --------------------------------------
    def loadInpData(self):
        self.dfrInp = self.loadData(pF=self.dPF['InpData'], iC=0)
        self.serNmerSeq = self.dfrInp[self.dITp['sCNmer']]
        lSIg = [self.dITp['sEffCode'], self.dITp['sCNmer']]
        if self.dITp['usedNmerSeq'] == self.dITp['sUnqList']:
            self.serNmerSeq = GF.iniPdSer(self.serNmerSeq.unique(),
                                          nameS=self.dITp['sCNmer'])
            lSer = []
            for cSeq in self.serNmerSeq:
                cDfr = self.dfrInp[self.dfrInp[self.dITp['sCNmer']] == cSeq]
                lSer.append(cDfr.iloc[0, :])
            self.dfrInp = GF.concLSer(lSer=lSer, ignIdx=True)
        self.X = self.dfrInp[[s for s in self.dfrInp.columns if s not in lSIg]]
        self.y = self.dfrInp[self.dITp['sEffCode']]

    # --- print methods -------------------------------------------------------
    def printX(self):
        print(GC.S_DS20, GC.S_NEWL, 'Training input samples:', sep='')
        print(self.X)
        print('Index:', self.X.index.to_list())
        print('Columns:', self.X.columns.to_list())
        print(GC.S_DS80)

    def printY(self):
        print(GC.S_DS20, GC.S_NEWL, 'Class labels:', sep='')
        print(self.y)
        print('Index:', self.y.index.to_list())
        print(GC.S_DS80)

    # --- methods for filling the result paths dictionary ---------------------
    def addValsToDPF(self):
        sFInp = self.dITp['sFInp'] + self.dITp['xtCSV']
        self.dPF['InpData'] = GF.joinToPath(self.dITp['pInp'], sFInp)
        # sFInp = self.dITp['sUSC'].join([self.dITp['sFInp'],
        #                                 self.dITp['usedNmerSeq']]) + sXt

    # --- methods for saving data ---------------------------------------------
    def saveResData(self):
        pass


###############################################################################
