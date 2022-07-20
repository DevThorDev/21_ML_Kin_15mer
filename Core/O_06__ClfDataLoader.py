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
        if self.dITp['useNmerNoCl']:
            self.getDictNmerNoCl()
        self.loadInpDataClf(iC=self.dITp['iCInpDataClf'])
        self.loadInpDataPrC(iC=self.dITp['iCInpDataPrC'])
        print('Initiated "DataLoader" base object.')

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPF(self):
        sJ, xtCSV = self.dITp['sUS02'], self.dITp['xtCSV']
        sFInpClf = self.dITp['sFInpClf'] + xtCSV
        sFInpPrC = self.dITp['sFInpPrC'] + xtCSV
        sFResComb = self.dITp['sFResComb'] + xtCSV
        lSF = [self.dITp['sFDictNmerSnips'], self.dITp['sFResComb']]
        sFDNS = GF.joinS(lSF, sJoin=sJ) + self.dITp['xtBIN']
        lSF = [self.dITp['sFDictNmerEffF'], self.dITp['sFResComb']]
        sFDNX = GF.joinS(lSF, sJoin=sJ) + self.dITp['xtBIN']
        sFNmerEffF = (GF.joinS([self.dITp['sNmerEffF'], self.dITp['sFInpClf']],
                               sJoin=sJ) + xtCSV)
        sFNmerSeqU = (GF.joinS([self.dITp['sNmerSeqU'], self.dITp['sFInpClf']],
                               sJoin=sJ) + xtCSV)
        sFInpDat = (GF.joinS([self.dITp['sInpData'], self.dITp['sFInpClf'],
                              self.dITp['sSglMltLbl']], sJoin=sJ) + xtCSV)
        sFX = (GF.joinS([self.dITp['sX'], self.dITp['sFInpClf']], sJoin=sJ) +
               xtCSV)
        sFY = (GF.joinS([self.dITp['sY'], self.dITp['sFInpClf'],
                         self.dITp['sSglMltLbl']], sJoin=sJ) + xtCSV)
        self.dPF = {}
        self.dPF['InpDataClf'] = GF.joinToPath(self.dITp['pInpClf'], sFInpClf)
        self.dPF['InpDataPrC'] = GF.joinToPath(self.dITp['pInpPrC'], sFInpPrC)
        self.dPF['ResComb'] = GF.joinToPath(self.dITp['pResComb'], sFResComb)
        self.dPF['DictNmerSnips'] = GF.joinToPath(self.dITp['pBinData'], sFDNS)
        self.dPF['DictNmerEffF'] = GF.joinToPath(self.dITp['pBinData'], sFDNX)
        self.dPF['NmerEffF'] = GF.joinToPath(self.dITp['pUnqNmer'], sFNmerEffF)
        self.dPF['NmerSeqU'] = GF.joinToPath(self.dITp['pUnqNmer'], sFNmerSeqU)
        self.dPF['InpData'] = GF.joinToPath(self.dITp['pInpData'], sFInpDat)
        self.dPF['X'] = GF.joinToPath(self.dITp['pInpData'], sFX)
        self.dPF['Y'] = GF.joinToPath(self.dITp['pInpData'], sFY)

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self):
        lBase, lAttr2None = ['serNmerSeq', 'dfrInp', 'X', 'Y', 'lSCl'], []
        for sTp in [self.dITp['sClf'], self.dITp['sPrC']]:
            lAttr2None += [s + sTp for s in lBase]
        lAttr2None += ['dNmerNoCl']
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)

    # --- method for generating the "no class" Nmer dictionary ----------------
    def getDictNmerNoCl(self):
        if GF.fileXist(pF=self.dPF['DictNmerSnips']):
            self.dNmerNoCl = GF.pickleLoadDict(pF=self.dPF['DictNmerSnips'])
        else:
            dfrComb = GF.readCSV(pF=self.dPF['ResComb'], iCol=0)
            assert self.dITp['sCCodeSeq'] in dfrComb.columns
            serFullUnq = GF.toSerUnique(pdSer=dfrComb[self.dITp['sCCodeSeq']])
            serNmerUnq = None
            if self.dITp['sCNmer'] in dfrComb.columns:
                serNmerUnq = GF.toSerUnique(dfrComb[self.dITp['sCNmer']])
            lNmerUnq = GF.toListUnqViaSer(serNmerUnq)
            self.dNmerNoCl = SF.getDSqNoCl(self.dITp, serFullSeqUnq=serFullUnq,
                                           lNmerSeqUnq=lNmerUnq)
            GF.pickleSaveDict(cD=self.dNmerNoCl, pF=self.dPF['DictNmerSnips'])

    # --- methods for loading input data --------------------------------------
    def procInpData(self, dfrInp, saveData=None):
        dNmerEffF, serNmerSeq = SF.preProcInp(self.dITp, dfrInp=dfrInp,
                                              dNmerNoCl=self.dNmerNoCl)
        dfrInp, X, Y, lSXCl = SF.procInp(self.dITp, dNmerEffF=dNmerEffF)
        if saveData is not None:
            sJ = self.dITp['sUSC']
            for cDat, s in zip([dNmerEffF, serNmerSeq, dfrInp, X, Y],
                               ['NmerEffF', 'NmerSeqU', 'InpData', 'X', 'Y']):
                pF = GF.modPF(self.dPF[s], sEnd=saveData, sJoin=sJ)
                if s == 'NmerEffF':
                    # nFam = max([0] + [len(cDat[cK]) for cK in cDat])
                    itC = [GF.joinS([self.dITp['sEffFam'], str(k + 1)]) for k
                           in range(max([0] + [len(cDat[cK]) for cK in cDat]))]
                    cDat = GF.dictLUneqLen2Dfr(cDat, itCol=itC, doTrans=True)
                self.saveData(cDat, pF=pF)
        return dNmerEffF, serNmerSeq, dfrInp, X, Y, lSXCl
    
    def loadInpDataClf(self, iC=0):
        if self.dfrInpClf is None:
            self.dfrInpClf = self.loadData(self.dPF['InpDataClf'], iC=iC)
        t = self.procInpData(dfrInp=self.dfrInpClf, saveData=self.dITp['sClf'])
        (self.dNmerEffFClf, self.serNmerSeqClf, self.dfrInpClf, self.XClf,
         self.YClf, self.lSXClClf) = t

    def loadInpDataPrC(self, iC=0):
        if (self.dfrInpPrC is None and self.dfrInpClf is not None and
            self.dPF['InpDataPrC'] == self.dPF['InpDataClf']):
            t = (self.dNmerEffFClf, self.serNmerSeqClf, self.dfrInpClf,
                 self.XClf, self.YClf, self.lSXClClf)
            (self.dNmerEffFPrC, self.serNmerSeqPrC, self.dfrInpPrC,
             self.XPrC, self.YPrC, self.lSXClPrC) = t
        elif ((self.dfrInpPrC is None and self.dfrInpClf is not None and
               self.dPF['InpDataPrC'] != self.dPF['InpDataClf']) or
              (self.dfrInpPrC is None and self.dfrInpClf is None)):
            self.dfrInpPrC = self.loadData(self.dPF['InpDataPrC'], iC=iC)
            t = self.procInpData(dfrInp=self.dfrInpPrC)
            (self.dNmerEffFPrC, self.serNmerSeqPrC, self.dfrInpPrC, self.XPrC,
             self.YPrC, self.lSXClPrC) = t

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

    def printDictNmerNoCl(self):
        print(GC.S_DS80, GC.S_NEWL, '"No class" Nmer sequences:', sep='')
        if self.dNmerNoCl is not None:
            for cK, cL in self.dNmerNoCl.items():
                print(cK, ': ', cL[:5], '...', sep='')
            nNmer = sum([len(cL) for cL in self.dNmerNoCl.values()])
            print('Number of "no class" Nmer sequences:', nNmer)
            print('Lengths of lists of "no class" Nmer sequences:',
                  {cK: len(cL) for cK, cL in self.dNmerNoCl.items()})
            print('Shares of "no class" Nmer sequences for each central',
                  'position amino acid:', {cK: round(len(cL)/nNmer*100, 2) for
                                           cK, cL in self.dNmerNoCl.items()})
        print(GC.S_DS80)

    def printLSClClf(self):
        print(GC.S_DS80, GC.S_NEWL, 'List of class strings for mode "Clf": ',
              self.lSXClClf, GC.S_NEWL, '(length = ', len(self.lSXClClf), ')',
              GC.S_NEWL, GC.S_DS80, sep='')

    def printLSClPrC(self):
        print(GC.S_DS80, GC.S_NEWL, 'List of class strings for mode "PrC": ',
              self.lSXClPrC, GC.S_NEWL, '(length = ', len(self.lSXClPrC), ')',
              GC.S_NEWL, GC.S_DS80, sep='')

    def getXorY(self, XorY=GC.S_CAP_X, sMd=GC.S_CLF):
        cDat = None
        if XorY == GC.S_CAP_X:
            if sMd == self.dITp['sClf']:
                cDat = self.XClf
            elif sMd == self.dITp['sPrC']:
                cDat = self.XPrC
        elif XorY == GC.S_CAP_Y:
            if sMd == self.dITp['sClf']:
                cDat = self.YClf
            elif sMd == self.dITp['sPrC']:
                cDat = self.YPrC
        return cDat

    def printX(self, sMd=GC.S_CLF):
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

    def printY(self, sMd=GC.S_CLF):
        print(GC.S_DS80, GC.S_NEWL, 'Class labels for mode "', sMd, '":',
              sep='')
        Y = self.getXorY(XorY=GC.S_CAP_Y, sMd=sMd)
        print(Y)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', Y.index.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printXY(self, sMd=GC.S_CLF):
        self.printX(sMd=sMd)
        self.printY(sMd=sMd)

    # --- methods for yielding data -------------------------------------------
    def yieldDPF(self):
        return self.dPF

    def yieldSerNmerSeqClf(self):
        return self.serNmerSeqClf

    def yieldSerNmerSeqPrC(self):
        return self.serNmerSeqPrC

    def yieldDfrInpClf(self):
        return self.dfrInpClf

    def yieldDfrInpPrC(self):
        return self.dfrInpPrC

    def yieldData(self, sMd=None):
        if sMd == self.dITp['sClf']:
            return (self.dfrInpClf, self.XClf, self.YClf, self.serNmerSeqClf,
                    self.lSXClClf)
        elif sMd == self.dITp['sPrC']:
            return (self.dfrInpPrC, self.XPrC, self.YPrC, self.serNmerSeqPrC,
                    self.lSXClPrC)
        else: return tuple([None]*5)

###############################################################################