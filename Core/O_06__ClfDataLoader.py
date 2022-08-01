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
        self.fillFPs()
        self.iniAttr()
        if self.dITp['useNmerNoCl']:
            self.getDictNmerNoCl()
        self.loadInpDataClf(iC=self.dITp['iCInpDataClf'])
        self.loadInpDataPrC(iC=self.dITp['iCInpDataPrC'])
        print('Initiated "DataLoader" base object.')

    # --- methods for filling the file paths ----------------------------------
    def fillFPs(self):
        d2PI, dIG, dITp = {}, self.dIG, self.dITp
        for sTp in ['Clf', 'PrC']:
            d2PI['InpData' + sTp] = {dIG['sPath']: dITp['pInp' + sTp],
                                     dIG['sLFC']: dITp['sFInp' + sTp],
                                     dIG['sFXt']: dIG['xtCSV']}
        d2PI['ResComb'] = {dIG['sPath']: dITp['pResComb'],
                           dIG['sLFC']: dITp['sFResComb'],
                           dIG['sFXt']: dIG['xtCSV']}
        for sTp in ['Snips', 'EffF']:
            d2PI['DictNmer' + sTp] = {dIG['sPath']: dITp['pBinData'],
                                      dIG['sLFC']: dITp['sFDictNmer' + sTp],
                                      dIG['sLFE']: dITp['sFResComb'],
                                      dIG['sLFJCE']: dITp['sUS02'],
                                      dIG['sFXt']: dIG['xtBIN']}
        for sTp in ['EffF', 'SeqU']:
            d2PI['Nmer' + sTp] = {dIG['sPath']: dITp['pUnqNmer'],
                                  dIG['sLFC']: dITp['sNmer' + sTp],
                                  dIG['sLFE']: dITp['sFInpClf'],
                                  dIG['sLFJCE']: dITp['sUS02'],
                                  dIG['sFXt']: dIG['xtCSV']}
        lSFE = ['sSglMltLbl', 'sXclEffFam', 'sSet', 'sRestr']
        for sTp in ['InpData', 'X', 'Y']:
            d2PI[sTp] = {dIG['sPath']: dITp['pInpData'],
                         dIG['sLFS']: dITp['s' + sTp],
                         dIG['sLFC']: dITp['sFInpClf'],
                         dIG['sLFE']: [dITp[s] for s in lSFE],
                         dIG['sLFJSC']: dITp['sUS02'],
                         dIG['sFXt']: dIG['xtCSV']}
        d2PI['X'][dIG['sLFE']] = lSFE[1:]
        self.FPs.addFPs(d2PI)

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
        sDSn, sCCS, sCNmer = 'DictNmerSnips', 'sCCodeSeq', 'sCNmer'
        if GF.fileXist(pF=self.FPs.dPF[sDSn]):
            self.dNmerNoCl = GF.pickleLoadDict(pF=self.FPs.dPF[sDSn])
        else:
            dfrComb = GF.readCSV(pF=self.FPs.dPF['ResComb'], iCol=0)
            assert self.dITp[sCCS] in dfrComb.columns
            serFullUnq = GF.toSerUnique(pdSer=dfrComb[self.dITp[sCCS]])
            serNmerUnq = None
            if self.dITp[sCNmer] in dfrComb.columns:
                serNmerUnq = GF.toSerUnique(dfrComb[self.dITp[sCNmer]])
            lNmerUnq = GF.toListUnqViaSer(serNmerUnq)
            self.dNmerNoCl = SF.getDSqNoCl(self.dITp, serFullSeqUnq=serFullUnq,
                                           lNmerSeqUnq=lNmerUnq)
            GF.pickleSaveDict(cD=self.dNmerNoCl, pF=self.FPs.dPF[sDSn])

    # --- methods for loading input data --------------------------------------
    def saveProcInpData(self, tData, lSKDPF, sMd=GC.S_CLF):
        sJ = self.dITp['sUSC']
        for cDat, s in zip(tData, lSKDPF):
            pF = GF.modPF(self.FPs.dPF[s], sEnd=sMd, sJoin=sJ)
            if s == 'NmerEffF':
                itC = [GF.joinS([self.dITp['sEffFam'], str(k + 1)]) for k in
                       range(max([0] + [len(cDat[cK]) for cK in cDat]))]
                cDat = GF.dictLUneqLen2Dfr(cDat, itCol=itC, doTrans=True)
            self.saveData(cDat, pF=pF)

    def procInpData(self, dfrInp, saveData=None):
        dNmerEffF, serNmerSeq = SF.preProcInp(self.dITp, dfrInp=dfrInp,
                                              dNmerNoCl=self.dNmerNoCl)
        dfrInp, X, Y, lSXCl = SF.procInp(self.dIG, self.dITp, dNmerEffF)
        if saveData is not None:
            t2Save = (dNmerEffF, serNmerSeq, dfrInp, X, Y)
            lSKeyDPF = 'NmerEffF', 'NmerSeqU', 'InpData', 'X', 'Y'
            self.saveProcInpData(tData=t2Save, lSKDPF=lSKeyDPF, sMd=saveData)
        return dNmerEffF, serNmerSeq, dfrInp, X, Y, lSXCl

    def loadInpDataClf(self, iC=0):
        if self.dfrInpClf is None:
            self.dfrInpClf = self.loadData(self.FPs.dPF['InpDataClf'], iC=iC)
        t = self.procInpData(dfrInp=self.dfrInpClf, saveData=self.dITp['sClf'])
        (self.dNmerEffFClf, self.serNmerSeqClf, self.dfrInpClf, self.XClf,
         self.YClf, self.lSXClClf) = t

    def loadInpDataPrC(self, iC=0):
        if (self.dfrInpPrC is None and self.dfrInpClf is not None and
            self.FPs.dPF['InpDataPrC'] == self.FPs.dPF['InpDataClf']):
            t = (self.dNmerEffFClf, self.serNmerSeqClf, self.dfrInpClf,
                 self.XClf, self.YClf, self.lSXClClf)
            (self.dNmerEffFPrC, self.serNmerSeqPrC, self.dfrInpPrC,
             self.XPrC, self.YPrC, self.lSXClPrC) = t
        elif ((self.dfrInpPrC is None and self.dfrInpClf is not None and
               self.FPs.dPF['InpDataPrC'] != self.FPs.dPF['InpDataClf']) or
              (self.dfrInpPrC is None and self.dfrInpClf is None)):
            self.dfrInpPrC = self.loadData(self.FPs.dPF['InpDataPrC'], iC=iC)
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
    def yieldFPs(self):
        return self.FPs

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