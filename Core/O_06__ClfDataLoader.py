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
    def __init__(self, inpDat, iTp=6, lITpUpd=[1]):
        super().__init__(inpDat)
        self.idO = 'O_06'
        self.descO = 'Loader of the classification input data'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.iniAttr()
        self.fillFPs()
        if self.dITp['useNmerNoCl'] is None or self.dITp['useNmerNoCl']:
            self.getDictNmerNoCl()
        self.loadInpDataClf(iC=self.dITp['iCInpDataClf'])
        self.loadInpDataPrC(iC=self.dITp['iCInpDataPrC'])
        print('Initiated "DataLoader" base object.')

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self):
        lBase = ['serNmerSeq', 'dfrInp', 'XS', 'XM', 'YS', 'YM', 'dClMap',
                 'dMltSt', 'lSXCl']
        lAttr2None = ['dMltStSClf', 'dMltStMClf', 'dNmerLoc']
        for sTp in [self.dITp['sClf'], self.dITp['sPrC']]:
            lAttr2None += [s + sTp for s in lBase]
        lAttr2None += ['dNmerNoCl', 'lI']
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)
        self.lIFES = [self.dITp[s] for s in self.dITp['lSIFEndS']]
        self.lIFET = [self.dITp[s] for s in self.dITp['lSIFEndT']]
        self.lIFEU = [self.dITp[s] for s in self.dITp['lSIFEndU']]
        self.lIFEX = [self.dITp[s] for s in self.dITp['lSIFEndX']]
        self.lIFEY = [self.dITp[s] for s in self.dITp['lSIFEndY']]

    # --- methods for filling the file paths ----------------------------------
    def fillFPs(self):
        d2PI, dIG, dITp = {}, self.dIG, self.dITp
        d2PI['ResComb'] = {dIG['sPath']: dITp['pResComb'],
                           dIG['sLFC']: dITp['sFResComb'],
                           dIG['sFXt']: dIG['xtCSV']}
        for sTp in ['Clf', 'PrC']:
            d2PI['InpData' + sTp] = {dIG['sPath']: dITp['pInp' + sTp],
                                     dIG['sLFC']: dITp['sFInp' + sTp],
                                     dIG['sFXt']: dIG['xtCSV']}
        for sTp in ['Snips', 'EffF']:
            d2PI['DictNmer' + sTp] = {dIG['sPath']: dITp['pBinData'],
                                      dIG['sLFS']: dITp['sFDictNmer' + sTp],
                                      dIG['sLFC']: dITp['sFResComb'],
                                      dIG['sLFE']: self.lIFEU,
                                      dIG['sLFJSC']: dITp['sUS02'],
                                      dIG['sLFJCE']: dITp['sUS02'],
                                      dIG['sFXt']: dIG['xtBIN']}
        for sTp in ['EffFam', 'SeqU']:
            d2PI['Nmer' + sTp] = {dIG['sPath']: dITp['pUnqNmer'],
                                  dIG['sLFS']: dITp['sNmer' + sTp],
                                  dIG['sLFC']: dITp['sFInpClf'],
                                  dIG['sLFE']: self.lIFEU,
                                  dIG['sLFJSC']: dITp['sUS02'],
                                  dIG['sLFJCE']: dITp['sUS02'],
                                  dIG['sFXt']: dIG['xtCSV']}
        d2PI['NOccTupU'] = {dIG['sPath']: dITp['pUnqNmer'],
                            dIG['sLFS']: dITp['sNOccTupUnq'],
                            dIG['sLFC']: dITp['sFInpClf'],
                            dIG['sLFE']: self.lIFEU,
                            dIG['sLFJSC']: dITp['sUS02'],
                            dIG['sLFJCE']: dITp['sUS02'],
                            dIG['sFXt']: dIG['xtCSV']}
        for sTp in ['InpData', 'XS', 'XM', 'YS', 'YM']:
            d2PI[sTp] = {dIG['sPath']: dITp['pInpData'],
                         dIG['sLFS']: dITp['s' + sTp],
                         dIG['sLFC']: dITp['sFInpClf'],
                         dIG['sLFE']: self.lIFES,
                         dIG['sLFJSC']: dITp['sUS02'],
                         dIG['sLFJCE']: dITp['sUS02'],
                         dIG['sFXt']: dIG['xtCSV']}
        d2PI['InpData'][dIG['sLFE']] = self.lIFET
        for sX in ['XS', 'XM']:
            d2PI[sX][dIG['sLFE']] = self.lIFEX
        for sY in ['YS', 'YM']:
            d2PI[sY][dIG['sLFE']] = self.lIFEY
        self.FPs.addFPs(d2PI)
        self.d2PInf = d2PI

    # --- method for generating the "no class" Nmer dictionary ----------------
    def getDictNmerNoCl(self):
        sDSn, sCCS, sCNmer = 'DictNmerSnips', 'sCCodeSeq', 'sCNmer'
        if self.dITp['useBinDicts'] and GF.fileXist(pF=self.FPs.dPF[sDSn]):
            self.dNmerNoCl = GF.pickleLoadDict(pF=self.FPs.dPF[sDSn])
        else:
            dfrComb = GF.readCSV(pF=self.FPs.dPF['ResComb'], iCol=0)
            assert self.dITp[sCCS] in dfrComb.columns
            serFullUnq = GF.toSerUnique(pdSer=dfrComb[self.dITp[sCCS]])
            serNmerUnq = None
            if self.dITp[sCNmer] in dfrComb.columns:
                serNmerUnq = GF.toSerUnique(dfrComb[self.dITp[sCNmer]])
            lNmerUnq = [SF.getCentSNmerDefLen(self.dITp, sSeq=sNmer) for sNmer
                        in GF.toListUnqViaSer(serNmerUnq)]
            self.dNmerNoCl = SF.getDSqNoCl(self.dITp, serFullSeqUnq=serFullUnq,
                                           lNmerSeqUnq=lNmerUnq)
            GF.pickleSaveDict(cD=self.dNmerNoCl, pF=self.FPs.dPF[sDSn])

    # --- methods for loading input data --------------------------------------
    def saveProcInpData(self, tData, tSKDPF, sMd=GC.S_CLF):
        assert len(tData) == len(tSKDPF)
        for cDat, s in zip(tData, tSKDPF):
            self.FPs.modFP(d2PI=self.d2PInf, kMn=s, kPos='sLFE', cS=sMd)
            if s == GC.S_N_MER_EFF_FAM:
                itC = [GF.joinS([self.dITp['sEffFam'], str(k + 1)]) for k in
                       range(max([0] + [len(cDat[cK]) for cK in cDat]))]
                cDat = GF.dictLUneqLen2Dfr(cDat, itCol=itC, doTrans=True)
            self.saveData(cDat, pF=self.FPs.dPF[s])

    def procInpData(self, dfrInp, saveData=None):
        dNmerEffF, serNmerSeq = SF.preProcInp(self.dITp, dfrInp=dfrInp,
                                              dNmerNoCl=self.dNmerNoCl)
        t = SF.procInp(self.dIG, self.dITp, dNmerEffF=dNmerEffF)
        dfrInp, XS, XM, YS, YM, dClMp, lSXCl = t
        if saveData is not None:
            t2Save = (dNmerEffF, dfrInp, YS, YM)
            tSKeyDPF = (GC.S_N_MER_EFF_FAM, 'InpData', 'YS', 'YM')
            self.saveProcInpData(tData=t2Save, tSKDPF=tSKeyDPF, sMd=saveData)
        return dNmerEffF, serNmerSeq, dfrInp, XS, XM, YS, YM, dClMp, lSXCl

    def genSaveDfrNmerSeq(self, dSubStr):
        maxNLoc = max([len(l) for l in dSubStr.values()])
        lIdx = ['loc_' + str(k) for k in range(1, maxNLoc + 1)]
        dfrNmer = GF.iniDfrFromDictIt(dSubStr, idxDfr=lIdx, doTrans=True)
        serK = dfrNmer.apply(GF.getKeyFromL, axis=1, sSep=self.dITp['sVBar'])
        dfrNmer[self.dITp['sLocKey']] = serK
        self.saveData(dfrNmer, pF=self.FPs.dPF['NmerSeqU'])
        self.dfrNmerSeq = dfrNmer
        return lIdx

    def procLocData(self, sMd=GC.S_CLF):
        dSubStr, lILoc, sLocKey = {}, [], self.dITp['sLocKey']
        self.FPs.modFP(d2PI=self.d2PInf, kMn='NmerSeqU', kPos='sLFE', cS=sMd)
        if not (GF.fileXist(self.FPs.dPF['NmerSeqU']) and
                GF.fileXist(self.FPs.dPF['NOccTupU'])):
            sPPI, sFPI = self.dITp['pProcInp'], self.dITp['sFProcInpNmer']
            dfrNmer = self.loadData(pF=GF.joinToPath(sPPI, sFPI), iC=0)
            lC = [self.dITp['sCCodeSeq'], self.dITp['sLoc']]
            dfrNmerLoc = dfrNmer.loc[:, lC].drop_duplicates(ignore_index=True)
            dNmerLoc = SF.genDictNmerLoc(self.dITp, dfrNL=dfrNmerLoc)
            dSubStr = SF.genDictSubStr(dNmerLoc, itSubStr=self.serNmerSeqClf)
            # SF.modDictSubStr(self.dITp, dSubStr)
            lILoc = self.genSaveDfrNmerSeq(dSubStr)
        else:
            self.dfrNmerSeq = self.loadData(self.FPs.dPF['NmerSeqU'], iC=0)
        dNmer = self.dfrNmerSeq.loc[:, [sLocKey]].to_dict(orient='index')
        self.dNmerLoc = {cK: list(cD.values())[0] for cK, cD in dNmer.items()}
        return dSubStr, lILoc

    def genDfrNOccTupUnq(self, tInf):
        dSubStr, i = tInf
        if not GF.fileXist(self.FPs.dPF['NOccTupU']):
            dUnNO = GF.unifyEquivItKeys(dInp=GF.getDNOccOfDictVals(cD=dSubStr))
            dfrTupUnq = SF.getDfrNOccTupUnq(self.dITp, dUNOcc=dUnNO, idxDfr=i)
            dfrTupUnq.sort_values(by=i, axis=0, ascending=True, inplace=True,
                                  ignore_index=True)
            self.saveData(dfrTupUnq, pF=self.FPs.dPF['NOccTupU'])
        # else:
        #     dfrTupUnq = self.loadData(self.FPs.dPF['NOccTupU'], iC=0)

    def extendXClfWithLoc(self):
        if self.dITp['useLocData'] == self.dITp['sWNmer']:
            lX = [self.XSClf, self.XMClf]
            for k, dfrX in enumerate(lX):
                serX = dfrX.apply(lambda x: ''.join([s for s in x]), axis=1)
                dNmerLocX = {sNmer: self.dNmerLoc[sNmer] for sNmer in serX}
                dfrXR = GF.iniPdDfr(dNmerLocX, lSNmR=[self.dITp['sLocKey']]).T
                iR, iC = dfrX.apply(lambda x: ''.join(x), axis=1), dfrX.columns
                dfrXT = GF.iniPdDfr(dfrX.to_numpy(), lSNmR=iR, lSNmC=iC)
                dfrX = GF.concPdDfrS([dfrXT, dfrXR], concAx=1)
                dfrX.reset_index(drop=True, inplace=True)
                lX[k] = dfrX
                # self.saveData(dfrX, pF='dfrX_'+str(k + 1))
            [self.XSClf, self.XMClf] = lX
            self.saveProcInpData(tData=tuple(lX), tSKDPF=('XS', 'XM'))

    def loadInpDataClf(self, iC=0):
        if self.dfrInpClf is None:
            self.dfrInpClf = self.loadData(self.FPs.dPF['InpDataClf'], iC=iC)
        t = self.procInpData(dfrInp=self.dfrInpClf, saveData=self.dITp['sClf'])
        (self.dNmerEffFClf, self.serNmerSeqClf, self.dfrInpClf, self.XSClf,
         self.XMClf, self.YSClf, self.YMClf, self.dClMapClf, self.lSXClClf) = t
        if self.dITp['useLocData'] is not None:
            self.genDfrNOccTupUnq(self.procLocData())
            self.extendXClfWithLoc()

    def loadInpDataPrC(self, iC=0):
        if (self.dfrInpPrC is None and self.dfrInpClf is not None and
            self.FPs.dPF['InpDataPrC'] == self.FPs.dPF['InpDataClf']):
            t = (self.dNmerEffFClf, self.serNmerSeqClf, self.dfrInpClf,
                 self.XSClf, self.XMClf, self.YSClf, self.YMClf,
                 self.dClMapClf, self.lSXClClf)
            (self.dNmerEffFPrC, self.serNmerSeqPrC, self.dfrInpPrC,
             self.XSPrC, self.XMPrC, self.YSPrC, self.YMPrC, self.dClMapPrC,
             self.lSXClPrC) = t
        elif ((self.dfrInpPrC is None and self.dfrInpClf is not None and
               self.FPs.dPF['InpDataPrC'] != self.FPs.dPF['InpDataClf']) or
              (self.dfrInpPrC is None and self.dfrInpClf is None)):
            self.dfrInpPrC = self.loadData(self.FPs.dPF['InpDataPrC'], iC=iC)
            t = self.procInpData(dfrInp=self.dfrInpPrC)
            (self.dNmerEffFPrC, self.serNmerSeqPrC, self.dfrInpPrC, self.XSPrC,
             self.XMPrC, self.YSPrC, self.YMPrC, self.dClMapPrC,
             self.lSXClPrC) = t

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
            print('Input data for proportion calculation and Classifier input',
                  ' data are both "', dfrPrC, '"!', sep='')
        elif dfrClf is not None and dfrPrC is None:
            print('Input data for proportion calculation is "', dfrPrC, '", ',
                  'while Classifier input data is', GC.S_NEWL, dfrClf, '!',
                  sep='')
        elif dfrClf is None and dfrPrC is not None:
            print('Classifier input is "', dfrClf, '", therefore unequal to ',
                  'input data for proportion calculation!', sep='')
        else:
            if dfrPrC.equals(dfrClf):
                print('Input data for proportion calculation is EQUAL to ',
                      'Classifier input data!', sep='')
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

    def printlSXClClf(self):
        print(GC.S_DS80, GC.S_NEWL, 'List of class strings for mode "Clf": ',
              self.lSXClClf, GC.S_NEWL, '(length = ', len(self.lSXClClf), ')',
              GC.S_NEWL, GC.S_DS80, sep='')

    def printlSXClPrC(self):
        print(GC.S_DS80, GC.S_NEWL, 'List of class strings for mode "PrC": ',
              self.lSXClPrC, GC.S_NEWL, '(length = ', len(self.lSXClPrC), ')',
              GC.S_NEWL, GC.S_DS80, sep='')

    def getXorY(self, XorY=GC.S_X, sMd=GC.S_CLF, sLbl=GC.S_SGL_LBL):
        cDat = None
        if XorY == GC.S_X:
            if sMd == self.dITp['sClf']:
                if sLbl == self.dITp['sSglLbl']:
                    cDat = self.XSClf
                else:
                    cDat = self.XMClf
            elif sMd == self.dITp['sPrC']:
                if sLbl == self.dITp['sSglLbl']:
                    cDat = self.XSPrC
                else:
                    cDat = self.XMPrC
        elif XorY == GC.S_Y:
            if sMd == self.dITp['sClf']:
                if sLbl == self.dITp['sSglLbl']:
                    cDat = self.YSClf
                else:
                    cDat = self.YMClf
            elif sMd == self.dITp['sPrC']:
                if sLbl == self.dITp['sSglLbl']:
                    cDat = self.YSPrC
                else:
                    cDat = self.YMPrC
        return cDat

    def printX(self, sMd=GC.S_CLF, sLbl=GC.S_SGL_LBL):
        print(GC.S_DS80, GC.S_NEWL, 'Training input samples for mode "', sMd,
              '":', sep='')
        X = self.getXorY(XorY=GC.S_X, sMd=sMd, sLbl=sLbl)
        print(X)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', X.index.to_list())
                print('Columns:', X.columns.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printY(self, sMd=GC.S_CLF, sLbl=GC.S_SGL_LBL):
        print(GC.S_DS80, GC.S_NEWL, 'Class labels for mode "', sMd, '":',
              sep='')
        Y = self.getXorY(XorY=GC.S_Y, sMd=sMd, sLbl=sLbl)
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

    # --- methods for getting multi-steps data --------------------------------
    def getDMltStS(self, YS=None):
        if self.dMltStSClf is None:
            self.dMltStSClf = SF.getIMltSt(self.dIG, self.dITp, Y=YS)
        return self.dMltStSClf

    def getDMltStM(self, YM=None):
        if self.dMltStMClf is None:
            self.dMltStMClf = SF.getIMltSt(self.dIG, self.dITp, Y=YM,
                                           sLbl=self.dITp['sMltLbl'])
        return self.dMltStMClf

    def getDMltSt(self, Y=None):
        if self.dITp['ILblSgl']:
            return self.getDMltStS(YS=Y)
        else:
            return self.getDMltStM(YM=Y)

    def getDMltStAll(self, YS=None, YM=None):
        return (self.getDMltStS(YS=YS), self.getDMltStM(YM=YM))

    # --- methods for yielding data -------------------------------------------
    def yieldFPs(self):
        return self.FPs

    def yieldSerNmerSeqClf(self):
        return self.serNmerSeqClf

    def yieldSerNmerSeqPrC(self):
        return self.serNmerSeqPrC

    def yieldSerNmerSeq(self, sMd=GC.S_CLF):
        if sMd == self.dITp['sClf']:
            return self.serNmerSeqClf
        else:
            return self.serNmerSeqPrC

    def yieldDfrInpClf(self):
        return self.dfrInpClf

    def yieldDfrInpPrC(self):
        return self.dfrInpPrC

    def yieldDfrInp(self, sMd=GC.S_CLF):
        if sMd == self.dITp['sClf']:
            return self.dfrInpClf
        else:
            return self.dfrInpPrC

    def yieldXSClf(self):
        return self.XSClf

    def yieldXMClf(self):
        return self.XMClf

    def yieldXClf(self, sLbl=GC.S_SGL_LBL):
        if sLbl == self.dITp['sSglLbl']:
            return self.yieldXSClf()
        else:
            return self.yieldXMClf()

    def yieldXSPrC(self):
        return self.XSPrC

    def yieldXMPrC(self):
        return self.XMPrC

    def yieldXPrC(self, sLbl=GC.S_SGL_LBL):
        if sLbl == self.dITp['sSglLbl']:
            return self.yieldXSPrC()
        else:
            return self.yieldXMPrC()

    def yieldX(self, sMd=GC.S_CLF, sLbl=GC.S_SGL_LBL):
        if sMd == self.dITp['sClf']:
            return self.yieldXClf(sLbl=sLbl)
        else:
            return self.yieldXPrC(sLbl=sLbl)

    def yieldYSClf(self):
        return self.YSClf

    def yieldYMClf(self):
        return self.YMClf

    def yieldYClf(self, sLbl=GC.S_SGL_LBL):
        if sLbl == self.dITp['sSglLbl']:
            return self.yieldYSClf()
        else:
            return self.yieldYMClf()

    def yieldYSPrC(self):
        return self.YSPrC

    def yieldYMPrC(self):
        return self.YMPrC

    def yieldYPrC(self, sLbl=GC.S_SGL_LBL):
        if sLbl == self.dITp['sSglLbl']:
            return self.yieldYSPrC()
        else:
            return self.yieldYMPrC()

    def yieldY(self, sMd=GC.S_CLF, sLbl=GC.S_SGL_LBL):
        if sMd == self.dITp['sClf']:
            return self.yieldYClf(sLbl=sLbl)
        else:
            return self.yieldYPrC(sLbl=sLbl)

    def yieldClfData(self, sLbl=GC.S_SGL_LBL):
        if sLbl is None:
            return (self.dfrInpClf, self.XSClf, self.XMClf, self.YSClf,
                    self.YMClf, self.serNmerSeqClf, self.dClMapClf,
                    self.lSXClClf)
        elif sLbl == self.dITp['sSglLbl']:
            return (self.dfrInpClf, self.XSClf, self.YSClf,
                    self.serNmerSeqClf, self.dClMapClf, self.lSXClClf)
        else:
            return (self.dfrInpClf, self.XMClf, self.YMClf,
                    self.serNmerSeqClf, self.dClMapClf, self.lSXClClf)

    def yieldPrCData(self, sLbl=GC.S_SGL_LBL):
        if sLbl is None:
            return (self.dfrInpPrC, self.XSPrC, self.XMPrC, self.YSPrC,
                    self.YMPrC, self.serNmerSeqPrC, self.dClMapPrC,
                    self.lSXClPrC)
        elif sLbl == self.dITp['sSglLbl']:
            return (self.dfrInpPrC, self.XSPrC, self.YSPrC,
                    self.serNmerSeqPrC, self.dClMapPrC, self.lSXClPrC)
        else:
            return (self.dfrInpPrC, self.XMPrC, self.YMPrC,
                    self.serNmerSeqPrC, self.dClMapPrC, self.lSXClPrC)

    def yieldData(self, sMd=None, sLbl=GC.S_SGL_LBL):
        if sMd == self.dITp['sClf']:
            return self.yieldClfData(sLbl=sLbl)
        elif sMd == self.dITp['sPrC']:
            return self.yieldPrCData(sLbl=sLbl)
        else: return None

###############################################################################