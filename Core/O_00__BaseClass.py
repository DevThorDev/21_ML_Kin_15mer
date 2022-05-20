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
        self.iniDicts()
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

    # --- methods for initialising the dictionaries and DataFrames ------------
    def iniDicts(self):
        self.dTpDfr = None
        self.dSnipS, self.dSnipX = None, None

    def iniDfrs(self):
        self.dfrKin, self.dfrNmer = None, None
        self.lDfrInp = [self.dfrKin, self.dfrNmer]
        self.dfrIEff, self.dfrInpSeq, self.dfrInpNmer = None, None, None
        self.dfrComb, self.dfrTrain, self.dfrTest = None, None, None
        self.dfrEmitProb = None
        self.dfrStartProb, self.dfrTransProb = None, None
        # self.dfrProbDetail, self.dfrProbFinal = None, None

    # --- methods for filling the DataFrame type dictionary -------------------
    def fillDTpDfrBase(self):
        sDfrC, sNmer = self.dITp['sCDfrComb'], self.dITp['sNmer']
        sEff, sEffF = self.dITp['sEff'], self.dITp['sEffF']
        self.dTpDfr = {self.dITp['sBase']: {},
                       self.dITp['sTrain']: {},
                       self.dITp['sTest']: {}}
        self.dTpDfr[self.dITp['sBase']][sDfrC] = self.dfrComb
        self.dTpDfr[self.dITp['sTrain']][sDfrC] = self.dfrTrain
        self.dTpDfr[self.dITp['sTest']][sDfrC] = self.dfrTest
        self.dTpDfr[self.dITp['sBase']][sNmer] = self.dITp['sImer']
        self.dTpDfr[self.dITp['sTrain']][sNmer] = self.dITp['sImerTrain']
        self.dTpDfr[self.dITp['sTest']][sNmer] = self.dITp['sImerTest']
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
            self.pDirResComb = self.dITp['pDirResComb_T']
            self.pDirResInfo = self.dITp['pDirResInfo_T']
            self.pDirResProb = self.dITp['pDirResProb_T']
        else:
            self.pDirProcInp = self.dITp['pDirProcInp']
            self.pDirResComb = self.dITp['pDirResComb']
            self.pDirResInfo = self.dITp['pDirResInfo']
            self.pDirResProb = self.dITp['pDirResProb']
        self.pDirResViterbi = self.dITp['pDirResViterbi']

    def loadData(self, pF, iC=None, dDTp=None, NAfillV=None, cSep=None):
        cDfr, sPrt = None, 'Path ' + pF + ' does not exist! Returning "None".'
        if cSep is None:
            cSep = self.dITp['cSep']
        if os.path.isfile(pF):
            cDfr = GF.readCSV(pF, iCol=iC, dDTp=dDTp, cSep=cSep)
            sPrt = 'Loading data from path ' + pF
        print(sPrt)
        if NAfillV is not None:
            return cDfr.fillna(NAfillV)
        return cDfr

    def loadSerOrList(self, pF, iC=None, toList=False, dDTp=None, cSep=None):
        cSer = self.loadData(pF, iC=iC, dDTp=dDTp, cSep=cSep).iloc[:, 0]
        if toList:
            return cSer.to_list()
        else:
            return cSer

    def saveDfr(self, cDfr, pF, saveIdx=True, idxLbl=None, dropDup=False,
                saveAnyway=True):
        if (not os.path.isfile(pF) or saveAnyway) and cDfr is not None:
            print('Saving DataFrame as *.csv file to path ' + pF)
            if idxLbl is not None:
                saveIdx = True
            GF.checkDupSaveCSV(cDfr, pF, cSep=self.dITp['cSep'],
                               saveIdx=saveIdx, iLbl=idxLbl, dropDup=dropDup)

    def saveListAsSer(self, cL, pF, saveIdx=True, lSIdx=None, sName=None,
                      saveAnyway=True):
        if (not os.path.isfile(pF) or saveAnyway) and cL is not None:
            print('Saving list as pandas Series *.csv file to path ' + pF)
            cSer = GF.iniPdSer(cL, lSNmI=lSIdx, nameS=sName)
            GF.saveCSV(cSer, pF=pF, cSep=self.dITp['cSep'], saveIdx=saveIdx)

# -----------------------------------------------------------------------------
class Seq:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, sSq=''):
        assert type(sSq) == str
        self.sSeq = sSq
        self.sPCent = None

    # --- print methods -------------------------------------------------------
    def __str__(self):
        sIn = (GC.S_WV80 + GC.S_NEWL + GC.S_SP04 + 'Sequence string: ' +
               self.sSeq + GC.S_NEWL + 'has central position ' +
               self.sPCent + GC.S_NEWL + GC.S_WV80)
        return sIn

# -----------------------------------------------------------------------------
class NmerSeq(Seq):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dITp, sSq='', iPCt=None):
        super().__init__(sSq=sSq)
        assert len(self.sSeq) == dITp['lenNmerDef']
        if iPCt is None:        # no index of centre given --> standalone Nmer
            iPCt = dITp['iCentNmer']
        self.iPCent = iPCt
        self.sPCent = sSq[dITp['iCentNmer']]
        self.lIPyl = [dITp['iCentNmer']]
        self.createProfileDict(dITp)

    # --- print methods -------------------------------------------------------
    def __str__(self):
        sIn = (GC.S_WV80 + GC.S_NEWL + GC.S_SP04 + 'Sequence string: ' +
               self.sSeq + GC.S_NEWL + 'has central position index ' +
               str(self.iPCent) + GC.S_NEWL + 'and central position ' +
               self.sPCent + GC.S_NEWL + 'The profile dictionary is:' +
               GC.S_NEWL + str(self.dPrf) + GC.S_NEWL + GC.S_WV80)
        return sIn

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

    # --- sub-method for storing the probabilities of snippets ----------------
    def getProbSnip(self, dITp, lInpSeq, maxLen=None):
        if GF.Xist(lInpSeq):
            dP = {}
            dProf = self.getProfileDict(maxLenSeq=maxLen)
            for sSnip in dProf.values():
                GF.fillDProbSnip(dProb=dP, lSeq=lInpSeq, sSnip=sSnip)
            return dP
        else:
            print('ERROR: List of input sequence strings is',
                  [cSeq.sSeq for cSeq in lInpSeq])
            assert False

    # --- method for storing the total probabilities of snippets --------------
    def getTotalProbSnip(self, dITp, lInpSeq):
        return self.getProbSnip(dITp, lInpSeq, maxLen=max(dITp['lLenNmer']))

    # --- method for storing the conditional probabilities of snippets --------
    def getCondProbSnip(self, dITp, lInpSeq, lFullSeq):
        maxLenNmer = min(max(dITp['lLenNmer']), len(self.sSeq) - 1)
        dCondP = self.getProbSnip(dITp, lInpSeq, maxLen=maxLenNmer)
        GF.fillDProbSnip(dProb=dCondP, lSeq=lFullSeq, sSnip=self.sSeq)
        return dCondP

# -----------------------------------------------------------------------------
class FullSeq(Seq):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dITp, sSq='', lIPosPyl=[]):
        super().__init__(sSq=sSq)
        iCNmer = dITp['iCentNmer']
        self.lIPyl = sorted([i for i in lIPosPyl
                             if (i >= iCNmer and i < len(self.sSeq) - iCNmer)])
        self.createNmerDict(dITp)

    # --- print methods -------------------------------------------------------
    def __str__(self):
        sIn = (GC.S_WV80 + GC.S_NEWL + GC.S_SP04 + 'Sequence string: ' +
               self.sSeq + GC.S_NEWL + 'has list of Pyl indices ' +
               str(self.lIPyl) + GC.S_NEWL +  GC.S_WV80)
        return sIn

    def printNmerDict(self, printSeq=False):
        if printSeq:
            print('The Nmer dictionary of the sequence', self.sSeq, 'is:')
        else:
            print('The Nmer dictionary corresponding to this sequence is:')
        for iPyl, cNmerSeq in self.dNmer.items():
            print('Pyl index ', iPyl, ':', sep='')
            print(cNmerSeq)

    # --- method for creating the Nmer dictionary ----------------------------
    def createNmerDict(self, dITp):
        self.dNmer, iCNmer = {}, dITp['iCentNmer']
        for iPyl in self.lIPyl:
            sNmerSeq = self.sSeq[(iPyl - iCNmer):(iPyl + iCNmer + 1)]
            self.dNmer[iPyl] = NmerSeq(dITp, sSq=sNmerSeq, iPCt=iPyl)

    # --- method for storing the positions of a sequence in the full sequence -
    def getDictPosSeq(self, lSSeq2F):
        dIPosSeq = {}
        for sSeq2F in lSSeq2F:
            lIPos = GF.getLCentPosSSub(self.sSeq, sSub=sSeq2F)
            if len(lIPos) > 0:
                lB = [(1 if iPos in self.lIPyl else 0) for iPos in lIPos]
                nPyl, nOcc = sum(lB), len(lB)
                cPrb = round(nPyl/nOcc, GC.R08)
                dIPosSeq[sSeq2F] = (lIPos, lB, nPyl, nOcc, cPrb)
        return dIPosSeq

    def getDictNmerSeqRmdr(self, dITp, lSSeq2F):
        dNmerRmdr, sRmdr, iCNmer = {}, '', dITp['iCentNmer']
        for sSeq2F in lSSeq2F:
            lIPos = GF.getLCentPosSSub(self.sSeq, sSub=sSeq2F)
            GF.addToDictCt(dNmerRmdr, cK=sSeq2F)
            if len(lIPos) > 0:
                for i in lIPos:
                    if i == lIPos[0]:
                        sRmdr += self.sSeq[:(i - iCNmer)]
                    elif i > lIPos[0] and i < lIPos[-1]:
                        sRmdr += self.sSeq[i:(i + 1)]
                    elif i == lIPos[-1]:
                        sRmdr += self.sSeq[(i + iCNmer + 1):]
            else:
                sRmdr = self.sSeq
            GF.addToDictCt(dNmerRmdr, cK=sRmdr)

# -----------------------------------------------------------------------------
class Timing:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, stT=None, rndDig=GC.R02):
        self.stT = stT
        self.rdDig=rndDig
        self.elT_02_1_getLInpSeq = 0.
        self.elT_02_2_genDLenSeq = 0.
        self.elT_02_3_performLhAnalysis = 0.
        self.elT_02_4_performProbAnalysis_A = 0.
        self.elT_02_5_performProbAnalysis_B = 0.
        self.elT_02_6_performProbAnalysis_C = 0.
        self.elT_02_7_performProbAnalysis_D = 0.
        self.elT_02_8_calcProbTable = 0.
        self.elT_02_9_getD2TotalProbSnip = 0.
        self.elT_02_10_getD2CondProbSnip = 0.
        self.elT_02_11_saveD2TCProbSnipAsDfr = 0.
        self.elT_02_12_getProbSglPos = 0.
        self.elT_05_13_ViterbiAlgorithm = 0.
        self.elT_Sum = 0.
        self.updateLElTimes()
        self.lSMth = ['getLInpSeq', 'genDLenSeq', 'performLhAnalysis',
                      'performProbAnalysis_A', 'performProbAnalysis_B',
                      'performProbAnalysis_C', 'performProbAnalysis_D',
                      'calcProbTable', 'getD2TotalProbSnip',
                      'getD2CondProbSnip', 'saveD2TCProbSnipAsDfr',
                      'getProbSglPos', 'ViterbiAlgorithm']
        assert len(self.lSMth) == len(self.lElT)

    # --- update methods ------------------------------------------------------
    def updateLElTimes(self):
        self.lElT = [self.elT_02_1_getLInpSeq, self.elT_02_2_genDLenSeq,
                     self.elT_02_3_performLhAnalysis,
                     self.elT_02_4_performProbAnalysis_A,
                     self.elT_02_5_performProbAnalysis_B,
                     self.elT_02_6_performProbAnalysis_C,
                     self.elT_02_7_performProbAnalysis_D,
                     self.elT_02_8_calcProbTable,
                     self.elT_02_9_getD2TotalProbSnip,
                     self.elT_02_10_getD2CondProbSnip,
                     self.elT_02_11_saveD2TCProbSnipAsDfr,
                     self.elT_02_12_getProbSglPos,
                     self.elT_05_13_ViterbiAlgorithm]

    def updateTimes(self, iMth=None, stTMth=None, endTMth=None):
        if stTMth is not None and endTMth is not None:
            elT = endTMth - stTMth
            if iMth == 1:
                self.elT_02_1_getLInpSeq += elT
            elif iMth == 2:
                self.elT_02_2_genDLenSeq += elT
            elif iMth == 3:
                self.elT_02_3_performLhAnalysis += elT
            elif iMth == 4:
                self.elT_02_4_performProbAnalysis_A += elT
            elif iMth == 5:
                self.elT_02_5_performProbAnalysis_B += elT
            elif iMth == 6:
                self.elT_02_6_performProbAnalysis_C += elT
            elif iMth == 7:
                self.elT_02_7_performProbAnalysis_D += elT
            elif iMth == 8:
                self.elT_02_8_calcProbTable += elT
            elif iMth == 9:
                self.elT_02_9_getD2TotalProbSnip += elT
            elif iMth == 10:
                self.elT_02_10_getD2CondProbSnip += elT
            elif iMth == 11:
                self.elT_02_11_saveD2TCProbSnipAsDfr += elT
            elif iMth == 12:
                self.elT_02_12_getProbSglPos += elT
            elif iMth == 13:
                self.elT_05_13_ViterbiAlgorithm += elT
            self.elT_Sum += elT
            self.updateLElTimes()

    # --- print methods -------------------------------------------------------
    def __str__(self):
        sIn = (GC.S_WV80 + GC.S_NEWL + GC.S_SP04 + 'Time (s) used in:' +
               GC.S_NEWL + 'Method 1 | "getLInpSeq":\t\t\t' +
               str(round(self.elT_02_1_getLInpSeq, self.rdDig)) + GC.S_NEWL +
               'Method 2 | "genDLenSeq":\t\t\t' +
               str(round(self.elT_02_2_genDLenSeq, self.rdDig)) + GC.S_NEWL +
               'Method 3 | "performLhAnalysis":\t\t' +
               str(round(self.elT_02_3_performLhAnalysis, self.rdDig)) +
               GC.S_NEWL + 'Method 4 | "performProbAnalysis_A":\t' +
               str(round(self.elT_02_4_performProbAnalysis_A, self.rdDig)) +
               GC.S_NEWL + 'Method 5 | "performProbAnalysis_B":\t' +
               str(round(self.elT_02_5_performProbAnalysis_B, self.rdDig)) +
               GC.S_NEWL + 'Method 6 | "performProbAnalysis_C":\t' +
               str(round(self.elT_02_6_performProbAnalysis_C, self.rdDig)) +
               GC.S_NEWL + 'Method 7 | "performProbAnalysis_D":\t' +
               str(round(self.elT_02_7_performProbAnalysis_D, self.rdDig)) +
               GC.S_NEWL + 'Method 8 | "calcProbTable":\t' +
               str(round(self.elT_02_8_calcProbTable, self.rdDig)) +
               GC.S_NEWL + 'Method 9 | "getD2TotalProbSnip":\t' +
               str(round(self.elT_02_9_getD2TotalProbSnip, self.rdDig)) +
               GC.S_NEWL + 'Method 10 | "getD2CondProbSnip":\t' +
               str(round(self.elT_02_10_getD2CondProbSnip, self.rdDig)) +
               GC.S_NEWL + 'Method 11 | "saveD2TCProbSnipAsDfr":\t' +
               str(round(self.elT_02_11_saveD2TCProbSnipAsDfr, self.rdDig)) +
               GC.S_NEWL + 'Method 12 | "getProbSglPos":\t' +
               str(round(self.elT_02_12_getProbSglPos, self.rdDig)) +
               GC.S_NEWL + 'Method 13 | "ViterbiAlgorithm":\t' +
               str(round(self.elT_05_13_ViterbiAlgorithm, self.rdDig)) +
               GC.S_NEWL + GC.S_WV80)
        return sIn

    def printRelTimes(self):
        assert len(self.lSMth) == len(self.lElT)
        if self.elT_Sum > 0:
            print(GC.S_WV80)
            for k, (sMth, cElT) in enumerate(zip(self.lSMth, self.lElT)):
                sX = str(round(cElT/self.elT_Sum*100., self.rdDig)) + '%'
                sWS = GC.S_SPACE*(6 - len(sX))
                print(GC.S_SP04 + sWS + sX + '\t(share of time in Method ' +
                      str(k + 1) + ' | "' + sMth + '")')
            print(GC.S_WV80)

###############################################################################
