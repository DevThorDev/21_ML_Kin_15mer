# -*- coding: utf-8 -*-
###############################################################################
# --- O_05__ViterbiLog.py -----------------------------------------------------
###############################################################################
import time

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Core.O_02__SeqAnalysis import SeqAnalysis

# -----------------------------------------------------------------------------
class ViterbiLog(SeqAnalysis):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=5, lITpUpd=[1, 2]):
        super().__init__(inpDat)
        self.idO = 'O_05'
        self.descO = 'Viterbi algoritm for sequences [using log(prob.)]'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.addValsToDPF()
        self.loadInpData()
        print('Initiated "ViterbiLog" base object.')

    # --- methods for loading input data --------------------------------------
    def loadInpData(self):
        sStPr, pFE = self.dITp['sStartProb'], self.dPF['EmitProb']
        pFS, pFT = self.dPF['StartProb'], self.dPF['TransProb']
        self.dfrFSeqInp = self.loadData(pF=self.dPF['FullSeq'], iC=0)
        self.dfrEmitProb = self.loadData(pF=pFE, iC=0, NAfillV=0.).T
        self.dfrStartProb = self.loadData(pF=pFS, iC=0, NAfillV=0.)
        self.dfrTransProb = self.loadData(pF=pFT, iC=0, NAfillV=0.)
        self.dEmitProb = self.dfrEmitProb.T.to_dict(orient='dict')
        self.dStartProb = self.dfrStartProb.to_dict(orient='dict')[sStPr]
        self.dTransProb = self.dfrTransProb.T.to_dict(orient='dict')
        self.dFSeq, self.dStatePath, self.dPosPyl = {}, {}, {}
        self.d3ProbDetail, self.dProbFinal, self.dRV = {}, {}, {}
        self.lStates = list(self.dStartProb)
        if self.dITp['sCCodeSeq'] in self.dfrFSeqInp.columns:
            arrFSeqUnq = self.dfrFSeqInp[self.dITp['sCCodeSeq']].unique()
            for k, sFSeq in enumerate(arrFSeqUnq):
                self.dFSeq[k] = list(sFSeq)

    # --- print methods -------------------------------------------------------
    def printDfrEmitProb(self):
        print(GC.S_EQ20, 'DataFrame of emission probabilities:')
        print(self.dfrEmitProb)
        print('Index:', self.dfrEmitProb.index.to_list())
        print('Columns:', self.dfrEmitProb.columns.to_list())
        print('Dict:\n', self.dEmitProb, sep='')
        print(GC.S_DS80)

    def printDfrStartProb(self):
        print(GC.S_EQ20, 'DataFrame of start probabilities:')
        print(self.dfrStartProb)
        print('Index:', self.dfrStartProb.index.to_list())
        print('Columns:', self.dfrStartProb.columns.to_list())
        print('Dict:\n', self.dStartProb, sep='')
        print(GC.S_DS80)

    def printDfrTransProb(self):
        print(GC.S_EQ20, 'DataFrame of transition probabilities:')
        print(self.dfrTransProb)
        print('Index:', self.dfrTransProb.index.to_list())
        print('Columns:', self.dfrTransProb.columns.to_list())
        print('Dict:\n', self.dTransProb, sep='')
        print(GC.S_DS80)

    # --- method printing the results of the Viterbi algorithm ----------------
    def printViterbiDetailedRes(self):
        for i, dSt in self.V.items():
            print(GC.S_ST04, 'Position (observation) index', i)
            for st, dPP in dSt.items():
                print(GC.S_DS04, 'State', st)
                lnProb = None
                if dPP[self.dITp['sProb']] is not None:
                    lnProb = round(dPP[self.dITp['sProb']], self.dIG['R08'])
                print(GC.S_SP04, 'ln(prob).:', lnProb,
                      GC.S_VBAR, 'Previous state selected:',
                      dPP[self.dITp['sPrev']])
        for sFSeq in self.dFSeq.values():
            print('Optimal state path:', self.dRV[sFSeq]['optStPath'])
            print('Maximal probability:', GF.X_exp(self.dRV[sFSeq]['maxProb']))
            print('Maximal ln(probability):', self.dRV[sFSeq]['maxProb'])

    # --- methods for filling the result paths dictionary ---------------------
    def addValsToDPF(self):
        pPI, pVi = self.pDirProcInp, self.pDirResViterbi
        sFE, sXt = self.dITp['sFESeqA'], self.dIG['xtCSV']
        self.dPF['EmitProb'] = self.dPF['ResRFreqSP']
        p = pPI
        if self.dITp['useFullSeqFrom'] == GC.S_COMB_INP:
            p = self.pDirResComb
        self.dPF['FullSeq'] = GF.joinToPath(p, self.dITp['sFInpFullSeq'] + sXt)
        for sF, sK in zip(['sFInpStartProb', 'sFInpTransProb'],
                          ['StartProb', 'TransProb']):
            self.dPF[sK] = GF.joinToPath(pPI, self.dITp[sF] + sXt)
        sFE = self.dITp['sUS02'].join([self.dITp['sNmerSeq'], sFE])
        lSF = ['sFOptStatePath', 'sFPosPyl', 'sFProbDetail', 'sFProbFinal']
        lSK = ['StatePath', 'PosPyl', 'ProbDetail', 'ProbFinal']
        for sF, sK in zip(lSF, lSK):
            self.dITp[sF] = self.dITp['sUSC'].join([self.dITp[sF], sFE]) + sXt
            self.dPF[sK] = GF.joinToPath(pVi, self.dITp[sF])

    # --- Viterbi algorithm related functions -------------------------------------
    def iniV(self, lObs=[]):
        assert len(lObs) > 0 and len(self.lStates) > 0
        self.V, i = {}, 0
        for st in self.lStates:
            cProb = GF.X_lnProd(self.dStartProb[st],
                                self.dEmitProb[st][lObs[i]])
            dData = {self.dITp['sProb']: cProb,
                     self.dITp['sPrev']: None}
            GF.addToDictD(self.V, cKMain=i, cKSub=st, cVSub=dData)

    def getTransProb(self, j, cSt, prevSt):
        return GF.X_sum(self.V[j][prevSt][self.dITp['sProb']],
                        GF.X_ln(self.dTransProb[prevSt][cSt]))

    def fillV(self, iFSeq=0, lObs=[]):
        st0 = self.lStates[0]
        for j, sAAc in enumerate(self.dFSeq[iFSeq][1:]):
            for st in self.lStates:
                maxTransProb = self.getTransProb(j, cSt=st, prevSt=st0)
                prevStSel = st0
                for prevSt in self.lStates[1:]:
                    transProb = self.getTransProb(j, cSt=st, prevSt=prevSt)
                    if GF.X_greater(transProb, maxTransProb):
                        maxTransProb = transProb
                        prevStSel = prevSt
                maxProb = GF.X_sum(maxTransProb,
                                   GF.X_ln(self.dEmitProb[st][sAAc]))
                dData = {self.dITp['sProb']: maxProb,
                         self.dITp['sPrev']: prevStSel}
                GF.addToDictD(self.V, cKMain=(j+1), cKSub=st, cVSub=dData)

    def getMostProbStWBacktrack(self, iFSeq=0, sFSeq=''):
        pOptSt, maxProb, bestSt = [], None, None
        # iLast = self.dITp['dIPos'][iFSeq][-1]
        iLast = len(self.dFSeq[iFSeq]) - 1
        for st, dData in self.V[iLast].items():
            if GF.X_greater(dData[self.dITp['sProb']], maxProb):
                maxProb = dData[self.dITp['sProb']]
                bestSt = st
        pOptSt.append(bestSt)
        GF.addToDictD(self.dRV, cKMain=sFSeq, cKSub='optStPath', cVSub=pOptSt)
        GF.addToDictD(self.dRV, cKMain=sFSeq, cKSub='maxProb', cVSub=maxProb)
        GF.addToDictD(self.dRV, cKMain=sFSeq, cKSub='prevSt', cVSub=bestSt)

    def followBacktrackTo1stObs(self, sFSeq=''):
        dRVSeq = self.dRV[sFSeq]
        for i in range(len(self.V) - 2, -1, -1):
            prevSt = self.V[i + 1][dRVSeq['prevSt']][self.dITp['sPrev']]
            dRVSeq['optStPath'].insert(0, prevSt)
            dRVSeq['prevSt'] = prevSt

    def ViterbiCore(self, iFSeq=0, lObs=[]):
        self.iniV(lObs=lObs)
        # run ViterbiCore for t > 0
        self.fillV(iFSeq=iFSeq, lObs=lObs)
        # Get most probable state and its backtrack
        sFullSeq = ''.join(lObs)
        self.getMostProbStWBacktrack(iFSeq=iFSeq, sFSeq=sFullSeq)
        # Follow the backtrack till the first observation
        self.followBacktrackTo1stObs(sFSeq=sFullSeq)
        return sFullSeq

    def runViterbiAlgorithm(self, cTim, stT=None):
        if self.dITp['doViterbi']:
            cT = time.time()
            print(GC.S_EQ80, GC.S_NEWL, GC.S_DS30, ' Viterbi Algorithm ',
                  GC.S_DS31, GC.S_NEWL, sep='')
            nFSeq, sFS = len(self.dFSeq), 'full sequences [ViterbiAlgorithm]'
            for k, (iFSeq, lAAcObs) in enumerate(self.dFSeq.items()):
                sFSeq = self.ViterbiCore(iFSeq=iFSeq, lObs=lAAcObs)
                # self.printViterbiDetailedRes()
                lO = [(s if s != self.dITp['sNotInNmer'] else
                       self.dITp['sX']) for s in self.dRV[sFSeq]['optStPath']]
                lIPyl = [i for i in range(len(lO)) if lO[i] == self.dITp['s0']]
                self.dStatePath[sFSeq], self.dPosPyl[sFSeq] = lO, sorted(lIPyl)
                for i, dSub in self.V.items():
                    for st, dData in dSub.items():
                        GF.addToD3(self.d3ProbDetail, cKL1=iFSeq, cKL2=i,
                                   cKL3=st, cVL3=dData[self.dITp['sProb']])
                GF.showProgress(N=nFSeq, n=k, modeDisp=self.dITp['mDsp'],
                                varText=sFS, startTime=stT)
            self.saveViterbiResData()
            cTim.updateTimes(tMth=(5, 1), stTMth=cT, endTMth=time.time())

    # --- methods for saving data ---------------------------------------------
    def saveViterbiResData(self):
        pFStP, pFPPy = self.dPF['StatePath'], self.dPF['PosPyl']
        pFPrD, pFPrF = self.dPF['ProbDetail'], self.dPF['ProbFinal']
        self.saveData(GF.iniDfrFromDictIt(self.dStatePath), pF=pFStP)
        self.saveData(GF.iniDfrFromDictIt(self.dPosPyl), pF=pFPPy)
        lC = [self.dITp['sFullSeq'], 'Index', self.dITp['sState'],
              self.dITp['sLnProb']]
        self.saveData(GF.iniDfrFromD3(self.d3ProbDetail, colDfr=lC), pF=pFPrD)
        lVProb = [GF.X_exp(self.dRV[sFSeq]['maxProb']) for sFSeq in self.dRV]
        lVLnProb = [self.dRV[sFSeq]['maxProb'] for sFSeq in self.dRV]
        dProbFin = {self.dITp['sFullSeq']: list(self.dRV),
                    self.dITp['sProb']: lVProb,
                    self.dITp['sLnProb']: lVLnProb}
        self.saveData(GF.iniDfrFromDictIt(dProbFin), pF=pFPrF)


###############################################################################
