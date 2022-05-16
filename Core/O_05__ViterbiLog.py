# -*- coding: utf-8 -*-
###############################################################################
# --- O_05__ViterbiLog.py ----------------------------------------------------
###############################################################################
# import os, time

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
# import Core.F_01__SpcFunctions as SF

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

    # --- print methods -------------------------------------------------------
    def printDfrEmitProb(self):
        print(GC.S_EQ20, 'DataFrame of emission probabilities:')
        print(self.dfrEmitProb)
        print('Index:', self.dfrEmitProb.index.to_list())
        print('Columns:', self.dfrEmitProb.columns.to_list())
        print('Dict:\n', self.dEmitProb, sep='')
        print(GC.S_DS80)

    def printDfrTransProb(self):
        print(GC.S_EQ20, 'DataFrame of transition probabilities:')
        print(self.dfrTransProb)
        print('Index:', self.dfrTransProb.index.to_list())
        print('Columns:', self.dfrTransProb.columns.to_list())
        print('Dict:\n', self.dTransProb, sep='')
        print(GC.S_DS80)

    # --- method printing the results of the Viterbi algorithm ----------------
    def printViterbiRes(self, dR, V):
        for i, dSt in V.items():
            print(GC.S_ST04, 'Position (observation) index', i)
            for st, dPP in dSt.items():
                print(GC.S_DS04, 'State', st)
                lnProb = None
                if dPP[self.dITp['sProb']] is not None:
                    lnProb = round(dPP[self.dITp['sProb']], self.dIG['R08'])
                print(GC.S_SP04, 'ln(prob).:', lnProb,
                      GC.S_VBAR, 'Previous state selected:',
                      dPP[self.dITp['sPrev']])
        print('Optimal state path:', dR['optStPath'])
        print('Maximal probability:', GF.X_exp(dR['maxProb']))
        print('Maximal ln(probability):', dR['maxProb'])

    # --- methods for filling the result paths dictionary ---------------------
    def addValsToDPF(self):
        pPI, pPr, pVi = self.pDirProcInp, self.pDirResProb, self.pDirResViterbi
        sFE, sXtCSV = self.dITp['sFESeqA'], self.dITp['xtCSV']
        self.dPF['EmitProb'] = self.dPF['ResRFreqSP']
        self.dITp['sFInpTransProb'] += sXtCSV
        self.dPF['TransProb'] = GF.joinToPath(pPI, self.dITp['sFInpTransProb'])
        sFE = self.dITp['sUS02'].join([self.dITp['sNmerSeq'], sFE])
        lSF = ['sFOptStatePath', 'sFProbDetail', 'sFProbFinal']
        lSK = ['StatePath', 'ProbDetail', 'ProbFinal']
        for sF, sK in zip(lSF, lSK):
            self.dITp[sF] += self.dITp['sUSC'].join([sF, sFE]) + sXtCSV
            self.dPF[sK] = GF.joinToPath(pVi, self.dITp[sF])

    # --- Viterbi algorithm related functions -------------------------------------
    def iniV(self, lObs=[]):
        assert len(lObs) > 0 and len(self.dITp['lStates']) > 0
        V, i = {}, 0
        for st in self.dITp['lStates']:
            cProb = GF.X_lnProd(self.dITp['startPr'][st],
                                self.dEmitProb[st][lObs[i]])
            dData = {self.dITp['sProb']: cProb,
                     self.dITp['sPrev']: None}
            GF.addToDictD(V, cKMain=i, cKSub=st, cVSub=dData)
        return V

    def getTransProb(self, V, i, cSt, prevSt):
        return GF.X_sum(V[i - 1][prevSt][self.dITp['sProb']],
                        GF.X_ln(self.dTransProb[prevSt][cSt]))

    def fillV(self, V, iLObs=0, lObs=[]):
        st0 = self.dITp['st0']
        for i in self.dITp['dIPos'][iLObs][1:]:
            for st in self.dITp['lStates']:
                maxTransProb = self.getTransProb(V, i, cSt=st, prevSt=st0)
                prevStSel = st0
                for prevSt in self.dITp['lStates'][1:]:
                    transProb = self.getTransProb(V, i, cSt=st, prevSt=prevSt)
                    if GF.X_greater(transProb, maxTransProb):
                        maxTransProb = transProb
                        prevStSel = prevSt
                maxProb = GF.X_sum(maxTransProb,
                                   GF.X_ln(self.dEmitProb[st][lObs[i]]))
                dData = {self.dITp['sProb']: maxProb,
                         self.dITp['sPrev']: prevStSel}
                GF.addToDictD(V, cKMain=i, cKSub=st, cVSub=dData)

    def getMostProbStWBacktrack(self, V, iLObs=0):
        optStPath, maxProb, bestSt = [], None, None
        iLast = self.dITp['dIPos'][iLObs][-1]
        for st, dData in V[iLast].items():
            if GF.X_greater(dData[self.dITp['sProb']], maxProb):
                maxProb = dData[self.dITp['sProb']]
                bestSt = st
        optStPath.append(bestSt)
        return {'optStPath': optStPath,
                'maxProb': maxProb,
                'previousSt': bestSt}

    def followBacktrackTo1stObs(self, dR, V):
        for i in range(len(V) - 2, -1, -1):
            previousSt = V[i + 1][dR['previousSt']][self.dITp['sPrev']]
            dR['optStPath'].insert(0, previousSt)
            dR['previousSt'] = previousSt

    def ViterbiCore(self, iLObs=0, lObs=[]):
        V = self.iniV(lObs=lObs)
        # run ViterbiCore for t > 0
        self.fillV(V, iLObs=iLObs, lObs=lObs)
        # Get most probable state and its backtrack
        dR = self.getMostProbStWBacktrack(V, iLObs=iLObs)
        # Follow the backtrack till the first observation
        self.followBacktrackTo1stObs(dR, V)
        return dR, V

    def runViterbiAlgorithm(self):
        if self.dITp['doViterbi']:
            print(GC.S_EQ80, GC.S_NEWL, GC.S_DS30, ' Viterbi Algorithm ',
                  GC.S_DS30, GC.S_NEWL, sep='')
            for iLObs, lObs in self.dITp['dObs'].items():
                print(GC.S_DS80, GC.S_NEWL, GC.S_EQ20, ' Observations ', iLObs,
                      GC.S_COL, sep='')
                dRes, V = self.ViterbiCore(iLObs=iLObs, lObs=lObs)
                self.printViterbiRes(dR=dRes, V=V)
            print(GC.S_DS80, GC.S_NEWL, GC.S_DS30, ' DONE ', GC.S_DS36, sep='')

    # --- methods for loading and saving data ---------------------------------
    def loadInpData(self):
        pFE, pFT = self.dPF['EmitProb'], self.dPF['TransProb']
        self.dfrEmitProb = self.loadData(pF=pFE, iC=0, NAfillV=0.).T
        self.dfrTransProb = self.loadData(pF=pFT, iC=0, NAfillV=0.)
        self.dEmitProb = self.dfrEmitProb.T.to_dict(orient='dict')
        self.dTransProb = self.dfrTransProb.T.to_dict(orient='dict')

###############################################################################
