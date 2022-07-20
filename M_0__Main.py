# -*- coding: utf-8 -*-
###############################################################################
# --- M_0__Main.py ------------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Control.A_00__GenInput import dictInpG
from Core.C_00__GenConstants import S_OBJINP
from Core.I_01__InpData import InputData
from Core.O_00__BaseClass import Timing
# from Core.O_01__ExpData import ExpData
# from Core.O_02__SeqAnalysis import SeqAnalysis
# from Core.O_03__Validation import Validation
# from Core.O_05__ViterbiLog import ViterbiLog
from Core.O_06__ClfDataLoader import DataLoader
# from Core.O_07__Classifier import RndForestClf, NNMLPClf, PropCalculator
from Core.O_07__Classifier import PropCalculator
from Core.O_80__Looper import Looper

# ### MAIN ####################################################################
startTime = GF.startSimu()
cTiming = Timing(stT=startTime)
cStTime = GF.showElapsedTime(startTime)

# -----------------------------------------------------------------------------
sFMain = ' M_0__Main.py '
print(GC.S_EQ80, GC.S_NEWL, GC.S_DS33, sFMain, GC.S_DS33, GC.S_NEWL, sep='')
inpDatG = InputData(dictInpG)
inpDatG.addObjTps(S_OBJINP)
print(GC.S_DS29, 'Added object types.', GC.S_DS30)

GF.printMode(inpDatG.dI['isTest'])
print(GC.S_EQ29, 'Starting analysis...', GC.S_EQ29)

# START - Loading and processing experimental data
# cData = ExpData(inpDatG)
# print(cData)
# GF.showElapsedTime(startTime)
# cData.procExpData()
# GF.showElapsedTime(startTime)
# cData.getInfoKinNmer(stT=startTime, tpDfr=GC.S_BASE)
# GF.showElapsedTime(startTime)
# cData.printDIG()
# cData.printDITp()
# cData.printInpDfrs()
# END - Loading and processing experimental data

# START - Sequence Analysis
# cSeqAnalysis = SeqAnalysis(inpDatG)
# GF.showElapsedTime(startTime)

# cSeqAnalysis.performLhAnalysis(lEff=[None, 'AT1G01140', 'AT4G23650'])
# cSeqAnalysis.performLhAnalysis(cTim=cTiming, lEff=[None], stT=startTime)

# cSeqAnalysis.performProbAnalysis(cTim=cTiming, lEff=[None], stT=startTime)
# cSeqAnalysis.calcProbTable(cTim=cTiming, stT=startTime)

# cSeqAnalysis.performTCProbAnalysis(cTim=cTiming, lEff=[None], stT=startTime)

# cSeqAnalysis.getProbSglPos(cTim=cTiming, lEff=[None], stT=startTime)
# END - Sequence Analysis

# START - Validation
# cValidation = Validation(inpDatG)
# cValidation.createResultsTrain(stT=startTime)
# cValidation.printTestObj(printDfrComb=True)
# END - Validation

# START - Viterbi algorithm (Hidden Markov Model)
# cViterbiAlg = ViterbiLog(inpDatG)
# cViterbiAlg.printDictPaths()
# cViterbiAlg.printDfrEmitProb()
# cViterbiAlg.printDfrStartProb()
# cViterbiAlg.printDfrTransProb()
# cViterbiAlg.runViterbiAlgorithm(cTim=cTiming, stT=startTime)
# END - Viterbi algorithm (Hidden Markov Model)

# cStTime = GF.showElapsedTime(startTime)
# cRFClf = RndForestClf(inpDatG)
# print(GC.S_EQ20, 'Fit quality of Random Forest Classifier:')
# cRFClf.ClfPred()
# cEndTime = GF.showElapsedTime(startTime)
# cTiming.updateTimes(iMth=15, stTMth=cStTime, endTMth=cEndTime)

# cStTime = GF.showElapsedTime(startTime)
# cMLPClf = NNMLPClf(inpDatG)
# print(GC.S_EQ20, 'Fit quality of NN MLP Classifier:')
# cMLPClf.ClfPred()
# cEndTime = GF.showElapsedTime(startTime)
# cTiming.updateTimes(iMth=16, stTMth=cStTime, endTMth=cEndTime)

# for cClf in [cRFClf, cMLPClf]:
#     cClf.printFitQuality()
#     cClf.plotConfMatrix()

# for cClf in [cRFClf, cMLPClf]:
#     print(cClf.descO, GC.S_VBAR, cClf.sMth)
#     cClf.calcPrintResPredict(X2Pre=cClf.getXY(getTrain=False)[0])

# Test parameter sets
# for sKeyPar in inpDatG.dI['d2Par_NNMLP']:
# for sKeyPar in list('ABC'):
#     print(GC.S_ST80, GC.S_NEWL, GC.S_EQ20, ' Parameter set ', sKeyPar, sep='')
#     cStTime = GF.showElapsedTime(startTime)
#     cMLPClf = NNMLPClf(inpDatG, sKPar=sKeyPar)
#     cMLPClf.ClfPred()
#     cMLPClf.printFitQuality()
#     # cMLPClf.calcPrintResPredict(X2Pre=cMLPClf.getXY(getTrain=False)[0])
#     cEndTime = GF.showElapsedTime(startTime)
#     cTiming.updateTimes(iMth=16, stTMth=cStTime, endTMth=cEndTime)

cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(iMth=99, stTMth=cStTime, endTMth=cEndTime)

cStTime = GF.showElapsedTime(startTime)
cDataLoader = DataLoader(inpDatG)
cDataLoader.printDictNmerNoCl()
# cDataLoader.printXY(sMd=GC.S_CLF)
# cDataLoader.printXY(sMd=GC.S_PRC)
# cDataLoader.printSerNmerSeqClf()
# cDataLoader.printSerNmerSeqPrC()
# cDataLoader.printDfrInpClf()
# cDataLoader.printDfrInpPrC()
# cDataLoader.printLSClClf()
# cDataLoader.printLSClPrC()
cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(iMth=14, stTMth=cStTime, endTMth=cEndTime)

cLooper = Looper(inpDatG, D=cDataLoader)
cLooper.doDoubleLoop(cTim=cTiming, stT=startTime)

cStTime = GF.showElapsedTime(startTime)
cPropCalc = PropCalculator(inpDatG, D=cDataLoader)
cPropCalc.calcPropAAc()
cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(iMth=19, stTMth=cStTime, endTMth=cEndTime)

GF.printMode(inpDatG.dI['isTest'])
print(cTiming)
cTiming.printRelTimes()

# -----------------------------------------------------------------------------
GF.endSimu(startTime)
###############################################################################