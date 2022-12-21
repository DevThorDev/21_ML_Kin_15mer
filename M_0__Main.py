# -*- coding: utf-8 -*-
###############################################################################
# --- M_0__Main.py ------------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Control.A_00__GenInput import dictInpG
from Core.C_00__GenConstants import S_OBJINP
from Core.I_01__InpData import InputData
# from Core.O_01__ExpData import ExpData
# from Core.O_02__SeqAnalysis import SeqAnalysis
# from Core.O_03__Validation import Validation
# from Core.O_05__ViterbiLog import ViterbiLog
from Core.O_06__ClfDataLoader import DataLoader
# from Core.O_07__Classifier import RFClf, MLPClf, PropCalculator
from Core.O_07__Classifier import PropCalculator
from Core.O_80__Looper import Looper
from Core.O_90__Evaluator import Evaluator
from Core.O_95__Timing import Timing

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

cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(tMth=(None, 9999), stTMth=cStTime, endTMth=cEndTime)

cStTime = GF.showElapsedTime(startTime)
cDataLoader = DataLoader(inpDatG)
cDataLoader.printDictNmerNoCl()

# cDataLoader.printXY(sMd=GC.S_CLF)
# cDataLoader.printXY(sMd=GC.S_PRC)
# cDataLoader.printSerNmerSeqClf()
# cDataLoader.printSerNmerSeqPrC()
# cDataLoader.printDfrInpClf()
# cDataLoader.printDfrInpPrC()
# cDataLoader.printlSXClClf()
# cDataLoader.printlSXClPrC()

cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(tMth=(6, 1), stTMth=cStTime, endTMth=cEndTime)

cLooper = Looper(inpDatG, D=cDataLoader)
cLooper.doQuadLoop(cTim=cTiming, stT=startTime)

cStTime = GF.showElapsedTime(startTime)
cPropCalc = PropCalculator(inpDatG, D=cDataLoader)
cPropCalc.calcPropAAc()
cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(tMth=(7, 1000), stTMth=cStTime, endTMth=cEndTime)

cStTime = GF.showElapsedTime(startTime)
cEval = Evaluator(inpDatG)

# tK1 = (GC.S_SMPL_RND_U_S, GC.S_A)
# tK1 = (GC.S_SMPL_RND_U_S,)
tK1 = (GC.S_FULL_FIT_S, GC.S_SMPL_NO_S)
tK2 = (GC.S_FULL_FIT_S, GC.S_SMPL_RND_U_S)
tK3 = (GC.S_PART_FIT_S + str(100), GC.S_SMPL_RND_U_S)
dMF = {tK1: [GC.S_MTH_ADA, GC.S_MTH_RF, GC.S_MTH_X_TR,
             GC.S_MTH_GR_B, GC.S_MTH_H_GR_B,
             GC.S_MTH_PA_A, GC.S_MTH_PCT, GC.S_MTH_SGD,
             GC.S_MTH_CT_NB, GC.S_MTH_CP_NB, GC.S_MTH_GS_NB,
             GC.S_MTH_MLP, GC.S_MTH_LSV],
       tK2: [GC.S_MTH_ADA, GC.S_MTH_RF, GC.S_MTH_X_TR,
             GC.S_MTH_GR_B, GC.S_MTH_H_GR_B,
             GC.S_MTH_GP,
             GC.S_MTH_PA_A, GC.S_MTH_PCT, GC.S_MTH_SGD,
             GC.S_MTH_CT_NB, GC.S_MTH_CP_NB, GC.S_MTH_GS_NB,
             GC.S_MTH_MLP, GC.S_MTH_LSV, GC.S_MTH_NSV],
       tK3: [GC.S_MTH_PA_A, GC.S_MTH_PCT, GC.S_MTH_SGD,
             GC.S_MTH_CT_NB, GC.S_MTH_CP_NB, GC.S_MTH_GS_NB,
             GC.S_MTH_MLP]}

cEval.printDDfrCmb(tK=tK1)
cEval.calcPredClassRes(dMthFlt=dMF)
# cEval.printDfrCl()
cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(tMth=(90, 1), stTMth=cStTime, endTMth=cEndTime)

GF.printMode(inpDatG.dI['isTest'])
print(cTiming)
cTiming.printRelTimes()

# -----------------------------------------------------------------------------
GF.endSimu(startTime)
###############################################################################