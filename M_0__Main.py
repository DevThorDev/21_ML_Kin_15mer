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
from Core.O_01__ExpData import ExpData
from Core.O_02__SeqAnalysis import SeqAnalysis
from Core.O_03__Validation import Validation
from Core.O_05__ViterbiLog import ViterbiLog
from Core.O_07__Classifier import RndForestClf, NNMLPClf, PropCalculator

# ### MAIN ####################################################################
startTime = GF.startSimu()
cTiming = Timing(stT=startTime)
# -----------------------------------------------------------------------------
sFMain = ' M_0__Main.py '
print(GC.S_EQ80, GC.S_NEWL, GC.S_DS33, sFMain, GC.S_DS33, GC.S_NEWL, sep='')
inpDatG = InputData(dictInpG)
inpDatG.addObjTps(S_OBJINP)
print(GC.S_DS29, 'Added object types.', GC.S_DS30)

GF.printMode(inpDatG.dI['isTest'])
print(GC.S_EQ29, 'Starting analysis...', GC.S_EQ29)
cData = ExpData(inpDatG)
print(cData)
GF.showElapsedTime(startTime)
cData.procExpData()
GF.showElapsedTime(startTime)
cData.getInfoKinNmer(stT=startTime, tpDfr=GC.S_BASE)
GF.showElapsedTime(startTime)
# cData.printDIG()
# cData.printDITp()
# cData.printInpDfrs()
cSeqAnalysis = SeqAnalysis(inpDatG)
GF.showElapsedTime(startTime)

# cSeqAnalysis.performLhAnalysis(lEff=[None, 'AT1G01140', 'AT4G23650'])
cSeqAnalysis.performLhAnalysis(cTim=cTiming, lEff=[None], stT=startTime)

cSeqAnalysis.performProbAnalysis(cTim=cTiming, lEff=[None], stT=startTime)
# cSeqAnalysis.calcProbTable(cTim=cTiming, stT=startTime)

cSeqAnalysis.performTCProbAnalysis(cTim=cTiming, lEff=[None], stT=startTime)

cSeqAnalysis.getProbSglPos(cTim=cTiming, lEff=[None], stT=startTime)

cValidation = Validation(inpDatG)
cValidation.createResultsTrain(stT=startTime)
# cValidation.printTestObj(printDfrComb=True)

# cViterbiAlg = ViterbiLog(inpDatG)
# cViterbiAlg.printDictPaths()
# cViterbiAlg.printDfrEmitProb()
# cViterbiAlg.printDfrStartProb()
# cViterbiAlg.printDfrTransProb()
# cViterbiAlg.runViterbiAlgorithm(cTim=cTiming, stT=startTime)

cStTime = GF.showElapsedTime(startTime)
cRFClf = RndForestClf(inpDatG)
# cRFClf.ClfPred(dat2Pre=[list('ACDADA'), list('BDBBCA'), list('CBDCCD'),
#                         list('DDBADC'), list('CDCDAA'), list('AAACCC'),
#                         list('CACCDB'), list('CACCDD'), list('DCBBDB'),
#                         list('DCABDB')])
print(GC.S_EQ20, 'Fit quality of Random Forest Classifier:')
cRFClf.ClfPred()
cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(iMth=14, stTMth=cStTime, endTMth=cEndTime)

cStTime = GF.showElapsedTime(startTime)
cMLPClf = NNMLPClf(inpDatG)
# cMLPClf.ClfPred(dat2Pre=[list('ACDADA'), list('BDBBCA'), list('CBDCCD'),
#                          list('DDBADC'), list('CDCDAA'), list('AAACCC'),
#                          list('CACCDB'), list('CACCDD'), list('DCBBDB'),
#                          list('DCABDB')])
print(GC.S_EQ20, 'Fit quality of NN MLP Classifier:')
cMLPClf.ClfPred()
cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(iMth=15, stTMth=cStTime, endTMth=cEndTime)

cStTime = GF.showElapsedTime(startTime)
cPropCalc = PropCalculator(inpDatG)
cPropCalc.calcPropAAc()
cEndTime = GF.showElapsedTime(startTime)
cTiming.updateTimes(iMth=16, stTMth=cStTime, endTMth=cEndTime)

for cClf in [cRFClf, cMLPClf]:
    cClf.printFitQuality()
    cClf.plotConfMatrix()

GF.printMode(inpDatG.dI['isTest'])
print(cTiming)
cTiming.printRelTimes()

# -----------------------------------------------------------------------------
GF.endSimu(startTime)
###############################################################################
