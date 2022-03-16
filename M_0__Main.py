# -*- coding: utf-8 -*-
###############################################################################
# --- M_0__Main.py ------------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Control.A_00__GenInput import dictInpG
from Core.C_00__GenConstants import S_OBJINP
from Core.I_01__InpData import InputData
from Core.O_01__ExpData import ExpData
from Core.O_02__SeqAnalysis import SeqAnalysis
from Core.O_03__Validation import Validation

# ### MAIN ####################################################################
startTime = GF.startSimu()
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
cData.getInfoKinNmer(stT=startTime, sIEff=GC.S_I_EFF, sIEffF=GC.S_I_EFF_F)
GF.showElapsedTime(startTime)
# cData.printDIG()
# cData.printDITp()
# cData.printInpDfrs()
cSeqAnalysis = SeqAnalysis(inpDatG)
GF.showElapsedTime(startTime)
cSeqAnalysis.performAnalysis(lEff=[None, 'AT1G01140', 'AT4G23650'])
GF.showElapsedTime(startTime)
cValidation = Validation(inpDatG)
cValidation.splitInpData()
cValidation.printTestObj(printDfrComb=True)
cValidation.createResultsTrain(stT=startTime)

# cPlotter = Plotter(inDG, calcDfrs=True)
GF.printMode(inpDatG.dI['isTest'])

# -----------------------------------------------------------------------------
GF.endSimu(startTime)
###############################################################################
