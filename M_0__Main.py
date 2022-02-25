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
cData.getInfoKin15mer()
GF.showElapsedTime(startTime)
# cData.printDIG()
# cData.printDITp()
cData.printInpDfrs()

# cPlotter = Plotter(inDG, calcDfrs=True)
GF.printMode(inpDatG.dI['isTest'])

# -----------------------------------------------------------------------------
GF.endSimu(startTime)
###############################################################################
