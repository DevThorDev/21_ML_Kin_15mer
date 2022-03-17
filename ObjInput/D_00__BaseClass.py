# -*- coding: utf-8 -*-
###############################################################################
# --- D_00__BaseClass.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Base class (D_00__BaseClass)'
sNmSpec = 'Data for the BaseClass class in O_00__BaseClass'

# --- data specific -----------------------------------------------------------
cSep = GC.S_SEMICOL

# --- names and paths of files and dirs ---------------------------------------

# --- strings -----------------------------------------------------------------
sBase = GC.S_BASE
sTrain = GC.S_TRAIN
sTest = GC.S_TEST
sCombined = GC.S_COMBINED
sCombinedInp = GC.S_COMBINED_INP
sCombinedOut = GC.S_COMBINED_OUT
sCDfrComb = GC.S_C_DFR_COMB
sBGenInfoNmer = GC.S_B_GEN_INFO_N_MER
sBGenInfoEff = GC.S_B_GEN_INFO_EFF
sNmer = GC.S_N_MER
sEff = GC.S_EFF
sEffF = GC.S_EFF_F
sImer = GC.S_I_MER
sImerTrain = GC.S_I_MER_TRAIN
sImerTest = GC.S_I_MER_TEST
sIEff = GC.S_I_EFF
sIEffTrain = GC.S_I_EFF_TRAIN
sIEffTest = GC.S_I_EFF_TEST
sIEffF = GC.S_I_EFF_F
sIEffFTrain = GC.S_I_EFF_F_TRAIN
sIEffFTest = GC.S_I_EFF_F_TEST
sIGen = GC.S_I_GEN

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- data specific
       'cSep': cSep,
       # --- names and paths of files and dirs
       # --- strings
       'sBase': sBase,
       'sTrain': sTrain,
       'sTest': sTest,
       'sCombined': sCombined,
       'sCombinedInp': sCombinedInp,
       'sCombinedOut': sCombinedOut,
       'sCDfrComb': sCDfrComb,
       'sBGenInfoNmer': sBGenInfoNmer,
       'sBGenInfoEff': sBGenInfoEff,
       'sNmer': sNmer,
       'sEff': sEff,
       'sEffF': sEffF,
       'sImer': sImer,
       'sImerTrain': sImerTrain,
       'sImerTest': sImerTest,
       'sIEff': sIEff,
       'sIEffTrain': sIEffTrain,
       'sIEffTest': sIEffTest,
       'sIEffF': sIEffF,
       'sIEffFTrain': sIEffFTrain,
       'sIEffFTest': sIEffFTest,
       'sIGen': sIGen}

###############################################################################
