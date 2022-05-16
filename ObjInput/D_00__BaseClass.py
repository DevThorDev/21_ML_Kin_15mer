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
sEff = GC.S_EFF
sEffF = GC.S_EFF_F
sNmer = GC.S_N_MER
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
sCapXS, sCapS, sCapM, sCapL = GC.S_CAP_XS, GC.S_CAP_S, GC.S_CAP_M, GC.S_CAP_L
sNmerSeq = GC.S_N_MER_SEQ
sFullSeq = GC.S_FULL_SEQ

# --- file name extensions ----------------------------------------------------
xtPY = GC.XT_PY
xtCSV = GC.XT_CSV
xtPDF = GC.XT_PDF
xtPTH = GC.XT_PTH
xtBIN = GC.XT_BIN

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
       'sEff': sEff,
       'sEffF': sEffF,
       'sNmer': sNmer,
       'sImer': sImer,
       'sImerTrain': sImerTrain,
       'sImerTest': sImerTest,
       'sIEff': sIEff,
       'sIEffTrain': sIEffTrain,
       'sIEffTest': sIEffTest,
       'sIEffF': sIEffF,
       'sIEffFTrain': sIEffFTrain,
       'sIEffFTest': sIEffFTest,
       'sIGen': sIGen,
       'sCapXS': sCapXS,
       'sCapS': sCapS,
       'sCapM': sCapM,
       'sCapL': sCapL,
       'sNmerSeq': sNmerSeq,
       'sFullSeq': sFullSeq,
       # --- file name extensions
       'xtPY': xtPY,
       'xtCSV': xtCSV,
       'xtPDF': xtPDF,
       'xtPTH': xtPTH,
       'xtBIN': xtBIN}

###############################################################################
