# -*- coding: utf-8 -*-
###############################################################################
# --- D_90__Evaluator.py ------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
# import Core.F_00__GenFunctions as GF

# --- general -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sMthNone = GC.S_MTH_NONE
sMthDummy = GC.S_MTH_DUMMY
sMthAda = GC.S_MTH_ADA
sMthRF = GC.S_MTH_RF
sMthXTr = GC.S_MTH_X_TR
sMthGrB = GC.S_MTH_GR_B
sMthHGrB = GC.S_MTH_H_GR_B
sMthGP = GC.S_MTH_GP
sMthPaA = GC.S_MTH_PA_A
sMthPct = GC.S_MTH_PCT
sMthSGD = GC.S_MTH_SGD
sMthCtNB = GC.S_MTH_CT_NB
sMthCpNB = GC.S_MTH_CP_NB
sMthGsNB = GC.S_MTH_GS_NB
sMthMLP = GC.S_MTH_MLP
sMthLSV = GC.S_MTH_LSV
sMthNSV = GC.S_MTH_NSV

# --- flow control (general) --------------------------------------------------
lSMthEvalCompositeClf = [sMthDummy, sMthAda, sMthRF, sMthXTr, sMthPaA, sMthPct,
                         sMthCtNB, sMthCpNB, sMthGsNB, sMthLSV, sMthNSV]

# --- names and paths of files and dirs ---------------------------------------
pInpDet = GC.P_DIR_RES_CLF_DETAILED

# --- lists -------------------------------------------------------------------

# === derived values and input processing =====================================

# === create input dictionary =================================================
dIO = {# --- general
       # --- flow control (general)
       'lSMthEvalCompositeClf': lSMthEvalCompositeClf,
       # --- names and paths of files and dir
       'pInpDet': pInpDet,
       # === derived values and input processing
       }

###############################################################################