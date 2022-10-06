# -*- coding: utf-8 -*-
###############################################################################
# --- D_90__Evaluator.py ------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

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
doEvaluation = True             # do evaluation?

lSetFIDet = [{sMthDummy, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthAda, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthRF, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthXTr, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthGrB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthHGrB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthGP, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthPaA, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthPct, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthSGD, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthCtNB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthCpNB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthGsNB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthMLP, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthLSV, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthNSV, GC.S_SMPL_RND_U_S, GC.S_A}]
lSetFIDet = [{sMthDummy, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthAda, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthRF, GC.S_SMPL_RND_U_S, GC.S_B},
             {sMthXTr, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthGrB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthHGrB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthGP, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthPaA, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthPct, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthSGD, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthCtNB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthCpNB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthGsNB, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthMLP, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthLSV, GC.S_SMPL_RND_U_S, GC.S_A},
             {sMthNSV, GC.S_SMPL_RND_U_S, GC.S_A}]
# lSetFIDet = [{sMthDummy, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthAda, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthRF, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthXTr, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthGrB, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthPaA, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthPct, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthCtNB, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthCpNB, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthGsNB, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthMLP, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthLSV, GC.S_SMPL_RND_U_S, GC.S_A},
#              {sMthNSV, GC.S_SMPL_RND_U_S, GC.S_A}]
# lSetFIDet = [{sMthDummy},
#              {sMthAda},
#              {sMthRF},
#              {sMthXTr},
#              {sMthPaA},
#              {sMthPct},
#              {sMthCtNB},
#              {sMthCpNB},
#              {sMthGsNB},
#              {sMthLSV},
#              {sMthNSV}]

# --- names and paths of files and dirs ---------------------------------------
sUnqNmer = GC.S_N_MER_SEQ_UNQ
sInpData = GC.S_INP_DATA
pInpUnqNmer = GC.P_DIR_RES_UNQ_N_MER
pInpData = GC.P_DIR_RES_INP_DATA_CLF_PRC
pInpDet = GC.P_DIR_RES_CLF_DETAILED
pOutEval = GC.P_DIR_RES_CLF_EVAL

# --- lists -------------------------------------------------------------------

# === derived values and input processing =====================================

# === create input dictionary =================================================
dIO = {# --- general
       # --- flow control (general)
       'doEvaluation': doEvaluation,
       'lSetFIDet': lSetFIDet,
       # --- names and paths of files and dir
       'sUnqNmer': sUnqNmer,
       'sInpData': sInpData,
       'pInpUnqNmer': pInpUnqNmer,
       'pInpData': pInpData,
       'pInpDet': pInpDet,
       'pOutEval': pOutEval,
       # === derived values and input processing
       }

###############################################################################