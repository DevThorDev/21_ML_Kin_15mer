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
doPredProbaEval = False         # calc. means of pred. cl. and proba results?
                                # [doPredProbaEval currently not needed]
doClsPredEval = True            # do evaluation of classes prediction?
doCompTruePred = False          # do comparison of true and predicted values?
                                # [doCompTruePred currently not usable]

sFF, sPF = GC.S_FULL_FIT_S, GC.S_PART_FIT_S + str(100)
sSmplRM1 = GC.S_SMPL_RND_U_S + GC.S_STRAT_REAL_MAJO_S + str(1)
sSmplRM10 = GC.S_SMPL_RND_U_S + GC.S_STRAT_REAL_MAJO_S + str(10)
lSetFIDet = [# (GC.S_SMPL_NO_S, sFF)
             {sMthDummy, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthAda, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthRF, GC.S_SMPL_NO_S, sFF, GC.S_B},
             {sMthXTr, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthGrB, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthHGrB, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthPaA, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthPct, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthSGD, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthCtNB, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthCpNB, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthGsNB, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthMLP, GC.S_SMPL_NO_S, sFF, GC.S_A},
             {sMthLSV, GC.S_SMPL_NO_S, sFF, GC.S_A},
             # (sSmplRM1, sFF)
             {sMthDummy, sSmplRM1, sFF, GC.S_A},
             {sMthAda, sSmplRM1, sFF, GC.S_A},
             {sMthRF, sSmplRM1, sFF, GC.S_B},
             {sMthXTr, sSmplRM1, sFF, GC.S_A},
             {sMthGrB, sSmplRM1, sFF, GC.S_A},
             {sMthHGrB, sSmplRM1, sFF, GC.S_A},
             {sMthGP, sSmplRM1, sFF, GC.S_B},
             {sMthPaA, sSmplRM1, sFF, GC.S_A},
             {sMthPct, sSmplRM1, sFF, GC.S_A},
             {sMthSGD, sSmplRM1, sFF, GC.S_A},
             {sMthCtNB, sSmplRM1, sFF, GC.S_A},
             {sMthCpNB, sSmplRM1, sFF, GC.S_A},
             {sMthGsNB, sSmplRM1, sFF, GC.S_A},
             {sMthMLP, sSmplRM1, sFF, GC.S_A},
             {sMthLSV, sSmplRM1, sFF, GC.S_A},
             {sMthNSV, sSmplRM1, sFF, GC.S_A},
             # (sSmplRM10, sFF)
             {sMthDummy, sSmplRM10, sFF, GC.S_A},
             {sMthAda, sSmplRM10, sFF, GC.S_A},
             {sMthRF, sSmplRM10, sFF, GC.S_B},
             {sMthXTr, sSmplRM10, sFF, GC.S_A},
             {sMthGrB, sSmplRM10, sFF, GC.S_A},
             {sMthHGrB, sSmplRM10, sFF, GC.S_A},
             {sMthGP, sSmplRM10, sFF, GC.S_B},
             {sMthPaA, sSmplRM10, sFF, GC.S_A},
             {sMthPct, sSmplRM10, sFF, GC.S_A},
             {sMthSGD, sSmplRM10, sFF, GC.S_A},
             {sMthCtNB, sSmplRM10, sFF, GC.S_A},
             {sMthCpNB, sSmplRM10, sFF, GC.S_A},
             {sMthGsNB, sSmplRM10, sFF, GC.S_A},
             {sMthMLP, sSmplRM10, sFF, GC.S_A},
             {sMthLSV, sSmplRM10, sFF, GC.S_A},
             {sMthNSV, sSmplRM10, sFF, GC.S_A},
             # (sSmplRM1, sPF)
             {sMthPaA, sSmplRM1, sPF, GC.S_A},
             {sMthPct, sSmplRM1, sPF, GC.S_A},
             {sMthSGD, sSmplRM1, sPF, GC.S_A},
             {sMthCtNB, sSmplRM1, sPF, GC.S_A},
             {sMthCpNB, sSmplRM1, sPF, GC.S_A},
             {sMthGsNB, sSmplRM1, sPF, GC.S_A},
             {sMthMLP, sSmplRM1, sPF, GC.S_A},
             # (sSmplRM10, sPF)
             {sMthPaA, sSmplRM10, sPF, GC.S_A},
             {sMthPct, sSmplRM10, sPF, GC.S_A},
             {sMthSGD, sSmplRM10, sPF, GC.S_A},
             {sMthCtNB, sSmplRM10, sPF, GC.S_A},
             {sMthCpNB, sSmplRM10, sPF, GC.S_A},
             {sMthGsNB, sSmplRM10, sPF, GC.S_A},
             {sMthMLP, sSmplRM10, sPF, GC.S_A}]

# --- names and paths of files and dirs ---------------------------------------
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
       'doPredProbaEval': doPredProbaEval,
       'doClsPredEval': doClsPredEval,
       'doCompTruePred': doCompTruePred,
       'lSetFIDet': lSetFIDet,
       # --- names and paths of files and dir
       'sInpData': sInpData,
       'pInpUnqNmer': pInpUnqNmer,
       'pInpData': pInpData,
       'pInpDet': pInpDet,
       'pOutEval': pOutEval,
       # === derived values and input processing
       }

###############################################################################