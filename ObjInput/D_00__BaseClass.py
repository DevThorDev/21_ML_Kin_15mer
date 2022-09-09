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

# --- numbers -----------------------------------------------------------------
lenNmerDef = GC.LEN_N_MER_DEF
iCentNmer = GC.I_CENT_N_MER

# --- strings -----------------------------------------------------------------
sSpace = GC.S_SPACE
sDot = GC.S_DOT
sComma = GC.S_COMMA
sSemicol = GC.S_SEMICOL
sColon = GC.S_COL
sDash = GC.S_DASH
sStar = GC.S_STAR
sUSC = GC.S_USC
sUS02 = GC.S_US02
sTab = GC.S_TAB
sNewl = GC.S_NEWL
sNULL = GC.S_NULL
s0, sC, sE, sX, sY = GC.S_0, GC.S_CAP_C, GC.S_CAP_E, GC.S_CAP_X, GC.S_CAP_Y
sXS, sS, sM, sL = GC.S_CAP_XS, GC.S_CAP_S, GC.S_CAP_M, GC.S_CAP_L
sVBarSep = GC.S_VBAR_SEP
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
sNoEff = GC.S_NO_EFF
sNoFam = GC.S_NO_FAM
sNmer = GC.S_N_MER
sINmer = GC.S_I_N_MER
sINmerTrain = GC.S_I_N_MER_TRAIN
sINmerTest = GC.S_I_N_MER_TEST
sIEff = GC.S_I_EFF
sIEffTrain = GC.S_I_EFF_TRAIN
sIEffTest = GC.S_I_EFF_TEST
sIEffF = GC.S_I_EFF_F
sIEffFTrain = GC.S_I_EFF_F_TRAIN
sIEffFTest = GC.S_I_EFF_F_TEST
sEffCode = GC.S_EFF_CODE
sEffCl = GC.S_EFF_CL
sXCl = GC.S_X_CL
sNone = GC.S_NONE
sClf, sPrC = GC.S_CLF, GC.S_PRC
sEffFam = GC.S_EFF_FAMILY
sTrueCl = GC.S_TRUE_CL
sPredCl = GC.S_PRED_CL
sIGen = GC.S_I_GEN
sCNmer = GC.S_C_N_MER
sAllSeq = GC.S_ALL_SEQ
sNmerSeq = GC.S_N_MER_SEQ
sFullSeq = GC.S_FULL_SEQ
sUnique = GC.S_UNIQUE
sDescNone = GC.S_DESC_NONE
sDescDummy = GC.S_DESC_DUMMY
sDescAda = GC.S_DESC_ADA
sDescRF = GC.S_DESC_RF
sDescXTr = GC.S_DESC_X_TR
sDescGrB = GC.S_DESC_GR_B
sDescHGrB = GC.S_DESC_H_GR_B
sDescGP = GC.S_DESC_GP
sDescPaA = GC.S_DESC_PA_A
sDescPct = GC.S_DESC_PCT
sDescSGD = GC.S_DESC_SGD
sDescCtNB = GC.S_DESC_CT_NB
sDescCpNB = GC.S_DESC_CP_NB
sDescGsNB = GC.S_DESC_GS_NB
sDescMLP = GC.S_DESC_MLP
sDescLSV = GC.S_DESC_LSV
sDescNSV = GC.S_DESC_NSV
sMthNoneL, sMthNone = GC.S_MTH_NONE_L, GC.S_MTH_NONE
sMthDummyL, sMthDummy = GC.S_MTH_DUMMY_L, GC.S_MTH_DUMMY
sMthAdaL, sMthAda = GC.S_MTH_ADA_L, GC.S_MTH_ADA
sMthRFL, sMthRF = GC.S_MTH_RF_L, GC.S_MTH_RF
sMthXTrL, sMthXTr = GC.S_MTH_X_TR_L, GC.S_MTH_X_TR
sMthGrBL, sMthGrB = GC.S_MTH_GR_B_L, GC.S_MTH_GR_B
sMthHGrBL, sMthHGrB = GC.S_MTH_H_GR_B_L, GC.S_MTH_H_GR_B
sMthGPL, sMthGP = GC.S_MTH_GP_L, GC.S_MTH_GP
sMthPaAL, sMthPaA = GC.S_MTH_PA_A_L, GC.S_MTH_PA_A
sMthPctL, sMthPct = GC.S_MTH_PCT_L, GC.S_MTH_PCT
sMthSGDL, sMthSGD = GC.S_MTH_SGD_L, GC.S_MTH_SGD
sMthCtNBL, sMthCtNB = GC.S_MTH_CT_NB_L, GC.S_MTH_CT_NB
sMthCpNBL, sMthCpNB = GC.S_MTH_CP_NB_L, GC.S_MTH_CP_NB
sMthGsNBL, sMthGsNB = GC.S_MTH_GS_NB_L, GC.S_MTH_GS_NB
sMthMLPL, sMthMLP = GC.S_MTH_MLP_L, GC.S_MTH_MLP
sMthLSVL, sMthLSV = GC.S_MTH_LSV_L, GC.S_MTH_LSV
sMthNSVL, sMthNSV = GC.S_MTH_NSV_L, GC.S_MTH_NSV
sPar, sGSRS = GC.S_PAR, GC.S_GSRS
sSummary = GC.S_SUMMARY
sCnfMat = GC.S_CNF_MAT
sDetailed = GC.S_DETAILED
sProba = GC.S_PROB
sProp = GC.S_PROP
sLbl, sStep = GC.S_LBL, GC.S_STEP
sClMapping = GC.S_CL_MAPPING
sClSteps = GC.S_CL_STEPS
sMaxLenS = GC.S_MAX_LEN_S
sRestr = GC.S_RESTR
sIPos = GC.S_I_POS
sLbl = GC.S_LBL
sNmerEffFam = GC.S_N_MER_EFF_FAM
sNmerSeqU = GC.S_N_MER_SEQ_UNQ
sInpData = GC.S_INP_DATA
sMltStep = GC.S_MLT_STEP
sStratRealMajo = GC.S_STRAT_REAL_MAJO
sStratShareMino = GC.S_STRAT_SHARE_MINO
sPartFit, sPartFitS = GC.S_PART_FIT, GC.S_PART_FIT_S

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- data specific
       'cSep': cSep,
       # --- names and paths of files and dirs
       # --- numbers
       'lenNmerDef': lenNmerDef,
       'iCentNmer': iCentNmer,
       # --- strings
       'sSpace': sSpace,
       'sDot': sDot,
       'sComma': sComma,
       'sSemicol': sSemicol,
       'sColon': sColon,
       'sDash': sDash,
       'sStar': sStar,
       'sUSC': sUSC,
       'sUS02': sUS02,
       'sTab': sTab,
       'sNewl': sNewl,
       'sNULL': sNULL,
       's0': s0,
       'sC': sC,
       'sE': sE,
       'sX': sX,
       'sY': sY,
       'sXS': sXS,
       'sS': sS,
       'sM': sM,
       'sL': sL,
       'sVBarSep': sVBarSep,
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
       'sNoEff': sNoEff,
       'sNoFam': sNoFam,
       'sNmer': sNmer,
       'sINmer': sINmer,
       'sINmerTrain': sINmerTrain,
       'sINmerTest': sINmerTest,
       'sIEff': sIEff,
       'sIEffTrain': sIEffTrain,
       'sIEffTest': sIEffTest,
       'sIEffF': sIEffF,
       'sIEffFTrain': sIEffFTrain,
       'sIEffFTest': sIEffFTest,
       'sEffCode': sEffCode,
       'sEffCl': sEffCl,
       'sXCl': sXCl,
       'sNone': sNone,
       'sClf': sClf,
       'sPrC': sPrC,
       'sEffFam': sEffFam,
       'sTrueCl': sTrueCl,
       'sPredCl': sPredCl,
       'sIGen': sIGen,
       'sCNmer': sCNmer,
       'sAllSeq': sAllSeq,
       'sNmerSeq': sNmerSeq,
       'sFullSeq': sFullSeq,
       'sUnique': sUnique,
       'sDescNone': sDescNone,
       'sDescDummy': sDescDummy,
       'sDescAda': sDescAda,
       'sDescRF': sDescRF,
       'sDescXTr': sDescXTr,
       'sDescGrB': sDescGrB,
       'sDescHGrB': sDescHGrB,
       'sDescGP': sDescGP,
       'sDescPaA': sDescPaA,
       'sDescPct': sDescPct,
       'sDescSGD': sDescSGD,
       'sDescCtNB': sDescCtNB,
       'sDescCpNB': sDescCpNB,
       'sDescGsNB': sDescGsNB,
       'sDescMLP': sDescMLP,
       'sDescLSV': sDescLSV,
       'sDescNSV': sDescNSV,
       'sMthNoneL': sMthNoneL,
       'sMthNone': sMthNone,
       'sMthDummyL': sMthDummyL,
       'sMthDummy': sMthDummy,
       'sMthAdaL': sMthAdaL,
       'sMthAda': sMthAda,
       'sMthRFL': sMthRFL,
       'sMthRF': sMthRF,
       'sMthXTrL': sMthXTrL,
       'sMthXTr': sMthXTr,
       'sMthGrBL': sMthGrBL,
       'sMthGrB': sMthGrB,
       'sMthHGrBL': sMthHGrBL,
       'sMthHGrB': sMthHGrB,
       'sMthGPL': sMthGPL,
       'sMthGP': sMthGP,
       'sMthPaAL': sMthPaAL,
       'sMthPaA': sMthPaA,
       'sMthPctL': sMthPctL,
       'sMthPct': sMthPct,
       'sMthSGDL': sMthSGDL,
       'sMthSGD': sMthSGD,
       'sMthCtNBL': sMthCtNBL,
       'sMthCtNB': sMthCtNB,
       'sMthCpNBL': sMthCpNBL,
       'sMthCpNB': sMthCpNB,
       'sMthGsNBL': sMthGsNBL,
       'sMthGsNB': sMthGsNB,
       'sMthMLPL': sMthMLPL,
       'sMthMLP': sMthMLP,
       'sMthLSVL': sMthLSVL,
       'sMthLSV': sMthLSV,
       'sMthNSVL': sMthNSVL,
       'sMthNSV': sMthNSV,
       'sPar': sPar,
       'sGSRS': sGSRS,
       'sSummary': sSummary,
       'sCnfMat': sCnfMat,
       'sDetailed': sDetailed,
       'sProba': sProba,
       'sProp': sProp,
       'sLbl': sLbl,
       'sStep': sStep,
       'sClMapping': sClMapping,
       'sClSteps': sClSteps,
       'sMaxLenS': sMaxLenS,
       'sRestr': sRestr,
       'sIPos': sIPos,
       'sLbl': sLbl,
       'sNmerEffFam': sNmerEffFam,
       'sNmerSeqU': sNmerSeqU,
       'sInpData': sInpData,
       'sMltStep': sMltStep,
       'sStratRealMajo': sStratRealMajo,
       'sStratShareMino': sStratShareMino,
       'sPartFit': sPartFit,
       'sPartFitS': sPartFitS,
       # --- numbers
       # --- lists
       'lSMth_L': [sMthDummyL, sMthAdaL, sMthRFL, sMthXTrL, sMthGrBL,
                   sMthHGrBL, sMthGPL, sMthPaAL, sMthPctL, sMthSGDL, sMthCtNBL,
                   sMthCpNBL, sMthGsNBL, sMthMLPL, sMthLSVL, sMthNSVL],
       'lSMth': [sMthDummy, sMthAda, sMthRF, sMthXTr, sMthGrB, sMthHGrB,
                 sMthGP, sMthPaA, sMthPct, sMthSGD, sMthCtNB, sMthCpNB,
                 sMthGsNB, sMthMLP, sMthLSV, sMthNSV],
       'lSMthPartFit': [sMthPaA, sMthPct, sMthSGD, sMthCtNB, sMthCpNB,
                        sMthGsNB, sMthMLP]}

###############################################################################