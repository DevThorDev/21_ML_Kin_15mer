# -*- coding: utf-8 -*-
###############################################################################
# --- D_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Nmer-sequence analysis (D_02__SeqAnalysis)'
sNmSpec = 'Input data for the SeqAnalysis class in O_02__SeqAnalysis'

# --- flow control ------------------------------------------------------------
calcWtLh = False
calcRelLh = False

calcSnipS = False
calcSnipX = False

saveAsDfrS = True
saveAsDfrX = False

calcWtProb = False
calcRelProb = False

convSnipXToProbTbl = True

calcTotalProb = False
calcCondProb = False

calcSglPos = False

useNmerSeqFrom = GC.S_COMB_INP    # S_I_EFF_INP / S_PROC_INP / S_COMB_INP

# --- names and paths of files and dirs ---------------------------------------
sFE = '2022_05_12'

sFSeqCheck = GC.S_F_PROC_INP_N_MER
if useNmerSeqFrom == GC.S_COMB_INP:
    sFSeqCheck = 'Combined_S_KinasesPho15mer_202202'
    # sFSeqCheck = 'Combined_S_KinasesPho15mer_202202__Train'

sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7_9_11_13_15'
sFProcInp = GC.S_F_PROC_INP_N_MER
sFCombInp = 'Combined_XS_KinasesPho15mer_202202'
# sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7__Train'
# sFCombInp = 'Combined_S_KinasesPho15mer_202202__Train'

sFE = GC.S_US02.join([sFSeqCheck.split(GC.S_USC)[0], sFE])
sFLNmerSeq = GC.S_US02.join(['ListNmerInputSeq'] + [sFE])
sFLFullSeq = GC.S_US02.join(['ListFullInputSeq'] + [sFE])

sFResWtLh = GC.S_US02.join(['WtLikelihood_' + GC.S_FULL_SEQ] + [sFE])
sFResRelLh = GC.S_US02.join(['RelLikelihood_' + GC.S_FULL_SEQ] + [sFE])
sFSnipDictS = GC.S_US02.join(['SnipDictS_' + GC.S_FULL_SEQ] + [sFE])
sFSnipDictX = GC.S_US02.join(['SnipDictX_' + GC.S_FULL_SEQ] + [sFE])
sFSnipDfrS = GC.S_US02.join(['SnipDfrS_' + GC.S_FULL_SEQ] + [sFE])
sFSnipDfrX = GC.S_US02.join(['SnipDfrX_' + GC.S_FULL_SEQ] + [sFE])
sFProbTblFS = GC.S_US02.join(['ProbTable_' + GC.S_FULL_SEQ] + [sFE])
sFProbTblTP = GC.S_US02.join(['ProbTable_' + GC.S_TOTAL_PROB] + [sFE])
sFProbTblCP = GC.S_US02.join(['ProbTable_' + GC.S_COND_PROB] + [sFE])
sFResWtProb = GC.S_US02.join(['WtProb_' + GC.S_FULL_SEQ] + [sFE])
sFResRelProb = GC.S_US02.join(['RelProb_' + GC.S_FULL_SEQ] + [sFE])
sFDictNOccSP = GC.S_US02.join(['Dict_' + GC.S_SGL_POS_OCC + GC.S_USC +
                               GC.S_N_MER_SEQ] + [sFE])
sFDictRFreqSP = GC.S_US02.join(['Dict_' + GC.S_SGL_POS_FRQ + GC.S_USC +
                                GC.S_N_MER_SEQ] + [sFE])
sFResNOccSP = GC.S_US02.join([GC.S_SGL_POS_OCC + GC.S_USC + GC.S_N_MER_SEQ] +
                             [sFE])
sFResRFreqSP = GC.S_US02.join([GC.S_SGL_POS_FRQ + GC.S_USC + GC.S_N_MER_SEQ] +
                              [sFE])

# --- numbers -----------------------------------------------------------------
iStartLInpNmerSeq = None            # number or None
iEndLInpNmerSeq = None              # number or None

mDsp = 200

# --- strings -----------------------------------------------------------------
sCNmer = GC.S_C_N_MER
sEffCode = GC.S_EFF_CODE
sSnippet = GC.S_SNIPPET
sLenSnip = GC.S_LEN_SNIP
sWtLikelihood = 'wtLikelihood'
sRelLikelihood = 'relLikelihood'
sTargSeq = 'seqTarget'
sSnipPyl = 'snipPyl'
sSnipTtl = 'snipTtl'
sSnipRelF = 'snipRelF'
sSnipProb = 'snipProb'
sTtlProb = GC.S_TOTAL_PROB
sCndProb = GC.S_COND_PROB
sWtProb = 'wtProb'
sRelProb = 'relProb'
sInCombRes = 'inCombRes'
sNotInNmer = 'AAcNotIn' + GC.S_N_MER
sAminoAcid = GC.S_AMINO_ACID
sSeqCheck = GC.S_SEQ_CHECK
sIEffInp = GC.S_I_EFF_INP
sProcInp = GC.S_PROC_INP
sCombInp = GC.S_COMB_INP
sAllSeqNmer = GC.S_ALL_SEQ_N_MER
sTrainData = GC.S_TRAIN_DATA
sTestData = GC.S_TEST_DATA

# --- lists -------------------------------------------------------------------
lIStartEnd = [iStartLInpNmerSeq, iEndLInpNmerSeq]
lSCDfrLhV = [sCNmer, sEffCode, sWtLikelihood, sInCombRes]
lSCDfrLhD = [sCNmer, sEffCode, sSnippet, sLenSnip, sRelLikelihood]

lSCDfrProbS = [sSnippet, sLenSnip, sSnipProb]
lSrtByDfrProbS = [sLenSnip, sSnipProb, sSnippet]
lSrtAscDfrProbS = [True, False, True]
lSCDfrProbX = [sSnippet, sSnipPyl, sSnipTtl, sSnipRelF, sSnipProb]
lSCDfrWFSProbX = [sSnippet, sTargSeq, sSnipPyl, sSnipTtl, sSnipRelF, sSnipProb]
lSrtByDfrProbX = [sSnippet, sSnipProb, sSnipPyl]
lSrtAscDfrProbX = [True, False, True]

lSCDfrProbV = [sCNmer, sEffCode, sWtProb, sInCombRes]
lSCDfrProbD = [sCNmer, sEffCode, sSnippet, sLenSnip, sRelProb]

# --- dictionaries ------------------------------------------------------------
dWtsLenSeq = {1: 0.1,
              3: 0.5,
              5: 0.25,
              7: 0.15,
              9: 0.0,
              11: 0.0,
              13: 0.0,
              15: 0.0}

# === assertions ==============================================================

# === derived values and input processing =====================================
lSKeyFRes = ['sFLNmerSeq', 'sFLFullSeq', 'sFResWtLh', 'sFResRelLh',
             'sFSnipDictS', 'sFSnipDictX', 'sFSnipDfrS', 'sFSnipDfrX',
             'sFProbTblFS', 'sFProbTblTP', 'sFProbTblCP', 'sFResWtProb',
             'sFResRelProb', 'sFDictNOccSP', 'sFDictRFreqSP', 'sFResNOccSP',
             'sFResRFreqSP']

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'calcWtLh': calcWtLh,
       'calcRelLh': calcRelLh,
       'calcSnipS': calcSnipS,
       'calcSnipX': calcSnipX,
       'saveAsDfrS': saveAsDfrS,
       'saveAsDfrX': saveAsDfrX,
       'calcWtProb': calcWtProb,
       'calcRelProb': calcRelProb,
       'convSnipXToProbTbl': convSnipXToProbTbl,
       'calcTotalProb': calcTotalProb,
       'calcCondProb': calcCondProb,
       'calcSglPos': calcSglPos,
       'useNmerSeqFrom': useNmerSeqFrom,
       # --- names and paths of files and dirs
       'sFESeqA': sFE,
       'sFSeqCheck': sFSeqCheck + GC.XT_CSV,
       'sFIEffInp': sFIEffInp + GC.XT_CSV,
       'sFProcInp': sFProcInp + GC.XT_CSV,
       'sFCombInp': sFCombInp + GC.XT_CSV,
       'sFLNmerSeq': sFLNmerSeq + GC.XT_CSV,
       'sFLFullSeq': sFLFullSeq + GC.XT_CSV,
       'sFResWtLh': sFResWtLh + GC.XT_CSV,
       'sFResRelLh': sFResRelLh + GC.XT_CSV,
       'sFSnipDictS': sFSnipDictS + GC.XT_BIN,
       'sFSnipDictX': sFSnipDictX + GC.XT_BIN,
       'sFSnipDfrS': sFSnipDfrS + GC.XT_CSV,
       'sFSnipDfrX': sFSnipDfrX + GC.XT_CSV,
       'sFProbTblFS': sFProbTblFS + GC.XT_CSV,
       'sFProbTblTP': sFProbTblTP + GC.XT_CSV,
       'sFProbTblCP': sFProbTblCP + GC.XT_CSV,
       'sFResWtProb': sFResWtProb + GC.XT_CSV,
       'sFResRelProb': sFResRelProb + GC.XT_CSV,
       'sFDictNOccSP': sFDictNOccSP + GC.XT_BIN,
       'sFDictRFreqSP': sFDictRFreqSP + GC.XT_BIN,
       'sFResNOccSP': sFResNOccSP + GC.XT_CSV,
       'sFResRFreqSP': sFResRFreqSP + GC.XT_CSV,
       # --- numbers
       'iStartLInpNmerSeq': iStartLInpNmerSeq,
       'iEndLInpNmerSeq': iEndLInpNmerSeq,
       'mDsp': mDsp,
       # --- strings
       'sCNmer': sCNmer,
       'sEffCode': sEffCode,
       'sSnippet': sSnippet,
       'sLenSnip': sLenSnip,
       'sTargSeq': sTargSeq,
       'sSnipPyl': sSnipPyl,
       'sSnipTtl': sSnipTtl,
       'sSnipRelF': sSnipRelF,
       'sSnipProb': sSnipProb,
       'sTtlProb': sTtlProb,
       'sCndProb': sCndProb,
       'sWtProb': sWtProb,
       'sRelProb': sRelProb,
       'sWtLikelihood': sWtLikelihood,
       'sRelLikelihood': sRelLikelihood,
       'sInCombRes': sInCombRes,
       'sNotInNmer': sNotInNmer,
       'sAminoAcid': sAminoAcid,
       'sSeqCheck': sSeqCheck,
       'sIEffInp': sIEffInp,
       'sProcInp': sProcInp,
       'sCombInp': sCombInp,
       'sAllSeqNmer': sAllSeqNmer,
       'sTrainData': sTrainData,
       'sTestData': sTestData,
       # --- lists
       'lIStartEnd': lIStartEnd,
       'lSCDfrLhV': lSCDfrLhV,
       'lSCDfrLhD': lSCDfrLhD,
       'lSCDfrProbS': lSCDfrProbS,
       'lSrtByDfrProbS': lSrtByDfrProbS,
       'lSrtAscDfrProbS': lSrtAscDfrProbS,
       'lSCDfrProbX': lSCDfrProbX,
       'lSCDfrWFSProbX': lSCDfrWFSProbX,
       'lSrtByDfrProbX': lSrtByDfrProbX,
       'lSrtAscDfrProbX': lSrtAscDfrProbX,
       'lSCDfrProbV': lSCDfrProbV,
       'lSCDfrProbD': lSCDfrProbD,
       # --- dictionaries
       'dWtsLenSeq': dWtsLenSeq,
       # === derived values and input processing
       'lSKeyFRes': lSKeyFRes}

###############################################################################
