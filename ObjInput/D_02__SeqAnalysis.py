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

calcSnipS = True
calcSnipX = True

saveAsDfrS = True
saveAsDfrX = False

calcTotalProb = True
calcCondProb = True

calcWtProb = True
calcRelProb = True

convSnipXToProbTbl = True

useNmerSeqFrom = GC.S_SEQ_CHECK    # S_SEQ_CHECK / S_I_EFF_INP / S_COMB_INP

# --- names and paths of files and dirs ---------------------------------------
# sFSeqCheck = GC.S_F_PROC_INP_N_MER
sFSeqCheck = 'Combined_S_KinasesPho15mer_202202'
# sFSeqCheck = 'Combined_S_KinasesPho15mer_202202__Train'

sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7_9_11_13_15'
sFCombInp = 'Combined_XS_KinasesPho15mer_202202'
# sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7__Train'
# sFCombInp = 'Combined_S_KinasesPho15mer_202202__Train'

# sFE = sFIEffInp.split(GC.S_US02)[1:]
sFE = [GC.S_US02.join([sFSeqCheck.split(GC.S_USC)[0], '2022_05_06'])]
sFLNmerSeq = GC.S_US02.join(['ListNmerInputSeq'] + sFE)
sFLFullSeq = GC.S_US02.join(['ListFullInputSeq'] + sFE)

sFResWtLh = GC.S_US02.join(['WtLikelihood_FullSeq'] + sFE)
sFResRelLh = GC.S_US02.join(['RelLikelihood_FullSeq'] + sFE)
sFSnipDictS = GC.S_US02.join(['SnipDictS_FullSeq'] + sFE)
sFSnipDictX = GC.S_US02.join(['SnipDictX_FullSeq'] + sFE)
sFSnipDfrS = GC.S_US02.join(['SnipDfrS_FullSeq'] + sFE)
sFSnipDfrX = GC.S_US02.join(['SnipDfrX_FullSeq'] + sFE)
sFProbTblFS = GC.S_US02.join(['ProbTable_FullSeq'] + sFE)
sFProbTblTP = GC.S_US02.join(['ProbTable_TotalProb'] + sFE)
sFProbTblCP = GC.S_US02.join(['ProbTable_CondProb'] + sFE)
sFResWtProb = GC.S_US02.join(['WtProb_FullSeq'] + sFE)
sFResRelProb = GC.S_US02.join(['RelProb_FullSeq'] + sFE)


# --- numbers -----------------------------------------------------------------
iStartLInpNmerSeq = None            # number or None
iEndLInpNmerSeq = None              # number or None

mDsp = 100

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
sSeqCheck = GC.S_SEQ_CHECK
sIEffInp = GC.S_I_EFF_INP
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
             'sFResRelProb']

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
       'calcTotalProb': calcTotalProb,
       'calcCondProb': calcCondProb,
       'calcWtProb': calcWtProb,
       'calcRelProb': calcRelProb,
       'convSnipXToProbTbl': convSnipXToProbTbl,
       'useNmerSeqFrom': useNmerSeqFrom,
       # --- names and paths of files and dirs
       'sFSeqCheck': sFSeqCheck + GC.S_DOT + GC.S_EXT_CSV,
       'sFIEffInp': sFIEffInp + GC.S_DOT + GC.S_EXT_CSV,
       'sFCombInp': sFCombInp + GC.S_DOT + GC.S_EXT_CSV,
       'sFLNmerSeq': sFLNmerSeq + GC.S_DOT + GC.S_EXT_CSV,
       'sFLFullSeq': sFLFullSeq + GC.S_DOT + GC.S_EXT_CSV,
       'sFResWtLh': sFResWtLh + GC.S_DOT + GC.S_EXT_CSV,
       'sFResRelLh': sFResRelLh + GC.S_DOT + GC.S_EXT_CSV,
       'sFSnipDictS': sFSnipDictS + GC.S_DOT + GC.S_EXT_BIN,
       'sFSnipDictX': sFSnipDictX + GC.S_DOT + GC.S_EXT_BIN,
       'sFSnipDfrS': sFSnipDfrS + GC.S_DOT + GC.S_EXT_CSV,
       'sFSnipDfrX': sFSnipDfrX + GC.S_DOT + GC.S_EXT_CSV,
       'sFProbTblFS': sFProbTblFS + GC.S_DOT + GC.S_EXT_CSV,
       'sFProbTblTP': sFProbTblTP + GC.S_DOT + GC.S_EXT_CSV,
       'sFProbTblCP': sFProbTblCP + GC.S_DOT + GC.S_EXT_CSV,
       'sFResWtProb': sFResWtProb + GC.S_DOT + GC.S_EXT_CSV,
       'sFResRelProb': sFResRelProb + GC.S_DOT + GC.S_EXT_CSV,
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
       'sSeqCheck': sSeqCheck,
       'sIEffInp': sIEffInp,
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
