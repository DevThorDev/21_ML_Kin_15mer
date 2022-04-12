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
calcWtProb = True
calcRelProb = True

useNmerSeqFrom = GC.S_COMB_INP    # S_SEQ_CHECK / S_I_EFF_INP / S_COMB_INP

# --- names and paths of files and dirs ---------------------------------------
sFSeqCheck = GC.S_F_PROC_INP_N_MER
# sFSeqCheck = 'Combined_S_KinasesPho15mer_202202__Train'
sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7'
sFCombInp = 'Combined_XS_KinasesPho15mer_202202'
# sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7__Train'
# sFCombInp = 'Combined_S_KinasesPho15mer_202202__Train'

sFResWtLh = GC.S_US02.join(['WtLikelihood'] + sFIEffInp.split(GC.S_US02)[1:])
sFResRelLh = GC.S_US02.join(['RelLikelihood'] + sFIEffInp.split(GC.S_US02)[1:])
sFResWtProb = GC.S_US02.join(['WtProb'] + sFIEffInp.split(GC.S_US02)[1:])
sFResRelProb = GC.S_US02.join(['RelProb'] + sFIEffInp.split(GC.S_US02)[1:])

# --- numbers -----------------------------------------------------------------
iStartLInpNmerSeq = None           # number or None
iEndLInpNmerSeq = 500          # number or None

mDsp = 100

# --- strings -----------------------------------------------------------------
sCNmer = GC.S_C_N_MER
sEffCode = GC.S_EFF_CODE
sSnippet = GC.S_SNIPPET
sLenSnip = GC.S_LEN_SNIP
sWtLikelihood = 'wtLikelihood'
sRelLikelihood = 'relLikelihood'
sWtProb = 'wtProb'
sRelProb = 'relProb'
sEstProb = 'estProb'
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

lSCDfrProbS = [sSnippet, sLenSnip, sEstProb]
lSrtByDfrProbS = [sLenSnip, sEstProb, sSnippet]
lSrtAscDfrProbS = [True, False, True]

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
lSKeyFRes = ['sFResWtLh', 'sFResRelLh', 'sFResWtProb', 'sFResRelProb']

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'calcWtLh': calcWtLh,
       'calcRelLh': calcRelLh,
       'calcWtProb': calcWtProb,
       'calcRelProb': calcRelProb,
       'useNmerSeqFrom': useNmerSeqFrom,
       # --- names and paths of files and dirs
       'sFSeqCheck': sFSeqCheck + GC.S_DOT + GC.S_EXT_CSV,
       'sFIEffInp': sFIEffInp + GC.S_DOT + GC.S_EXT_CSV,
       'sFCombInp': sFCombInp + GC.S_DOT + GC.S_EXT_CSV,
       'sFResWtLh': sFResWtLh + GC.S_DOT + GC.S_EXT_CSV,
       'sFResRelLh': sFResRelLh + GC.S_DOT + GC.S_EXT_CSV,
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
       'lSCDfrProbV': lSCDfrProbV,
       'lSCDfrProbD': lSCDfrProbD,
       # --- dictionaries
       'dWtsLenSeq': dWtsLenSeq,
       # === derived values and input processing
       'lSKeyFRes': lSKeyFRes}

###############################################################################
