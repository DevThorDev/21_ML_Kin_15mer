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
saveAsDfrX = True
calcWtProb = False
calcRelProb = False

reduceFullToNmerSeq = True
convSnipXToProbTbl = True

useNmerSeqFrom = GC.S_COMB_INP    # S_SEQ_CHECK / S_I_EFF_INP / S_COMB_INP

# --- names and paths of files and dirs ---------------------------------------
# sFSeqCheck = GC.S_F_PROC_INP_N_MER
sFSeqCheck = 'Combined_S_KinasesPho15mer_202202'
# sFSeqCheck = 'Combined_S_KinasesPho15mer_202202__Train'

sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7'
sFCombInp = 'Combined_XS_KinasesPho15mer_202202'
# sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7__Train'
# sFCombInp = 'Combined_S_KinasesPho15mer_202202__Train'

sFE = sFIEffInp.split(GC.S_US02)[1:]
sFLInpSeq = GC.S_US02.join(['ListNmerInputSeq'] + sFE)

sFResWtLh = GC.S_US02.join(['WtLikelihood_FullSeq'] + sFE)
sFResRelLh = GC.S_US02.join(['RelLikelihood_FullSeq'] + sFE)
sFSnipDictS = GC.S_US02.join(['SnipDictS_FullSeq'] + sFE)
sFSnipDictX = GC.S_US02.join(['SnipDictX_FullSeq'] + sFE)
sFSnipDfrS = GC.S_US02.join(['SnipDfrS_FullSeq'] + sFE)
sFSnipDfrX = GC.S_US02.join(['SnipDfrX_FullSeq'] + sFE)
sFProbTbl = GC.S_US02.join(['ProbTable_FullSeq'] + sFE)
sFResWtProb = GC.S_US02.join(['WtProb_FullSeq'] + sFE)
sFResRelProb = GC.S_US02.join(['RelProb_FullSeq'] + sFE)
if reduceFullToNmerSeq:
    sFResWtLh = GC.S_US02.join(['WtLikelihood_RedSeq'] + sFE)
    sFResRelLh = GC.S_US02.join(['RelLikelihood_RedSeq'] + sFE)
    sFSnipDictS = GC.S_US02.join(['SnipDictS_RedSeq'] + sFE)
    sFSnipDictX = GC.S_US02.join(['SnipDictX_RedSeq'] + sFE)
    sFSnipDfrS = GC.S_US02.join(['SnipDfrS_RedSeq'] + sFE)
    sFSnipDfrX = GC.S_US02.join(['SnipDfrX_RedSeq'] + sFE)
    sFProbTbl = GC.S_US02.join(['ProbTable_RedSeq'] + sFE)
    sFResWtProb = GC.S_US02.join(['WtProb_RedSeq'] + sFE)
    sFResRelProb = GC.S_US02.join(['RelProb_RedSeq'] + sFE)

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
sFullSeq = 'seqTarget'
sSnipPyl = 'snipPyl'
sSnipTtl = 'snipTtl'
sSnipProb = 'snipProb'
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
lSCDfrProbX = [sSnippet, sSnipPyl, sSnipTtl, sSnipProb]
lSCDfrWFullSProbX = [sSnippet, sFullSeq, sSnipPyl, sSnipTtl, sSnipProb]
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
lSKeyFRes = ['sFLInpSeq', 'sFResWtLh', 'sFResRelLh', 'sFSnipDictS',
             'sFSnipDictX', 'sFSnipDfrS', 'sFSnipDfrX', 'sFResWtProb',
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
       'calcWtProb': calcWtProb,
       'calcRelProb': calcRelProb,
       'reduceFullToNmerSeq': reduceFullToNmerSeq,
       'convSnipXToProbTbl': convSnipXToProbTbl,
       'useNmerSeqFrom': useNmerSeqFrom,
       # --- names and paths of files and dirs
       'sFSeqCheck': sFSeqCheck + GC.S_DOT + GC.S_EXT_CSV,
       'sFIEffInp': sFIEffInp + GC.S_DOT + GC.S_EXT_CSV,
       'sFCombInp': sFCombInp + GC.S_DOT + GC.S_EXT_CSV,
       'sFLInpSeq': sFLInpSeq + GC.S_DOT + GC.S_EXT_CSV,
       'sFResWtLh': sFResWtLh + GC.S_DOT + GC.S_EXT_CSV,
       'sFResRelLh': sFResRelLh + GC.S_DOT + GC.S_EXT_CSV,
       'sFSnipDictS': sFSnipDictS + GC.S_DOT + GC.S_EXT_BIN,
       'sFSnipDictX': sFSnipDictX + GC.S_DOT + GC.S_EXT_BIN,
       'sFSnipDfrS': sFSnipDfrS + GC.S_DOT + GC.S_EXT_CSV,
       'sFSnipDfrX': sFSnipDfrX + GC.S_DOT + GC.S_EXT_CSV,
       'sFProbTbl': sFProbTbl + GC.S_DOT + GC.S_EXT_CSV,
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
       'sFullSeq': sFullSeq,
       'sSnipPyl': sSnipPyl,
       'sSnipTtl': sSnipTtl,
       'sSnipProb': sSnipProb,
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
       'lSCDfrWFullSProbX': lSCDfrWFullSProbX,
       'lSrtByDfrProbX': lSrtByDfrProbX,
       'lSrtAscDfrProbX': lSrtAscDfrProbX,
       'lSCDfrProbV': lSCDfrProbV,
       'lSCDfrProbD': lSCDfrProbD,
       # --- dictionaries
       'dWtsLenSeq': dWtsLenSeq,
       # === derived values and input processing
       'lSKeyFRes': lSKeyFRes}

###############################################################################
