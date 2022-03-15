# -*- coding: utf-8 -*-
###############################################################################
# --- D_02__SeqAnalysis.py ----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
# import Core.F_00__GenFunctions as GF

# --- general -----------------------------------------------------------------
sOType = 'Nmer-sequence analysis (D_02__SeqAnalysis)'
sNmSpec = 'Input data for the SeqAnalysis class in O_02__SeqAnalysis'

# --- flow control ------------------------------------------------------------
analyseSeq = True

# --- names and paths of files and dirs ---------------------------------------
sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3_5_7'
sFResWtLh = GC.S_US02.join(['WtLikelihood', sFIEffInp])
sFResRelLh = GC.S_US02.join(['RelLikelihood', sFIEffInp])

# --- numbers -----------------------------------------------------------------
iStartLInpNmerSeq = 0          # number or None
iEndLInpNmerSeq = 1000          # number or None

# --- strings -----------------------------------------------------------------
sCNmer = GC.S_C_N_MER
sEffCode = GC.S_EFF_CODE
sSnippet = GC.S_SNIPPET
sLenSnip = GC.S_LEN_SNIP
sWtLikelihood = 'wtLikelihood'
sRelLikelihood = 'relLikelihood'

# --- lists -------------------------------------------------------------------
lIStartEnd = [iStartLInpNmerSeq, iEndLInpNmerSeq]
lSCDfrLhV = [sCNmer, sEffCode, sWtLikelihood]
lSCDfrLhD = [sCNmer, sEffCode, sSnippet, sLenSnip, sRelLikelihood]

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

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'analyseSeq': analyseSeq,
       # --- names and paths of files and dirs
       'sFIEffInp': sFIEffInp + GC.S_DOT + GC.S_EXT_CSV,
       'sFResWtLh': sFResWtLh + GC.S_DOT + GC.S_EXT_CSV,
       'sFResRelLh': sFResRelLh + GC.S_DOT + GC.S_EXT_CSV,
       # --- numbers
       'iStartLInpNmerSeq': iStartLInpNmerSeq,
       'iEndLInpNmerSeq': iEndLInpNmerSeq,
       # --- strings
       'sCNmer': sCNmer,
       'sEffCode': sEffCode,
       'sSnippet': sSnippet,
       'sLenSnip': sLenSnip,
       'sWtLikelihood': sWtLikelihood,
       'sRelLikelihood': sRelLikelihood,
       # --- lists
       'lIStartEnd': lIStartEnd,
       'lSCDfrLhV': lSCDfrLhV,
       'lSCDfrLhD': lSCDfrLhD,
       # --- dictionaries
       'dWtsLenSeq': dWtsLenSeq,
       # === derived values and input processing
       }

###############################################################################
