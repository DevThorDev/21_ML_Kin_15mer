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

# --- lists (1) ---------------------------------------------------------------

# --- names and paths of files and dirs ---------------------------------------
sFIEffInp = 'InfoEff_Prop15mer_202202__Full__1_3'

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------

# --- lists (2) ---------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------
dWtsLenSeq = {1: 0.1,
              3: 0.9}

# === assertions ==============================================================

# === derived values and input processing =====================================

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'analyseSeq': analyseSeq,
       # --- lists (1)
       # --- names and paths of files and dirs
       'sFIEffInp': sFIEffInp + GC.S_DOT + GC.S_EXT_CSV,
       # --- numbers
       # --- strings
       # --- lists (2)
       # --- dictionaries
       # === derived values and input processing
       }

###############################################################################
