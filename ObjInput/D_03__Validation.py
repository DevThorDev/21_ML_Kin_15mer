# -*- coding: utf-8 -*-
###############################################################################
# --- D_03__Validation.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
# import Core.F_00__GenFunctions as GF

# --- general -----------------------------------------------------------------
sOType = 'Validation of Nmer-sequence analysis (D_03__Validation)'
sNmSpec = 'Input data for the Validation class in O_03__Validation'

# --- flow control ------------------------------------------------------------
doValidation = True

# --- names and paths of files and dirs ---------------------------------------
sFCombInp = 'Combined_S_KinasesPho15mer_202202'

# sFResWtLh = GC.S_US02.join(['WtLikelihood'] + sFIEffInp.split(GC.S_US02)[1:])
# sFResRelLh = GC.S_US02.join(['RelLikelihood'] + sFIEffInp.split(GC.S_US02)[1:])

# --- numbers -----------------------------------------------------------------
share4Test = 0.4        # share of input records reserved for test data

# --- strings -----------------------------------------------------------------
sTest = GC.S_TEST
sTrain = GC.S_TRAIN

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
assert share4Test >= 0 and share4Test <= 1

# === derived values and input processing =====================================
share4Train = 1. - share4Test

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'doValidation': doValidation,
       # --- names and paths of files and dirs
       'sFCombInp': sFCombInp + GC.S_DOT + GC.S_EXT_CSV,
       # 'sFResWtLh': sFResWtLh + GC.S_DOT + GC.S_EXT_CSV,
       # 'sFResRelLh': sFResRelLh + GC.S_DOT + GC.S_EXT_CSV,
       # --- numbers
       'share4Test': share4Test,
       # --- strings
       'sTest': sTest,
       'sTrain': sTrain,
       # --- lists
       # --- dictionaries
       # === derived values and input processing
       'share4Train': share4Train,
       }

###############################################################################
