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
predictWTrain = True

saveCombTrain = True
saveCombTest = True

genInfoNmerTrain = True
genInfoNmerTest = True      # not used yet

genInfoEffTrain = True
genInfoEffTest = True       # not used yet

# --- names and paths of files and dirs ---------------------------------------
sFCombInp = 'Combined_S_KinasesPho15mer_202202'

# sFResWtLh = GC.S_US02.join(['WtLikelihood'] + sFIEffInp.split(GC.S_US02)[1:])
# sFResRelLh = GC.S_US02.join(['RelLikelihood'] + sFIEffInp.split(GC.S_US02)[1:])

# --- numbers -----------------------------------------------------------------
share4Test = 0.4        # share of input records reserved for test data

# --- strings -----------------------------------------------------------------

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
assert share4Test >= 0 and share4Test <= 1

# === derived values and input processing =====================================
lSSpl = sFCombInp.split(GC.S_USC)
lSStart, lSEnd = [GC.S_USC.join(lSSpl[:2])], [GC.S_USC.join(lSSpl[2:])]
sFCombTrain = GC.S_US02.join(lSStart + [GC.S_TRAIN] + lSEnd)
sFCombTest = GC.S_US02.join(lSStart + [GC.S_TEST] + lSEnd)

share4Train = 1. - share4Test

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'predictWTrain': predictWTrain,
       'saveCombTrain': saveCombTrain,
       'saveCombTest': saveCombTest,
       'genInfoNmerTrain': genInfoNmerTrain,
       'genInfoNmerTest': genInfoNmerTest,
       'genInfoEffTrain': genInfoEffTrain,
       'genInfoEffTest': genInfoEffTest,
       # --- names and paths of files and dirs
       'sFCombInp': sFCombInp + GC.S_DOT + GC.S_EXT_CSV,
       # 'sFResWtLh': sFResWtLh + GC.S_DOT + GC.S_EXT_CSV,
       # 'sFResRelLh': sFResRelLh + GC.S_DOT + GC.S_EXT_CSV,
       # --- numbers
       'share4Test': share4Test,
       # --- strings
       # --- lists
       # --- dictionaries
       # === derived values and input processing
       'sFCombTrain': sFCombTrain + GC.S_DOT + GC.S_EXT_CSV,
       'sFCombTest': sFCombTest + GC.S_DOT + GC.S_EXT_CSV,
       'share4Train': share4Train,
       }

###############################################################################
