# -*- coding: utf-8 -*-
###############################################################################
# --- D_03__Validation.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Validation of Nmer-sequence analysis (D_03__Validation)'
sNmSpec = 'Input data for the Validation class in O_03__Validation'

# --- flow control ------------------------------------------------------------
predictWTrain = False

saveCombTrain = True
saveCombTest = True

genInfoNmerTrain = True
genInfoNmerTest = True      # not used yet

genInfoEffTrain = True
genInfoEffTest = True       # not used yet

# --- names and paths of files and dirs ---------------------------------------
sFCombInp = 'Combined_XS_KinasesPho15mer_202202'

# --- numbers -----------------------------------------------------------------
share4Test = 0.25        # share of input records reserved for test data

# --- strings -----------------------------------------------------------------

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
assert share4Test >= 0 and share4Test <= 1

# === derived values and input processing =====================================
sFCombTrain = GC.S_US02.join([sFCombInp, GC.S_TRAIN])
sFCombTest = GC.S_US02.join([sFCombInp, GC.S_TEST])

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
       'sFCombInp': sFCombInp + GC.XT_CSV,
       # 'sFResWtLh': sFResWtLh + GC.XT_CSV,
       # 'sFResRelLh': sFResRelLh + GC.XT_CSV,
       # --- numbers
       'share4Test': share4Test,
       # --- strings
       # --- lists
       # --- dictionaries
       # === derived values and input processing
       'sFCombTrain': sFCombTrain + GC.XT_CSV,
       'sFCombTest': sFCombTest + GC.XT_CSV,
       'share4Train': share4Train,
       }

###############################################################################
