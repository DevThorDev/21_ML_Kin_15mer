# -*- coding: utf-8 -*-
###############################################################################
# --- D_07__Classifier.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Classifiers for data classification (D_07__Classifier)'
sNmSpec = 'Input data for the Classifier class in O_07__Classifier'

# --- flow control ------------------------------------------------------------
doRndForestClf = True
doNNMLPClf = True

useFullSeqFrom = GC.S_COMB_INP      # S_PROC_INP / S_COMB_INP
usedNmerSeq = GC.S_FULL_LIST        # S_FULL_LIST / S_UNQ_LIST

# --- names and paths of files and dirs ---------------------------------------
sFInp = 'TestCategorical'

if useFullSeqFrom == GC.S_COMB_INP:
    sFInp = 'TestCategorical'

pInp = GC.P_DIR_TEMP

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

# --- sets --------------------------------------------------------------------
setFeat = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
# for lObs in dObs.values():
#     for cObs in lObs:
#         assert cObs in setFeat

# assert sum(startPr.values()) == 1
# for dTransPr in transPr.values():
#     assert sum(dTransPr.values()) == 1
# for dEmitPr in emitPr.values():
#     assert (sum(dEmitPr.values()) > 1. - GC.MAX_DELTA and
#             sum(dEmitPr.values()) < 1. + GC.MAX_DELTA)

# === derived values and input processing =====================================

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'doRndForestClf': doRndForestClf,
       'doNNMLPClf': doNNMLPClf,
       'useFullSeqFrom': useFullSeqFrom,
       'usedNmerSeq': usedNmerSeq,
       # --- names and paths of files and dirs
       'sFInp': sFInp,
       'pInp': pInp,
       # --- numbers
       # --- strings
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       # --- sets
       'setFeat': setFeat,
       # --- lists
       # --- dictionaries
       # === derived values and input processing
       }

###############################################################################
