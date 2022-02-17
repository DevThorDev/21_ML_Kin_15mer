# -*- coding: utf-8 -*-
###############################################################################
# --- A_00__GenInput.py -------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
isTest = True              # True (just for testing) / False (standard run)
cMode = GC.S_MD_B           # mode: S_MD_A / S_MD_B / S_MD_C
nDigObj = GC.N_DIG_OBJ_2    # number of digits reserved for all input objects

# --- file and folder names, extensions ---------------------------------------
sObjInp = GC.S_OBJINP
sObjInpPre = GC.S_OBJINP_PRE

sExtPY = GC.S_EXT_PY
sExtCSV = GC.S_EXT_CSV
sExtPDF = GC.S_EXT_PDF
sExtPTH = GC.S_EXT_PTH

# --- predefined numbers ------------------------------------------------------
kDigRnd04 = GC.R04

# === create input dictionary =================================================
dictInpG = {# --- general
            'isTest': isTest,
            'cMode': cMode,
            'nDigObj': nDigObj,
            # --- file and folder names, extensions
            'sObjInp': sObjInp,
            'sObjInpPre': sObjInpPre,
            'sExtPY': sExtPY,
            'sExtCSV': sExtCSV,
            'sExtPDF': sExtPDF,
            'sExtPTH': sExtPTH,
            # --- predefined numbers
            'kDigRnd04': kDigRnd04}

###############################################################################
