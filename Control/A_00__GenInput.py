# -*- coding: utf-8 -*-
###############################################################################
# --- A_00__GenInput.py -------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
isTest = False              # True (just for testing) / False (standard run)
cMode = GC.S_0              # mode: S_0
nDigObj = GC.N_DIG_OBJ_2    # number of digits reserved for all input objects

# --- file and folder names, extensions ---------------------------------------
sObjInp = GC.S_OBJINP
sObjInpPre = GC.S_OBJINP_PRE

sFXtCSV = GC.S_EXT_CSV
sFXtPDF = GC.S_EXT_PDF
sFXtPTH = GC.S_EXT_PTH

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
            'sFXtCSV': sFXtCSV,
            'sFXtPDF': sFXtPDF,
            'sFXtPTH': sFXtPTH,
            # --- predefined numbers
            'kDigRnd04': kDigRnd04}

###############################################################################
