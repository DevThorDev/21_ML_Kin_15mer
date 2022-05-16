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
R01 = GC.R01
R02 = GC.R02
R03 = GC.R03
R04 = GC.R04
R08 = GC.R08
R24 = GC.R24
maxDelta = GC.MAX_DELTA

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
            'R01': R01,
            'R02': R02,
            'R03': R03,
            'R04': R04,
            'R08': R08,
            'R24': R24,
            'maxDelta': maxDelta}

###############################################################################
