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

xtPY = GC.XT_PY
xtCSV = GC.XT_CSV
xtPDF = GC.XT_PDF
xtPTH = GC.XT_PTH
xtBIN = GC.XT_BIN

# --- predefined numbers ------------------------------------------------------
R01 = GC.R01
R02 = GC.R02
R03 = GC.R03
R04 = GC.R04
R08 = GC.R08
R24 = GC.R24
maxDelta = GC.MAX_DELTA

# --- predefined strings ------------------------------------------------------
sPath = GC.S_PATH
sLFCS = GC.S_L_F_S
sLFCC = GC.S_L_F_C
sLFCE = GC.S_L_F_E
sLFCJS = GC.S_L_F_J + GC.S_S
sLFCJC = GC.S_L_F_J + GC.S_C
sLFCJE = GC.S_L_F_J + GC.S_E
sLFCJSC = GC.S_L_F_J + GC.S_S + GC.S_C
sLFCJCE = GC.S_L_F_J + GC.S_C + GC.S_E
sFXt = GC.S_F_XT

# === create input dictionary =================================================
dictInpG = {# --- general
            'isTest': isTest,
            'cMode': cMode,
            'nDigObj': nDigObj,
            # --- file and folder names, extensions
            'sObjInp': sObjInp,
            'sObjInpPre': sObjInpPre,
            'xtPY': xtPY,
            'xtCSV': xtCSV,
            'xtPDF': xtPDF,
            'xtPTH': xtPTH,
            'xtBIN': xtBIN,
            # --- predefined numbers
            'R01': R01,
            'R02': R02,
            'R03': R03,
            'R04': R04,
            'R08': R08,
            'R24': R24,
            'maxDelta': maxDelta,
            # --- predefined strings
            'sPath': sPath,
            'sLFS': sLFCS,
            'sLFC': sLFCC,
            'sLFE': sLFCE,
            'sLFJS': sLFCJS,
            'sLFJC': sLFCJC,
            'sLFJE': sLFCJE,
            'sLFJSC': sLFCJSC,
            'sLFJCE': sLFCJCE,
            'sFXt': sFXt}

###############################################################################