# -*- coding: utf-8 -*-
###############################################################################
# --- D_01__ExpData.py --------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

# --- general -----------------------------------------------------------------
sOType = 'Experimental data (D_01__ExpData)'
sNmSpec = 'Experimenal data for the ExpData class in O_01__ExpData'

# --- flow control ------------------------------------------------------------

# --- names and paths of files and dirs ---------------------------------------
sFRawInpKin = GC.S_F_RAW_INP_KIN + GC.S_DOT + GC.S_EXT_CSV
sFRawInp15mer = GC.S_F_RAW_INP_15MER + GC.S_DOT + GC.S_EXT_CSV
sFProcInpKin = 'KinasesAndTargets_202202'
sFProcInp15mer = 'Pho15mer_202202'

pDirRawInp = GC.P_DIR_RAW_INP
pDirRawInp_Test = GC.P_DIR_RAW_INP_TEST
pDirProcInp = GC.P_DIR_PROC_INP
pDirProcInp_Test = GC.P_DIR_PROC_INP_TEST

# --- numbers -----------------------------------------------------------------
lenSnippet = 15

mDsp = 1000

# --- strings -----------------------------------------------------------------
s0 = GC.S_0
sNULL = 'NULL'
lSCKinF = ['Effector', 'Target']
sC15mer = 'c15mer'

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

# === derived values and input processing =====================================

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       # --- names and paths of files and dirs
       'sFProcInpKin': sFProcInpKin + GC.S_DOT + GC.S_EXT_CSV,
       'sFProcInp15mer': sFProcInp15mer + GC.S_DOT + GC.S_EXT_CSV,
       'pFRawInpKin': GF.joinToPath(pDirRawInp, sFRawInpKin),
       'pFRawInpKin_T': GF.joinToPath(pDirRawInp_Test, sFRawInpKin),
       'pFRawInp15mer': GF.joinToPath(pDirRawInp, sFRawInp15mer),
       'pFRawInp15mer_T': GF.joinToPath(pDirRawInp_Test, sFRawInp15mer),
       'pDirProcInp': pDirProcInp,
       'pDirProcInp_T': pDirProcInp_Test,
       # --- numbers
       'lenSDef': lenSnippet,
       'mDsp': mDsp,
       # --- strings
       's0': s0,
       'sNULL': sNULL,
       'lSCKinF': lSCKinF,
       'sC15mer': sC15mer,
       # --- lists
       # --- dictionaries
       # === derived values and input processing
       }

###############################################################################
