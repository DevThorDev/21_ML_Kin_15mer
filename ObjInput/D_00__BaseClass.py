# -*- coding: utf-8 -*-
###############################################################################
# --- D_00__BaseClass.py -----------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Base class (D_00__BaseClass)'
sNmSpec = 'Data for the BaseClass class in O_00__BaseClass'

# --- data specific -----------------------------------------------------------
cSep = GC.S_SEMICOL

# --- names and paths of files and dirs ---------------------------------------
sFXtCSV = GC.S_DOT + GC.S_EXT_CSV
sFXtPDF = GC.S_DOT + GC.S_EXT_PDF
sFXtPTH = GC.S_DOT + GC.S_EXT_PTH

# --- strings -----------------------------------------------------------------

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- data specific
       'cSep': cSep,
       # --- names and paths of files and dirs
       'sFXtCSV': sFXtCSV,
       'sFXtPDF': sFXtPDF,
       'sFXtPTH': sFXtPTH,
       # --- strings
       }

###############################################################################
