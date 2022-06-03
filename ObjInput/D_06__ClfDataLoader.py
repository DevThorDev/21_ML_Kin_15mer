# -*- coding: utf-8 -*-
###############################################################################
# --- D_06__ClfDataLoader.py --------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

# --- general -----------------------------------------------------------------
sOType = 'Loader of data for classification (D_06__ClfDataLoader)'
sNmSpec = 'Input data for the data loader class in O_06__ClfDataLoader'

lvlOut = 1      # higher level --> print more information (0: no printing)

# --- flow control ------------------------------------------------------------
useFullSeqFrom = GC.S_COMB_INP      # S_COMB_INP
usedNmerSeq = GC.S_UNQ_LIST        # S_FULL_LIST / S_UNQ_LIST

usedAAcType = GC.S_AAC        # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

usedClType = GC.S_NEW + GC.S_CL     # GC.S_NEW/GC.S_OLD + GC.S_CL

# --- names and paths of files and dirs ---------------------------------------
sFInpStart = usedClType + 'Orig70'     # '' / 'AllCl' / 'NewClOrig70'
sFResCombS = 'Combined_S_KinasesPho15mer_202202'

sFInpClf, sFInpPrC = None, None
if useFullSeqFrom == GC.S_COMB_INP:     # currently only option implemented
    sFInpClf = GF.joinS([sFInpStart, usedAAcType, sFResCombS])
    sFInpPrC = sFInpClf

pInpClf = GC.P_DIR_INP_CLF
pInpPrC = GC.P_DIR_INP_CLF

# --- numbers -----------------------------------------------------------------
iCInpData = 0
maxLenNmer = None           # odd number between 1 and 15 or None (max. len)

# --- strings -----------------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

sCY = GC.S_EFF_CL

# --- sets --------------------------------------------------------------------
setNmerLen = set(range(1, 15 + 1, 2))

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================
if maxLenNmer not in setNmerLen:
    maxLenNmer = max(setNmerLen)

# === derived values and input processing =====================================
maxPosNmer = maxLenNmer//2
rngPosNmer = range(-maxPosNmer, maxPosNmer + 1)
lSCX = [str(n) for n in rngPosNmer]

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       'lvlOut': lvlOut,
       # --- flow control
       'useFullSeqFrom': useFullSeqFrom,
       'usedNmerSeq': usedNmerSeq,
       'usedAAcType': usedAAcType,
       'usedClType': usedClType,
       # --- names and paths of files and dirs
       'sFInpClf': sFInpClf,
       'sFInpPrC': sFInpPrC,
       'pInpClf': pInpClf,
       'pInpPrC': pInpPrC,
       # --- numbers
       'iCInpData': iCInpData,
       'maxLenNmer': maxLenNmer,
       # --- strings
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       'sCY': sCY,
       # --- sets
       # --- lists
       # --- dictionaries
       # === derived values and input processing
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCX': lSCX}

###############################################################################
