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

# --- flow control (classifier) -----------------------------------------------
useFullSeqFromClf = GC.S_COMB_INP    # S_COMB_INP
usedNmerSeqClf = GC.S_UNQ_LIST       # S_FULL_LIST / S_UNQ_LIST

usedAAcTypeClf = GC.S_AAC            # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

usedClTypeClf = GC.S_NEW + GC.S_CL   # GC.S_NEW/GC.S_OLD + GC.S_CL

# --- flow control (proportion calculator) ------------------------------------
useFullSeqFromPrC = GC.S_COMB_INP    # S_COMB_INP
usedNmerSeqPrC = GC.S_UNQ_LIST       # S_FULL_LIST / S_UNQ_LIST

usedAAcTypePrC = usedAAcTypeClf

usedClTypePrC = usedClTypeClf

# --- names and paths of files and dirs (general) -----------------------------
sSet = GC.S_SET_03

# --- names and paths of files and dirs (classifier) --------------------------
# sFInpStartClf = usedClTypeClf + 'Orig70'     # '' / 'AllCl' / 'NewClOrig70'
# sFResCombSClf = 'Combined_S_KinasesPho15mer_202202'
sFInpStartClf = 'InpClf'
sFInpBaseClf = 'Combined_XS_KinasesPho15mer_202202'

sFInpClf = None
if useFullSeqFromClf == GC.S_COMB_INP:    # currently only option implemented
    # sFInpClf = GF.joinS([sFInpStartClf, usedAAcTypeClf, sFResCombSClf])
    sFInpClf = GF.joinS([sFInpStartClf, sFInpBaseClf])
sFInpDClClf = GF.joinS([sFInpStartClf, GC.S_CL_MAPPING, sSet])

pInpClf = GC.P_DIR_INP_CLF

# --- names and paths of files and dirs (proportion calculator) ---------------
sFInpStartPrC = sFInpStartClf
sFInpBasePrC = sFInpBaseClf

sFInpPrC = None
if useFullSeqFromPrC == GC.S_COMB_INP:    # currently only option implemented
    # sFInpPrC = GF.joinS([sFInpStartPrC, usedAAcTypePrC, sFResCombSPrC])
    sFInpPrC = GF.joinS([sFInpStartPrC, sFInpBasePrC])
sFInpDClPrC = sFInpDClClf

pInpPrC = pInpClf

# --- boolean values (general) ------------------------------------------------
printDClasses = True

# --- numbers (general) -------------------------------------------------------
maxLenNmer = None           # odd number between 1 and 15 or None (max. len)
# maxLenNmer = 9           # odd number between 1 and 15 or None (max. len)

# --- numbers (classifier) -------------------------------------------------------
iCInpDataClf = 0

# --- numbers (proportion calculator) -------------------------------------------------------
iCInpDataPrC = 0

# --- strings (general) -------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST

# --- strings (classifier) ----------------------------------------------------
sCYClf = GC.S_EFF_CL

# --- strings (proportion calculator) -----------------------------------------
sCYPrC = GC.S_EFF_CL

# --- sets --------------------------------------------------------------------
setNmerLen = set(range(1, GC.LEN_N_MER_DEF + 1, 2))

# --- dictionaries ------------------------------------------------------------
dAAcPosRestr = None     # None or dict. {iPos: lAAc restrict}
# dAAcPosRestr = {0: ['S', 'T', 'Y']}     # None or dict. {iPos: lAAc restrict}

# === assertions ==============================================================

if maxLenNmer is not None:
    maxLenNmer = max(1, round(maxLenNmer))
    if 2*(maxLenNmer//2) >= maxLenNmer:     # i.e. maxLenNmer is even
        maxLenNmer += 1
if maxLenNmer not in setNmerLen:
    maxLenNmer = max(setNmerLen)

# === derived values and input processing =====================================
maxPosNmer = maxLenNmer//2
rngPosNmer = range(-maxPosNmer, maxPosNmer + 1)
lSCXClf = [str(n) for n in rngPosNmer]
lSCXPrC = lSCXClf

sMaxLenNmer, sAAcRestr, sMxLen = '', '', str(maxLenNmer)
if maxLenNmer is not None and maxLenNmer != GC.LEN_N_MER_DEF:
    sMaxLenNmer = GC.S_MAX_LEN_S + str(GC.S_0)*(2 - len(sMxLen)) + sMxLen
if dAAcPosRestr is not None:
    for sK, lV in dAAcPosRestr.items():
        sAAcRestr += GF.joinS([GC.S_RESTR, sK, GF.joinS(lV)])

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       'lvlOut': lvlOut,
       # --- flow control (classifier)
       'useFullSeqFromClf': useFullSeqFromClf,
       'usedNmerSeqClf': usedNmerSeqClf,
       'usedAAcTypeClf': usedAAcTypeClf,
       'usedClTypeClf': usedClTypeClf,
       # --- flow control (proportion calculator)
       'useFullSeqFromPrC': useFullSeqFromPrC,
       'usedNmerSeqPrC': usedNmerSeqPrC,
       'usedAAcTypePrC': usedAAcTypePrC,
       'usedClTypePrC': usedClTypePrC,
       # --- names and paths of files and dirs (general)
       'sSet': sSet,
       # --- names and paths of files and dirs (classifier)
       'sFInpBaseClf': sFInpBaseClf,
       'sFInpClf': sFInpClf,
       'sFInpDClClf': sFInpDClClf,
       'sFInpBasePrC': sFInpBasePrC,
       'pInpClf': pInpClf,
       # --- names and paths of files and dirs (proportion calculator)
       'sFInpPrC': sFInpPrC,
       'sFInpDClPrC': sFInpDClPrC,
       'pInpPrC': pInpPrC,
       # --- boolean values (general)
       'printDClasses': printDClasses,
       # --- numbers (general)
       'maxLenNmer': maxLenNmer,
       # --- numbers (classifier)
       'iCInpDataClf': iCInpDataClf,
       # --- numbers (proportion calculator)
       'iCInpDataPrC': iCInpDataPrC,
       # --- strings (general)
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       # --- strings (classifier)
       'sCYClf': sCYClf,
       # --- strings (proportion calculator)
       'sCYPrC': sCYPrC,
       # --- sets
       # --- dictionaries
       'dAAcPosRestr': dAAcPosRestr,
       # === derived values and input processing
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCXClf': lSCXClf,
       'lSCXPrC': lSCXPrC,
       'sMaxLenNmer': sMaxLenNmer,
       'sAAcRestr': sAAcRestr}

###############################################################################