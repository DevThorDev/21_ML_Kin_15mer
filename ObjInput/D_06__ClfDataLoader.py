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

# --- names and paths of files and dirs (classifier) --------------------------
sSetClf = 'Set01_11Cl'
# sFInpStartClf = usedClTypeClf + 'Orig70'     # '' / 'AllCl' / 'NewClOrig70'
# sFResCombSClf = 'Combined_S_KinasesPho15mer_202202'
sFInpStartClf = 'InpClf'
sFInpBaseClf = 'Combined_XS_KinasesPho15mer_202202'

sFInpClf = None
if useFullSeqFromClf == GC.S_COMB_INP:    # currently only option implemented
    # sFInpClf = GF.joinS([sFInpStartClf, usedAAcTypeClf, sFResCombSClf])
    sFInpClf = GF.joinS([sFInpStartClf, sFInpBaseClf])

pInpClf = GC.P_DIR_INP_CLF

# --- names and paths of files and dirs (proportion calculator) ---------------
sSetPrC = sSetClf

sFInpStartPrC = sFInpStartClf
sFInpBasePrC = sFInpBaseClf

sFInpPrC = None
if useFullSeqFromPrC == GC.S_COMB_INP:    # currently only option implemented
    # sFInpPrC = GF.joinS([sFInpStartPrC, usedAAcTypePrC, sFResCombSPrC])
    sFInpPrC = GF.joinS([sFInpStartPrC, sFInpBasePrC])

pInpPrC = pInpClf

# --- numbers (general) -------------------------------------------------------
maxLenNmer = None           # odd number between 1 and 15 or None (max. len)

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
setNmerLen = set(range(1, 15 + 1, 2))

# --- lists (classifier) ------------------------------------------------------
lCl = ['AGC', 'CDK', 'CDPK', 'CK_II', 'LRR', 'MAPNK', 'MAPK', 'RLCK', 'SLK',
       'SnRK', 'soluble']
lXCl = ['X_' + s for s in lCl]

# --- dictionaries (classifier) -----------------------------------------------
dClasses = {'AGC': 'X_AGC',
            'CDK': 'X_CDK',
            'CDPK': 'X_CDPK',
            'CK_II': 'X_CK_II',
            'LRR_1': 'X_LRR',
            'LRR_10': 'X_LRR',
            'LRR_11': 'X_LRR',
            'LRR_12': 'X_LRR',
            'LRR_2': 'X_LRR',
            'LRR_3': 'X_LRR',
            'LRR_6B': 'X_LRR',
            'LRR_7': 'X_LRR',
            'LRR_7A': 'X_LRR',
            'LRR_8A': 'X_LRR',
            'LRR_8B': 'X_LRR',
            'LRR_8C': 'X_LRR',
            'LRR_9': 'X_LRR',
            'LRR_9A': 'X_LRR',
            'MAP2K': 'X_MAPNK',
            'MAP3K': 'X_MAPNK',
            'MAPK': 'X_MAPK',
            'RLCK_2': 'X_RLCK',
            'RLCK_6': 'X_RLCK',
            'RLCK_7': 'X_RLCK',
            'SLK': 'X_SLK',
            'SnRK1': 'X_SnRK',
            'SnRK2': 'X_SnRK',
            'SnRK3': 'X_SnRK',
            'soluble': 'X_soluble'}

# === assertions ==============================================================
if maxLenNmer not in setNmerLen:
    maxLenNmer = max(setNmerLen)

# === derived values and input processing =====================================
maxPosNmer = maxLenNmer//2
rngPosNmer = range(-maxPosNmer, maxPosNmer + 1)
lSCXClf = [str(n) for n in rngPosNmer]
lSCXPrC = lSCXClf

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
       # --- names and paths of files and dirs (classifier)
       'sSetClf': sSetClf,
       'sFInpBaseClf': sFInpBaseClf,
       'sFInpClf': sFInpClf,
       'sSetPrC': sSetPrC,
       'sFInpBasePrC': sFInpBasePrC,
       'pInpClf': pInpClf,
       # --- names and paths of files and dirs (proportion calculator)
       'sFInpPrC': sFInpPrC,
       'pInpPrC': pInpPrC,
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
       # --- lists (classifier)
       'lCl': lCl,
       'lXCl': lXCl,
       # --- dictionaries (classifier)
       'dClasses': dClasses,
       # === derived values and input processing
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCXClf': lSCXClf,
       'lSCXPrC': lSCXPrC}

###############################################################################
