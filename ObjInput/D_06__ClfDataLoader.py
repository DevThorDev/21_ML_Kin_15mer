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

# --- flow control (general) --------------------------------------------------
sSet = GC.S_SET_07

# --- flow control (classifier) -----------------------------------------------
useFullSeqFromClf = GC.S_COMB_INP # S_COMB_INP

noExclEffFam = True               # do not use Nmers with excl. effFam?
useNmerNoCl = None                # use "no class" Nmers from full seq.?
                                  # True: use "no class" Nmers as class
                                  # False: do not use any "no class" Nmers
                                  # None: "no class" Nmers used but not a class

I_Lbl = GC.S_MLT_LBL              # use multi-label or single-label input?
P_Lbl = GC.S_MLT_LBL_01           # multi-label or single-label predictions?
                                  # GC.S_SGL_LBL / GC.S_MLT_LBL
                                  # GC.S_MLT_LBL_01 (only no/single class)

usedNmerSeqClf = GC.S_UNQ_LIST    # S_FULL_LIST / S_UNQ_LIST

usedAAcTypeClf = GC.S_AAC         # {[S_AAC], S_AAC_CHARGE, S_AAC_POLAR}

usedClTypeClf = GC.S_NEW + GC.S_CL    # GC.S_NEW/GC.S_OLD + GC.S_CL

# --- flow control (proportion calculator) ------------------------------------
useFullSeqFromPrC = GC.S_COMB_INP # S_COMB_INP
useBinDicts = False               # use binary dicts (in case they exist)?
usedNmerSeqPrC = GC.S_UNQ_LIST    # S_FULL_LIST / S_UNQ_LIST

usedAAcTypePrC = usedAAcTypeClf

usedClTypePrC = usedClTypeClf

# --- names and paths of files and dirs (general) -----------------------------
sFResComb = 'Combined_S_KinasesPho15mer_202202'
sFDictNmerSnips = 'DictNmerSnipsNoCl'
sFDictNmerEffF = 'DictNmerEffF'

pResComb = GC.P_DIR_RES_COMB
pBinData = GC.P_DIR_BIN_DATA
pUnqNmer = GC.P_DIR_RES_UNQ_N_MER
pInpData = GC.P_DIR_RES_INP_DATA_CLF_PRC

# --- names and paths of files and dirs (classifier) --------------------------
sFInpSClf = 'InpClf'
sFInpBClf = 'Combined_XS_KinasesPho15mer_202202'
sFInpStIClf = '2StA'

sFInpClf = None
if useFullSeqFromClf == GC.S_COMB_INP:    # currently only option implemented
    sFInpClf = GF.joinS([sFInpSClf, sFInpBClf])
sFInpDClMpClf = GF.joinS([sFInpSClf, GC.S_CL_MAPPING, sSet])
if useNmerNoCl is None:
    sFInpDClMpClf = GF.joinS([sFInpDClMpClf, GC.S_NO_CL_AS_0])
sFInpDClStClf = GF.joinS([sFInpSClf, GC.S_CL_STEPS, sFInpStIClf, sSet])

pInpClf = GC.P_DIR_INP_CLF

# --- names and paths of files and dirs (proportion calculator) ---------------
sFInpSPrC = sFInpSClf
sFInpBPrC = sFInpBClf
sFInpStIPrC = sFInpStIClf

sFInpPrC = None
if useFullSeqFromPrC == GC.S_COMB_INP:    # currently only option implemented
    sFInpPrC = GF.joinS([sFInpSPrC, sFInpBPrC])
sFInpDClMpPrC = sFInpDClMpClf
sFInpDClStPrC = sFInpDClStClf

pInpPrC = pInpClf

# --- boolean values (general) ------------------------------------------------
printDClMap = True

# --- numbers (general) -------------------------------------------------------
maxLenNmer = None           # odd number between 1 and 15 or None (max. len)
# maxLenNmer = 9

# --- numbers (classifier) -------------------------------------------------------
iCInpDataClf = 0
modDispIni = 1000

# --- numbers (proportion calculator) -------------------------------------------------------
iCInpDataPrC = 0

# --- strings (general) -------------------------------------------------------
sFullList = GC.S_FULL_LIST
sUnqList = GC.S_UNQ_LIST
sCCodeSeq = 'code_seq'

# --- strings (classifier) ----------------------------------------------------
sCYClf = GC.S_EFF_CL

# --- strings (proportion calculator) -----------------------------------------
sCYPrC = GC.S_EFF_CL

# --- sets --------------------------------------------------------------------
setNmerLen = set(range(1, GC.LEN_N_MER_DEF + 1, 2))

# --- lists -------------------------------------------------------------------
lIPosUsed = None                # None or list of positions used for classific.
# lIPosUsed = [-7, -5, -3, -2, -1, 1, 2, 3, 5, 7]

# list of keys for file name end strings
lSIFEndS = ['sSet', 'sRestr', 'sSglMltLbl', 'sXclEffFam', 'sCls']
lSIFEndT = ['sSet', 'sRestr', 'sXclEffFam', 'sCls']
lSIFEndU = ['sRestr', 'sXclEffFam', 'sCls']
# lSIFEndV = ['sRestr', 'sCls']

# --- dictionaries ------------------------------------------------------------
# dAAcPosRestr = None     # None or dict. {iPos: lAAc restrict}
dAAcPosRestr = {0: ['S', 'T', 'Y']}     # None or dict. {iPos: lAAc restrict}

# === assertions ==============================================================
if maxLenNmer is not None:
    maxLenNmer = max(1, round(maxLenNmer))
    if 2*(maxLenNmer//2) >= maxLenNmer:     # i.e. maxLenNmer is even
        maxLenNmer += 1
if maxLenNmer not in setNmerLen:
    maxLenNmer = max(setNmerLen)

# === derived values and input processing =====================================
sFInpSzClf = sFInpBClf.split(GC.S_USC)[1]
sFInpSzPrC = sFInpSzClf

sXclEffFam, sCls = GC.S_WITH_EXCL_EFF_FAM, GC.S_X_CL
if noExclEffFam:
    sXclEffFam = GC.S_NO_EXCL_EFF_FAM
if useNmerNoCl is None:
    sCls = GC.S_X_NO_CL_AS_0
else:
    if useNmerNoCl:
        sCls = GC.S_X_NO_CL
sSglMltLbl = GC.S_I + I_Lbl + GC.S_P + P_Lbl

ILblSgl = (I_Lbl == GC.S_SGL_LBL)

maxPosNmer = maxLenNmer//2
rngPosNmer = range(-maxPosNmer, maxPosNmer + 1)
if lIPosUsed is None:
    lIPosUsed = list(rngPosNmer)
else:
    lIPosUsed = sorted(list(set(lIPosUsed) & set(rngPosNmer)))
lSCXClf = [str(n) for n in lIPosUsed]
lSCXPrC = lSCXClf

sRestr, sMxLen = '', str(maxLenNmer)
if maxLenNmer is not None and maxLenNmer != GC.LEN_N_MER_DEF:
    pass
    # sMaxLenNmer = GC.S_MAX_LEN_S + str(GC.S_0)*(2 - len(sMxLen)) + sMxLen
if dAAcPosRestr is not None:
    for sK, lV in dAAcPosRestr.items():
        sRestr += GF.joinS([GC.S_RESTR, sK, GF.joinS(lV)])
if lIPosUsed == list(range(lIPosUsed[0], lIPosUsed[-1] + 1)):
    sPosUsed = GF.joinS([lIPosUsed[0], GC.S_TO, lIPosUsed[-1]])
    sRestr = GF.joinS([sRestr, GC.S_I_POS, sPosUsed])
else:
    if set(lIPosUsed) < set(range(-GC.I_CENT_N_MER, GC.I_CENT_N_MER + 1)):
        sRestr = GF.joinS([sRestr, GC.S_I_POS, GF.joinS(lIPosUsed)])

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       'lvlOut': lvlOut,
       # --- flow control (general)
       'sSet': sSet,
       # --- flow control (classifier)
       'useFullSeqFromClf': useFullSeqFromClf,
       'noExclEffFam': noExclEffFam,
       'useNmerNoCl': useNmerNoCl,
       'I_Lbl': I_Lbl,
       'P_Lbl': P_Lbl,
       'usedNmerSeqClf': usedNmerSeqClf,
       'usedAAcTypeClf': usedAAcTypeClf,
       'usedClTypeClf': usedClTypeClf,
       # --- flow control (proportion calculator)
       'useFullSeqFromPrC': useFullSeqFromPrC,
       'useBinDicts': useBinDicts,
       'usedNmerSeqPrC': usedNmerSeqPrC,
       'usedAAcTypePrC': usedAAcTypePrC,
       'usedClTypePrC': usedClTypePrC,
       # --- names and paths of files and dirs (general)
       'sFResComb': sFResComb,
       'sFDictNmerSnips': sFDictNmerSnips,
       'sFDictNmerEffF': sFDictNmerEffF,
       'pResComb': pResComb,
       'pBinData': pBinData,
       'pUnqNmer': pUnqNmer,
       'pInpData': pInpData,
       # --- names and paths of files and dirs (classifier)
       'sFInpBClf': sFInpBClf,
       'sFInpStIClf': sFInpStIClf,
       'sFInpClf': sFInpClf,
       'sFInpDClMpClf': sFInpDClMpClf,
       'sFInpDClStClf': sFInpDClStClf,
       'sFInpBPrC': sFInpBPrC,
       'pInpClf': pInpClf,
       # --- names and paths of files and dirs (proportion calculator)
       'sFInpPrC': sFInpPrC,
       'sFInpDClMpPrC': sFInpDClMpPrC,
       'sFInpDClStPrC': sFInpDClStPrC,
       'pInpPrC': pInpPrC,
       # --- boolean values (general)
       'printDClMap': printDClMap,
       # --- numbers (general)
       'maxLenNmer': maxLenNmer,
       # --- numbers (classifier)
       'iCInpDataClf': iCInpDataClf,
       'modDispIni': modDispIni,
       # --- numbers (proportion calculator)
       'iCInpDataPrC': iCInpDataPrC,
       # --- strings (general)
       'sFullList': sFullList,
       'sUnqList': sUnqList,
       'sCCodeSeq': sCCodeSeq,
       # --- strings (classifier)
       'sCYClf': sCYClf,
       # --- strings (proportion calculator)
       'sCYPrC': sCYPrC,
       # --- sets
       # --- lists
       'lIPosUsed': lIPosUsed,
       'lSIFEndS': lSIFEndS,
       'lSIFEndT': lSIFEndT,
       'lSIFEndU': lSIFEndU,
       # 'lSIFEndV': lSIFEndV,
       # --- dictionaries
       'dAAcPosRestr': dAAcPosRestr,
       # === derived values and input processing
       'sFInpSzClf': sFInpSzClf,
       'sFInpSzPrC': sFInpSzPrC,
       'sSglLbl': GC.S_SGL_LBL,
       'sMltLbl': GC.S_MLT_LBL,
       'sXclEffFam': sXclEffFam,
       'sCls': sCls,
       'sSglMltLbl': sSglMltLbl,
       'ILblSgl': ILblSgl,
       'maxPosNmer': maxPosNmer,
       'rngPosNmer': rngPosNmer,
       'lSCXClf': lSCXClf,
       'lSCXPrC': lSCXPrC,
       'sRestr': sRestr}

###############################################################################