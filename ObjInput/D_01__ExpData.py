# -*- coding: utf-8 -*-
###############################################################################
# --- D_01__ExpData.py --------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

# --- general -----------------------------------------------------------------
sOType = 'Experimental data (D_01__ExpData)'
sNmSpec = 'Input data for the ExpData class in O_01__ExpData'

# --- flow control ------------------------------------------------------------
procInput = False
dBDoDfrRes = {GC.S_X_SHORT: False,
              GC.S_SHORT: False,
              GC.S_MED: False,
              GC.S_LONG: False}
genInfoNmer = False
genInfoEff = False
genInfoGen = False

# --- lists (1) ---------------------------------------------------------------
lLenNmer = None             # None: all lengths considered
# lLenNmer = [1, 3, 5, 7]     # None: all lengths considered

# --- names and paths of files and dirs ---------------------------------------
sFRawInpKin = GC.S_F_RAW_INP_KIN
sFRawInpNmer = GC.S_F_RAW_INP_N_MER

sFProcInpKin = GC.S_F_PROC_INP_KIN
sFProcInpNmer = GC.S_F_PROC_INP_N_MER

sFResCombXS = 'Combined_XS_KinasesPho15mer_202202'
sFResCombS = 'Combined_S_KinasesPho15mer_202202'
sFResCombM = 'Combined_M_KinasesPho15mer_202202'
sFResCombL = 'Combined_L_KinasesPho15mer_202202'

sFResIGen = 'InfoGen_Kin_15mer_202202'
sFResINmer = 'Info15mer_NOcc_202202'
sFResIEff = 'InfoEff_Prop15mer_202202'

pDirRawInp = GC.P_DIR_RAW_INP
pDirRawInp_Test = GC.P_DIR_RAW_INP_TEST
pDirProcInp = GC.P_DIR_PROC_INP
pDirProcInp_Test = GC.P_DIR_PROC_INP_TEST
pDirResComb = GC.P_DIR_RES_COMB
pDirResComb_Test = GC.P_DIR_RES_COMB_TEST
pDirResInfo = GC.P_DIR_RES_INFO
pDirResInfo_Test = GC.P_DIR_RES_INFO_TEST
pDirResProb = GC.P_DIR_RES_PROB
pDirResProb_Test = GC.P_DIR_RES_PROB_TEST
pDirResViterbi = GC.P_DIR_RES_VITERBI
pBinData = GC.P_DIR_BIN_DATA
pDirTemp = GC.P_DIR_TEMP

# --- numbers -----------------------------------------------------------------
rndDigProp = GC.R08

# --- strings -----------------------------------------------------------------
sSeq = 'Seq'
sMer = GC.S_MER
sFull = GC.S_FULL
sEffCode = GC.S_EFF_CODE
sEffSeq = 'EffectorSeq'
sTargCode = 'Target'
sTargSeq = 'TargetSeq'
sPSite = 'pSite'
sPosPSite = GC.S_POS_P_SITE
sPep = 'pep'
sPepMod = 'pep_mod'
sCCode = 'code'
sCCodeSeq = 'code_seq'
sPepPIP = 'pep_pos_in_prot'
sLenSnip = GC.S_LEN_SNIP
sExpName = GC.S_EXP_NAME
sCodeTrunc = GC.S_CODE_TRUNC
sLoc = GC.S_LOCATION
sAnyEff = GC.S_ANY_EFF
sNumOcc = GC.S_NUM_OCC
sFullSeq = GC.S_FULL_SEQ
sNmer = GC.S_N_MER
sLenNmer = GC.S_LEN_N_MER
sNOcc = GC.S_NUM_OCC
sRowRaw = 'nRows_Raw'
sRowProcI = 'nRows_ProcI'
sRowCombL = 'nRows_CombL'
sRowCombM = 'nRows_CombM'
sRowCombS = 'nRows_CombS'
sRowCombXS = 'nRows_CombXS'
sTotalRaw = 'Total_Raw'
sTotalProcI = 'Total_ProcI'
sTotalCombL = 'Total_CombL'
sTotalCombM = 'Total_CombM'
sTotalCombS = 'Total_CombS'
sTotalCombXS = 'Total_CombXS'
sNaNRaw = 'nNaN_Raw'
sNaNProcI = 'nNaN_ProcI'
sNaNCombL = 'nNaN_CombL'
sNaNCombM = 'nNaN_CombM'
sNaNCombS = 'nNaN_CombS'
sNaNCombXS = 'nNaN_CombXS'

# --- sets --------------------------------------------------------------------
setSDig = GC.SET_S_DIG

# --- lists (2) ---------------------------------------------------------------
lSCKinF = [sEffCode, sTargCode]
lCXclDfrNmerM = [sExpName, sCodeTrunc]
lCXclDfrNmerS = [sPep, sPepMod, sCCode] + lCXclDfrNmerM
lCXclDfrNmerXS = [sCCodeSeq, sPepPIP] + lCXclDfrNmerS
lSortCDfrNmer = [sEffCode, sLenNmer, sNOcc, sNmer]
lSortDirAscDfrNmer = [True, True, False, True]
lRowsResIGen = [sRowRaw, sTotalRaw, sNaNRaw,
                sRowProcI, sTotalProcI, sNaNProcI,
                sRowCombL, sTotalCombL, sNaNCombL,
                sRowCombM, sTotalCombM, sNaNCombM,
                sRowCombS, sTotalCombS, sNaNCombS,
                sRowCombXS, sTotalCombXS, sNaNCombXS]
lKCmb = [GC.S_X_SHORT, GC.S_SHORT, GC.S_MED, GC.S_LONG]
lVCmb = [GC.S_XS, GC.S_S, GC.S_M, GC.S_L]

# === assertions ==============================================================
assert len(lKCmb) == len(lVCmb)

# --- dictionaries ------------------------------------------------------------
dMer = GC.D_MER
dCmb = {cKCmb: cVCmb for cKCmb, cVCmb in zip(lKCmb, lVCmb)}

# === derived values and input processing =====================================
if lLenNmer is None:
    lLenNmer = list(range(1, GC.LEN_N_MER_DEF + 1, 2))
lSCXt = ['', GC.S_PYL, GC.S_TOTAL, GC.S_REL_FREQ, GC.S_PROB]
lSCMer = [str(n) + GC.S_MER for n in lLenNmer]
lSCMerPy = [s + GC.S_USC + lSCXt[1] for s in lSCMer]
lSCMerTt = [s + GC.S_USC + lSCXt[2] for s in lSCMer]
lSCMerRF = [s + GC.S_USC + lSCXt[3] for s in lSCMer]
lSCMerPr = [s + GC.S_USC + lSCXt[4] for s in lSCMer]
dSCMerAll = {n: [sCS, sCY, sCT, sCR, sCP] for (n, sCS, sCY, sCT, sCR, sCP) in
             zip(lLenNmer, lSCMer, lSCMerPy, lSCMerTt, lSCMerRF, lSCMerPr)}
lSCMerAll = GF.flattenIt(dSCMerAll.values())
iStXS = lRowsResIGen.index(sRowCombXS)
iStS = lRowsResIGen.index(sRowCombS)
iStM = lRowsResIGen.index(sRowCombM)
iStL = lRowsResIGen.index(sRowCombL)

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'procInput': procInput,
       'dBDoDfrRes': dBDoDfrRes,
       'genInfoNmer': genInfoNmer,
       'genInfoEff': genInfoEff,
       'genInfoGen': genInfoGen,
       # --- lists (1)
       'lLenNmer': lLenNmer,
       'lSLenNmer': [str(n) for n in lLenNmer],
       'lSCXt': lSCXt,
       'lSCMer': lSCMer,
       'lSCMerPy': lSCMerPy,
       'lSCMerTt': lSCMerTt,
       'lSCMerRF': lSCMerRF,
       'lSCMerPr': lSCMerPr,
       'lSCMerAll': lSCMerAll,
       # --- names and paths of files and dirs
       'sFProcInpKin': sFProcInpKin,
       'sFProcInpNmer': sFProcInpNmer,
       'sFResCombXS': sFResCombXS,
       'sFResCombS': sFResCombS,
       'sFResCombM': sFResCombM,
       'sFResCombL': sFResCombL,
       'sFResIGen': sFResIGen,
       'sFResINmer': sFResINmer,
       'sFResIEff': sFResIEff,
       'pFRawInpKin': GF.joinToPath(pDirRawInp, sFRawInpKin),
       'pFRawInpKin_T': GF.joinToPath(pDirRawInp_Test, sFRawInpKin),
       'pFRawInpNmer': GF.joinToPath(pDirRawInp, sFRawInpNmer),
       'pFRawInpNmer_T': GF.joinToPath(pDirRawInp_Test, sFRawInpNmer),
       'pDirProcInp': pDirProcInp,
       'pDirProcInp_T': pDirProcInp_Test,
       'pDirResComb': pDirResComb,
       'pDirResComb_T': pDirResComb_Test,
       'pDirResInfo': pDirResInfo,
       'pDirResInfo_T': pDirResInfo_Test,
       'pDirResProb': pDirResProb,
       'pDirResProb_T': pDirResProb_Test,
       'pDirResViterbi': pDirResViterbi,
       'pBinData': pBinData,
       'pDirTemp': pDirTemp,
       # --- numbers
       'rndDigProp': rndDigProp,
       # --- strings
       'sSeq': sSeq,
       'sMer': sMer,
       'sFull': sFull,
       'sEffCode': sEffCode,
       'sEffSeq': sEffSeq,
       'sTargCode': sTargCode,
       'sTargSeq': sTargSeq,
       'sPSite': sPSite,
       'sPosPSite': sPosPSite,
       'sPep': sPep,
       'sPepMod': sPepMod,
       'sCCode': sCCode,
       'sCCodeSeq': sCCodeSeq,
       'sPepPIP': sPepPIP,
       'sLenSnip': sLenSnip,
       'sExpName': sExpName,
       'sCodeTrunc': sCodeTrunc,
       'sLoc': sLoc,
       'sAnyEff': sAnyEff,
       'sNumOcc': sNumOcc,
       'sFullSeq': sFullSeq,
       'sNmer': sNmer,
       'sLenNmer': sLenNmer,
       'sNOcc': sNOcc,
       # --- sets
       'setSDig': setSDig,
       # --- lists (2)
       'lSCKinF': lSCKinF,
       'lCXclDfrNmerXS': lCXclDfrNmerXS,
       'lCXclDfrNmerS': lCXclDfrNmerS,
       'lCXclDfrNmerM': lCXclDfrNmerM,
       'lSCDfrNmer': [sEffCode, sNmer, sLenNmer, sNOcc],
       'lSortCDfrNmer': lSortCDfrNmer,
       'lSortDirAscDfrNmer': lSortDirAscDfrNmer,
       'lRResIG': lRowsResIGen,
       'lKCmb': lKCmb,
       'lVCmb': lVCmb,
       # --- dictionaries
       'dMer': dMer,
       'dCmb': dCmb,
       'dSCMerAll': dSCMerAll,
       # === derived values and input processing
       'iStXS': iStXS,
       'iStS': iStS,
       'iStM': iStM,
       'iStL': iStL}
dIO['lSFProcInp'] = [dIO['sFProcInpKin'], dIO['sFProcInpNmer']]

###############################################################################