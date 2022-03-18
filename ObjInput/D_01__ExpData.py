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
dBDoDfrRes = {GC.S_SHORT: False,
              GC.S_MED: False,
              GC.S_LONG: False}
genInfoNmer = False
genInfoEff = False
genInfoGen = False

# --- lists (1) ---------------------------------------------------------------
lLenNMer4ResIEff = [1, 3, 5, 7]       # empty list: all lengths considered

# --- names and paths of files and dirs ---------------------------------------
sFRawInpKin = GC.S_F_RAW_INP_KIN + GC.S_DOT + GC.S_EXT_CSV
sFRawInpNmer = GC.S_F_RAW_INP_N_MER + GC.S_DOT + GC.S_EXT_CSV

sFProcInpKin = GC.S_F_PROC_INP_KIN
sFProcInpNmer = GC.S_F_PROC_INP_N_MER

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
pDirRes = GC.P_DIR_RES
pDirRes_Test = GC.P_DIR_RES_TEST

# --- numbers -----------------------------------------------------------------
lenSnippetDef = GC.LEN_SNIPPET_DEF
iCentNmer = GC.I_CENT_N_MER
rndDigProp = GC.R08

# --- strings -----------------------------------------------------------------
sUSC = GC.S_USC
sUS02 = GC.S_US02
s0 = GC.S_0
sNULL = 'NULL'
sSeq = 'Seq'
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
sCNmer = GC.S_C_N_MER
sLenSnip = GC.S_LEN_SNIP
sExpName = GC.S_EXP_NAME
sCodeTrunc = GC.S_CODE_TRUNC
sAnyEff = GC.S_ANY_EFF
sNmer = GC.S_N_MER
sLenNmer = GC.S_LEN_N_MER
sNOcc = GC.S_NUM_OCC
sRowRaw = 'nRows_Raw'
sRowProcI = 'nRows_ProcI'
sRowCombL = 'nRows_CombL'
sRowCombM = 'nRows_CombM'
sRowCombS = 'nRows_CombS'
sTotalRaw = 'Total_Raw'
sTotalProcI = 'Total_ProcI'
sTotalCombL = 'Total_CombL'
sTotalCombM = 'Total_CombM'
sTotalCombS = 'Total_CombS'
sNaNRaw = 'nNaN_Raw'
sNaNProcI = 'nNaN_ProcI'
sNaNCombL = 'nNaN_CombL'
sNaNCombM = 'nNaN_CombM'
sNaNCombS = 'nNaN_CombS'

# --- lists (2) ---------------------------------------------------------------
lSCKinF = [sEffCode, sTargCode]
lCXclDfrNmerM = [sExpName, sCodeTrunc]
lCXclDfrNmerS = [sPep, sPepMod, sCCode, sCCodeSeq, sPepPIP] + lCXclDfrNmerM
lSortCDfrNMer = [sEffCode, sLenNmer, sNOcc, sNmer]
lSortDirAscDfrNMer = [True, True, False, True]
lRowsResIGen = [sRowRaw, sTotalRaw, sNaNRaw,
                sRowProcI, sTotalProcI, sNaNProcI,
                sRowCombL, sTotalCombL, sNaNCombL,
                sRowCombM, sTotalCombM, sNaNCombM,
                sRowCombS, sTotalCombS, sNaNCombS]

# --- dictionaries ------------------------------------------------------------
dMer = GC.D_MER

# === assertions ==============================================================

# === derived values and input processing =====================================

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
       'lLenNMer': lLenNMer4ResIEff,
       'lSLenNMer': [str(n) for n in lLenNMer4ResIEff],
       # --- names and paths of files and dirs
       'sFProcInpKin': sFProcInpKin + GC.S_DOT + GC.S_EXT_CSV,
       'sFProcInpNmer': sFProcInpNmer + GC.S_DOT + GC.S_EXT_CSV,
       'sFResCombS': sFResCombS + GC.S_DOT + GC.S_EXT_CSV,
       'sFResCombM': sFResCombM + GC.S_DOT + GC.S_EXT_CSV,
       'sFResCombL': sFResCombL + GC.S_DOT + GC.S_EXT_CSV,
       'sFResIGen': sFResIGen + GC.S_DOT + GC.S_EXT_CSV,
       'sFResINmer': sFResINmer + GC.S_DOT + GC.S_EXT_CSV,
       'sFResIEff': sFResIEff + GC.S_DOT + GC.S_EXT_CSV,
       'pFRawInpKin': GF.joinToPath(pDirRawInp, sFRawInpKin),
       'pFRawInpKin_T': GF.joinToPath(pDirRawInp_Test, sFRawInpKin),
       'pFRawInpNmer': GF.joinToPath(pDirRawInp, sFRawInpNmer),
       'pFRawInpNmer_T': GF.joinToPath(pDirRawInp_Test, sFRawInpNmer),
       'pDirProcInp': pDirProcInp,
       'pDirProcInp_T': pDirProcInp_Test,
       'pDirRes': pDirRes,
       'pDirRes_T': pDirRes_Test,
       # --- numbers
       'lenSDef': lenSnippetDef,
       'iCentNmer': iCentNmer,
       'rndDigProp': rndDigProp,
       # --- strings
       'sUSC': sUSC,
       'sUS02': sUS02,
       's0': s0,
       'sNULL': sNULL,
       'sSeq': sSeq,
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
       'sCNmer': sCNmer,
       'sLenSnip': sLenSnip,
       'sExpName': sExpName,
       'sCodeTrunc': sCodeTrunc,
       'sAnyEff': sAnyEff,
       'sNmer': sNmer,
       'sLenNmer': sLenNmer,
       'sNOcc': sNOcc,
       # --- lists (2)
       'lSCKinF': lSCKinF,
       'lCXclDfrNmerS': lCXclDfrNmerS,
       'lCXclDfrNmerM': lCXclDfrNmerM,
       'lSCDfrNMer': [sEffCode, sNmer, sLenNmer, sNOcc],
       'lSortCDfrNMer': lSortCDfrNMer,
       'lSortDirAscDfrNMer': lSortDirAscDfrNMer,
       'lRResIG': lRowsResIGen,
       # --- dictionaries
       'dMer': dMer,
       # === derived values and input processing
       }
dIO['lSFProcInp'] = [dIO['sFProcInpKin'], dIO['sFProcInpNmer']]

###############################################################################
