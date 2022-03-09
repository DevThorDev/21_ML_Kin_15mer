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
procInput = False
dBDoDfrRes = {GC.S_SHORT: False,
              GC.S_MED: False,
              GC.S_LONG: False}

# --- lists (1) ---------------------------------------------------------------
lLenNMer4ResIEff = [1]       # empty list: all lengths considered

# --- names and paths of files and dirs ---------------------------------------
sFRawInpKin = GC.S_F_RAW_INP_KIN + GC.S_DOT + GC.S_EXT_CSV
sFRawInp15mer = GC.S_F_RAW_INP_15MER + GC.S_DOT + GC.S_EXT_CSV
sFProcInpKin = 'KinasesAndTargets_202202'
sFProcInp15mer = 'Pho15mer_202202'
sFResCombS = 'Combined_S_KinasesPho15mer_202202'
sFResCombM = 'Combined_M_KinasesPho15mer_202202'
sFResCombL = 'Combined_L_KinasesPho15mer_202202'
sFResI15mer = 'Info15mer_NOcc_202202'
sFResIEff = 'InfoEff_Prop15mer_202202'

pDirRawInp = GC.P_DIR_RAW_INP
pDirRawInp_Test = GC.P_DIR_RAW_INP_TEST
pDirProcInp = GC.P_DIR_PROC_INP
pDirProcInp_Test = GC.P_DIR_PROC_INP_TEST
pDirRes = GC.P_DIR_RES
pDirRes_Test = GC.P_DIR_RES_TEST

# --- numbers -----------------------------------------------------------------
lenSnippetDef = GC.LEN_SNIPPET_DEF
iCent15mer = GC.I_CENT_15MER
rndDigProp = GC.R08

mDsp = 1000

# --- strings -----------------------------------------------------------------
sUSC = GC.S_USC
sUS02 = GC.S_US02
s0 = GC.S_0
sNULL = 'NULL'
sSeq = 'Seq'
sEffCode = 'Effector'
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
sC15mer = 'c15mer'
sLenSnip = GC.S_LEN_SNIP
sExpName = GC.S_EXP_NAME
sCodeTrunc = GC.S_CODE_TRUNC
sAnyEff = GC.S_ANY_EFF
sNmer = GC.S_N_MER
sLenNmer = GC.S_LEN_N_MER
sNOcc = GC.S_NUM_OCC

# --- lists (2) ---------------------------------------------------------------
lSCKinF = [sEffCode, sTargCode]
lCXclDfr15merM = [sExpName, sCodeTrunc]
lCXclDfr15merS = [sPep, sPepMod, sCCode, sCCodeSeq, sPepPIP] + lCXclDfr15merM
lSortCDfrNMer = [sEffCode, sLenNmer, sNOcc, sNmer]
lSortDirAscDfrNMer = [True, True, False, True]

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
       # --- lists (1)
       'lLenNMer': lLenNMer4ResIEff,
       'lSLenNMer': [str(n) for n in lLenNMer4ResIEff],
       # --- names and paths of files and dirs
       'sFProcInpKin': sFProcInpKin + GC.S_DOT + GC.S_EXT_CSV,
       'sFProcInp15mer': sFProcInp15mer + GC.S_DOT + GC.S_EXT_CSV,
       'sFResCombS': sFResCombS + GC.S_DOT + GC.S_EXT_CSV,
       'sFResCombM': sFResCombM + GC.S_DOT + GC.S_EXT_CSV,
       'sFResCombL': sFResCombL + GC.S_DOT + GC.S_EXT_CSV,
       'sFResI15mer': sFResI15mer + GC.S_DOT + GC.S_EXT_CSV,
       'sFResIEff': sFResIEff + GC.S_DOT + GC.S_EXT_CSV,
       'pFRawInpKin': GF.joinToPath(pDirRawInp, sFRawInpKin),
       'pFRawInpKin_T': GF.joinToPath(pDirRawInp_Test, sFRawInpKin),
       'pFRawInp15mer': GF.joinToPath(pDirRawInp, sFRawInp15mer),
       'pFRawInp15mer_T': GF.joinToPath(pDirRawInp_Test, sFRawInp15mer),
       'pDirProcInp': pDirProcInp,
       'pDirProcInp_T': pDirProcInp_Test,
       'pDirRes': pDirRes,
       'pDirRes_T': pDirRes_Test,
       # --- numbers
       'lenSDef': lenSnippetDef,
       'iCent15mer': iCent15mer,
       'rndDigProp': rndDigProp,
       'mDsp': mDsp,
       # --- strings
       'sUSC': sUSC,
       'sUS02': sUS02,
       's0': s0,
       'sNULL': sNULL,
       'sSeq': sSeq,
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
       'sC15mer': sC15mer,
       'sLenSnip': sLenSnip,
       'sExpName': sExpName,
       'sCodeTrunc': sCodeTrunc,
       'sAnyEff': sAnyEff,
       'sNmer': sNmer,
       'sLenNmer': sLenNmer,
       'sNOcc': sNOcc,
       # --- lists (2)
       'lSCKinF': lSCKinF,
       'lCXclDfr15merS': lCXclDfr15merS,
       'lCXclDfr15merM': lCXclDfr15merM,
       'lSCDfrNMer': [sEffCode, sNmer, sLenNmer, sNOcc],
       'lSortCDfrNMer': lSortCDfrNMer,
       'lSortDirAscDfrNMer': lSortDirAscDfrNMer,
       # --- dictionaries
       'dMer': dMer,
       # === derived values and input processing
       }
dIO['lSFProcInp'] = [dIO['sFProcInpKin'], dIO['sFProcInp15mer']]

###############################################################################
