# -*- coding: utf-8 -*-
###############################################################################
# --- C_00__GenConstants.py ---------------------------------------------------
###############################################################################
import os

# --- general -----------------------------------------------------------------
N_DIG_OBJ_2 = 2

# --- predefined strings (1) --------------------------------------------------
S_OBJINP = 'ObjInput'
S_OBJINP_PRE = 'D_'

S_SPACE = ' '
S_DOT = '.'
S_P_SMALL = 'p'
S_PERC = '%'
S_COMMA = ','
S_SEMICOL = ';'
S_COL = ':'
S_DASH = '-'
S_EQ = '='
S_PLUS = '+'
S_STAR = '*'
S_USC = '_'
S_WAVE = '~'
S_VBAR = '|'
S_SLASH = '/'
S_DBLBACKSL = '\\'
S_QMK = '?'
S_TAB = '\t'
S_NEWL = '\n'
S_NULL = 'NULL'
S_0, S_1 = '0', '1'
S_CAP_C = 'C'
S_CAP_E = 'E'
S_CAP_X = 'X'
S_CAP_Y = 'Y'
S_CAP_XS, S_CAP_S, S_CAP_M, S_CAP_L = S_CAP_X + 'S', 'S', 'M', 'L'

S_US02 = S_USC*2
S_DT03 = S_DOT*3
S_SP04 = S_SPACE*4
S_DS04 = S_DASH*4
S_PL04 = S_PLUS*4
S_ST04 = S_STAR*4
S_DT20 = S_DOT*20
S_DS20 = S_DASH*20
S_PL20 = S_PLUS*20
S_EQ20 = S_EQ*20
S_DS24 = S_DASH*24
S_PL24 = S_PLUS*24
S_ST24 = S_STAR*24
S_ST25 = S_STAR*25
S_DS29 = S_DASH*29
S_EQ29 = S_EQ*29
S_DS30 = S_DASH*30
S_DS31 = S_DASH*31
S_ST31 = S_STAR*31
S_ST32 = S_STAR*32
S_DS33 = S_DASH*33
S_ST33 = S_STAR*33
S_ST34 = S_STAR*34
S_ST35 = S_STAR*35
S_DS36 = S_DASH*36
S_DS37 = S_DASH*37
S_DT80 = S_DOT*80
S_DS80 = S_DASH*80
S_EQ80 = S_EQ*80
S_ST80 = S_STAR*80
S_WV80 = S_WAVE*80

S_ARR_LR = '-->'
S_ARR_RL = '<--'
S_MER = 'mer'

S_X_SHORT = 'Xshort'
S_SHORT = 'short'
S_MED = 'med'
S_LONG = 'long'
S_FULL = 'Full'
S_GEN = 'Gen'
S_COMB = 'Comb'
S_COMBINED = S_COMB + 'ined'
S_COMBINED_INP = S_COMBINED + 'Input'
S_COMBINED_OUT = S_COMBINED + 'Output'
S_PROC = 'Proc'
S_INFO = 'Info'
S_PYL = 'Pyl'
S_TOTAL = 'Total'
S_REL_FREQ = 'RelF'
S_COND = 'Cond'
S_PROB = 'Prob'
S_TOTAL_PROB = S_TOTAL + S_PROB
S_COND_PROB = S_COND + S_PROB
S_LN_PROB = 'ln(' + S_PROB + ')'
S_N_MER = 'N' + S_MER
S_NO = 'No'
S_EFF = 'Eff'
S_EFF_F = S_EFF + S_FULL
S_NO_EFF = S_NO + S_EFF
S_FAM = 'Fam'
S_FAMILY = S_FAM + 'ily'
S_EFF_FAM = S_EFF + S_FAM
S_NO_FAM = S_NO + S_FAM
S_BASE = 'Base'
S_TRAIN = 'Train'
S_TEST = 'Test'
S_C_DFR_COMB = 'cDfr' + S_COMB
S_I_GEN = S_INFO + S_GEN
S_I_N_MER = S_INFO + S_N_MER
S_I_EFF = S_INFO + S_EFF
S_I_N_MER_TRAIN = S_USC.join([S_I_N_MER, S_TRAIN])
S_I_N_MER_TEST = S_USC.join([S_I_N_MER, S_TEST])
S_I_EFF_TRAIN = S_USC.join([S_I_EFF, S_TRAIN])
S_I_EFF_TEST = S_USC.join([S_I_EFF, S_TEST])
S_I_EFF_F = S_I_EFF + S_FULL
S_I_EFF_F_TRAIN = S_USC.join([S_I_EFF_F, S_TRAIN])
S_I_EFF_F_TEST = S_USC.join([S_I_EFF_F, S_TEST])
S_B_GEN_INFO_N_MER = 'b' + S_GEN + S_I_N_MER
S_B_GEN_INFO_EFF = 'b' + S_GEN + S_I_EFF

# --- file names, paths and extensions ----------------------------------------
S_F_RAW_INP_KIN = 'KinaseTarget_202202'
S_F_RAW_INP_N_MER = 'phosphat_m15'

S_F_PROC_INP_KIN = 'KinasesAndTargets_202202'
S_F_PROC_INP_N_MER = 'Pho15mer_202202'

DIR_INP = '13_Sysbio03_Phospho15mer'
DIR_RAW_INP = '01_RawData'
DIR_RAW_INP_TEST = '02_RawData' + S_USC + S_TEST
DIR_PROC_INP = '11_ProcInpData'
DIR_PROC_INP_TEST = '12_ProcInpData' + S_USC + S_TEST
DIR_INP_CLF = '21_InpDataClf'
DIR_RES_COMB = '31_Res' + S_COMBINED
DIR_RES_COMB_TEST = '32_Res' + S_COMBINED + S_USC + S_TEST
DIR_RES_INFO = '33_Res' + S_INFO
DIR_RES_INFO_TEST = '34_Res' + S_INFO + S_USC + S_TEST
DIR_RES_PROB = '35_Res' + S_PROB
DIR_RES_PROB_TEST = '36_Res' + S_PROB + S_USC + S_TEST
DIR_RES_VITERBI = '37_ResViterbi'
DIR_RES_CLF = '39_ResClf'
DIR_BIN_DATA = '45_BinData'
DIR_PARS = '00_Pars'
DIR_SUMMARIES = '01_Summaries'
DIR_UNQ_N_MER = '11_UniqueNmer'
DIR_INP_DATA_CLF_PRC = '12_InpDataClfPrC'
DIR_CNF_MAT = '21_CnfMat'
DIR_DETAILED = '31_Detailed'
DIR_PROP = '41_Prop'
DIR_TEMP = '98_TEMP_CSV'

P_DIR_INP = os.path.join('..', '..', DIR_INP)
P_DIR_RAW_INP = os.path.join(P_DIR_INP, DIR_RAW_INP)
P_DIR_RAW_INP_TEST = os.path.join(P_DIR_INP, DIR_RAW_INP_TEST)
P_DIR_PROC_INP = os.path.join(P_DIR_INP, DIR_PROC_INP)
P_DIR_PROC_INP_TEST = os.path.join(P_DIR_INP, DIR_PROC_INP_TEST)
P_DIR_INP_CLF = os.path.join(P_DIR_INP, DIR_INP_CLF)
P_DIR_RES_COMB = os.path.join(P_DIR_INP, DIR_RES_COMB)
P_DIR_RES_COMB_TEST = os.path.join(P_DIR_INP, DIR_RES_COMB_TEST)
P_DIR_RES_INFO = os.path.join(P_DIR_INP, DIR_RES_INFO)
P_DIR_RES_INFO_TEST = os.path.join(P_DIR_INP, DIR_RES_INFO_TEST)
P_DIR_RES_PROB = os.path.join(P_DIR_INP, DIR_RES_PROB)
P_DIR_RES_PROB_TEST = os.path.join(P_DIR_INP, DIR_RES_PROB_TEST)
P_DIR_RES_VITERBI = os.path.join(P_DIR_INP, DIR_RES_VITERBI)
P_DIR_RES_CLF = os.path.join(P_DIR_INP, DIR_RES_CLF)
P_DIR_BIN_DATA = os.path.join(P_DIR_INP, DIR_BIN_DATA)
P_DIR_RES_CLF_PARS = os.path.join(P_DIR_RES_CLF, DIR_PARS)
P_DIR_RES_CLF_SUMMARIES = os.path.join(P_DIR_RES_CLF, DIR_SUMMARIES)
P_DIR_RES_UNQ_N_MER = os.path.join(P_DIR_RES_CLF, DIR_UNQ_N_MER)
P_DIR_RES_INP_DATA_CLF_PRC = os.path.join(P_DIR_RES_CLF, DIR_INP_DATA_CLF_PRC)
P_DIR_RES_CLF_CNF_MAT = os.path.join(P_DIR_RES_CLF, DIR_CNF_MAT)
P_DIR_RES_CLF_DETAILED = os.path.join(P_DIR_RES_CLF, DIR_DETAILED)
P_DIR_RES_CLF_PROP = os.path.join(P_DIR_RES_CLF, DIR_PROP)
P_DIR_TEMP = os.path.join(P_DIR_INP, DIR_TEMP)

S_EXT_PY = 'py'
S_EXT_CSV = 'csv'
S_EXT_PDF = 'pdf'
S_EXT_PTH = 'pth'
S_EXT_BIN = 'bin'

# --- file name extensions ----------------------------------------------------
XT_PY = S_DOT + S_EXT_PY
XT_CSV = S_DOT + S_EXT_CSV
XT_PDF = S_DOT + S_EXT_PDF
XT_PTH = S_DOT + S_EXT_PTH
XT_BIN = S_DOT + S_EXT_BIN

# --- predefined strings (2) --------------------------------------------------
S_MODE = 'Mode'
S_TYPE = 'Type'
S_DATA = 'Data'
S_SEQ = 'Seq'
S_INP = 'Inp'
S_OUT = 'Out'
S_CL = 'Cl'
S_X_CL = S_CAP_X + S_CL
S_CLF, S_PRC = 'Clf', 'PrC'
S_NONE = 'None'
S_OLD = 'Old'
S_NEW = 'New'
S_MEAN = 'Mean'
S_SD = 'SD'
S_SEM = 'SEM'
S_PATH = 'Path'
S_START, S_CENTRE, S_END = 'Start', 'Centre', 'End'
S_JOIN = 'Join'
S_L_F_S = 'lFileComp' + S_START
S_L_F_C = 'lFileComp' + S_CENTRE
S_L_F_E = 'lFileComp' + S_END
S_L_F_J = 'lFileComp' + S_JOIN
S_F_XT = 'FileXt'
S_FULL_SEQ = S_FULL + S_SEQ
S_N_MER_SEQ = S_N_MER + S_SEQ
S_LEN_N_MER = 'len' + S_N_MER
S_EFF_CODE = S_EFF + 'ector'
S_EFF_CL = S_EFF_CODE + S_CL
S_EFF_FAMILY = S_EFF_FAM + 'ily'
S_TRUE_CL = 'True' + S_CL
S_PRED_CL = 'Pred' + S_CL
S_C_N_MER = 'c15mer'
S_POS_P_SITE = 'PosPSite'
S_SNIPPET = 'Snippet'
S_LEN_SNIP = 'lenSnip'
S_EXP_NAME = 'exp_name'
S_CODE_TRUNC = 'code_trunc'
S_ANY_EFF = 'Any' + S_EFF_CODE
S_NUM_OCC = 'nOcc'
S_ALL_SEQ = 'AllSeq'
S_ALL_SEQ_N_MER = S_ALL_SEQ + S_C_N_MER[1:]
S_AMINO_ACID = 'AminoAcid'
S_SEQ_CHECK = S_SEQ + 'Check'
S_I_EFF_INP = 'I' + S_EFF + S_INP
S_PROC_INP = S_PROC + S_INP
S_COMB_INP = S_COMB + S_INP
S_TRAIN_DATA = S_TRAIN + S_DATA
S_TEST_DATA = S_TEST + S_DATA
S_PREV = 'prev'
S_STATE = 'State'
S_START_PROB = 'Start' + S_PROB
S_SGL = 'Sgl'
S_MLT = 'Mlt'
S_SGL_POS_OCC = S_SGL + 'PosNOcc'
S_SGL_POS_FRQ = S_SGL + 'PosRFreq'
S_LIST = 'List'
S_UNQ = 'Unq'
S_UNIQUE = 'Unique'
S_FULL_LIST = S_FULL + S_LIST
S_UNQ_LIST = S_UNIQUE + S_LIST
S_AAC = 'AAc'
S_AAC_CHARGE = S_AAC + 'Charge'
S_AAC_POLAR = S_AAC + 'Polar'
S_MTH_RF_L = 'RandomForest'
S_MTH_RF = 'RF'
S_MTH_MLP_L = 'NeuralNetworkMLP'
S_MTH_MLP = 'MLP'
S_PAR = 'Par'
S_SUMMARY = 'Summary'
S_CNF_MAT = 'CnfMat'
S_DETAILED = 'Detailed'
S_PROP = 'Prop'
S_CL_MAPPING = S_CL + 'Mapping'
S_CL_STEPS = S_CL + 'Steps'
S_MAX_LEN_S = 'mxL'
S_RESTR = 'restr'
S_I_POS = 'iPos'
S_WITH_EXCL_EFF_FAM = 'withXcl' + S_EFF_FAM
S_NO_EXCL_EFF_FAM = 'noXcl' + S_EFF_FAM
S_LBL, S_STEP = 'Lbl', 'Step'
S_SGL_LBL = S_SGL + S_LBL
S_MLT_LBL = S_MLT + S_LBL
S_N_MER_EFF_FAM = S_N_MER + S_EFF_FAM
S_N_MER_SEQ_UNQ = S_N_MER + S_SEQ + S_UNQ
S_INP_DATA = S_INP + S_DATA
S_MLT_STEP = S_MLT + S_STEP
S_STRAT_REAL_MAJO = 'RealMajo'

# --- sets for X class mapping ------------------------------------------------
S_SET_01 = 'Set01_11Cl'
S_SET_02 = 'Set02_06Cl'
S_SET_03 = 'Set03_05Cl'
S_SET_04 = 'Set04_14Cl'
S_SET_05 = 'Set05_09Cl'
S_SET_06 = 'Set06_06Cl'

# --- predefined numbers ------------------------------------------------------
LEN_N_MER_DEF = 15
I_CENT_N_MER = LEN_N_MER_DEF//2
R01 = 1
R02 = 2
R03 = 3
R04 = 4
R08 = 8
R24 = 24
MAX_DELTA = 1.0E-14
MAX_LEN_L_DSP = 5

# --- predefined sets ---------------------------------------------------------
SET_S_DIG = {str(k) for k in range(10)}

# --- predefined lists --------------------------------------------------------
L_MER = [1, 3, 5, 7, 9, 11, 13, 15]
L_S_DIG = []

# --- predefined dictionaries -------------------------------------------------
D_MER = {n: str(n) + S_MER for n in L_MER}

# --- predefined colours ------------------------------------------------------
CLR_TAB_RED = 'tab:red'
CLR_TAB_GREEN = 'tab:green'
CLR_TAB_BLUE = 'tab:blue'

###############################################################################