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
S_NEWL = '\n'
S_0 = '0'
S_CAP_XS, S_CAP_S, S_CAP_M, S_CAP_L = 'XS', 'S', 'M', 'L'

S_US02 = S_USC*2
S_DT03 = S_DOT*3
S_SP04 = S_SPACE*4
S_DS04 = S_DASH*4
S_PL04 = S_PLUS*4
S_ST04 = S_STAR*4
S_DT20 = S_DOT*20
S_DS20 = S_DASH*20
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
S_INFO = 'Info'
S_PYL = 'Pyl'
S_TOTAL = 'Total'
S_COND = 'Cond'
S_PROB = 'Prob'
S_TOTAL_PROB = S_TOTAL + S_PROB
S_COND_PROB = S_COND + S_PROB
S_N_MER = 'N' + S_MER
S_EFF = 'Eff'
S_EFF_F = S_EFF + S_FULL
S_BASE = 'Base'
S_TRAIN = 'Train'
S_TEST = 'Test'
S_C_DFR_COMB = 'cDfr' + S_COMB
S_I_GEN = S_INFO + S_GEN
S_I_MER = S_INFO + S_N_MER
S_I_EFF = S_INFO + S_EFF
S_I_MER_TRAIN = S_USC.join([S_I_MER, S_TRAIN])
S_I_MER_TEST = S_USC.join([S_I_MER, S_TEST])
S_I_EFF_TRAIN = S_USC.join([S_I_EFF, S_TRAIN])
S_I_EFF_TEST = S_USC.join([S_I_EFF, S_TEST])
S_I_EFF_F = S_I_EFF + S_FULL
S_I_EFF_F_TRAIN = S_USC.join([S_I_EFF_F, S_TRAIN])
S_I_EFF_F_TEST = S_USC.join([S_I_EFF_F, S_TEST])
S_B_GEN_INFO_N_MER = 'b' + S_GEN + S_I_MER
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
DIR_RES_COMB = '31_Res' + S_COMBINED
DIR_RES_COMB_TEST = '32_Res' + S_COMBINED + S_USC + S_TEST
DIR_RES_INFO = '33_Res' + S_INFO
DIR_RES_INFO_TEST = '34_Res' + S_INFO + S_USC + S_TEST
DIR_RES_PROB = '35_Res' + S_PROB
DIR_RES_PROB_TEST = '36_Res' + S_PROB + S_USC + S_TEST

P_DIR_INP = os.path.join('..', '..', DIR_INP)
P_DIR_RAW_INP = os.path.join(P_DIR_INP, DIR_RAW_INP)
P_DIR_RAW_INP_TEST = os.path.join(P_DIR_INP, DIR_RAW_INP_TEST)
P_DIR_PROC_INP = os.path.join(P_DIR_INP, DIR_PROC_INP)
P_DIR_PROC_INP_TEST = os.path.join(P_DIR_INP, DIR_PROC_INP_TEST)
P_DIR_RES_COMB = os.path.join(P_DIR_INP, DIR_RES_COMB)
P_DIR_RES_COMB_TEST = os.path.join(P_DIR_INP, DIR_RES_COMB_TEST)
P_DIR_RES_INFO = os.path.join(P_DIR_INP, DIR_RES_INFO)
P_DIR_RES_INFO_TEST = os.path.join(P_DIR_INP, DIR_RES_INFO_TEST)
P_DIR_RES_PROB = os.path.join(P_DIR_INP, DIR_RES_PROB)
P_DIR_RES_PROB_TEST = os.path.join(P_DIR_INP, DIR_RES_PROB_TEST)

S_EXT_PY = 'py'
S_EXT_CSV = 'csv'
S_EXT_PDF = 'pdf'
S_EXT_PTH = 'pth'
S_EXT_BIN = 'bin'

# --- predefined strings (2) --------------------------------------------------
S_MODE = 'Mode'
S_TYPE = 'Type'
S_DATA = 'Data'
S_LEN_N_MER = 'len' + S_N_MER
S_EFF_CODE = 'Effector'
S_C_N_MER = 'c15mer'
S_POS_P_SITE = 'PosPSite'
S_SNIPPET = 'Snippet'
S_LEN_SNIP = 'lenSnip'
S_EXP_NAME = 'exp_name'
S_CODE_TRUNC = 'code_trunc'
S_ANY_EFF = 'AnyEffector'
S_NUM_OCC = 'nOcc'
S_ALL_SEQ_N_MER = 'AllSeq15mer'
S_SEQ_CHECK = 'SeqCheck'
S_I_EFF_INP = 'IEffInp'
S_COMB_INP = S_COMB + 'Inp'
S_TRAIN_DATA = S_TRAIN + S_DATA
S_TEST_DATA = S_TEST + S_DATA

# --- predefined numbers ------------------------------------------------------
LEN_N_MER_DEF = 15
I_CENT_N_MER = LEN_N_MER_DEF//2
R01 = 1
R02 = 2
R03 = 3
R04 = 4
R06 = 6
R08 = 8
R24 = 24
MAX_DELTA = 1.0E-14
MAX_LEN_L_DSP = 5

# --- predefined lists --------------------------------------------------------
L_MER = [1, 3, 5, 7, 9, 11, 13, 15]

# --- predefined dictionaries -------------------------------------------------
D_MER = {n: str(n) + S_MER for n in L_MER}

# --- predefined colours ------------------------------------------------------
CLR_TAB_RED = 'tab:red'
CLR_TAB_GREEN = 'tab:green'
CLR_TAB_BLUE = 'tab:blue'

###############################################################################
