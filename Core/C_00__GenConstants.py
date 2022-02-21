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

S_US02 = S_USC*2
S_SP04 = S_SPACE*4
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
S_0 = '0'

S_CAP_A = 'A'
S_CAP_B = 'B'
S_CAP_C = 'C'

# --- file names, paths and extensions ----------------------------------------
S_F_RAW_INP_KIN = 'KinaseTarget_202202'
S_F_RAW_INP_15MER = 'phosphat_m15'

DIR_INP = '13_Sysbio03_Phospho15mer'
DIR_RAW_INP = '01_RawData'
DIR_RAW_INP_TEST = '02_RawData_Test'
DIR_PROC_INP = '11_ProcInpData'
DIR_PROC_INP_TEST = '12_ProcInpData_Test'
DIR_RES = '31_ResCombined'
DIR_RES_TEST = '32_ResCombined_Test'

P_DIR_INP = os.path.join('..', '..', DIR_INP)
P_DIR_RAW_INP = os.path.join(P_DIR_INP, DIR_RAW_INP)
P_DIR_RAW_INP_TEST = os.path.join(P_DIR_INP, DIR_RAW_INP_TEST)
P_DIR_PROC_INP = os.path.join(P_DIR_INP, DIR_PROC_INP)
P_DIR_PROC_INP_TEST = os.path.join(P_DIR_INP, DIR_PROC_INP_TEST)
P_DIR_RES = os.path.join(P_DIR_INP, DIR_RES)
P_DIR_RES_TEST = os.path.join(P_DIR_INP, DIR_RES_TEST)

S_EXT_PY = 'py'
S_EXT_CSV = 'csv'
S_EXT_PDF = 'pdf'
S_EXT_PTH = 'pth'

# --- predefined strings (2) --------------------------------------------------
S_MODE = 'Mode'
S_TYPE = 'Type'
S_MD_A = S_CAP_A
S_MD_B = S_CAP_B
S_MD_C = S_CAP_C
S_POS_P_SITE = 'PosPSite'
S_LEN_SNIP = 'lenSnip'
S_CODE_TRUNC = 'code_trunc'

# --- predefined numbers ------------------------------------------------------
R01 = 1
R02 = 2
R03 = 3
R04 = 4
R08 = 8
MAX_DELTA = 1.0E-14

# --- predefined dictionaries -------------------------------------------------

# --- predefined colours ------------------------------------------------------
CLR_TAB_RED = 'tab:red'
CLR_TAB_GREEN = 'tab:green'
CLR_TAB_BLUE = 'tab:blue'

###############################################################################
