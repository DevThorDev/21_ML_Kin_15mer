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
S_A, S_B, S_C, S_D, S_E = 'A', 'B', 'C', 'D', 'E'
S_F, S_G, S_H, S_I, S_J = 'F', 'G', 'H', 'I', 'J'
S_K, S_L, S_M, S_N, S_O = 'K', 'L', 'M', 'N', 'O'
S_P, S_Q, S_R, S_S, S_T = 'P', 'Q', 'R', 'S', 'T'
S_U, S_V, S_W, S_X, S_Y, S_Z = 'U', 'V', 'W', 'X', 'Y', 'Z'
S_XS, S_XM, S_YS, S_YM = S_X + S_S, S_X + S_M, S_Y + S_S, S_Y + S_M

S_US02 = S_USC*2
S_DT03 = S_DOT*3
S_SP04 = S_SPACE*4
S_EQ04 = S_EQ*4
S_DS04 = S_DASH*4
S_PL04 = S_PLUS*4
S_ST04 = S_STAR*4
S_SP08 = S_SPACE*8
S_EQ08 = S_EQ*8
S_DS08 = S_DASH*8
S_PL08 = S_PLUS*8
S_ST08 = S_STAR*8
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

S_VBAR_SEP = S_SPACE + S_VBAR + S_SPACE
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
S_N_MER = S_N + S_MER
S_NO, S_ANY, S_ALL, S_MULTI = 'No', 'Any', 'All', 'Multi'
S_EFF, S_EFF_S = 'Eff', S_E
S_EFF_F = S_EFF + S_FULL
S_NO_EFF = S_NO + S_EFF
S_FAM, S_FAM_S = 'Fam', S_F
S_FAMILY = S_FAM + 'ily'
S_EFF_FAM, S_EFF_FAM_S = S_EFF + S_FAM, S_EFF_S + S_FAM_S
S_NO_FAM, S_MULTI_FAM = S_NO + S_FAM, S_MULTI + S_FAM
S_BASE = 'Base'
S_TRAIN = 'Train'
S_TEST = 'Test'
S_REP, S_REP_S = 'Rep', S_R
S_FOLD, S_FOLD_S = 'Fold', S_F
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
S_F_PROC_INP_N_MER = 'Pho15mer_wLoc_202202'

DIR_INP = '13_Sysbio03_Phospho15mer'
DIR_RES = DIR_INP
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
DIR_EVAL = '90_Eval'
DIR_TEMP = '98_TEMP_CSV'

P_DIR_INP = os.path.join('..', '..', DIR_INP)
P_DIR_RES = os.path.join('..', '..', DIR_RES)
P_DIR_RAW_INP = os.path.join(P_DIR_INP, DIR_RAW_INP)
P_DIR_RAW_INP_TEST = os.path.join(P_DIR_INP, DIR_RAW_INP_TEST)
P_DIR_PROC_INP = os.path.join(P_DIR_INP, DIR_PROC_INP)
P_DIR_PROC_INP_TEST = os.path.join(P_DIR_INP, DIR_PROC_INP_TEST)
P_DIR_INP_CLF = os.path.join(P_DIR_INP, DIR_INP_CLF)
P_DIR_RES_COMB = os.path.join(P_DIR_INP, DIR_RES_COMB)
P_DIR_RES_COMB_TEST = os.path.join(P_DIR_INP, DIR_RES_COMB_TEST)
P_DIR_RES_INFO = os.path.join(P_DIR_RES, DIR_RES_INFO)
P_DIR_RES_INFO_TEST = os.path.join(P_DIR_RES, DIR_RES_INFO_TEST)
P_DIR_RES_PROB = os.path.join(P_DIR_RES, DIR_RES_PROB)
P_DIR_RES_PROB_TEST = os.path.join(P_DIR_RES, DIR_RES_PROB_TEST)
P_DIR_RES_VITERBI = os.path.join(P_DIR_RES, DIR_RES_VITERBI)
P_DIR_RES_CLF = os.path.join(P_DIR_RES, DIR_RES_CLF)
P_DIR_BIN_DATA = os.path.join(P_DIR_INP, DIR_BIN_DATA)
P_DIR_RES_CLF_PARS = os.path.join(P_DIR_RES_CLF, DIR_PARS)
P_DIR_RES_CLF_SUMMARIES = os.path.join(P_DIR_RES_CLF, DIR_SUMMARIES)
P_DIR_RES_UNQ_N_MER = os.path.join(P_DIR_RES_CLF, DIR_UNQ_N_MER)
P_DIR_RES_INP_DATA_CLF_PRC = os.path.join(P_DIR_RES_CLF, DIR_INP_DATA_CLF_PRC)
P_DIR_RES_CLF_CNF_MAT = os.path.join(P_DIR_RES_CLF, DIR_CNF_MAT)
P_DIR_RES_CLF_DETAILED = os.path.join(P_DIR_RES_CLF, DIR_DETAILED)
P_DIR_RES_CLF_PROP = os.path.join(P_DIR_RES_CLF, DIR_PROP)
P_DIR_RES_CLF_EVAL = os.path.join(P_DIR_RES_CLF, DIR_EVAL)
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
S_X_CL, S_NO_CL = S_X + S_CL, S_NO + S_CL
S_X_NO_CL = S_X + S_NO_CL
S_AS_0 = 'As' + S_0
S_X_NO_CL_AS_0 = S_X_NO_CL + S_AS_0
S_NO_CL_AS_0 = S_NO_CL + S_AS_0
S_CLF, S_PRC = 'Clf', 'PrC'
S_NONE = 'None'
S_OLD = 'Old'
S_NEW = 'New'
S_MEAN, S_AV = 'Mean', 'Av'
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
S_TRUE, S_PRED_S, S_PRED = 'True', 'pred', 'Pred'
S_N_PRED = 'n' + S_PRED
S_PROBA = S_PROB + 'a'
S_PRED_PROBA, S_AV_PROBA = S_PRED + S_PROBA, S_AV + S_PROBA
S_TRUE_CL = S_TRUE + S_CL
S_PRED_CL = S_PRED + S_CL
S_MULTI_CL = S_MULTI + S_CL
S_C_N_MER = 'c15mer'
S_POS_P_SITE = 'PosPSite'
S_SNIPPET = 'Snippet'
S_LEN_SNIP = 'lenSnip'
S_EXP_NAME = 'exp_name'
S_CODE_TRUNC = 'code_trunc'
S_LOC, S_LOCATION = 'Loc', 'location'
S_ANY_EFF = S_ANY + S_EFF_CODE
S_NUM_OCC = 'nOcc'
S_TUP = 'Tup'
S_ALL_SEQ = S_ALL + S_SEQ
S_ALL_SEQ_N_MER = S_ALL_SEQ + S_C_N_MER[1:]
S_AMINO_ACID = 'AminoAcid'
S_SEQ_CHECK = S_SEQ + 'Check'
S_I_EFF_INP = S_I + S_EFF + S_INP
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
S_LAST = 'Last'
S_UNQ, S_UNIQUE = 'Unq', 'Unique'
S_FULL_LIST = S_FULL + S_LIST
S_UNQ_LIST = S_UNIQUE + S_LIST
S_AAC = 'AAc'
S_AAC_CHARGE = S_AAC + 'Charge'
S_AAC_POLAR = S_AAC + 'Polar'
S_KEY = 'Key'
S_LOC_KEY = S_LOC + S_KEY
S_W_N_MER = S_W + S_N_MER
S_SEPARATE = 'Separate'
S_DESC_NONE = 'No Classifier'
S_DESC_DUMMY = 'Dummy Classifier'
S_DESC_ADA = 'AdaBoost Classifier'
S_DESC_RF = 'Random Forest Classifier'
S_DESC_X_TR = 'Extra Trees Classifier'
S_DESC_GR_B = 'Gradient Boosting Classifier'
S_DESC_H_GR_B = 'Hist Gradient Boosting Classifier'
S_DESC_GP = 'Gaussian Process Classifier'
S_DESC_PA_A = 'Passive Aggressive Classifier'
S_DESC_PCT = 'Perceptron Classifier'
S_DESC_SGD = 'Stochastic Gradient Descent Classifier'
S_DESC_CT_NB = 'Categorical Naive Bayes Classifier'
S_DESC_CP_NB = 'Complement Naive Bayes Classifier'
S_DESC_GS_NB = 'Gaussian Naive Bayes Classifier'
S_DESC_MLP = 'Neural Network MLP Classifier'
S_DESC_LSV = 'Linear Support Vector Classifier'
S_DESC_NSV = 'Nu-Support Vector Classifier'
S_MTH_NONE_L, S_MTH_NONE = S_NO + 'Method', S_NO
S_MTH_DUMMY_L, S_MTH_DUMMY = 'DummyMethod', 'Dummy'
S_MTH_ADA_L, S_MTH_ADA = 'AdaBoost', 'Ada'
S_MTH_RF_L, S_MTH_RF = 'RandomForest', 'RF'
S_MTH_X_TR_L, S_MTH_X_TR = 'ExtraTrees', 'XTr'
S_MTH_GR_B_L, S_MTH_GR_B = 'GradientBoosting', 'GrB'
S_MTH_H_GR_B_L, S_MTH_H_GR_B = 'HistGradientBoosting', 'HGrB'
S_MTH_GP_L, S_MTH_GP = 'GaussianProcess', 'GP'
S_MTH_PA_A_L, S_MTH_PA_A = 'PassiveAggressive', 'PaA'
S_MTH_PCT_L, S_MTH_PCT = 'Perceptron', 'Pct'
S_MTH_SGD_L, S_MTH_SGD = 'StochasticGradientDescent', 'SGD'
S_MTH_CT_NB_L, S_MTH_CT_NB = 'CategoricalNaiveBayes', 'CtNB'
S_MTH_CP_NB_L, S_MTH_CP_NB = 'ComplementNaiveBayes', 'CpNB'
S_MTH_GS_NB_L, S_MTH_GS_NB = 'GaussianNaiveBayes', 'GsNB'
S_MTH_MLP_L, S_MTH_MLP = 'NeuralNetworkMLP','MLP'
S_MTH_LSV_L, S_MTH_LSV = 'LinearSupportVector','LSV'
S_MTH_NSV_L, S_MTH_NSV = 'Nu-SupportVector','NSV'
S_PAR, S_GSRS = 'Par', 'GSRS'
S_SUMMARY = 'Summary'
S_CNF_MAT = 'CnfMat'
S_DETAILED = 'Detailed'
S_PROP = 'Prop'
S_EVAL_CL_PRED = 'Eval' + S_CL + S_PRED
S_CLS_CL_PRED = 'Cls' + S_CL + S_PRED
S_CLFR_CL_PRED = 'Clfr' + S_CL + S_PRED
S_LBL, S_STEP = 'Lbl', 'Step'
S_SGL_LBL = S_SGL + S_LBL
S_MLT_LBL = S_MLT + S_LBL
S_N_MER_EFF_FAM = S_N_MER + S_EFF_FAM
S_N_MER_SEQ_UNQ = S_N_MER + S_SEQ + S_UNQ
S_N_OCC_TUP_UNQ = S_NUM_OCC + S_LOC + S_TUP + S_UNQ
S_INP_DATA = S_INP + S_DATA
S_MLT_STEP = S_MLT + S_STEP
S_MAPPING = 'Mapping'
S_CL_MAPPING = S_CL + S_MAPPING
S_LOC_KEY_MAPPING = S_LOC + S_KEY + S_MAPPING
S_CL_STEPS = S_CL + S_STEP + 's'
S_MAX_LEN_S = 'mxL'
S_RESTR = 'rst'
S_I_POS = 'iPos'
S_TO = 'to'
S_WITH_EXCL_EFF_FAM = 'wXcl' + S_EFF_FAM_S
S_NO_EXCL_EFF_FAM = 'noXcl' + S_EFF_FAM_S
S_K_FOLD, S_GROUP_K_FOLD = 'KFold', 'GroupKFold'
S_STRAT_K_FOLD, S_STRAT_GROUP_K_FOLD = 'StratKFold', 'StratGroupKFold'
S_ONE_HOT, S_ORDINAL = 'OneHot', 'Ordinal'
S_SAMPLER, S_SAMPLER_S = 'Sampler', 'Smpl'
S_SMPL_NO, S_SMPL_NO_S = (S_NO + S_SAMPLER, S_NO + S_SAMPLER_S)
S_SMPL_CL_CTR, S_SMPL_CL_CTR_S = ('ClusterCentroids', 'ClCtr' + S_SAMPLER_S)
S_SMPL_ALL_KNN, S_SMPL_ALL_KNN_S = (S_ALL + 'KNN', 'KNN' + S_SAMPLER_S)
S_SMPL_NBH_CL_R, S_SMPL_NBH_CL_R_S = ('NeighbourhoodCleaningRule',
                                      'NbhClR' + S_SAMPLER_S)
S_SMPL_RND_U, S_SMPL_RND_U_S = ('RandomUnderSampler', 'RndU' + S_SAMPLER_S)
S_SMPL_TOM_LKS, S_SMPL_TOM_LKS_S = ('TomekLinks', 'TomLks' + S_SAMPLER_S)
S_STRAT_REAL_MAJO, S_STRAT_REAL_MAJO_S = 'RealMajo', S_R + S_M
S_STRAT_SHARE_MINO, S_STRAT_SHARE_MINO_S = 'ShareMino', S_S + S_M
S_FULL_FIT, S_FULL_FIT_S = 'FullFit', 'FF'
S_PART_FIT, S_PART_FIT_S = 'PartFit', 'PF'
S_0_PRED, S_1_PRED = S_USC.join([S_0, S_PRED_S]), S_USC.join([S_1, S_PRED_S])
S_NOT_PRED = S_USC.join(['not', S_PRED_S])

# --- sets for X class mapping ------------------------------------------------
S_SET_01 = 'Set01_11Cl'
S_SET_02 = 'Set02_06Cl'
S_SET_03 = 'Set03_05Cl'
S_SET_04 = 'Set04_14Cl'
S_SET_05 = 'Set05_09Cl'
S_SET_06 = 'Set06_06Cl'
S_SET_07 = 'Set07_05Cl'
S_SET_08 = 'Set08_07Cl'

# --- predefined numbers ------------------------------------------------------
LEN_N_MER_DEF = 15
I_CENT_N_MER = LEN_N_MER_DEF//2
N_REP_0 = 0
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

# === derived values and input processing =====================================
D_S_SMPL = {S_SMPL_NO: S_SMPL_NO_S,
            S_SMPL_CL_CTR: S_SMPL_CL_CTR_S,
            S_SMPL_ALL_KNN: S_SMPL_ALL_KNN_S,
            S_SMPL_NBH_CL_R: S_SMPL_NBH_CL_R_S,
            S_SMPL_RND_U: S_SMPL_RND_U_S,
            S_SMPL_TOM_LKS: S_SMPL_TOM_LKS_S}
D_S_STRAT_TO_S = {S_STRAT_REAL_MAJO: S_STRAT_REAL_MAJO_S,
                  S_STRAT_SHARE_MINO: S_STRAT_SHARE_MINO_S}

###############################################################################