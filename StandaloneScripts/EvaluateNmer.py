# -*- coding: utf-8 -*-
###############################################################################
# --- EvaluateNmer.py ---------------------------------------------------------
###############################################################################
import os

# import numpy as np
import pandas as pd

# ### CONSTANTS ###############################################################
# --- strings (1) -------------------------------------------------------------
S_X_SHORT = 'Xshort'
S_SHORT = 'short'
S_MED = 'med'
S_LONG = 'long'
S_COMBINED = 'Combined'
S_I_MER = 'InfoNmer'
S_I_EFF = 'InfoEff'

# --- file names, paths and extensions ----------------------------------------
DIR_INP = '13_Sysbio03_Phospho15mer'
DIR_PROC_INP = '11_ProcInpData'
DIR_RES = '31_Res' + S_COMBINED

# P_DIR_INP = os.path.join('..', '..', DIR_INP)
P_DIR_INP = os.path.join('..', '..', '..', DIR_INP)
P_DIR_PROC_INP = os.path.join(P_DIR_INP, DIR_PROC_INP)
P_DIR_RES = os.path.join(P_DIR_INP, DIR_RES)

# --- strings (2) -------------------------------------------------------------
S_DOT = '.'
S_SEMICOL = ';'
S_NEWL = '\n'
S_DASH = '-'
S_USC = '_'
S_CSV = 'csv'
S_PDF = 'pdf'

S_0 = '0'
S_CAP_O, S_CAP_S, S_CAP_D = 'O', 'S', 'D'
S_IN = 'In'
S_OUT = 'Out'
S_X_DAT = 'XDat'
S_MN_SD_DAT = 'MnSDDat'
S_DATA_FRAME_REP = 'DataFrameRep'

S_RES_TYPE = 'ResultType'
S_MIX_TYPE = 'MixType'
S_CLASS = 'SubstanceClass'
S_GT = 'Genotype'

S_X_DAT_TP = S_0

S_MN, S_SD = 'Mean', 'SD'
S_MIX_TP_A, S_MIX_TP_B, S_MIX_TP_C = 'A', 'B', 'C'
S_MET, S_PHO = 'Metabolite', 'Phosphopeptide'
S_GT0, S_GT1, S_GT5 = 'GT0', 'GT1', 'GT5'

S_OV = 'Ov'
S_ROW = 'Row'
S_COL = 'Col'
S_REP = 'Rep'
S_REPS = S_REP + 's'

S_OV_MN_ALL = S_OV + 'MeanAll'
S_OV_SD_ALL = S_OV + 'SDAll'
S_AV_ARI_SD_ALL = 'AriAvSD'
S_AV_GEO_SD_ALL = 'GeoAvSD'
S_OV_MIN_ALL = S_OV + 'MinAll'
S_OV_MAX_ALL = S_OV + 'MaxAll'

S_OV_MN_REP = S_OV + 'MeanRep'
S_OV_SD_REP = S_OV + 'SDRep'
S_OV_MIN_REP = S_OV + 'MinRep'
S_OV_MAX_REP = S_OV + 'MaxRep'

S_ROW_MN_REP = S_ROW + 'MeanRep'
S_ROW_SD_REP = S_ROW + 'SDRep'
S_COL_MN_REP = S_COL + 'MeanRep'
S_COL_SD_REP = S_COL + 'SDRep'

S_EFF = 'Effector'
S_ANY_EFF = 'AnyEffector'
S_N_MER = 'Nmer'
S_LEN_N_MER = 'lenNmer'

# --- predefined numbers ------------------------------------------------------
LEN_SNIPPET_DEF = 15
I_CENT_N_MER = LEN_SNIPPET_DEF//2

# --- file name extensions ----------------------------------------------------
xtCSV = S_DOT + S_CSV
xtPDF = S_DOT + S_PDF

# --- lists -------------------------------------------------------------------
L_S_MN_SD = [S_MN, S_SD]
L_S_MIX_TP = [S_MIX_TP_A, S_MIX_TP_B, S_MIX_TP_C]
L_S_XD_MIX_TP = [S_X_DAT_TP] + L_S_MIX_TP
L_S_MET_PHO = [S_MET, S_PHO]
L_S_GT = [S_GT0, S_GT1, S_GT5]
L_S_OV_STATS_REP = [S_OV_MN_REP, S_OV_SD_REP, S_OV_MIN_REP, S_OV_MAX_REP]

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
readProcInpKin = False
readProcInpNmer = True
readResCombS = False
readResPropNmer = False

# --- names and paths of files and dirs ---------------------------------------
pDirProcInp = P_DIR_PROC_INP
pDirRes = P_DIR_RES
sFProcInpKin = 'KinasesAndTargets_202202'
sFProcInpNmer = 'Pho15mer_202202'
sFResINmerNOcc = 'Info15mer_NOcc_202202__1_3_5_7__Train'
sFResCombS = 'Combined_S_KinasesPho15mer_202202'
sFResPropNmer = 'InfoEff_Prop15mer_202202__1_3_5_7'

# --- numbers -----------------------------------------------------------------
lenSnippetDef = LEN_SNIPPET_DEF
iCentNmer = I_CENT_N_MER
iStartCS = 0
iEndCS = 10

# --- lists -------------------------------------------------------------------
lSEff = [None]

# --- dictionaries ------------------------------------------------------------

# === derived input and assertions ============================================
if None in lSEff:
    lSEff = [S_ANY_EFF] + [sEff for sEff in lSEff if sEff is not None]

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'readProcInpKin': readProcInpKin,
        'readProcInpNmer': readProcInpNmer,
        'readResCombS': readResCombS,
        'readResPropNmer': readResPropNmer,
        # --- names and paths of files and dirs -------------------------------
        'pDirProcInp': pDirProcInp,
        'pDirRes': pDirRes,
        'sFProcInpKin': sFProcInpKin,
        'sFProcInpNmer': sFProcInpNmer,
        'sFResCombS': sFResCombS,
        'sFResPropNmer': sFResPropNmer,
        'pFProcInpKin': os.path.join(pDirProcInp, sFProcInpKin + xtCSV),
        'pFProcInpNmer': os.path.join(pDirProcInp, sFProcInpNmer + xtCSV),
        'pFResINmerNOcc': os.path.join(pDirRes, sFResINmerNOcc + xtCSV),
        'pFResCombS': os.path.join(pDirRes, sFResCombS + xtCSV),
        'pFResPropNmer': os.path.join(pDirRes, sFResPropNmer + xtCSV),
        # --- strings ---------------------------------------------------------
        's0': S_0,
        'sO': S_CAP_O,
        'sS': S_CAP_S,
        'sD': S_CAP_D,
        'sIn': S_IN,
        'sOut': S_OUT,
        'sXDat': S_X_DAT,
        'sMnSDDat': S_MN_SD_DAT,
        'sDfrRep': S_DATA_FRAME_REP,
        'sDfrRepO': S_DATA_FRAME_REP + S_CAP_O,
        'sDfrRepS': S_DATA_FRAME_REP + S_CAP_S,
        'sDfrRepD': S_DATA_FRAME_REP + S_CAP_D,
        'sResType': S_RES_TYPE,
        'sMixType': S_MIX_TYPE,
        'sClass': S_CLASS,
        'sGT': S_GT,
        'sXDatTp': S_X_DAT_TP,
        'sMn': S_MN,
        'sSD': S_SD,
        'sMixTpA': S_MIX_TP_A,
        'sMixTpB': S_MIX_TP_B,
        'sMixTpC': S_MIX_TP_C,
        'sMet': S_MET,
        'sPho': S_PHO,
        'sGT0': S_GT0,
        'sGT1': S_GT1,
        'sGT5': S_GT5,
        'sOv': S_OV,
        'sRow': S_ROW,
        'sCol': S_COL,
        'sRep': S_REP,
        'sReps': S_REPS,
        'sOvMnAll': S_OV_MN_ALL,
        'sOvSDAll': S_OV_SD_ALL,
        'sAvAriSDAll': S_AV_ARI_SD_ALL,
        'sAvGeoSDAll': S_AV_GEO_SD_ALL,
        'sOvMinAll': S_OV_MIN_ALL,
        'sOvMaxAll': S_OV_MAX_ALL,
        'sOvMnRep': S_OV_MN_REP,
        'sOvSDRep': S_OV_SD_REP,
        'sOvMinRep': S_OV_MIN_REP,
        'sOvMaxRep': S_OV_MAX_REP,
        'sRowMnRep': S_ROW_MN_REP,
        'sRowSDRep': S_ROW_SD_REP,
        'sColMnRep': S_COL_MN_REP,
        'sColSDRep': S_COL_SD_REP,
        # --- file name extensions --------------------------------------------
        'xtCSV': xtCSV,
        'xtPDF': xtPDF,
        # --- lists -----------------------------------------------------------
        'lSMnSD': L_S_MN_SD,
        'lSMixTp': L_S_MIX_TP,
        'lSXDMixTp': L_S_XD_MIX_TP,
        'lSMetPho': L_S_MET_PHO,
        'lSGT': L_S_GT,
        'lSOvStatsRep': L_S_OV_STATS_REP,
        'lSEff': lSEff,
        # --- numbers ---------------------------------------------------------
        'lenSnippetDef': lenSnippetDef,
        'iCentNmer': iCentNmer,
        'iStartCS': iStartCS,
        'iEndCS': iEndCS,
        'NSit': len(L_S_MN_SD)*len(L_S_MIX_TP)*len(L_S_MET_PHO)*len(L_S_GT),
        # --- dictionaries ----------------------------------------------------
        }

# ### FUNCTIONS ###############################################################
# --- General file system related functions -----------------------------------
def createDir(pF):
    if not os.path.isdir(pF):
        os.mkdir(pF)

def joinToPath(pF='', nmF='Dummy.txt'):
    if len(pF) > 0:
        createDir(pF)
        return os.path.join(pF, nmF)
    else:
        return nmF

def readCSV(pF, iCol=None, dDTp=None, cSep=S_SEMICOL):
    if os.path.isfile(pF):
        return pd.read_csv(pF, sep=cSep, index_col=iCol, dtype=dDTp)

def saveAsCSV(pdDfr, pF, reprNA='', cSep=S_SEMICOL):
    if pdDfr is not None:
        pdDfr.to_csv(pF, sep=cSep, na_rep=reprNA)

# --- Functions for filling dictionaries --------------------------------------
def fillDNumBothDir(d1Dir):
    pass

def getDNmer(dfrINmerNOcc, dINmerNOccSL):
    print('Dictionary columns:')
    print(dfrINmerNOcc.columns.to_list())
    lNmer, lLenNmer = list(dfrINmerNOcc[S_N_MER]), list(dfrINmerNOcc[S_LEN_N_MER])
    assert len(lNmer) == len(lLenNmer)
    for sNmer, cLenNmer in zip(lNmer, lLenNmer):
        addToDictL(dINmerNOccSL, cLenNmer, sNmer)

def scanCodeSeq():
    pass

# # --- Helper functions --------------------------------------------------------
# def conVDRepToList(dInpMnSD, tSit, iT=1):
#     assert len(dInpMnSD[tSit]) >= iT + 1
#     dLRep = dInpMnSD[tSit][iT]
#     return sorted(list(set.union(*[set(cL) for cL in dLRep.values()])))
def findAllSSubInStr(sFull, sSub, overLap=False):
    i = sFull.find(sSub)
    while i >= 0:
        yield i
        i = sFull.find(sSub, i + (1 if overLap else len(sSub)))

def startPosToCentPos(iPSt, sSeq):
    return iPSt + len(sSeq)//2

def addToDictL(cD, cK, cE, lUniqEl=False):
    if cK in cD:
        if not lUniqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def printDINmerNOcc(cDINmerNOcc, maxLenL=3):
    for cK, cL in cDINmerNOcc.items():
        print(cK, ': (', len(cL), ' elements)', cL[:maxLenL], '...', sep='')
    print('Total number of sequences:', sum([len(cL) for cL in
                                             cDINmerNOcc.values()]))

# # --- Functions initialising numpy arrays and pandas DataFrames ---------------
# def iniNpArr(data=None, shape=(0, 0), fillV=np.nan):
#     if data is None:
#         return np.full(shape, fillV)
#     else:       # ignore shape
#         return np.array(data)

# def iniDStatsRepS(dI, nRep):
#     dStats = {}
#     for sTp in dI['lSMixTp']:
#         for sK in dI['lSOvStatsRep']:
#             dStats[S_USC.join([sTp, sK])] = iniNpArr(shape=(nRep,))
#     return dStats

# # --- Functions calculating result dictionaries -------------------------------
# def getDataSit(dI, d2Dfr, dLRep, tSit, sTp):
#     (sResTp, sClass, sGT), sP = tSit, dI['dPRel'][dI['sIn']][dI['sMnSDDat']]
#     d2Dfr[sTp] = {}
#     for sF in os.listdir(sP):
#         lSpl = sF.split(S_DOT)
#         assert len(lSpl) >= 2
#         sNmF, sXt = S_DOT.join(lSpl[:-1]), lSpl[-1]
#         if sXt == S_CSV:
#             lSplNm = sNmF.split(S_USC)
#             if len(lSplNm) in [6, 7]:           # 6: sXDatTp; 7: sMixTp...
#                 if len(lSplNm) == 6:
#                     (cResTp, cTp, cClass), cGT = lSplNm[:3], lSplNm[-1]
#                     sRp = dI['sXDatTp']         # exp. data --> sRp == sXDatTp
#                 else:
#                     (cResTp, cTp, sRp, cClass), cGT = lSplNm[:4], lSplNm[-1]
#                 if (cResTp == sResTp and cTp in [dI['sXDatTp'], sTp] and
#                     cClass == sClass and cGT == sGT):
#                     addToDictL(dLRep, cTp, int(sRp), lUniqEl=True)
#                     d2Dfr[sTp][int(sRp)] = readCSV(joinToPath(sP, sF), iCol=0)
#             else:
#                 print('ERROR: lSplNm =', lSplNm, 'with length', len(lSplNm))

# def getDMnSDDat(dI):
#     dInpMnSD, k = {}, 0
#     for sResTp in dI['lSMnSD']:
#         for sClass in dI['lSMetPho']:
#             for sGT in dI['lSGT']:
#                 tSit, d2DfrSit, dLRep = (sResTp, sClass, sGT), {}, {}
#                 for sTp in dI['lSMixTp']:
#                     k += 1
#                     # tSit = (sResTp, sTp, sClass, sGT)
#                     getDataSit(dI, d2DfrSit, dLRep, tSit, sTp)
#                     print('Retrieved mean and SD data [', k, 'of', dI['NSit'],
#                           '] for situation', tSit, 'and type', sTp)
#                 dInpMnSD[tSit] = (d2DfrSit, dLRep)
#     return dInpMnSD

# # --- Functions calculating means and standard deviations ---------------------
# def getStatsCRep(dI, dStatsRep, cDfr, sTp, iL=0):
#     dStatsRep[S_USC.join([sTp, dI['sOvMnRep']])][iL] = cDfr.stack().mean()
#     dStatsRep[S_USC.join([sTp, dI['sOvSDRep']])][iL] = cDfr.stack().std()
#     dStatsRep[S_USC.join([sTp, dI['sOvMinRep']])][iL] = cDfr.stack().min()
#     dStatsRep[S_USC.join([sTp, dI['sOvMaxRep']])][iL] = cDfr.stack().max()

# def fillDfrStatsRepS(dI, dInpMnSD, dDfrRepS, tSit, k=0, iT=0):
#     lRep = conVDRepToList(dInpMnSD, tSit)
#     dStats = iniDStatsRepS(dI, len(lRep))
#     for sTp in dI['lSMixTp']:
#         k += 1
#         for cRep in lRep:
#             cDfr = dInpMnSD[tSit][iT][sTp][cRep]
#             getStatsCRep(dI, dStats, cDfr, sTp=sTp, iL=cRep)
#         print('Filled and saved DataFrame [', k, 'of', dI['NSit'],
#               '] for situation', tSit, 'and type', sTp)
#     dDfrRepS[tSit] = pd.DataFrame(dStats, index=lRep)
#     return k

# def getStatsOfReps(dI, dInpMnSD):
#     dDfrRepS, k = {}, 0
#     for sResTp in dI['lSMnSD']:
#         for sClass in dI['lSMetPho']:
#             for sGT in dI['lSGT']:
#                 tSit = (sResTp, sClass, sGT)
#                 k = fillDfrStatsRepS(dI, dInpMnSD, dDfrRepS, tSit, k)
#                 saveDfrStatsRepS(dI, dDfrRepS, tSit)

# # --- Functions saving result DataFrames --------------------------------------
# def saveDfrStatsRepS(dI, dDfrRepS, tSit):
#     sF = S_USC.join([dI['sDfrRepS']] + list(tSit)) + dI['xtCSV']
#     pF = joinToPath(dI['dPRel'][dI['sOut']][dI['sS']], nmF=sF)
#     saveAsCSV(dDfrRepS[tSit], pF=pF)
#     print('Saved DataFrame of situation', tSit, 'to', sF)

# ### MAIN ####################################################################
print('='*80, '\n', '-'*31, ' EvaluateNMer.py ', '-'*32, '\n', sep='')
if dInp['readProcInpKin']:
    dfrKinIn = readCSV(dInp['pFProcInpKin'], iCol=0)
    serKin = dfrKinIn['Effector'].unique()
    print('Number of different Kinases:', serKin.size)
    # print(dfrKinIn)
if dInp['readProcInpNmer']:
    dfrNmerIn = readCSV(dInp['pFProcInpNmer'], iCol=0)
    serNmer = dfrNmerIn['c15mer'].unique()
    serCodeSeq = dfrNmerIn['code_seq'].unique()
    dfrINmerNOcc = readCSV(dInp['pFResINmerNOcc'], iCol=0)
    dfrINmerNOccF = dfrINmerNOcc[dfrINmerNOcc[S_EFF].isin(dInp['lSEff'])]
    # dINmerNOccFS creation BEGIN
    dINmerNOccFS = {}
    getDNmer(dfrINmerNOcc=dfrINmerNOccF, dINmerNOccSL=dINmerNOccFS)
    printDINmerNOcc(dINmerNOccFS)
    # dINmerNOccFS creation END
    lSnipL3 = dINmerNOccFS[3]
    print('Number of different Nmer chains:', serNmer.size)
    print('Number of different code sequences:', serCodeSeq.size)
    # print(dfrNmerIn)
if dInp['readResCombS']:
    dfrResCombS = readCSV(dInp['pFResCombS'], iCol=0)
    print('Shape of dfrResCombS:', dfrResCombS.shape)
if dInp['readResPropNmer']:
    dfrResPropNmer = readCSV(dInp['pFResPropNmer'], iCol=0)
    print('Shape of dfrResPropNmer:', dfrResPropNmer.shape)

###############################################################################
