# -*- coding: utf-8 -*-
###############################################################################
# --- ExtractUniqueProteinSeq.py ----------------------------------------------
###############################################################################
import os

import pandas as pd

# ### CONSTANTS ###############################################################
# --- strings -----------------------------------------------------------------
S_USC = '_'
S_DOT = '.'
S_SEMICOL = ';'
S_NEWL = '\n'
S_GREATER = '>'
S_CSV = 'csv'
S_M = 'M'
S_COMBINED = 'Combined'
S_INFO = 'Info'
S_SEQ = 'seq'
S_C_CODE_SEQ = 'code' + S_USC + S_SEQ
S_C_N_MER = 'c15mer'
S_FASTA = 'FASTA'

# --- file names, paths and extensions ----------------------------------------
F_COMB_M = S_COMBINED + S_USC + S_M + S_USC + 'KinasesPho15mer_202202'

DIR_INP = '13_Sysbio03_Phospho15mer'
DIR_COMB = '31_Res' + S_COMBINED
DIR_INFO = '33_Res' + S_INFO

P_DIR_INP = os.path.join('..', '..', '..', DIR_INP)
P_DIR_COMB = os.path.join(P_DIR_INP, DIR_COMB)
P_DIR_INFO = os.path.join(P_DIR_INP, DIR_INFO)

# --- predefined numbers ------------------------------------------------------

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV
XT_FASTA = S_DOT + S_FASTA

# --- lists -------------------------------------------------------------------

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
extractUnqProtSeq = True
saveAsFASTA = True

# --- names and paths of files and dirs ---------------------------------------
sFComb = F_COMB_M
sFUnqProtSeq = 'UniqueProteinSeq'
sFUnqProtSeqFASTA = 'UniqueProteinSeq' + S_USC + S_FASTA

pDirComb = P_DIR_COMB
pDirUnqProtSeq = P_DIR_INFO

# --- numbers -----------------------------------------------------------------

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === derived input and assertions ============================================

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- strings ---------------------------------------------------------
        'sUSC': S_USC,
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sNewl': S_NEWL,
        'sGreater': S_GREATER,
        'sCSV': S_CSV,
        'sM': S_M,
        'sCombined': S_COMBINED,
        'sInfo': S_INFO,
        'sSeq': S_SEQ,
        'sCCodeSeq': S_C_CODE_SEQ,
        'sCNmer': S_C_N_MER,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        'xtFASTA': XT_FASTA,
        # --- flow control ----------------------------------------------------
        'extractUnqProtSeq': extractUnqProtSeq,
        'saveAsFASTA': saveAsFASTA,
        # --- names and paths of files and dirs -------------------------------
        'sFComb': sFComb,
        'sFUnqProtSeq': sFUnqProtSeq,
        'sFUnqProtSeqFASTA': sFUnqProtSeqFASTA,
        'pDirComb': pDirComb,
        'pDirUnqProtSeq': pDirUnqProtSeq,
        'pFComb': os.path.join(pDirComb, sFComb + XT_CSV),
        'pFUnqProtSeq': os.path.join(pDirUnqProtSeq, sFUnqProtSeqFASTA + XT_CSV),
        'pFUnqProtSeqFASTA': os.path.join(pDirUnqProtSeq,
                                          sFUnqProtSeqFASTA + XT_FASTA),
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

def pathXist(pD):
    return os.path.isdir(pD)

def fileXist(pF):
    return os.path.isfile(pF)

def readCSV(pF, iCol=None, dDTp=None, cSep=S_SEMICOL, sXt=XT_CSV):
    if not pF.endswith(sXt):
        pF += sXt
    if fileXist(pF):
        return pd.read_csv(pF, sep=cSep, index_col=iCol, dtype=dDTp)
    else:
        print('File "', pF, '" does not exist!', sep='')

def saveCSV(pdObj, pF, reprNA='', cSep=S_SEMICOL, sXt=XT_CSV,
            saveIdx=True, iLbl=None):
    if not pF.endswith(sXt):
        pF += sXt
    if pdObj is not None:
        pdObj.to_csv(pF, sep=cSep, na_rep=reprNA, index=saveIdx,
                     index_label=iLbl)

def savePdSerAsFASTA(dI, pdSer):
    s = ''
    for k, cEl in enumerate(pdSer):
        s += (dI['sGreater'] + dI['sSeq'] + str(k + 1) + dI['sNewl'] +
              str(cEl) + dI['sNewl'])
    with open(dI['pFUnqProtSeqFASTA'], 'w') as f: 
        f.write(s)

# --- Functions for filling dictionaries --------------------------------------
def fillDProtSeq2Nmer(cD):
    pass

# --- Helper functions --------------------------------------------------------
def addToDictL(cD, cK, cE, lUnqEl=False):
    if cK in cD:
        if not lUnqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def toSerUnique(pdSer, sName=None):
    nameS = pdSer.name
    if sName is not None:
        nameS = sName
    return pd.Series(pdSer.unique(), name=nameS)

# ### MAIN ####################################################################
print('='*80, '\n', '-'*26, ' ExtractUniqueProteinSeq.py ',
      '-'*26, '\n', sep='')
if dInp['extractUnqProtSeq']:
    dfrCombM = readCSV(dInp['pFComb'], iCol=0)
    serProtSeq = toSerUnique(dfrCombM[dInp['sCCodeSeq']])
    saveCSV(serProtSeq, pF=dInp['pFUnqProtSeq'], cSep=S_SEMICOL, sXt=XT_CSV)
    print('Number of different protein sequences:', serProtSeq.size)
if dInp['saveAsFASTA']:
    savePdSerAsFASTA(dInp, serProtSeq)

###############################################################################
