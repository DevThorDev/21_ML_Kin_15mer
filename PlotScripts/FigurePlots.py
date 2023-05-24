# -*- coding: utf-8 -*-
###############################################################################
# --- FigurePlots.py ----------------------------------------------------------
###############################################################################
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Constants ===============================================================
S_SPACE = ' '
S_DOT = '.'
S_SEMICOL = ';'
S_USC = '_'
S_PERC = '%'
S_CSV = 'csv'
S_PDF = 'pdf'

# === Input for all figures ===================================================
# --- file and directory paths ------------------------------------------------
pPltDat = os.path.join('..', 'PlotInputData')
pPltFig = os.path.join('..', 'PlottedFigures')

# --- flow control ------------------------------------------------------------
plotF2 = False
plotF3 = False
plotF4 = False
plotF5 = False
plotF6 = True

# --- predefined colours ------------------------------------------------------

# === General input for all plots ---------------------------------------------
sCol0PlotRow, sCol1Metric = 'PlotIt', 'Metric'
lSColNoDat = [sCol0PlotRow, sCol1Metric]

# === Plot-specific input for figure 2 (wo/w loation) =========================
sFPltDat_F2 = 'Data_Figure2'
sPlt_F2 = 'Figure2'
tFigSz_F2 = (6.4, 7.2)
layoutTp_F2 = 'constrained'
lblX_F2, lblY_F2 = None, None
lnWdAxY_F2 = 1
lnStyAxY_F2 = '-'
lnClrAxY_F2 = 'k'
sSupTtl_F2 = None
wdBar_F2 = 1.6

# subplot-specific
nRowSubPl_F2 = 3
nColSubPl_F2 = 2
lPosInSubPl_F2 = [1, 2, 4, 5]
lLblXInSubPl_F2 = ['- loc', '+ loc', '- loc', '+ loc']
lblYInSubPl_F2 = None
lWdBarInSubPl_F2 = [wdBar_F2]*len(lPosInSubPl_F2)
lClrInSubPl_F2 = [(0.85, 0.5, 0.0, 0.85), (0.8, 0.0, 0.0, 0.75),
                  (0.0, 0.5, 0.75, 0.85), (0.0, 0.0, 0.8, 0.75)]
tXLimInSubPl_F2 = (0, 6)

lLPosSubPl_F2 = [lPosInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lLLblXSubPl_F2 = [lLblXInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lLLblYSubPl_F2 = [lblYInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lWdBarSubPl_F2 = [lWdBarInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lLClrSubPl_F2 = [lClrInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lTXLimSubPl_F2 = [tXLimInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)

lYTckSubPl12_F2 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lYTckSubPl34_F2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
lYTckSubPl56_F2 = [0.7, 0.8, 0.9]
lTYLimSubPl_F2 = [(lYTckSubPl12_F2[0], lYTckSubPl12_F2[-1]),
                  (lYTckSubPl12_F2[0], lYTckSubPl12_F2[-1]),
                  (lYTckSubPl34_F2[0], lYTckSubPl34_F2[-1]),
                  (lYTckSubPl34_F2[0], lYTckSubPl34_F2[-1]),
                  (lYTckSubPl56_F2[0], lYTckSubPl56_F2[-1]),
                  (lYTckSubPl56_F2[0], lYTckSubPl56_F2[-1])]
lLYTckSubPl_F2 = [lYTckSubPl12_F2, lYTckSubPl12_F2, lYTckSubPl34_F2,
                  lYTckSubPl34_F2, lYTckSubPl56_F2, lYTckSubPl56_F2]
lXLblSubPl_F2 = ['accuracy', 'balanced accuracy', 'minimum accuracy',
                 'average sensitivity', 'average specificity', 'ROC AUC']
dTxtSubPl_F2 = {}

# === Plot-specific input for figure 3 (classifiers) ==========================
sFPltDat_F3 = 'Data_Figure3'
sPlt_F3 = 'Figure3'
tFigSz_F3 = (6.4, 7.2)
layoutTp_F3 = 'constrained'
lblX_F3, lblY_F3 = None, None
lnWdAxY_F3 = 1
lnStyAxY_F3 = '-'
lnClrAxY_F3 = 'k'
sSupTtl_F3 = None
wdBar_F3 = 0.9

# subplot-specific
nRowSubPl_F3 = 3
nColSubPl_F3 = 2
lPosInSubPl_F3, lPosInSubPl_F3_AUC = [1, 2, 3, 4], [2, 3, 4]
lLblXInSubPl_F3 = ['PaA PF', 'CtNB FF', 'CpNB FF', 'MLP PF']
lblYInSubPl_F3 = None
lWdBarInSubPl_F3 = [wdBar_F3]*len(lPosInSubPl_F3)
lClrInSubPl_F3 = [(0.4, 0.4, 0.4, 1.), (0.75, 0.55, 0.0, 1.),
                  (0.9, 0.0, 0.0, 1.), (0.0, 0.0, 0.9, 1.)]
tXLimInSubPl_F3 = (0.45, 4.55)

lLPosSubPl_F3 = [lPosInSubPl_F3]*(nRowSubPl_F3*nColSubPl_F3)
lLLblXSubPl_F3 = [lLblXInSubPl_F3]*(nRowSubPl_F3*nColSubPl_F3)
lLLblYSubPl_F3 = [lblYInSubPl_F3]*(nRowSubPl_F3*nColSubPl_F3)
lWdBarSubPl_F3 = [lWdBarInSubPl_F3]*(nRowSubPl_F3*nColSubPl_F3)
lLClrSubPl_F3 = [lClrInSubPl_F3]*(nRowSubPl_F3*nColSubPl_F3)
lTXLimSubPl_F3 = [tXLimInSubPl_F3]*(nRowSubPl_F3*nColSubPl_F3)
# special case last subplot: no first bar (no ROC AUC for PaA classifier)
lLPosSubPl_F3[-1] = lPosInSubPl_F3_AUC
lLLblXSubPl_F3[-1] = lLblXInSubPl_F3[1:]
lWdBarSubPl_F3[-1] = [wdBar_F3]*len(lPosInSubPl_F3_AUC)
lLClrSubPl_F3[-1] = lClrInSubPl_F3[1:]

lYTckSubPl13_F3 = [0.6, 0.7, 0.8, 0.9, 1.0]
lYTckSubPl34_F3 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
lYTckSubPl56_F3 = [0.7, 0.8, 0.9]
lTYLimSubPl_F3 = [(lYTckSubPl13_F3[0], lYTckSubPl13_F3[-1]),
                  (lYTckSubPl13_F3[0], lYTckSubPl13_F3[-1]),
                  (lYTckSubPl34_F3[0], lYTckSubPl34_F3[-1]),
                  (lYTckSubPl34_F3[0], lYTckSubPl34_F3[-1]),
                  (lYTckSubPl56_F3[0], lYTckSubPl56_F3[-1]),
                  (lYTckSubPl56_F3[0], lYTckSubPl56_F3[-1])]
lLYTckSubPl_F3 = [lYTckSubPl13_F3, lYTckSubPl13_F3, lYTckSubPl34_F3,
                  lYTckSubPl34_F3, lYTckSubPl56_F3, lYTckSubPl56_F3]
lXLblSubPl_F3 = ['accuracy', 'balanced accuracy', 'minimum accuracy',
                 'average sensitivity', 'average specificity', 'ROC AUC']
dTxtSubPl_F3 = {}

# === Plot-specific input for figure 4 (sampler and fit types) ================
sFPltDat_F4 = 'Data_Figure4'
sPlt_F4 = 'Figure4'
tFigSz_F4 = (6.4, 7.2)
layoutTp_F4 = 'constrained'
lblX_F4, lblY_F4 = None, None
lnWdAxY_F4 = 1
lnStyAxY_F4 = '-'
lnClrAxY_F4 = 'k'
sSupTtl_F4 = None
wdBar_F4 = 0.9

# subplot-specific
nRowSubPl_F4 = 3
nColSubPl_F4 = 2
lPosInSubPl_F4 = list(range(1, 6)) + list(range(7, 12))
lLblXInSubPl_F4 = ['No/FF', '1.5:1/PF', '1.33:1/PF', '1.1:1/PF', '1:1/PF']*2
lblYInSubPl_F4 = None
lWdBarInSubPl_F4 = [wdBar_F4]*len(lPosInSubPl_F4)
lClrInSubPl_F4 = [(0.55, 0.35, 0.0, 1.), (0.65, 0.45, 0.0, 1.),
                  (0.75, 0.55, 0.0, 1.), (0.85, 0.65, 0.0, 1.),
                  (0.95, 0.75, 0.0, 1.),
                  (0.0, 0.0, 0.4, 1.), (0.0, 0.0, 0.55, 1.),
                  (0.0, 0.0, 0.7, 1.), (0.0, 0.0, 0.85, 1.),
                  (0.0, 0.0, 1., 1.)]
tXLimInSubPl_F4 = (0.45, 11.55)

lLPosSubPl_F4 = [lPosInSubPl_F4]*(nRowSubPl_F4*nColSubPl_F4)
lLLblXSubPl_F4 = [lLblXInSubPl_F4]*(nRowSubPl_F4*nColSubPl_F4)
lLLblYSubPl_F4 = [lblYInSubPl_F4]*(nRowSubPl_F4*nColSubPl_F4)
lWdBarSubPl_F4 = [lWdBarInSubPl_F4]*(nRowSubPl_F4*nColSubPl_F4)
lLClrSubPl_F4 = [lClrInSubPl_F4]*(nRowSubPl_F4*nColSubPl_F4)
lTXLimSubPl_F4 = [tXLimInSubPl_F4]*(nRowSubPl_F4*nColSubPl_F4)

lYTckSubPl12_F4 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lYTckSubPl34_F4 = [0.0, 0.2, 0.4, 0.6, 0.8]
lYTckSubPl56_F4 = [0.7, 0.8, 0.9]
lTYLimSubPl_F4 = [(lYTckSubPl12_F4[0], lYTckSubPl12_F4[-1]),
                  (lYTckSubPl12_F4[0], lYTckSubPl12_F4[-1]),
                  (lYTckSubPl34_F4[0], lYTckSubPl34_F4[-1]),
                  (lYTckSubPl34_F4[0], lYTckSubPl34_F4[-1]),
                  (lYTckSubPl56_F4[0], lYTckSubPl56_F4[-1]),
                  (lYTckSubPl56_F4[0], lYTckSubPl56_F4[-1])]
lLYTckSubPl_F4 = [lYTckSubPl12_F4, lYTckSubPl12_F4, lYTckSubPl34_F4,
                  lYTckSubPl34_F4, lYTckSubPl56_F4, lYTckSubPl56_F4]
lXLblSubPl_F4 = ['accuracy', 'balanced accuracy', 'minimum accuracy',
                 'average sensitivity', 'average specificity', 'ROC AUC']
x = 0.5*(len(lLblXInSubPl_F4)/(2*(len(lLblXInSubPl_F4) + 1)))
dTxtSubPl_F4 = {'ClfA': {'xPos': x, 'yPos': 0.1, 'sTxt': 'CtNB',
                         'szFnt': 12, 'hAln': 'center', 'vAln': 'bottom',
                         'bBox': {'boxstyle': 'round', 'fc': 'w', 'ec': 'k',
                                  'alpha': 0.85}},
                'ClfB': {'xPos': 1 - x, 'yPos': 0.1, 'sTxt': 'MLP',
                         'szFnt': 12, 'hAln': 'center', 'vAln': 'bottom',
                         'bBox': {'boxstyle': 'round', 'fc': 'w', 'ec': 'k',
                                  'alpha': 0.85}}}

# === Plot-specific input for figure 5 (varying threshold binary case) ========
dSFPltDat_F5 = {'Kin1': 'Data_Figure5_Kin1_AGC',
                'Kin2': 'Data_Figure5_Kin2_CDK',
                'Kin3': 'Data_Figure5_Kin3_CDP',
                'Kin4': 'Data_Figure5_Kin4_CK2',
                'Kin5': 'Data_Figure5_Kin5_LRR',
                'Kin6': 'Data_Figure5_Kin6_MPK',
                'Kin7': 'Data_Figure5_Kin7_Sn2',
                'Kin8': 'Data_Figure5_Kin8_sol'}
sPlt_F5 = 'Figure5'
tFigSz_F5 = (6.4, 9.0)
tFigMarg_F5 = (0.1, 0.1, 0.1, 0.5)      # (left, right, top, bottom) in inches
layoutTp_F5 = 'constrained'
lblX_F5, lblY_F5 = None, None
lnWdAxY_F5 = 1
lnStyAxY_F5 = '-'
lnClrAxY_F5 = 'k'
sSupTtl_F5 = None

# subplot-specific
nRowSubPl_F5 = 4
nColSubPl_F5 = 2
lLblXInSubPl_F5 = None
lblYInSubPl_F5 = None
lStyLnInSubPl_F5 = ['-']*6
lWdLnInSubPl_F5 = [1.5]*6
lClrLnInSubPl_F5 = [(0.7, 0.0, 0.8, 1.), (0.9, 0.0, 0.0, 1.),
                    (0.0, 0.7, 0.8, 1.), (0.0, 0.0, 0.9, 1.),
                    (0.8, 0.7, 0.0, 1.), (0.4, 0.4, 0.4, 1.)]
lStyMkInSubPl_F5 = ['o', 'x', 'd']*2
lSzMkInSubPl_F5 = [6]*6
lWdMkInSubPl_F5 = [1]*6
lFillStyMkInSubPl_F5 = ['none']*6
lClrMkInSubPl_F5 = [(0.6, 0.0, 0.7, 1.), (0.8, 0.0, 0.0, 1.),
                    (0.0, 0.6, 0.7, 1.), (0.0, 0.0, 0.8, 1.),
                    (0.7, 0.6, 0.0, 1.), (0.3, 0.3, 0.3, 1.)]
tXLimInSubPl_F5 = (0.1, 0.5)
lLegInSubPl_F5 = ['accuracy', 'balanced accuracy', 'minimum accuracy']*2

lLLblXSubPl_F5 = [lLblXInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLLblYSubPl_F5 = [lblYInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLStyLnSubPl_F5 = [lStyLnInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLWdLnSubPl_F5 = [lWdLnInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLClrLnSubPl_F5 = [lClrLnInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLStyMkSubPl_F5 = [lStyMkInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLSzMkSubPl_F5 = [lSzMkInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLWdMkSubPl_F5 = [lWdMkInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLFillStyMkSubPl_F5 = [lFillStyMkInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLClrMkSubPl_F5 = [lClrMkInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lTXLimSubPl_F5 = [tXLimInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLLegSubPl_F5 = [lLegInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)

lXTckSubPl_F5 = [0.1, 0.2, 0.3, 0.4, 0.5]
lYTckSubPl12_F5 = [0.2, 0.4, 0.6, 0.8, 1.0]
lYTckSubPl34_F5 = [0.2, 0.4, 0.6, 0.8, 1.0]
lYTckSubPl56_F5 = [0.2, 0.4, 0.6, 0.8, 1.0]
lYTckSubPl78_F5 = [0.2, 0.4, 0.6, 0.8, 1.0]
lTYLimSubPl_F5 = [(lYTckSubPl12_F5[0], lYTckSubPl12_F5[-1]),
                  (lYTckSubPl12_F5[0], lYTckSubPl12_F5[-1]),
                  (lYTckSubPl34_F5[0], lYTckSubPl34_F5[-1]),
                  (lYTckSubPl34_F5[0], lYTckSubPl34_F5[-1]),
                  (lYTckSubPl56_F5[0], lYTckSubPl56_F5[-1]),
                  (lYTckSubPl56_F5[0], lYTckSubPl56_F5[-1]),
                  (lYTckSubPl78_F5[0], lYTckSubPl78_F5[-1]),
                  (lYTckSubPl78_F5[0], lYTckSubPl78_F5[-1])]
lLXTckSubPl_F5 = [lXTckSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLYTckSubPl_F5 = [lYTckSubPl12_F5, lYTckSubPl12_F5, lYTckSubPl34_F5,
                  lYTckSubPl34_F5, lYTckSubPl56_F5, lYTckSubPl56_F5,
                  lYTckSubPl78_F5, lYTckSubPl78_F5]
lSTxtSubPl = ['AGC', 'CDK', 'CDPK', 'CK2', 'LRR', 'MAPK', 'SnRK2', 'soluble']
dTxtSubPl_F5 = {'sKin': {'xPos': 0.1, 'yPos': 0.1, 'lSTxt': lSTxtSubPl,
                         'szFnt': 12, 'hAln': 'left', 'vAln': 'bottom',
                         'bBox': {'boxstyle': 'round', 'fc': (0.9, 0.9, 0.9),
                                  'ec': 'k', 'alpha': 0.85}}}
assert len(lTYLimSubPl_F5) == nRowSubPl_F5*nColSubPl_F5
assert len(lLYTckSubPl_F5) == nRowSubPl_F5*nColSubPl_F5

# === Plot-specific input for figure 6 (multi-class double threshold heatmap) =
sFPltDat_F6 = 'Data_Figure6'
sPlt_F6 = 'Figure6'
tFigSz_F6 = (6.4, 3.6)
layoutTp_F6 = 'constrained'
lblX_F6, lblY_F6 = None, None
lnWdAxY_F6 = 1
lnStyAxY_F6 = '-'
lnClrAxY_F6 = 'k'
sSupTtl_F6 = None

# subplot-specific
nRowSubPl_F6 = 1
nColSubPl_F6 = 1
lLblXInSubPl_F5 = None
lblYInSubPl_F5 = None
lLblXInSubPl_F6 = ['No/FF', '1.5:1/PF', '1.33:1/PF', '1.1:1/PF', '1:1/PF']*2
lblYInSubPl_F6 = None
lClrInSubPl_F6 = [(0.55, 0.35, 0.0, 1.), (0.65, 0.45, 0.0, 1.),
                  (0.75, 0.55, 0.0, 1.), (0.85, 0.65, 0.0, 1.),
                  (0.95, 0.75, 0.0, 1.),
                  (0.0, 0.0, 0.4, 1.), (0.0, 0.0, 0.55, 1.),
                  (0.0, 0.0, 0.7, 1.), (0.0, 0.0, 0.85, 1.),
                  (0.0, 0.0, 1., 1.)]

lLLblXSubPl_F6 = [lLblXInSubPl_F6]*(nRowSubPl_F6*nColSubPl_F6)
lLLblYSubPl_F6 = [lblYInSubPl_F6]*(nRowSubPl_F6*nColSubPl_F6)
lLClrSubPl_F6 = [lClrInSubPl_F6]*(nRowSubPl_F6*nColSubPl_F6)

lXTckSubPl_F6 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
lYTckSubPl_F6 = [0.1, 0.2, 0.3, 0.4]
tXLimSubPl_F6 = (lXTckSubPl_F6[0], lXTckSubPl_F6[-1])
tYLimSubPl_F6 = (lYTckSubPl_F6[0], lYTckSubPl_F6[-1])
# x = 0.5*(len(lLblXInSubPl_F6)/(2*(len(lLblXInSubPl_F6) + 1)))
dTxtSubPl_F6 = {}

# === Create input and plot dictionaries ======================================
# --- create input dictionary -------------------------------------------------
dInp = {# === Constants
        'sSpace': S_SPACE,
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sUsc': S_USC,
        'sPerc': S_PERC,
        'sCSV': S_CSV,
        'sPDF': S_PDF,
        # === Input for all figures
        # --- file and directory paths
        'pPltDat': pPltDat,
        'pPltFig': pPltFig,
        # --- flow control
        'plotF2': plotF2,
        'plotF3': plotF3,
        'plotF4': plotF4,
        'plotF5': plotF5,
        'plotF6': plotF6}

# --- create plot dictionary of general input data ----------------------------
dPltG = {'sCol0Dfr': sCol0PlotRow,
         'sCol1Dfr': sCol1Metric,
         'lSColDfrNoDat': lSColNoDat}

# --- create plot dictionary of data for plot 2 -------------------------------
dPlt2 = {'pFDat': os.path.join(pPltDat, sFPltDat_F2 + S_DOT + S_CSV),
         'pFPlt': os.path.join(pPltFig, sPlt_F2 + S_DOT + S_PDF),
         'tFigSz': tFigSz_F2,
         'layoutTp': layoutTp_F2,
         'lblX': lblX_F2,
         'lblY': lblY_F2,
         'lnWdAxY': lnWdAxY_F2,
         'lnStyAxY': lnStyAxY_F2,
         'lnClrAxY': lnClrAxY_F2,
         'sSupTtl': sSupTtl_F2,
         'wdBar': wdBar_F2,
         'nRowSubPl': nRowSubPl_F2,
         'nColSubPl': nColSubPl_F2,
         'dSubPl': {'lLPos': lLPosSubPl_F2,
                    'lLLblX': lLLblXSubPl_F2,
                    'lLLblY': lLLblYSubPl_F2,
                    'lWdBar': lWdBarSubPl_F2,
                    'lLClr': lLClrSubPl_F2,
                    'lTXLim': lTXLimSubPl_F2,
                    'lTYLim': lTYLimSubPl_F2,
                    'lLYTck': lLYTckSubPl_F2,
                    'lXLbl': lXLblSubPl_F2,
                    'dTxt': dTxtSubPl_F2}}
dPlt2 = dPltG | dPlt2

# --- create plot dictionary of data for plot 3 -------------------------------
dPlt3 = {'pFDat': os.path.join(pPltDat, sFPltDat_F3 + S_DOT + S_CSV),
         'pFPlt': os.path.join(pPltFig, sPlt_F3 + S_DOT + S_PDF),
         'tFigSz': tFigSz_F3,
         'layoutTp': layoutTp_F3,
         'lblX': lblX_F3,
         'lblY': lblY_F3,
         'lnWdAxY': lnWdAxY_F3,
         'lnStyAxY': lnStyAxY_F3,
         'lnClrAxY': lnClrAxY_F3,
         'sSupTtl': sSupTtl_F3,
         'wdBar': wdBar_F3,
         'nRowSubPl': nRowSubPl_F3,
         'nColSubPl': nColSubPl_F3,
         'dSubPl': {'lLPos': lLPosSubPl_F3,
                    'lLLblX': lLLblXSubPl_F3,
                    'lLLblY': lLLblYSubPl_F3,
                    'lWdBar': lWdBarSubPl_F3,
                    'lLClr': lLClrSubPl_F3,
                    'lTXLim': lTXLimSubPl_F3,
                    'lTYLim': lTYLimSubPl_F3,
                    'lLYTck': lLYTckSubPl_F3,
                    'lXLbl': lXLblSubPl_F3,
                    'dTxt': dTxtSubPl_F3}}
dPlt3 = dPltG | dPlt3

# --- create plot dictionary of data for plot 4 -------------------------------
dPlt4 = {'pFDat': os.path.join(pPltDat, sFPltDat_F4 + S_DOT + S_CSV),
         'pFPlt': os.path.join(pPltFig, sPlt_F4 + S_DOT + S_PDF),
         'tFigSz': tFigSz_F4,
         'layoutTp': layoutTp_F4,
         'lblX': lblX_F4,
         'lblY': lblY_F4,
         'lnWdAxY': lnWdAxY_F4,
         'lnStyAxY': lnStyAxY_F4,
         'lnClrAxY': lnClrAxY_F4,
         'sSupTtl': sSupTtl_F4,
         'wdBar': wdBar_F4,
         'nRowSubPl': nRowSubPl_F4,
         'nColSubPl': nColSubPl_F4,
         'dSubPl': {'lLPos': lLPosSubPl_F4,
                    'lLLblX': lLLblXSubPl_F4,
                    'lLLblY': lLLblYSubPl_F4,
                    'lWdBar': lWdBarSubPl_F4,
                    'lLClr': lLClrSubPl_F4,
                    'lTXLim': lTXLimSubPl_F4,
                    'lTYLim': lTYLimSubPl_F4,
                    'lLYTck': lLYTckSubPl_F4,
                    'lXLbl': lXLblSubPl_F4,
                    'dTxt': dTxtSubPl_F4}}
dPlt4 = dPltG | dPlt4

# --- create plot dictionary of data for plot 5 -------------------------------
dPlt5 = {'dPFDat': {sK: os.path.join(pPltDat, dSFPltDat_F5[sK] + S_DOT + S_CSV)
                    for sK in dSFPltDat_F5},
         'pFPlt': os.path.join(pPltFig, sPlt_F5 + S_DOT + S_PDF),
         'tFigSz': tFigSz_F5,
         'tFigMarg': tFigMarg_F5,
         'layoutTp': layoutTp_F5,
         'lblX': lblX_F5,
         'lblY': lblY_F5,
         'lnWdAxY': lnWdAxY_F5,
         'lnStyAxY': lnStyAxY_F5,
         'lnClrAxY': lnClrAxY_F5,
         'sSupTtl': sSupTtl_F5,
         'nRowSubPl': nRowSubPl_F5,
         'nColSubPl': nColSubPl_F5,
         'dSubPl': {'lLLblX': lLLblXSubPl_F5,
                    'lLLblY': lLLblYSubPl_F5,
                    'lLStyLn': lLStyLnSubPl_F5,
                    'lLWdLn': lLWdLnSubPl_F5,
                    'lLClrLn': lLClrLnSubPl_F5,
                    'lLStyMk': lLStyMkSubPl_F5,
                    'lLSzMk': lLSzMkSubPl_F5,
                    'lLWdMk': lLWdMkSubPl_F5,
                    'lLFillStyMk': lLFillStyMkSubPl_F5,
                    'lLClrMk': lLClrMkSubPl_F5,
                    'lTXLim': lTXLimSubPl_F5,
                    'lTYLim': lTYLimSubPl_F5,
                    'lLXTck': lLXTckSubPl_F5,
                    'lLYTck': lLYTckSubPl_F5,
                    'lLLeg': lLLegSubPl_F5,
                    'dTxt': dTxtSubPl_F5}}
dPlt5 = dPltG | dPlt5

# --- create plot dictionary of data for plot 6 -------------------------------
dPlt6 = {'pFDat': os.path.join(pPltDat, sFPltDat_F6 + S_DOT + S_CSV),
         'pFPlt': os.path.join(pPltFig, sPlt_F6 + S_DOT + S_PDF),
         'tFigSz': tFigSz_F6,
         'layoutTp': layoutTp_F6,
         'lblX': lblX_F6,
         'lblY': lblY_F6,
         'lnWdAxY': lnWdAxY_F6,
         'lnStyAxY': lnStyAxY_F6,
         'lnClrAxY': lnClrAxY_F6,
         'sSupTtl': sSupTtl_F6,
         'nRowSubPl': nRowSubPl_F6,
         'nColSubPl': nColSubPl_F6,
         'dSubPl': {'lLLblX': lLLblXSubPl_F6,
                    'lLLblY': lLLblYSubPl_F6,
                    'lLClr': lLClrSubPl_F6,
                    'lXTck': lXTckSubPl_F6,
                    'lYTck': lYTckSubPl_F6,
                    'tXLim': tXLimSubPl_F6,
                    'tYLim': tYLimSubPl_F6,
                    'dTxt': dTxtSubPl_F6}}
dPlt6 = dPltG | dPlt6

# === Dataframe manipulation functions ========================================
# --- General dataframe manipulation functions --------------------------------
def reduceDfr(pdDfr, sCSel=None, lSNoDat=None):
    if sCSel is not None:
        pdDfr = pdDfr[pdDfr[sCSel] == 1]
    if lSNoDat is None:
        return pdDfr
    else:
        lDat = [s for s in pdDfr.columns if s not in lSNoDat]
        return pdDfr.loc[:, lDat].reset_index(drop=True)

# --- Figures 2, 3 and 4 dataframe manipulation -------------------------------
def procDataForFig234(dInp, dPlt):
    dfrF = pd.read_csv(dPlt['pFDat'], sep=dInp['sSemicol'])
    return reduceDfr(pdDfr=dfrF, sCSel=dPlt['sCol0Dfr'],
                     lSNoDat=dPlt['lSColDfrNoDat'])

# --- Figure 5 dataframe manipulation -----------------------------------------
def procDataForFig5(dInp, dPlt):
    dDfr = {sK: pd.read_csv(sPF, sep=dInp['sSemicol']) for sK, sPF in
            dPlt['dPFDat'].items()}
    for sK, dfrF in dDfr.items():
        dDfr[sK] = reduceDfr(pdDfr=dfrF, sCSel=dPlt['sCol0Dfr'],
                             lSNoDat=dPlt['lSColDfrNoDat'])
    return dDfr

# --- Figure 6 dataframe manipulation -----------------------------------------
def procDataForFig6(dInp, dPlt):
    return reduceDfr(pdDfr=pd.read_csv(dPlt['pFDat'], sep=dInp['sSemicol']))

# === Plotting functions ======================================================
# --- Plotting helper functions -----------------------------------------------
def setFigMargins(cFig, tMargins):
    # set figure margins from edges as (left, right, top, bottom) in inches
    assert len(tMargins) == 4
    leftSp, rightSp, topSp, bottomSp = tMargins
    cWd, cHt = cFig.get_size_inches()
    #convert to figure coordinates:
    leftSp, rightSp = leftSp/cWd, 1 - rightSp/cWd
    bottomSp, topSp = bottomSp/cHt, 1 - topSp/cHt
    #get the layout engine and convert to its desired format
    cLOEng = cFig.get_layout_engine()
    #set and recompute the layout
    cRectSp = (leftSp, bottomSp, rightSp - leftSp, topSp - bottomSp)
    cLOEng.set(rect=cRectSp)
    cLOEng.execute(cFig)

def procAxBar(dSub, pdDfr, sglAx, k):
    serHeight = pdDfr.loc[k, :][pdDfr.loc[k, :].notna()]
    assert len(dSub['lLPos'][k]) == serHeight.shape[0]
    sglAx.bar(x=dSub['lLPos'][k], height=serHeight, width=dSub['lWdBar'][k],
              bottom=0, align='center', color=dSub['lLClr'][k])
    sglAx.set_xlim(dSub['lTXLim'][k])
    sglAx.set_ylim(dSub['lTYLim'][k])
    lSYTck = [str(round(x*100)) + dInp['sPerc'] for x in dSub['lLYTck'][k]]
    sglAx.set_xticks(ticks=dSub['lLPos'][k], labels=dSub['lLLblX'][k],
                     rotation='vertical')
    sglAx.set_yticks(dSub['lLYTck'][k], labels=lSYTck)
    sglAx.set_xlabel(xlabel=dSub['lXLbl'][k], loc='center')

def procAxLines(dPlt, dSub, pdDfr, sglAx, k):
    serX = pdDfr.loc[0, :][pdDfr.loc[0, :].notna()]
    for l in range(pdDfr.shape[0] - 1):
        serY = pdDfr.loc[l + 1, :][pdDfr.loc[l + 1, :].notna()]
        assert serX.shape[0] == serY.shape[0]
        cLbl = (dSub['lLLeg'][k][l] if k == 0 else None)
        sglAx.plot(serX, serY, label=cLbl, ls=dSub['lLStyLn'][k][l],
                   lw=dSub['lLWdLn'][k][l], color=dSub['lLClrLn'][k][l],
                   marker=dSub['lLStyMk'][k][l], ms=dSub['lLSzMk'][k][l],
                   mew=dSub['lLWdMk'][k][l],
                   fillstyle=dSub['lLFillStyMk'][k][l],
                   mfc=dSub['lLClrMk'][k][l], mec=dSub['lLClrMk'][k][l])
    sglAx.set_xlim(dSub['lTXLim'][k])
    sglAx.set_ylim(dSub['lTYLim'][k])
    lSXTck = [str(round(x*100)) + dInp['sPerc'] for x in dSub['lLXTck'][k]]
    sglAx.set_xticks(dSub['lLXTck'][k], labels=lSXTck)
    lSYTck = [str(round(x*100)) + dInp['sPerc'] for x in dSub['lLYTck'][k]]
    sglAx.set_yticks(dSub['lLYTck'][k], labels=lSYTck)
    if k >= dPlt['nColSubPl']*(dPlt['nRowSubPl'] - 1):
        sglAx.set_xlabel(xlabel='Classification threshold', loc='center')

def procAxHeatmap(dSub, pdDfr, sglAx):
    serX = pdDfr.iloc[0, 1:][pdDfr.iloc[0, 1:].notna()].convert_dtypes()
    serY = pd.Series(pdDfr.iloc[1:, 0], name=pdDfr.columns[0], dtype=float)
    arrDat = pdDfr.iloc[1:, 1:].to_numpy()
    im = sglAx.imshow(arrDat)

def addTxtToAx(dTxt, sglAx, k=0):
    for cD in dTxt.values():
        assert (('sTxt' in cD) or (('lSTxt' in cD) and k < len(cD['lSTxt'])))
        sTxt = (cD['sTxt'] if 'sTxt' in cD else cD['lSTxt'][k])
        sglAx.text(x=cD['xPos'], y=cD['yPos'], s=sTxt,
                   transform=sglAx.transAxes, fontsize=cD['szFnt'],
                   horizontalalignment=cD['hAln'],
                   verticalalignment=cD['vAln'], bbox=cD['bBox'])

def doBarSubplot(dSub, dTxt, pdDfr, cAx, k, m, n):
    procAxBar(dSub, pdDfr, sglAx=cAx[m, n], k=k)
    addTxtToAx(dTxt, sglAx=cAx[m, n], k=k)

def doLinesSubplot(dPlt, dSub, dTxt, pdDfr, cAx, k, m, n):
    procAxLines(dPlt, dSub, pdDfr, sglAx=cAx[m, n], k=k)
    addTxtToAx(dTxt, sglAx=cAx[m, n], k=k)

def doHeatmapPlot(dSub, dTxt, pdDfr, cAx):
    procAxHeatmap(dSub, pdDfr, sglAx=cAx)
    addTxtToAx(dTxt, sglAx=cAx)

def pltYHalfAxHist(dPlt, cSer=None, yBase=0., yMax=1.):
    if cSer is not None:
        if yBase is None:
            yBase = cSer.min()
        if yMax is None:
            yMax = cSer.max()
    else:
        if yBase is None:
            yBase = 0.
        if yMax is None:
            yMax = 1.
    plt.plot([0., 0.], [yBase, yMax], lw=dPlt['lnWdAxY'],
             ls=dPlt['lnStyAxY'], color=dPlt['lnClrAxY'])

def decorateSavePlot(dPlt, sPF, serDt=None, sTtl=None, xLim=None, yLim=None,
                     yMax=None, pltAx=False, addLeg=True, saveFig=True):
    if pltAx and yLim is None:
        pltYHalfAxHist(dPlt, serDt, yMax=yMax)
    elif pltAx and yLim is not None:
        if len(yLim) == 2:
            pltYHalfAxHist(dPlt, serDt, yMax=yLim[1])
    if sTtl is not None:
        plt.title(sTtl)
    if dPlt['lblX'] is not None:
        plt.xlabel(dPlt['lblX'])
    if dPlt['lblY'] is not None:
        plt.ylabel(dPlt['lblY'])
    if xLim is not None and len(xLim) == 2:
        plt.xlim(xLim)
    if yLim is not None and len(yLim) == 2:
        plt.ylim(yLim)
    if addLeg:
        plt.legend(loc='best')
    if saveFig:
        plt.savefig(dPlt[sPF])

# --- Plotting bar plots (figures 2, 3 and 4) ---------------------------------
def plotFig234(dInp, dPlt, pdDfr, sTtl=None, savePlt=True):
    assert pdDfr.shape[0] == dPlt['nRowSubPl']*dPlt['nColSubPl']
    cFig, cAx = plt.subplots(nrows=dPlt['nRowSubPl'], ncols=dPlt['nColSubPl'],
                             figsize=dPlt['tFigSz'], layout=dPlt['layoutTp'])
    dSub, dTxt = dPlt['dSubPl'], dPlt['dSubPl']['dTxt']
    for m in range(dPlt['nRowSubPl']):
        for n in range(dPlt['nColSubPl']):
            k = m*dPlt['nColSubPl'] + n
            doBarSubplot(dSub, dTxt, pdDfr, cAx=cAx, k=k, m=m, n=n)
    decorateSavePlot(dPlt, sPF='pFPlt', sTtl=sTtl, addLeg=False,
                     saveFig=savePlt)
    plt.close()

# --- Plotting xy-plots (figure 5) --------------------------------------------
def plotFig5(dInp, dPlt, dDfr, sTtl=None, savePlt=True):
    assert len(dDfr) == dPlt['nRowSubPl']*dPlt['nColSubPl']
    cFig, cAx = plt.subplots(nrows=dPlt['nRowSubPl'], ncols=dPlt['nColSubPl'],
                             figsize=dPlt['tFigSz'], layout=dPlt['layoutTp'])
    setFigMargins(cFig, tMargins=dPlt['tFigMarg'])
    dSub, dTxt, k, m, n = dPlt['dSubPl'], dPlt['dSubPl']['dTxt'], 0, 0, 0
    for pdDfr in dDfr.values():
        doLinesSubplot(dPlt, dSub, dTxt, pdDfr, cAx=cAx, k=k, m=m, n=n)
        k += 1
        if n < dPlt['nColSubPl'] - 1:
            n += 1
        else:
            n = 0
            m += 1
    cFig.legend(bbox_to_anchor=(0.085, 0.01, 0.85, .1), loc='lower center',
                ncols=3, mode='expand', borderaxespad=0.5)
    decorateSavePlot(dPlt, sPF='pFPlt', sTtl=sTtl, addLeg=False,
                     saveFig=savePlt)
    plt.close()

# --- Plotting a heatmap (figure 6) -------------------------------------------
def plotFig6(dInp, dPlt, pdDfr, sTtl=None, savePlt=True):
    cFig, cAx = plt.subplots(nrows=dPlt['nRowSubPl'], ncols=dPlt['nColSubPl'],
                             figsize=dPlt['tFigSz'], layout=dPlt['layoutTp'])
    dSub, dTxt = dPlt['dSubPl'], dPlt['dSubPl']['dTxt']
    doHeatmapPlot(dSub, dTxt, pdDfr, cAx)
    plt.show()
    decorateSavePlot(dPlt, sPF='pFPlt', sTtl=sTtl, addLeg=False,
                     saveFig=savePlt)
    plt.close()

# === Main ====================================================================
if dInp['plotF2']:
    plotFig234(dInp, dPlt=dPlt2, pdDfr=procDataForFig234(dInp, dPlt=dPlt2))
    print('Plotted figure 2.')
if dInp['plotF3']:
    plotFig234(dInp, dPlt=dPlt3, pdDfr=procDataForFig234(dInp, dPlt=dPlt3))
    print('Plotted figure 3.')
if dInp['plotF4']:
    plotFig234(dInp, dPlt=dPlt4, pdDfr=procDataForFig234(dInp, dPlt=dPlt4))
    print('Plotted figure 4.')
if dInp['plotF5']:
    plotFig5(dInp, dPlt=dPlt5, dDfr=procDataForFig5(dInp, dPlt=dPlt5))
    print('Plotted figure 5.')
if dInp['plotF6']:
    plotFig6(dInp, dPlt=dPlt6, pdDfr=procDataForFig6(dInp, dPlt=dPlt6))
    print('Plotted figure 6.')

# #############################################################################