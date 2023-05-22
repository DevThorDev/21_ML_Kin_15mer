# -*- coding: utf-8 -*-
###############################################################################
# --- FigurePlots.py ----------------------------------------------------------
###############################################################################
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
plotF4 = True
plotF5 = True

# --- predefined colours ------------------------------------------------------

# === General input for all plots ---------------------------------------------
sCol0PlotRow, sCol1Metric = 'PlotIt', 'Metric'
lSColNoDat = [sCol0PlotRow, sCol1Metric]

# === Plot-specific input for figure 2 (wo/w loation) =========================
sFPltDat_F2 = 'Data_Figure2'
sPlt_F2 = 'Figure2'
tFigSz_F2 = (6.4, 7.2)
layoutTp_F2 = 'constrained'
wdBar_F2 = 1.6
lblX_F2, lblY_F2 = None, None
lnWdAxY_F2 = 1
lnStyAxY_F2 = '-'
lnClrAxY_F2 = 'k'
sSupTtl_F2 = None
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
wdBar_F3 = 0.9
lblX_F3, lblY_F3 = None, None
lnWdAxY_F3 = 1
lnStyAxY_F3 = '-'
lnClrAxY_F3 = 'k'
sSupTtl_F3 = None
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
wdBar_F4 = 0.9
lblX_F4, lblY_F4 = None, None
lnWdAxY_F4 = 1
lnStyAxY_F4 = '-'
lnClrAxY_F4 = 'k'
sSupTtl_F4 = None
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
tFigSz_F5 = (6.4, 7.2)
layoutTp_F5 = 'constrained'
wdBar_F5 = 0.9
lblX_F5, lblY_F5 = None, None
lnWdAxY_F5 = 1
lnStyAxY_F5 = '-'
lnClrAxY_F5 = 'k'
sSupTtl_F5 = None
# subplot-specific
nRowSubPl_F5 = 3
nColSubPl_F5 = 2
lPosInSubPl_F5 = None
lLblXInSubPl_F5 = None
lblYInSubPl_F5 = None
lWdBarInSubPl_F5 = [wdBar_F5]*len(lPosInSubPl_F5)
lClrInSubPl_F5 = [(0.55, 0.35, 0.0, 1.), (0.65, 0.45, 0.0, 1.),
                  (0.75, 0.55, 0.0, 1.), (0.85, 0.65, 0.0, 1.),
                  (0.95, 0.75, 0.0, 1.),
                  (0.0, 0.0, 0.4, 1.), (0.0, 0.0, 0.55, 1.),
                  (0.0, 0.0, 0.7, 1.), (0.0, 0.0, 0.85, 1.),
                  (0.0, 0.0, 1., 1.)]
tXLimInSubPl_F5 = (0.45, 11.55)
lLPosSubPl_F5 = [lPosInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLLblXSubPl_F5 = [lLblXInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLLblYSubPl_F5 = [lblYInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lWdBarSubPl_F5 = [lWdBarInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lLClrSubPl_F5 = [lClrInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)
lTXLimSubPl_F5 = [tXLimInSubPl_F5]*(nRowSubPl_F5*nColSubPl_F5)

lYTckSubPl12_F5 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lYTckSubPl34_F5 = [0.0, 0.2, 0.4, 0.6, 0.8]
lYTckSubPl56_F5 = [0.7, 0.8, 0.9]
lTYLimSubPl_F5 = [(lYTckSubPl12_F5[0], lYTckSubPl12_F5[-1]),
                  (lYTckSubPl12_F5[0], lYTckSubPl12_F5[-1]),
                  (lYTckSubPl34_F5[0], lYTckSubPl34_F5[-1]),
                  (lYTckSubPl34_F5[0], lYTckSubPl34_F5[-1]),
                  (lYTckSubPl56_F5[0], lYTckSubPl56_F5[-1]),
                  (lYTckSubPl56_F5[0], lYTckSubPl56_F5[-1])]
lLYTckSubPl_F5 = [lYTckSubPl12_F5, lYTckSubPl12_F5, lYTckSubPl34_F5,
                  lYTckSubPl34_F5, lYTckSubPl56_F5, lYTckSubPl56_F5]
lXLblSubPl_F5 = ['accuracy', 'balanced accuracy', 'minimum accuracy',
                 'average sensitivity', 'average specificity', 'ROC AUC']
x = 0.5*(len(lLblXInSubPl_F5)/(2*(len(lLblXInSubPl_F5) + 1)))
dTxtSubPl_F5 = {'ClfA': {'xPos': x, 'yPos': 0.1, 'sTxt': 'CtNB',
                         'szFnt': 12, 'hAln': 'center', 'vAln': 'bottom',
                         'bBox': {'boxstyle': 'round', 'fc': 'w', 'ec': 'k',
                                  'alpha': 0.85}},
                'ClfB': {'xPos': 1 - x, 'yPos': 0.1, 'sTxt': 'MLP',
                         'szFnt': 12, 'hAln': 'center', 'vAln': 'bottom',
                         'bBox': {'boxstyle': 'round', 'fc': 'w', 'ec': 'k',
                                  'alpha': 0.85}}}

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
        'plotF5': plotF5}

# --- create plot dictionary of general input data ----------------------------
dPltG = {'sCol0Dfr': sCol0PlotRow,
         'sCol1Dfr': sCol1Metric,
         'lSColDfrNoDat': lSColNoDat}

# --- create plot dictionary of data for plot 2 -------------------------------
dPlt2 = {'pFDat': os.path.join(pPltDat, sFPltDat_F2 + S_DOT + S_CSV),
         'pFPlt': os.path.join(pPltFig, sPlt_F2 + S_DOT + S_PDF),
         'tFigSz': tFigSz_F2,
         'layoutTp': layoutTp_F2,
         'wdBar': wdBar_F2,
         'lblX': lblX_F2,
         'lblY': lblY_F2,
         'lnWdAxY': lnWdAxY_F2,
         'lnStyAxY': lnStyAxY_F2,
         'lnClrAxY': lnClrAxY_F2,
         'sSupTtl': sSupTtl_F2,
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
         'wdBar': wdBar_F3,
         'lblX': lblX_F3,
         'lblY': lblY_F3,
         'lnWdAxY': lnWdAxY_F3,
         'lnStyAxY': lnStyAxY_F3,
         'lnClrAxY': lnClrAxY_F3,
         'sSupTtl': sSupTtl_F3,
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
         'wdBar': wdBar_F4,
         'lblX': lblX_F4,
         'lblY': lblY_F4,
         'lnWdAxY': lnWdAxY_F4,
         'lnStyAxY': lnStyAxY_F4,
         'lnClrAxY': lnClrAxY_F4,
         'sSupTtl': sSupTtl_F4,
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
         'layoutTp': layoutTp_F5,
         'wdBar': wdBar_F5,
         'lblX': lblX_F5,
         'lblY': lblY_F5,
         'lnWdAxY': lnWdAxY_F5,
         'lnStyAxY': lnStyAxY_F5,
         'lnClrAxY': lnClrAxY_F5,
         'sSupTtl': sSupTtl_F5,
         'nRowSubPl': nRowSubPl_F5,
         'nColSubPl': nColSubPl_F5,
         'dSubPl': {'lLPos': lLPosSubPl_F5,
                    'lLLblX': lLLblXSubPl_F5,
                    'lLLblY': lLLblYSubPl_F5,
                    'lWdBar': lWdBarSubPl_F5,
                    'lLClr': lLClrSubPl_F5,
                    'lTXLim': lTXLimSubPl_F5,
                    'lTYLim': lTYLimSubPl_F5,
                    'lLYTck': lLYTckSubPl_F5,
                    'lXLbl': lXLblSubPl_F5,
                    'dTxt': dTxtSubPl_F5}}
dPlt5 = dPltG | dPlt5

# === Data manipulation functions =============================================
# --- Figures 2, 3 and 4 data manipulation ---------------------------------------
def procDataForPlot(dInp, dPlt):
    dfrF2 = pd.read_csv(dPlt['pFDat'], sep=dInp['sSemicol'])
    dfrF2Red = dfrF2[dfrF2[dPlt['sCol0Dfr']] == 1]
    lDat = [s for s in dfrF2Red.columns if s not in dPlt['lSColDfrNoDat']]
    return dfrF2Red.loc[:, lDat].reset_index(drop=True)

# === Plotting functions ======================================================
# --- Plotting helper functions -----------------------------------------------
def procAx(dPlt, dSub, pdDfr, sglAx, k):
    serHeight = pdDfr.loc[k, :][pdDfr.loc[k, :].notna()]
    assert len(dSub['lLPos'][k]) == serHeight.shape[0]
    sglAx.bar(x=dSub['lLPos'][k], height=serHeight, width=dSub['lWdBar'][k],
              bottom=0, align='center', color=dSub['lLClr'][k])
    sglAx.set_xlim(dSub['lTXLim'][k])
    sglAx.set_ylim(dSub['lTYLim'][k])
    lS = [str(round(x*100)) + dInp['sPerc'] for x in dSub['lLYTck'][k]]
    sglAx.set_xticks(ticks=dSub['lLPos'][k], labels=dSub['lLLblX'][k],
                     rotation='vertical')
    sglAx.set_yticks(dSub['lLYTck'][k], labels=lS)
    sglAx.set_xlabel(xlabel=dSub['lXLbl'][k], loc='center')

def addTxtToAx(dTxt, sglAx):
    for cD in dTxt.values():
        sglAx.text(x=cD['xPos'], y=cD['yPos'], s=cD['sTxt'],
                   transform=sglAx.transAxes, fontsize=cD['szFnt'],
                   horizontalalignment=cD['hAln'],
                   verticalalignment=cD['vAln'], bbox=cD['bBox'])

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
def plotFigures234(dInp, dPlt, pdDfr, sTtl=None, savePlt=True):
    assert pdDfr.shape[0] == dPlt['nRowSubPl']*dPlt['nColSubPl']
    cFig, cAx = plt.subplots(nrows=dPlt['nRowSubPl'], ncols=dPlt['nColSubPl'],
                             figsize=dPlt['tFigSz'], layout=dPlt['layoutTp'])
    dSub, dTxt = dPlt['dSubPl'], dPlt['dSubPl']['dTxt']
    for m in range(dPlt['nRowSubPl']):
        for n in range(dPlt['nColSubPl']):
            k = m*dPlt['nColSubPl'] + n
            procAx(dPlt, dSub, pdDfr, sglAx=cAx[m, n], k=k)
            addTxtToAx(dTxt, sglAx=cAx[m, n])
    decorateSavePlot(dPlt, sPF='pFPlt', sTtl=sTtl, addLeg=False,
                     saveFig=savePlt)
    plt.close()

## --- Plotting xy-plots (figure 5) -------------------------------------------
def plotFigure5(dInp, dPlt, pdDfr, sTtl=None, savePlt=True):
    assert pdDfr.shape[0] == dPlt['nRowSubPl']*dPlt['nColSubPl']
    cFig, cAx = plt.subplots(nrows=dPlt['nRowSubPl'], ncols=dPlt['nColSubPl'],
                             figsize=dPlt['tFigSz'], layout=dPlt['layoutTp'])
    dSub, dTxt = dPlt['dSubPl'], dPlt['dSubPl']['dTxt']
    for m in range(dPlt['nRowSubPl']):
        for n in range(dPlt['nColSubPl']):
            k = m*dPlt['nColSubPl'] + n
            procAx(dPlt, dSub, pdDfr, sglAx=cAx[m, n], k=k)
            addTxtToAx(dTxt, sglAx=cAx[m, n])
    decorateSavePlot(dPlt, sPF='pFPlt', sTtl=sTtl, addLeg=False,
                     saveFig=savePlt)
    plt.close()

# === Main ====================================================================
if dInp['plotF2']:
    plotFigures234(dInp, dPlt=dPlt2, pdDfr=procDataForPlot(dInp, dPlt=dPlt2))
    print('Plotted figure 2.')
if dInp['plotF3']:
    plotFigures234(dInp, dPlt=dPlt3, pdDfr=procDataForPlot(dInp, dPlt=dPlt3))
    print('Plotted figure 3.')
if dInp['plotF4']:
    plotFigures234(dInp, dPlt=dPlt4, pdDfr=procDataForPlot(dInp, dPlt=dPlt4))
    print('Plotted figure 4.')
if dInp['plotF5']:
    plotFigure5(dInp, dPlt=dPlt5, pdDfr=procDataForPlot(dInp, dPlt=dPlt5))
    print('Plotted figure 5.')

# #############################################################################