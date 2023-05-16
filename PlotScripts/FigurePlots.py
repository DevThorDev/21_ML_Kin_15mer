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
plotF2 = True
plotF3 = False
plotF4 = False
plotF5 = False

# --- predefined colours ------------------------------------------------------

# --- data for all plots ------------------------------------------------------

# === Plot-specific input for figure 2 ========================================
sFPltDat_F2 = 'Data_Figure2'
sCol0NoDat_F2, sCol1NoDat_F2 = 'PlotIt', 'Metric'
lSColNoDat_F2 = [sCol0NoDat_F2, sCol1NoDat_F2]
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
lColInSubPl_F2 = [(0.85, 0.5, 0.0, 0.85), (0.8, 0.0, 0.0, 0.75),
                  (0.0, 0.5, 0.75, 0.85), (0.0, 0.0, 0.8, 0.75)]
lLPosSubPl_F2 = [lPosInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lLLblXSubPl_F2 = [lLblXInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lLLblYSubPl_F2 = [lblYInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lWdBarSubPl_F2 = [lWdBarInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
lLColSubPl_F2 = [lColInSubPl_F2]*(nRowSubPl_F2*nColSubPl_F2)
tYLimSubPl12_F2 = (0.5, 1.0)
tYLimSubPl34_F2 = (0.1, 0.7)
tYLimSubPl56_F2 = (0.7, 0.9)
lTYLimSubPl_F2 = [tYLimSubPl12_F2, tYLimSubPl12_F2, tYLimSubPl34_F2,
                  tYLimSubPl34_F2, tYLimSubPl56_F2, tYLimSubPl56_F2]
lYTckSubPl12_F2 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lYTckSubPl34_F2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
lYTckSubPl56_F2 = [0.7, 0.8, 0.9]
lLYTckSubPl_F2 = [lYTckSubPl12_F2, lYTckSubPl12_F2, lYTckSubPl34_F2,
                  lYTckSubPl34_F2, lYTckSubPl56_F2, lYTckSubPl56_F2]
lXLblSubPl_F2 = ['accuracy', 'bal. accuracy', 'min. accuracy',
                 'av. sensitivity', 'av. specificity', 'ROC AUC']

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
        }

# --- create plot dictionaries ------------------------------------------------
dPlt2 = {'pFDat': os.path.join(pPltDat, sFPltDat_F2 + S_DOT + S_CSV),
         'pFPlt': os.path.join(pPltFig, sPlt_F2 + S_DOT + S_PDF),
         'sCol0NoDat': sCol0NoDat_F2,
         'sCol1NoDat': sCol1NoDat_F2,
         'lSColNoDat': lSColNoDat_F2,
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
                    'lLCol': lLColSubPl_F2,
                    'lTYLim': lTYLimSubPl_F2,
                    'lLYTck': lLYTckSubPl_F2,
                    'lXLbl': lXLblSubPl_F2}}

# === General functions =======================================================
# --- String manipulation functions -------------------------------------------
def joinS(itS, cJ=S_USC):
    if type(itS) in [str, int, float]:    # convert to 1-element list
        itS = [str(itS)]
    if len(itS) == 0:
        return ''
    elif len(itS) == 1:
        return str(itS[0])
    lS2J = [str(s) for s in itS if (s is not None and len(str(s)) > 0)]
    if type(cJ) == str:
        return cJ.join(lS2J).strip()
    elif type(cJ) in [tuple, list, range]:
        sRet = str(itS[0])
        if len(cJ) == 0:
            cJ = [cJ]*(len(itS) - 1)
        else:
            cJ = [str(j) for j in cJ]
        if len(cJ) >= len(itS) - 1:
            for k, sJ in zip(range(1, len(itS)), cJ[:(len(itS) - 1)]):
                sRet = joinS([sRet, str(itS[k])], cJ=sJ)
        else:
            # cJ too short --> only use first element of cJ
            sRet = cJ[0].join(lS2J).strip()
        return sRet

def getSTxtAnnotBar(dInp, cVSl, cVMix, cY, sTxt, nTp, RD=None):
    cNum, cVRd = cVSl, cVMix*100
    if nTp == 0:
        if RD is not None:
            cVRd = round(cVRd, RD)
        cNum, cY = str(cVRd) + dInp['sPerc'], max(0., cVMix/2.)
        if cVRd == 0:
            return cNum, cY
    return joinS([sTxt, cNum], sJoin=dInp['sSpace']), cY

# === Data manipulation functions =============================================
# --- Fig. 2 data manipulation ------------------------------------------------
def procF2Data(dInp, dPlt):
    dfrF2 = pd.read_csv(dPlt['pFDat'], sep=dInp['sSemicol'])
    dfrF2Red = dfrF2[dfrF2[dPlt['sCol0NoDat']] == 1]
    lDat = [s for s in dfrF2Red.columns if s not in dPlt['lSColNoDat']]
    return dfrF2Red.loc[:, lDat].reset_index(drop=True)

# === Plotting functions ======================================================
# --- Plotting helper functions -----------------------------------------------
def annotBar(dPlt, dVSl, cX, cVMix, sBO, RD=None):
    for nTp, (cY, cCSysY, sTxt) in dPlt['dAnnot'][sBO].items():
        bboxProps = dict(boxstyle=dPlt['dBoxSty'][sBO], fc=dPlt['dBoxFC'][sBO],
                         ec=dPlt['dBoxEC'][sBO], alpha=dPlt['dBoxAlpha'][sBO])
        sTxtAn, cY = getSTxtAnnotBar(dVSl[sBO], cVMix, cY, sTxt, nTp, RD=RD)
        plt.annotate(sTxtAn, xy=(cX, cY), xytext=(cX, cY),
                     xycoords=(dPlt['coordSysX'], cCSysY),
                     textcoords=(dPlt['coordSysX'], cCSysY),
                     size=dPlt['dTxtSz'][sBO], horizontalalignment=dPlt['hAl'],
                     verticalalignment=dPlt['vAl'], bbox=bboxProps)

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

# --- Fig. 2: plotting bar plots for no location / location data comparison ---
def plotFigure2(dInp, dPlt, pdDfr, sTtl=None, savePlt=True):
    assert pdDfr.shape[0] == dPlt['nRowSubPl']*dPlt['nColSubPl']
    cFig, cAx = plt.subplots(nrows=dPlt['nRowSubPl'], ncols=dPlt['nColSubPl'],
                             figsize=dPlt['tFigSz'], layout=dPlt['layoutTp'])
    dSub = dPlt['dSubPl']
    for m in range(dPlt['nRowSubPl']):
        for n in range(dPlt['nColSubPl']):
            k = m*dPlt['nColSubPl'] + n
            cAx[m, n].bar(x=dSub['lLPos'][k], height=pdDfr.loc[k, :],
                          width=dSub['lWdBar'][k], bottom=0, align='center',
                          color=dSub['lLCol'][k])
            cAx[m, n].set_ylim(dSub['lTYLim'][k])
            lS = [str(round(x*100)) + dInp['sPerc'] for x in dSub['lLYTck'][k]]
            cAx[m, n].set_yticks(dSub['lLYTck'][k], labels=lS)
            cAx[m, n].set_xticks(ticks=dSub['lLPos'][k],
                                 labels=dSub['lLLblX'][k], rotation='vertical')
            cAx[m, n].set_xlabel(xlabel=dSub['lXLbl'][k], loc='center')
    # cFig.suptitle('Location comparison')
    decorateSavePlot(dPlt, sPF='pFPlt', sTtl=sTtl, addLeg=False,
                     saveFig=savePlt)
    plt.close()

# === Main ====================================================================
plotFigure2(dInp, dPlt=dPlt2, pdDfr=procF2Data(dInp, dPlt=dPlt2))

# #############################################################################