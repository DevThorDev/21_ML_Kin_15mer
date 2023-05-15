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
S_PDF = 'pdf'

# === Input ===================================================================
# --- general -----------------------------------------------------------------
sOType = 'Plotter (D_11__Plotter)'
sNmSpec = 'Data for the Plotter class in O_11__Plotter'
dClrBO = {}
pPltFig = os.path.join('..', 'PlottedFigures')

# --- plot-specific: figure 2 -------------------------------------------------
sPltFig2 = 'Figure2'
nDigYPcAx = 2
tFigSz = (6.4, 4.8)
layoutTp = 'compressed'
wdBar = 1.4
lblX = None
lblY = None
clrOverlMixSh = 1
alphaOverlMixSh = 1
nDigVMix = 2
lnWdAxY = 1
lnStyAxY = '-'
lnClrAxY = 'k'

# --- predefined colours ------------------------------------------------------

# --- data for all plots ------------------------------------------------------

# === Create input and plot dictionaries ======================================
# --- create input dictionary -------------------------------------------------
dInp = {# --- constants
        'sSpace': S_SPACE,
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sUsc': S_USC,
        'sPerc': S_PERC,
        'sPDF': S_PDF,
        # --- general
        'sOType': sOType,
        'sNmSpec': sNmSpec,
        'dClrBO': dClrBO,
        'pPltFig': pPltFig}

dPlt2 = {'pFPlt2': os.path.join(pPltFig, sPltFig2 + S_DOT + S_PDF),
         'nDigYPcAx': nDigYPcAx,
         'tFigSz': tFigSz,
         'layoutTp': layoutTp,
         'wdBar': wdBar,
         'lblX': lblX,
         'lblY': lblY,
         'clrOverlMixSh': clrOverlMixSh,
         'alphaOverlMixSh': alphaOverlMixSh,
         'nDigVMix': nDigVMix,
         'lnWdAxY': lnWdAxY,
         'lnStyAxY': lnStyAxY,
         'lnClrAxY': lnClrAxY,
         }

# --- create plot dictionary -------------------------------------------------

# === Functions ===============================================================
# --- Plotting bar plots for no location / location data comparison -----------
# --- String selection and manipulation functions -----------------------------
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

def plotFigure2(dInp, dPlt, pdDfr, sTtl=None, savePlt=True):
    # nDY = dPlt['nDigYPcAx']
    cFig, cAx = plt.subplots(nrows=2, ncols=2, figsize=dPlt['tFigSz'],
                             layout=dPlt['layoutTp'])
    # cAx.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=nDY))
    lClr = ['r', 'b', 'y', 'g']
    cAx[0, 0].bar(x=[1, 2, 4, 5], height=pdDfr.loc[0, :], width=1.5,
                  bottom=0, align='center', color=lClr)
    cAx[0, 1].bar(x=[1, 2, 4, 5], height=pdDfr.loc[1, :], width=1.5,
                  bottom=0, align='center', color=lClr)
    cAx[1, 0].bar(x=[1, 2, 4, 5], height=pdDfr.loc[2, :], width=1.5,
                  bottom=0, align='center', color=lClr)
    cAx[1, 1].bar(x=[1, 2, 4, 5], height=pdDfr.loc[14, :], width=1.5,
                  bottom=0, align='center', color=lClr)
    cFig.suptitle('Location comparison')
    decorateSavePlot(dPlt, sPF='pFPlt2', sTtl=sTtl, addLeg=False,
                     saveFig=savePlt)
    plt.close()

# === Main ====================================================================
dfrFig2 = pd.read_csv("C:\\Users\\tstefan\\Documents\\25_Papers\\01_FirstAuthor\\09_SysBio_Classification\\9_Figures\\Data_Figure2.csv",
                      sep=dInp['sSemicol'])
dfrFig2Red = dfrFig2[dfrFig2['PlotIt'] == 1]
lDat = list(dfrFig2Red.columns)[2:]
dfrFig2Red = dfrFig2Red.loc[:, lDat]
plotFigure2(dInp, dPlt=dPlt2, pdDfr=dfrFig2Red)

# #############################################################################