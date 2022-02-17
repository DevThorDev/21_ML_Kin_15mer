# -*- coding: utf-8 -*-
###############################################################################
# --- F_02__PltFunctions.py ---------------------------------------------------
###############################################################################
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

# --- Plotter help functions --------------------------------------------------
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

def pltYeqX(dPlt, sZL):
    if dPlt['pltYeqX']:
        plt.plot(dPlt['dXLim'][sZL], dPlt['dXLim'][sZL], ls=dPlt['lnStyYeqX'],
                 lw=dPlt['lnWdYeqX'], color=dPlt['clrYeqX'])

def pltHLine(dPlt, yLim):
    if dPlt['pltHLn']:
        for cX in dPlt['lXPosHLn']:
            plt.plot((cX, cX), yLim, ls=dPlt['lnStyHLn'], lw=dPlt['lnWdHLn'],
                     color=dPlt['clrHLn'])

def annotPlot(dITp, dPlt, lX, lY, lAnnot, sID, sGT, sZL, sBO, yLim, RD=None):
    dVAn, dTxtOf = dITp['dMix4Plt']['dVAnnot'], dITp['dMix4Plt']['dTxtOffsFr']
    cSetVAnnot, xLim = dVAn[sID][sGT][sZL][sBO], dPlt['dXLim'][sZL]
    for cX, cY, cA in zip(lX, lY, lAnnot):
        if GF.isInSeqSet(cA, cSetVAnnot):
            if RD is not None:
                cA = round(cA, RD)
            cTOX = cX + (xLim[1] - xLim[0])*dTxtOf[sID][sGT][sZL][sBO][0]
            cTOY = cY + (yLim[1] - yLim[0])*dTxtOf[sID][sGT][sZL][sBO][1]
            bboxProps = dict(boxstyle=dPlt['dBoxSty'][sBO],
                             fc=dPlt['dBoxFC'][sBO], ec=dPlt['dBoxEC'][sBO],
                             alpha=dPlt['dBoxAlpha'][sBO])
            arrowProps = dict(arrowstyle=dPlt['dArrSty'][sBO],
                              connectionstyle=dPlt['dConnSty'][sBO])
            plt.annotate(cA, xy=(cX, cY), xytext=(cTOX, cTOY),
                         xycoords=(dPlt['coordSysX'], dPlt['coordSysY']),
                         textcoords=(dPlt['coordSysX'], dPlt['coordSysY']),
                         size=dPlt['dTxtSz'][sBO],
                         bbox=bboxProps, arrowprops=arrowProps)

def getSTxtAnnotBar(cVSl, cVMix, cY, sTxt, nTp, RD=None):
    cNum, cVRd = cVSl, cVMix*100
    if nTp == 0:
        if RD is not None:
            cVRd = round(cVRd, RD)
        cNum, cY = str(cVRd) + GC.S_PERC, max(0., cVMix/2.)
        if cVRd == 0:
            return cNum, cY
    return GF.joinS([sTxt, cNum], sJoin=GC.S_SPACE), cY

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

def decorateSavePlot(dPlt, pF, serDt=None, sTtl=None, xLim=None, yLim=None,
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
        plt.savefig(pF)

# --- Functions plotting histograms for experimental and "mixed" data ---------
def plotHistVXDt(dITp, dPlt, vXD, pF, sTtl=None, savePlt=True):
    plt.hist(vXD, bins=dPlt['nBinsHist'], alpha=dPlt['alpha'],
             density=dPlt['isDensity'])
    for sDist, dDist in dPlt['dDistPlt'].items():
        if dDist['pltDist']:
            x, y = SF.getPDFct(dITp, dPlt, vDt=vXD, dist=sDist)
            plt.plot(x, y, lw=dPlt['lwdPD'], color=dDist['clr'], label=sDist)
    decorateSavePlot(dPlt, pF, sTtl=sTtl, saveFig=savePlt)
    plt.close()

# --- Functions plotting histograms for binary operations result data ---------
def barPlotVBOs(dPlt, dIHist, dSLeg, serX, dfrY, pF, sTtl=None, savePlt=True):
    for _, serY in dfrY.items():
        plt.bar(serX, serY, width=dIHist['binWd'], alpha=dIHist['alpha'],
                label=dSLeg[serY.name])
    decorateSavePlot(dPlt, pF, sTtl=sTtl, yLim=dIHist['yLim'],
                     yMax=dfrY.stack().max(), pltAx=True, saveFig=savePlt)
    plt.close()

# --- Functions plotting histograms for comparisons between BOs ---------------
def plotXYCmpBOs(dITp, dPlt, d2DfrBH, sID, sGT, sZL, sCX, sCY, sCA, pF,
                 sTtl=None, savePlt=True):
    nDX, nDY = dPlt['nDigXPcAx'], dPlt['nDigYPcAx']
    cFig, cAx = plt.subplots(figsize=dPlt['tFigSz'])
    cAx.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=nDX))
    cAx.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=nDY))
    plt.subplots_adjust(left=dPlt['padLeft'], right=dPlt['padRight'],
                        top=dPlt['padTop'], bottom=dPlt['padBottom'])
    dDfr, cYLim = d2DfrBH[sGT], dITp['dMix4Plt']['dYLim'][sID][sGT][sZL]
    for sBO in dITp['lBOsUsed']:
        if sCY in dDfr[sBO][0].columns:
            sLbl, serA = dITp['dSNmBOPlt'][sBO], dDfr[sBO][0][sCA]
            serX, serY = dDfr[sBO][0][sCX], dDfr[sBO][0][sCY]
            plt.plot(serX, serY, ls=dPlt['lnSty'], lw=dPlt['lnWd'],
                     marker=dPlt['mkTp'], ms=dPlt['mkSz'],
                     color=dITp['dClrBO'][sBO], label=sLbl)
            pltYeqX(dPlt, sZL)
            pltHLine(dPlt, cYLim)
            annotPlot(dITp, dPlt, serX, serY, serA, sID, sGT, sZL, sBO, cYLim,
                      RD=GC.R02)
    decorateSavePlot(dPlt, pF, sTtl=sTtl, xLim=dPlt['dXLim'][sZL], yLim=cYLim,
                     saveFig=savePlt)
    plt.close()

# --- Functions plotting bar plots for comparisons between BOs ----------------
def plotBarCompBOs(dITp, dPlt, d2DfrBH, sID, sGT, sC0, sCMix, sCSl, pF,
                   sTtl=None, savePlt=True):
    nDY = dPlt['nDigYPcAx']
    cFig, cAx = plt.subplots(figsize=dPlt['tFigSz'],
                             tight_layout=dPlt['tightLay'])
    cAx.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=nDY))
    dVSl = dITp['dMix4Plt']['dBOValSel'][sID][sGT]
    lBOsPlt = GF.iterIntersect(dITp['lBOsUsed'], list(dVSl))
    dDfr = {sBO: d2DfrBH[sGT][sBO][0] for sBO in lBOsPlt}
    lSX = [dITp['dSNmBOPlt'][sBO] for sBO in lBOsPlt]
    lVY = [GF.getSglElWSelC(dDfr[sBO], sC0, sCSl, dVSl[sBO], RD=GC.R08)
           for sBO in lBOsPlt]
    lVM = [GF.getSglElWSelC(dDfr[sBO], sCMix, sCSl, dVSl[sBO], RD=GC.R08)
           for sBO in lBOsPlt]
    lClr, lVX = [dITp['dClrBO'][sBO] for sBO in lBOsPlt], range(len(lVY))
    plt.bar(lVX, lVY, width=dPlt['wdBar'], color=lClr)
    plt.bar(lVX, lVM, width=dPlt['wdBar'], color=dPlt['clrOverlMixSh'],
            alpha=dPlt['alphaOverlMixSh'], tick_label=lSX)
    for sBO, cX, cVCMix in zip(lBOsPlt, lVX, lVM):
        annotBar(dPlt, dVSl, cX, cVCMix, sBO, RD=dPlt['nDigVMix'])
    decorateSavePlot(dPlt, pF, sTtl=sTtl, addLeg=False, saveFig=savePlt)
    plt.close()

###############################################################################
