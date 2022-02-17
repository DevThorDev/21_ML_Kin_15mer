# -*- coding: utf-8 -*-
###############################################################################
# --- Tests.py ----------------------------------------------------------------
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ### COMPARE NORMAL DISTRIBUTION TO FITTED GAMMA DISTRIBUTION ################
# --- input -------------------------------------------------------------------
cMnND, cSDND = 0., 1.
cSizeArrRN = 1000

lwCurves = 2
alphaCurves = .6
alphaBars = .3

sNmPlt = 'TestPlot'
sXtPDF = '.pdf'

# --- main -------------------------------------------------------------------
print('='*80, '\n', '-'*30, ' Tests.py started.  ', '-'*30, '\n', sep='')
arrRNND = stats.norm.rvs(loc=cMnND, scale=cSDND, size=cSizeArrRN)

locN, sclN = stats.norm.fit(arrRNND)
sh1G, locG, sclG = stats.gamma.fit(arrRNND)

x = np.linspace(stats.norm.ppf(0.01, loc=cMnND, scale=cSDND),
                stats.norm.ppf(0.99, loc=cMnND, scale=cSDND), 100)

plt.plot(x, stats.norm.pdf(x, loc=cMnND, scale=cSDND), 'r-', lw=lwCurves,
         alpha=alphaCurves, label='given normal pdf')
plt.plot(x, stats.norm.pdf(x, loc=locN, scale=sclN), 'm-', lw=lwCurves,
         alpha=alphaCurves, label='fitted normal pdf')
plt.plot(x, stats.gamma.pdf(x, a=sh1G, loc=locG, scale=sclG), 'g-',
         lw=lwCurves, alpha=alphaCurves, label='fitted gamma pdf')
plt.hist(arrRNND, density=True, histtype='stepfilled', alpha=alphaBars)
plt.legend(loc='best')
plt.savefig(sNmPlt + sXtPDF)
print('Saved figure to file "', sNmPlt + sXtPDF, '".', sep='')
plt.close()

print('-'*30, ' Tests.py finished. ', '-'*30, '\n', '='*80, sep='')

###############################################################################
