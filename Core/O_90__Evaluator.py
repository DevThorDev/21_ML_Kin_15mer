# -*- coding: utf-8 -*-
###############################################################################
# --- O_90__Evaluator.py ------------------------------------------------------
###############################################################################
# import Core.C_00__GenConstants as GC
# import Core.F_00__GenFunctions as GF
# import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

# -----------------------------------------------------------------------------
class Evaluator(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=90):
        super().__init__(inpDat)
        self.idO = 'O_90'
        self.descO = 'Evaluator'
        self.inpD = inpDat
        self.getDITp(iTp=iTp)
        self.D2E = None
        print('Initiated "Evaluator" base object.')

    # --- methods for filling the file paths ----------------------------------
    def fillFPs(self):
        d2PI, dIG, dITp = {}, self.dIG, self.dITp
        d2PI['DetR'] = {dIG['sPath']: dITp['pInpDet']}
        d2PI['PrbR'] = {dIG['sPath']: dITp['pInpDet']}

        for sK in ['DetR', 'PrbR']:
            dIG:
            d2PI[sK]['sLFS'] = {dIG['sPath']: dITp['pInp' + sTp],
                                     dIG['sLFC']: dITp['sFInp' + sTp],
                                     dIG['sFXt']: dIG['xtCSV']}
        for sTp in ['Snips', 'EffF']:
            d2PI['DictNmer' + sTp] = {dIG['sPath']: dITp['pBinData'],
                                      dIG['sLFC']: dITp['sFDictNmer' + sTp],
                                      dIG['sLFE']: dITp['sFResComb'],
                                      dIG['sLFJCE']: dITp['sUS02'],
                                      dIG['sFXt']: dIG['xtBIN']}
        for sTp in ['EffFam', 'SeqU']:
            d2PI['Nmer' + sTp] = {dIG['sPath']: dITp['pUnqNmer'],
                                  dIG['sLFC']: dITp['sNmer' + sTp],
                                  dIG['sLFE']: dITp['sFInpClf'],
                                  dIG['sLFJE']: dITp['sUS02'],
                                  dIG['sLFJCE']: dITp['sUS02'],
                                  dIG['sFXt']: dIG['xtCSV']}
        for sTp in ['InpData', 'X', 'Y']:
            d2PI[sTp] = {dIG['sPath']: dITp['pInpData'],
                         dIG['sLFS']: dITp['s' + sTp],
                         dIG['sLFC']: dITp['sFInpClf'],
                         dIG['sLFE']: self.lIFE,
                         dIG['sLFJSC']: dITp['sUS02'],
                         dIG['sLFJCE']: dITp['sUS02'],
                         dIG['sFXt']: dIG['xtCSV']}
        d2PI['X'][dIG['sLFE']] = self.lIFEX
        self.FPs.addFPs(d2PI)
        self.d2PInf = d2PI

    def loadInpData(self, iC=0):
        if self.D2E is None:
            dfrCmbR = self.loadData(self.FPs.dPF['InpDataClf'], iC=iC)
            self.dfrInpClf = self.loadData(self.FPs.dPF['InpDataClf'], iC=iC)

    # --- print methods -------------------------------------------------------

###############################################################################