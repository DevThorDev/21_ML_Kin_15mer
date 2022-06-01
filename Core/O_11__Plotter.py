# -*- coding: utf-8 -*-
###############################################################################
# --- O_11__Plotter.py --------------------------------------------------------
###############################################################################
from Core.O_00__BaseClass import BaseClass

# -----------------------------------------------------------------------------
class Plotter(BaseClass):
# --- initialisation of the class ---------------------------------------------
    def __init__(self, inpDat, iTp=11, lITpUpd=[]):
        super().__init__(inpDat)
        self.idO = 'O_11'
        self.descO = 'Plotter'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        print('Initiated "Plotter" base object.')

###############################################################################
