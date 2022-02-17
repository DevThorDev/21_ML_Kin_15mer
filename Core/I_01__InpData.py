# -*- coding: utf-8 -*-
###############################################################################
# --- I_01__InpData.py --------------------------------------------------------
###############################################################################
import os, pprint

from importlib import import_module

import Core.C_00__GenConstants as GC

# -----------------------------------------------------------------------------
class InputData:
# --- initialisation of the class ---------------------------------------------
    def __init__(self, inpDat, lVals=[]):
        dInp = {}
        if type(inpDat) is dict:
            for cKey in inpDat:
                dInp[cKey] = inpDat[cKey]
        elif type(inpDat) is list:
            lKeys = inpDat
            nKeys, nVals = len(lKeys), len(lVals)
            if nKeys == nVals:
                for cIdx in range(nKeys):
                    dInp[lKeys[cIdx]] = lVals[cIdx]
            else:
                print('ERROR: Cannot add to input dictionary.')
                print('Length of keys and values lists:', nKeys, '!=', nVals)
                assert False
        self.dI = dInp

# --- print methods -----------------------------------------------------------
    def __str__(self):
        sIn = (GC.S_ST31 + ' "InputData" type ' + GC.S_ST31 + GC.S_NEWL +
               'Input dictionary:' + GC.S_NEWL + str(self.dI))
        return sIn

    def printInputData(self):
        pprint.pprint(self.dI)

# --- methods adding data by importing modules --------------------------------
    def addObjTps(self, nmDObjInp):
        nmPre, pyX = GC.S_OBJINP_PRE, GC.S_EXT_PY
        for nmF in os.listdir(nmDObjInp):
            if len(nmF) >= len(nmPre) + self.dI['nDigObj'] + len(pyX):
                if nmF.startswith(nmPre) and nmF.endswith(pyX):
                    nmMd, iTp = nmDObjInp + GC.S_DOT + nmF[:(-len(pyX) - 1)], 0
                    sBID = nmF[len(nmPre):len(nmPre) + self.dI['nDigObj']]
                    try:
                        iTp = int(sBID)
                        cMd = import_module(nmMd)
                        print('Imported module', nmMd)
                        self.dI[iTp] = getattr(cMd, 'dIO')
                        self.dI[iTp]['iTp'] = iTp
                    except:
                        print('ERROR: Cannot convert', sBID, 'to an integer.')
                        print('Object with type index', iTp, 'not imported.')
                        print('Name of module:', nmMd)
                        assert False

# --- methods yielding values, lists of values and dictionaries from input ----
    def yieldOneVal(self, cKey):
        retVal = None
        if cKey in self.dI:
            retVal = self.dI[cKey]
        else:
            print('ERROR: Key', cKey, 'not in input dictionary.')
            assert False
        return retVal

    def yieldValList(self, lKeys):
        retList = [None]*len(lKeys)
        for cIdx, cKey in enumerate(lKeys):
            retList[cIdx] = self.yieldOneVal(cKey)
        return retList

    def yieldDict(self, lKeys):
        retDict = {}
        for cKey in lKeys:
            retDict[cKey] = self.yieldOneVal(cKey)
        return retDict

###############################################################################
