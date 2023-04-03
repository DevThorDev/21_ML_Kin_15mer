# -*- coding: utf-8 -*-
###############################################################################
# --- O_95__Timing.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# -----------------------------------------------------------------------------
class Timing:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, stT=None, rndDig=GC.R02):
        self.stT = stT
        self.rdDig=rndDig
        self.elT_02_0001_getLInpSeq = 0.
        self.elT_02_0002_genDLenSeq = 0.
        self.elT_02_0003_performLhAnalysis = 0.
        self.elT_02_0004_performProbAnalysis_A = 0.
        self.elT_02_0005_performProbAnalysis_B = 0.
        self.elT_02_0006_performProbAnalysis_C = 0.
        self.elT_02_0007_performProbAnalysis_D = 0.
        self.elT_02_0008_calcProbTable = 0.
        self.elT_02_0009_getD2TotalProbSnip = 0.
        self.elT_02_0010_getD2CondProbSnip = 0.
        self.elT_02_0011_saveD2TCProbSnipAsDfr = 0.
        self.elT_02_0012_getProbSglPos = 0.
        self.elT_05_0001_ViterbiAlgorithm = 0.
        self.elT_06_0001_ClfDataLoader = 0.
        self.elT_07_0001_DummyClf_Ini = 0.
        self.elT_07_0002_DummyClf_Pred = 0.
        self.elT_07_0011_AdaClf_Ini = 0.
        self.elT_07_0012_AdaClf_Pred = 0.
        self.elT_07_0021_RFClf_Ini = 0.
        self.elT_07_0022_RFClf_Pred = 0.
        self.elT_07_0031_XTrClf_Ini = 0.
        self.elT_07_0032_XTrClf_Pred = 0.
        self.elT_07_0041_GrBClf_Ini = 0.
        self.elT_07_0042_GrBClf_Pred = 0.
        self.elT_07_0051_HGrBClf_Ini = 0.
        self.elT_07_0052_HGrBClf_Pred = 0.
        self.elT_07_0061_GPClf_Ini = 0.
        self.elT_07_0062_GPClf_Pred = 0.
        self.elT_07_0071_PaAClf_Ini = 0.
        self.elT_07_0072_PaAClf_Pred = 0.
        self.elT_07_0081_PctClf_Ini = 0.
        self.elT_07_0082_PctClf_Pred = 0.
        self.elT_07_0091_SGDClf_Ini = 0.
        self.elT_07_0092_SGDClf_Pred = 0.
        self.elT_07_0101_CtNBClf_Ini = 0.
        self.elT_07_0102_CtNBClf_Pred = 0.
        self.elT_07_0111_CpNBClf_Ini = 0.
        self.elT_07_0112_CpNBClf_Pred = 0.
        self.elT_07_0121_GsNBClf_Ini = 0.
        self.elT_07_0122_GsNBClf_Pred = 0.
        self.elT_07_0131_MLPClf_Ini = 0.
        self.elT_07_0132_MLPClf_Pred = 0.
        self.elT_07_0141_LSVClf_Ini = 0.
        self.elT_07_0142_LSVClf_Pred = 0.
        self.elT_07_0151_NSVClf_Ini = 0.
        self.elT_07_0152_NSVClf_Pred = 0.
        self.elT_07_0161_CSVClf_Ini = 0.
        self.elT_07_0162_CSVClf_Pred = 0.
        self.elT_07_1000_PropCalculator = 0.
        self.elT_90_0001_Evaluator_ClPred = 0.
        self.elT_XX_9999_Other = 0.
        self.elT_Sum = 0.
        self.updateLElTimes()
        self.lSMth = ['getLInpSeq', 'genDLenSeq', 'performLhAnalysis',
                      'performProbAnalysis_A', 'performProbAnalysis_B',
                      'performProbAnalysis_C', 'performProbAnalysis_D',
                      'calcProbTable', 'getD2TotalProbSnip',
                      'getD2CondProbSnip', 'saveD2TCProbSnipAsDfr',
                      'getProbSglPos', 'ViterbiAlgorithm', 'ClfDataLoader',
                      'DummyClf_Ini', 'DummyClf_Pred', 'AdaClf_Ini',
                      'AdaClf_Pred', 'RFClf_Ini', 'RFClf_Pred', 'XTrClf_Ini',
                      'XTrClf_Pred', 'GrBClf_Ini', 'GrBClf_Pred',
                      'HGrBClf_Ini', 'HGrBClf_Pred', 'GPClf_Ini', 'GPClf_Pred',
                      'PaAClf_Ini', 'PaAClf_Pred', 'PctClf_Ini', 'PctClf_Pred',
                      'SGDClf_Ini', 'SGDClf_Pred', 'CtNBClf_Ini',
                      'CtNBClf_Pred', 'CpNBClf_Ini', 'CpNBClf_Pred',
                      'GsNBClf_Ini', 'GsNBClf_Pred', 'MLPClf_Ini',
                      'MLPClf_Pred', 'LSVClf_Ini', 'LSVClf_Pred', 'NSVClf_Ini',
                      'NSVClf_Pred', 'CSVClf_Ini', 'CSVClf_Pred',
                      'PropCalculator', 'Evaluator_ClPred', 'Other']
        assert len(self.lSMth) == len(self.lElT)

    # --- update methods ------------------------------------------------------
    def updateLElTimes(self):
        self.lElT = [self.elT_02_0001_getLInpSeq,
                     self.elT_02_0002_genDLenSeq,
                     self.elT_02_0003_performLhAnalysis,
                     self.elT_02_0004_performProbAnalysis_A,
                     self.elT_02_0005_performProbAnalysis_B,
                     self.elT_02_0006_performProbAnalysis_C,
                     self.elT_02_0007_performProbAnalysis_D,
                     self.elT_02_0008_calcProbTable,
                     self.elT_02_0009_getD2TotalProbSnip,
                     self.elT_02_0010_getD2CondProbSnip,
                     self.elT_02_0011_saveD2TCProbSnipAsDfr,
                     self.elT_02_0012_getProbSglPos,
                     self.elT_05_0001_ViterbiAlgorithm,
                     self.elT_06_0001_ClfDataLoader,
                     self.elT_07_0001_DummyClf_Ini,
                     self.elT_07_0002_DummyClf_Pred,
                     self.elT_07_0011_AdaClf_Ini,
                     self.elT_07_0012_AdaClf_Pred,
                     self.elT_07_0021_RFClf_Ini,
                     self.elT_07_0022_RFClf_Pred,
                     self.elT_07_0031_XTrClf_Ini,
                     self.elT_07_0032_XTrClf_Pred,
                     self.elT_07_0041_GrBClf_Ini,
                     self.elT_07_0042_GrBClf_Pred,
                     self.elT_07_0051_HGrBClf_Ini,
                     self.elT_07_0052_HGrBClf_Pred,
                     self.elT_07_0061_GPClf_Ini,
                     self.elT_07_0062_GPClf_Pred,
                     self.elT_07_0071_PaAClf_Ini,
                     self.elT_07_0072_PaAClf_Pred,
                     self.elT_07_0081_PctClf_Ini,
                     self.elT_07_0082_PctClf_Pred,
                     self.elT_07_0091_SGDClf_Ini,
                     self.elT_07_0092_SGDClf_Pred,
                     self.elT_07_0101_CtNBClf_Ini,
                     self.elT_07_0102_CtNBClf_Pred,
                     self.elT_07_0111_CpNBClf_Ini,
                     self.elT_07_0112_CpNBClf_Pred,
                     self.elT_07_0121_GsNBClf_Ini,
                     self.elT_07_0122_GsNBClf_Pred,
                     self.elT_07_0131_MLPClf_Ini,
                     self.elT_07_0132_MLPClf_Pred,
                     self.elT_07_0141_LSVClf_Ini,
                     self.elT_07_0142_LSVClf_Pred,
                     self.elT_07_0151_NSVClf_Ini,
                     self.elT_07_0152_NSVClf_Pred,
                     self.elT_07_0161_CSVClf_Ini,
                     self.elT_07_0162_CSVClf_Pred,
                     self.elT_07_1000_PropCalculator,
                     self.elT_90_0001_Evaluator_ClPred,
                     self.elT_XX_9999_Other]

    def updateTimes(self, tMth=None, stTMth=None, endTMth=None):
        if stTMth is not None and endTMth is not None:
            elT = endTMth - stTMth
            if tMth == (2, 1):
                self.elT_02_0001_getLInpSeq += elT
            elif tMth == (2, 2):
                self.elT_02_0002_genDLenSeq += elT
            elif tMth == (2, 3):
                self.elT_02_0003_performLhAnalysis += elT
            elif tMth == (2, 4):
                self.elT_02_0004_performProbAnalysis_A += elT
            elif tMth == (2, 5):
                self.elT_02_0005_performProbAnalysis_B += elT
            elif tMth == (2, 6):
                self.elT_02_0006_performProbAnalysis_C += elT
            elif tMth == (2, 7):
                self.elT_02_0007_performProbAnalysis_D += elT
            elif tMth == (2, 8):
                self.elT_02_0008_calcProbTable += elT
            elif tMth == (2, 9):
                self.elT_02_0009_getD2TotalProbSnip += elT
            elif tMth == (2, 10):
                self.elT_02_0010_getD2CondProbSnip += elT
            elif tMth == (2, 11):
                self.elT_02_0011_saveD2TCProbSnipAsDfr += elT
            elif tMth == (2, 12):
                self.elT_02_0012_getProbSglPos += elT
            elif tMth == (5, 1):
                self.elT_05_0001_ViterbiAlgorithm += elT
            elif tMth == (6, 1):
                self.elT_06_0001_ClfDataLoader += elT
            elif tMth == (7, 1):
                self.elT_07_0001_DummyClf_Ini += elT
            elif tMth == (7, 2):
                self.elT_07_0002_DummyClf_Pred += elT
            elif tMth == (7, 11):
                self.elT_07_0011_AdaClf_Ini += elT
            elif tMth == (7, 12):
                self.elT_07_0012_AdaClf_Pred += elT
            elif tMth == (7, 21):
                self.elT_07_0021_RFClf_Ini += elT
            elif tMth == (7, 22):
                self.elT_07_0022_RFClf_Pred += elT
            elif tMth == (7, 31):
                self.elT_07_0031_XTrClf_Ini += elT
            elif tMth == (7, 32):
                self.elT_07_0032_XTrClf_Pred += elT
            elif tMth == (7, 41):
                self.elT_07_0041_GrBClf_Ini += elT
            elif tMth == (7, 42):
                self.elT_07_0042_GrBClf_Pred += elT
            elif tMth == (7, 51):
                self.elT_07_0051_HGrBClf_Ini += elT
            elif tMth == (7, 52):
                self.elT_07_0052_HGrBClf_Pred += elT
            elif tMth == (7, 61):
                self.elT_07_0061_GPClf_Ini += elT
            elif tMth == (7, 62):
                self.elT_07_0062_GPClf_Pred += elT
            elif tMth == (7, 71):
                self.elT_07_0071_PaAClf_Ini += elT
            elif tMth == (7, 72):
                self.elT_07_0072_PaAClf_Pred += elT
            elif tMth == (7, 81):
                self.elT_07_0081_PctClf_Ini += elT
            elif tMth == (7, 82):
                self.elT_07_0082_PctClf_Pred += elT
            elif tMth == (7, 91):
                self.elT_07_0091_SGDClf_Ini += elT
            elif tMth == (7, 92):
                self.elT_07_0092_SGDClf_Pred += elT
            elif tMth == (7, 101):
                self.elT_07_0101_CtNBClf_Ini += elT
            elif tMth == (7, 102):
                self.elT_07_0102_CtNBClf_Pred += elT
            elif tMth == (7, 111):
                self.elT_07_0111_CpNBClf_Ini += elT
            elif tMth == (7, 112):
                self.elT_07_0112_CpNBClf_Pred += elT
            elif tMth == (7, 121):
                self.elT_07_0121_GsNBClf_Ini += elT
            elif tMth == (7, 122):
                self.elT_07_0122_GsNBClf_Pred += elT
            elif tMth == (7, 131):
                self.elT_07_0131_MLPClf_Ini += elT
            elif tMth == (7, 132):
                self.elT_07_0132_MLPClf_Pred += elT
            elif tMth == (7, 141):
                self.elT_07_0141_LSVClf_Ini += elT
            elif tMth == (7, 142):
                self.elT_07_0142_LSVClf_Pred += elT
            elif tMth == (7, 151):
                self.elT_07_0151_NSVClf_Ini += elT
            elif tMth == (7, 152):
                self.elT_07_0152_NSVClf_Pred += elT
            elif tMth == (7, 161):
                self.elT_07_0161_CSVClf_Ini += elT
            elif tMth == (7, 162):
                self.elT_07_0162_CSVClf_Pred += elT
            elif tMth == (7, 1000):
                self.elT_07_1000_PropCalculator += elT
            elif tMth == (90, 1):
                self.elT_90_0001_Evaluator_ClPred += elT
            elif tMth == (None, 9999):
                self.elT_XX_9999_Other += elT
            self.elT_Sum += elT
            self.updateLElTimes()

    # --- print methods -------------------------------------------------------
    def __str__(self):
        sIn = (GC.S_WV80 + GC.S_NEWL + GC.S_SP04 + 'Time (s) used in:' +
               GC.S_NEWL + 'Method 02_0001 | "getLInpSeq":\t\t\t' +
               str(round(self.elT_02_0001_getLInpSeq, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0002 | "genDLenSeq":\t\t\t' +
               str(round(self.elT_02_0002_genDLenSeq, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0003 | "performLhAnalysis":\t\t' +
               str(round(self.elT_02_0003_performLhAnalysis, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0004 | "performProbAnalysis_A":\t' +
               str(round(self.elT_02_0004_performProbAnalysis_A, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0005 | "performProbAnalysis_B":\t' +
               str(round(self.elT_02_0005_performProbAnalysis_B, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0006 | "performProbAnalysis_C":\t' +
               str(round(self.elT_02_0006_performProbAnalysis_C, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0007 | "performProbAnalysis_D":\t' +
               str(round(self.elT_02_0007_performProbAnalysis_D, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0008 | "calcProbTable":\t' +
               str(round(self.elT_02_0008_calcProbTable, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0009 | "getD2TotalProbSnip":\t' +
               str(round(self.elT_02_0009_getD2TotalProbSnip, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0010 | "getD2CondProbSnip":\t' +
               str(round(self.elT_02_0010_getD2CondProbSnip, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0011 | "saveD2TCProbSnipAsDfr":\t' +
               str(round(self.elT_02_0011_saveD2TCProbSnipAsDfr, self.rdDig)) +
               GC.S_NEWL + 'Method 02_0012 | "getProbSglPos":\t' +
               str(round(self.elT_02_0012_getProbSglPos, self.rdDig)) +
               GC.S_NEWL + 'Method 05_0001 | "ViterbiAlgorithm":\t' +
               str(round(self.elT_05_0001_ViterbiAlgorithm, self.rdDig)) +
               GC.S_NEWL + 'Method 06_0001 | "ClfDataLoader":\t' +
               str(round(self.elT_06_0001_ClfDataLoader, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0001 | "DummyClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0001_DummyClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0002 | "DummyClf_Predict":\t' +
               str(round(self.elT_07_0002_DummyClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0011 | "AdaClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0011_AdaClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0012 | "AdaClf_Predict":\t' +
               str(round(self.elT_07_0012_AdaClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0021 | "RFClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0021_RFClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0022 | "RFClf_Predict":\t' +
               str(round(self.elT_07_0022_RFClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0031 | "XTrClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0031_XTrClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0032 | "XTrClf_Predict":\t' +
               str(round(self.elT_07_0032_XTrClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0041 | "GrBClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0041_GrBClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0042 | "GrBClf_Predict":\t' +
               str(round(self.elT_07_0042_GrBClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0051 | "HGrBClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0051_HGrBClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0052 | "HGrBClf_Predict":\t' +
               str(round(self.elT_07_0052_HGrBClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0061 | "GPClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0061_GPClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0062 | "GPClf_Predict":\t' +
               str(round(self.elT_07_0062_GPClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0071 | "PaAClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0071_PaAClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0072 | "PaAClf_Predict":\t' +
               str(round(self.elT_07_0072_PaAClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0081 | "PctClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0081_PctClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0082 | "PctClf_Predict":\t' +
               str(round(self.elT_07_0082_PctClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0091 | "SGDClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0091_SGDClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0092 | "SGDClf_Predict":\t' +
               str(round(self.elT_07_0092_SGDClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0101 | "CtNBClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0101_CtNBClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0102 | "CtNBClf_Predict":\t' +
               str(round(self.elT_07_0102_CtNBClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0111 | "CpNBClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0111_CpNBClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0112 | "CpNBClf_Predict":\t' +
               str(round(self.elT_07_0112_CpNBClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0121 | "GsNBClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0121_GsNBClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0122 | "GsNBClf_Predict":\t' +
               str(round(self.elT_07_0122_GsNBClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0131 | "MLPClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0131_MLPClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0132 | "MLPClf_Predict":\t' +
               str(round(self.elT_07_0132_MLPClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0141 | "LSVClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0141_LSVClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0142 | "LSVClf_Predict":\t' +
               str(round(self.elT_07_0142_LSVClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0151 | "NSVClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0151_NSVClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0152 | "NSVClf_Predict":\t' +
               str(round(self.elT_07_0152_NSVClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0161 | "CSVClf_Ini_ImbSmpl":\t' +
               str(round(self.elT_07_0161_CSVClf_Ini, self.rdDig)) +
               GC.S_NEWL + 'Method 07_0162 | "CSVClf_Predict":\t' +
               str(round(self.elT_07_0162_CSVClf_Pred, self.rdDig)) +
               GC.S_NEWL + 'Method 07_1000 | "PropCalculator":\t' +
               str(round(self.elT_07_1000_PropCalculator, self.rdDig)) +
               GC.S_NEWL + 'Method 90_0001 | "Evaluator_ClPred":\t' +
               str(round(self.elT_90_0001_Evaluator_ClPred, self.rdDig)) +
               GC.S_NEWL + 'Method XX_9999 | "Other":\t' +
               str(round(self.elT_XX_9999_Other, self.rdDig)) +
               GC.S_NEWL + GC.S_WV80)
        return sIn

    def printRelTimes(self):
        if self.elT_Sum > 0:
            print(GC.S_WV80)
            for sMth, cElT in zip(self.lSMth, self.lElT):
                sX = str(round(cElT/self.elT_Sum*100., self.rdDig)) + '%'
                sWS = GC.S_SPACE*(6 - len(sX))
                print(GC.S_SP04 + sWS + sX + '\t(share of time in Method "' +
                      sMth + '")')
            print(GC.S_WV80)

###############################################################################