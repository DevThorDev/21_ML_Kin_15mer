# -*- coding: utf-8 -*-
###############################################################################
# --- SKLearnTest_MLPClassifier.py --------------------------------------------
###############################################################################
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_TEMP = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer', '98_TEMP')

S_F_INP_MLP_CLF = 'SglPosAAc_Red__Combined_S_KinasesPho15mer_202202'
S_F_OUT_MLP_CLF = S_F_INP_MLP_CLF + '_Results'

# --- strings -----------------------------------------------------------------
S_SPACE = ' '
S_DOT = '.'
S_SEMICOL = ';'
S_DASH = '-'
S_PLUS = '+'
S_EQ = '='
S_STAR = '*'
S_USC = '_'
S_VLINE = '|'
S_NEWL = '\n'

S_CSV = 'csv'

S_DS08 = S_DASH*8
S_EQ08 = S_EQ*8
S_ST08 = S_STAR*8
S_DS20 = S_DASH*20
S_DS30 = S_DASH*30
S_DS44 = S_DASH*44
S_DS80 = S_DASH*80
S_EQ80 = S_EQ*80

S_SPECIES = 'Species'

S_TYPE = 'Type'
S_PRED, S_RESPONSE = 'Predictor', 'Response'
S_TRAIN, S_TEST = 'Train', 'Test'
S_SCALED, S_UNSCD = 'Scaled', 'Unscaled'

S_TP_PRED_RESPONSE = S_TYPE + S_USC + S_PRED + S_RESPONSE
S_TP_TRAIN_TEST = S_TYPE + S_USC + S_TRAIN + S_TEST
S_TP_SCALD_UNSCD = S_TYPE + S_USC + S_SCALED + S_UNSCD

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
R08 = 8

# --- sets --------------------------------------------------------------------

# --- lists -------------------------------------------------------------------

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
doMLP_Clf = True
usedData = 'SglPosAAc_Red'  # 'IrisData' / 'SglPosAAc_Red'

# --- boolean variables -------------------------------------------------------
scaleX=False
scaleY=False

# --- numbers -----------------------------------------------------------------
rndState = None            # None (random) or integer (reproducible)
propTestData = .2
sizeHiddenLayers = (256, 128, 64, 32)

# --- strings -----------------------------------------------------------------
sActivation = 'relu'
sPltSupTtl = 'Confusion Matrix for Iris Dataset'

sHdrCY = 'EffType'     # KinaseFamily / 'EffType'

# --- lists -------------------------------------------------------------------
lPltLbls = ['Setosa', 'Versicolor', 'Virginica']

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

# === derived values and input processing =====================================
pFInpMLP_Clf = os.path.join(P_TEMP, S_F_INP_MLP_CLF + XT_CSV)
pFOutMLP_Clf = os.path.join(P_TEMP, S_F_OUT_MLP_CLF + XT_CSV)

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'doMLP_Clf': doMLP_Clf,
        'usedData': usedData,
        # --- boolean variables -----------------------------------------------
        'scaleX': scaleX,
        'scaleY': scaleY,
        # --- files, directories and paths ------------------------------------
        'pTemp': P_TEMP,
        'sFInpMLP_Clf': S_F_INP_MLP_CLF + XT_CSV,
        'sFOutMLP_Clf': S_F_OUT_MLP_CLF + XT_CSV,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers
        'R08': R08,
        'rndState': rndState,
        'propTest': propTestData,
        'sizeHL': sizeHiddenLayers,
        # --- sets
        # --- lists
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sCSV': S_CSV,
        'sSpecies': S_SPECIES,
        'sActivation': sActivation,
        'sPltSupTtl': sPltSupTtl,
        'sHdrCY': sHdrCY,
        # --- numbers ---------------------------------------------------------
        # --- lists -----------------------------------------------------------
        'lPltLbls': lPltLbls,
        # --- dictionaries ----------------------------------------------------
        # === derived values and input processing =============================
        'pFInpMLP_Clf': pFInpMLP_Clf,
        'pFOutMLP_Clf': pFOutMLP_Clf}

# ### FUNCTIONS ###############################################################
# --- General file system related functions -----------------------------------
def createDir(pF):
    if not os.path.isdir(pF):
        os.mkdir(pF)

def joinToPath(pF='', nmF='Dummy.txt'):
    if len(pF) > 0:
        createDir(pF)
        return os.path.join(pF, nmF)
    else:
        return nmF

def readCSV(pF, iCol=None, dDTp=None, cSep=S_SEMICOL):
    if os.path.isfile(pF):
        return pd.read_csv(pF, sep=cSep, index_col=iCol, dtype=dDTp)

def saveAsCSV(pdDfr, pF, reprNA='', cSep=S_SEMICOL):
    if pdDfr is not None:
        pdDfr.to_csv(pF, sep=cSep, na_rep=reprNA)

# --- Functions loading input DataFrames --------------------------------------
def loadIrisData(dI):
    iris_data = load_iris()
    X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    # y = iris_data.target
    y = pd.Series(iris_data.target, name=dI['sSpecies'])
    return X, y

def loadSKLearnTestData(dI, iCol=None):
    dfrInp = readCSV(pF=dI['pFInpMLP_Clf'], iCol=iCol)
    assert dI['sHdrCY'] in dfrInp.columns
    dfrX = dfrInp[[s for s in dfrInp.columns if s != dI['sHdrCY']]]
    serY = dfrInp[dI['sHdrCY']]
    return dfrX, serY

# --- Functions handling dictionaries -----------------------------------------
def addToDictCt(cD, cK, cIncr=1):
    if cK in cD:
        cD[cK] += cIncr
    else:
        cD[cK] = cIncr

def addToDictL(cD, cK, cE, lUniqEl=False):
    if cK in cD:
        if not lUniqEl or cE not in cD[cK]:
            cD[cK].append(cE)
    else:
        cD[cK] = [cE]

def addToDictD(cD, cKMain, cKSub, cVSub=[], allowRpl=False):
    if cKMain in cD:
        if cKSub not in cD[cKMain]:
            cD[cKMain][cKSub] = cVSub
        else:
            if allowRpl:
                cD[cKMain][cKSub] = cVSub
            else:
                print('ERROR: Key', cKSub, 'already in', cD[cKMain])
                assert False
    else:
        cD[cKMain] = {cKSub: cVSub}

# --- MLP_Clf algorithm related functions -------------------------------------

# --- Function printing the results -------------------------------------------

# ### CLASSES #################################################################
# -----------------------------------------------------------------------------
class TrainTestData:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dInp, X, y):
        self.idO = 'TrTeData'
        self.descO = 'Data for training and test'
        self.dI = dInp
        self.iniData()
        self.getTrainTestDS(X, y)
        print('Initiated "TrainTestData" base object.')

    def iniData(self):
        self.X_Train, self.X_Test = None, None
        self.y_Train, self.y_Test = None, None
        self.X_Train_Sc, self.X_Test_Sc = None, None
        self.y_Train_Sc, self.y_Test_Sc = None, None

    # --- method for filling objects with training and test data --------------
    def getTrainTestDS(self, X, y):
        tTrTe = train_test_split(X, y, random_state=self.dI['rndState'],
                                 test_size=self.dI['propTest'])
        self.X_Train, self.X_Test, self.y_Train, self.y_Test = tTrTe
        if self.dI['scaleX']:
            sc_X = StandardScaler()
            self.X_Train_Sc = sc_X.fit_transform(self.X_Train)
            self.X_Test_Sc = sc_X.transform(self.X_Test)
        if self.dI['scaleY']:
            sc_Y = StandardScaler()
            self.y_Train_Sc = sc_Y.fit_transform(self.y_Train)
            self.y_Test_Sc = sc_Y.transform(self.y_Test)
    
# -----------------------------------------------------------------------------
class Pred_MLPClf:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dInp, TTD):
        self.idO = 'Pred_MPL'
        self.descO = 'Predictor based on the MLP classifier'
        self.dI = dInp
        self.TTD = TTD
        self.predictMLP()
        self.calcConfusionMatrix()
        print('Initiated "Pred_MLPClf" base object.')

    # --- method for getting classifier score and predictions -----------------
    def predictMLP(self):
        XTrain, XTest = self.TTD.X_Train, self.TTD.X_Test
        if self.dI['scaleX']:
            XTrain, XTest = self.TTD.X_Train_Sc, self.TTD.X_Test_Sc
        tXY = (XTrain, self.TTD.y_Train)
        self.Clf = MLPClassifier(hidden_layer_sizes=self.dI['sizeHL'],
                                 activation=self.dI['sActivation'],
                                 random_state=self.dI['rndState']).fit(*tXY)
        self.ScoreClf = self.Clf.score(XTest, self.TTD.y_Test)
        self.y_Pred = self.Clf.predict(XTest)

    # --- method for calculating the confusion matrix -------------------------
    def calcConfusionMatrix(self):
        self.confusMatrix = confusion_matrix(y_true=self.TTD.y_Test,
                                             y_pred=self.y_Pred)

    # --- print methods -------------------------------------------------------
    def printFitQuality(self):
        print('True/Test y-values:', self.TTD.y_Test.to_numpy())
        print('Predicted y-values:', self.y_Pred)
        print('Classification score for the Test data:',
              round(self.ScoreClf, self.dI['R08']))
        print('Confusion matrix:', S_NEWL, self.confusMatrix, sep='')

# -----------------------------------------------------------------------------
class Plotter:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dInp):
        self.idO = 'Pltr'
        self.descO = 'MatPlotLib plotter'
        self.dI = dInp
        print('Initiated "Plotter" base object.')

    # --- method for plotting the confusion matrix ----------------------------
    def plotConfusionMatrix(self, prMLPC):
        cDisp = ConfusionMatrixDisplay(confusion_matrix=prMLPC.confusMatrix,
                                       display_labels=self.dI['lPltLbls'])
        cDisp.plot()
        cDisp.figure_.suptitle(self.dI['sPltSupTtl'])
        plt.show()

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' SKLearnTest_MLPClassifier.py ', S_DS20, S_NEWL,
      sep='')
if doMLP_Clf:
    X, y = None, None
    if dInp['usedData'] == 'IrisData':
        X, y = loadIrisData(dI=dInp)
    elif dInp['usedData'] == 'SglPosAAc_Red':
        X, y = loadSKLearnTestData(dI=dInp, iCol=0)
        
    print('X:', S_NEWL, X, S_NEWL, S_DS80, S_NEWL, 'y:', S_NEWL, y, sep='')
    print('type(X):', type(X), '| type(y):', type(y))
    print('shape of X:', X.shape, '| shape of y:', y.shape)
    
    TrTeData = TrainTestData(dInp=dInp, X=X, y=y)
    cMLPPred = Pred_MLPClf(dInp=dInp, TTD=TrTeData)
    cMLPPred.printFitQuality()
    
    Pltr = Plotter(dInp=dInp)
    Pltr.plotConfusionMatrix(prMLPC=cMLPPred)
    
print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
