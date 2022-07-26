# -*- coding: utf-8 -*-
###############################################################################
# --- SKLearnTest_MLPClassifier.py --------------------------------------------
###############################################################################
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_TEMP = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                      '98_TEMP_CSV')

S_F_INP_MLP_CLF = 'SglPosAAc_Red__Combined_S_KinasesPho15mer_202202'
S_F_INP_RF_CLF = 'TestCategorical'
S_F_OUT_MLP_CLF = S_F_INP_MLP_CLF + '_Results'
S_F_OUT_RF_CLF = S_F_INP_RF_CLF + '_Results'

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
doMLP_Clf = False
doRF_Clf = True

# 'IrisData' / 'SglPosAAc_Red' / 'TestCategorical' / 'MakeClassifier'
usedData = 'TestCategorical'

# --- boolean variables -------------------------------------------------------
scaleX=False
scaleY=False

# --- numbers -----------------------------------------------------------------
rndState = None            # None (random) or integer (reproducible)

propTestData = .2
sizeHiddenLayers = (256, 128, 64, 32)

nSamplesMC = 100
nFeaturesMC = 4
nInformativeMC = 2
nRedundantMC = 0
shuffleItMC = False
maxDepthClfMC = 2

llFeaturesMC = [['A', 'C', 'D', 'A', 'D', 'A'],
                ['B', 'D', 'B', 'B', 'C', 'A']]

# --- strings -----------------------------------------------------------------
sActivation = 'relu'
sPltSupTtl = 'Confusion Matrix for Iris Dataset'

sHdrCY_SglPosAAc_Red = 'EffType'     # 'KinaseFamily' / 'EffType'
sHdrCY_TestCategorical = 'Class'     # 'Class'

# --- lists -------------------------------------------------------------------
lPltLbls = ['Setosa', 'Versicolor', 'Virginica']

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

# === derived values and input processing =====================================
pFInpMLP_Clf = os.path.join(P_TEMP, S_F_INP_MLP_CLF + XT_CSV)
pFInpRF_Clf = os.path.join(P_TEMP, S_F_INP_RF_CLF + XT_CSV)
pFOutMLP_Clf = os.path.join(P_TEMP, S_F_OUT_MLP_CLF + XT_CSV)
pFOutRF_Clf = os.path.join(P_TEMP, S_F_OUT_RF_CLF + XT_CSV)

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'doMLP_Clf': doMLP_Clf,
        'doRF_Clf': doRF_Clf,
        'usedData': usedData,
        # --- boolean variables -----------------------------------------------
        'scaleX': scaleX,
        'scaleY': scaleY,
        # --- files, directories and paths ------------------------------------
        'pTemp': P_TEMP,
        'sFInpMLP_Clf': S_F_INP_MLP_CLF + XT_CSV,
        'sFInpRF_Clf': S_F_INP_RF_CLF + XT_CSV,
        'sFOutMLP_Clf': S_F_OUT_MLP_CLF + XT_CSV,
        'sFOutRF_Clf': S_F_OUT_RF_CLF + XT_CSV,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers
        'R08': R08,
        'rndState': rndState,
        'propTest': propTestData,
        'sizeHL': sizeHiddenLayers,
        'nSamples': nSamplesMC,
        'nFeatures': nFeaturesMC,
        'nInformative': nInformativeMC,
        'nRedundant': nRedundantMC,
        'shuffleIt': shuffleItMC,
        'maxDepthClfMC': maxDepthClfMC,
        'llFeaturesMC': llFeaturesMC,
        # --- sets
        # --- lists
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sCSV': S_CSV,
        'sSpecies': S_SPECIES,
        'sActivation': sActivation,
        'sPltSupTtl': sPltSupTtl,
        'sHdrCY_SglPosAAc_Red': sHdrCY_SglPosAAc_Red,
        'sHdrCY_TestCategorical': sHdrCY_TestCategorical,
        # --- numbers ---------------------------------------------------------
        # --- lists -----------------------------------------------------------
        'lPltLbls': lPltLbls,
        # --- dictionaries ----------------------------------------------------
        # === derived values and input processing =============================
        'pFInpMLP_Clf': pFInpMLP_Clf,
        'pFInpRF_Clf': pFInpRF_Clf,
        'pFOutMLP_Clf': pFOutMLP_Clf,
        'pFOutRF_Clf': pFOutRF_Clf}

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

def loadClfTestData(dI, iCol=None):
    pF, sHdCY = dI['pFInpMLP_Clf'], dI['sHdrCY_SglPosAAc_Red']
    if dI['usedData'] == 'SglPosAAc_Red':
        pF, sHdCY = dI['pFInpMLP_Clf'], dI['sHdrCY_SglPosAAc_Red']
    elif dI['usedData'] == 'TestCategorical':
        pF, sHdCY = dI['pFInpRF_Clf'], dI['sHdrCY_TestCategorical']
    dfrInp = readCSV(pF=pF, iCol=iCol)
    print('Input DataFrame:', S_NEWL, dfrInp, sep='')
    assert sHdCY in dfrInp.columns
    dfrX = dfrInp[[s for s in dfrInp.columns if s != sHdCY]]
    serY = dfrInp[sHdCY]
    return dfrX, serY

def getDataFromMakeClassification(dI):
    return make_classification(n_samples=dI['nSamples'],
                               n_features=dI['nFeatures'],
                               n_informative=dI['nInformative'],
                               n_redundant=dI['nRedundant'],
                               random_state=dI['rndState'],
                               shuffle=dI['shuffleIt'])

def getDataXy(dI):
    X, y = None, None
    if dI['usedData'] == 'IrisData':
        X, y = loadIrisData(dI=dI)
    elif dI['usedData'] in ['SglPosAAc_Red', 'TestCategorical']:
        X, y = loadClfTestData(dI=dI, iCol=0)
    elif dI['usedData'] == 'MakeClassifier':
        X, y = getDataFromMakeClassification(dI=dI)
    return X, y

# --- Functions handling dictionaries -----------------------------------------
def addToDictCt(cD, cK, cIncr=1):
    if cK in cD:
        cD[cK] += cIncr
    else:
        cD[cK] = cIncr

def addToDictL(cD, cK, cE, lUnqEl=False):
    if cK in cD:
        if not lUnqEl or cE not in cD[cK]:
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
class Fitted_RFClf:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dInp, X, y):
        self.idO = 'Pred_RF'
        self.descO = 'Predictor based on the Random Forest classifier'
        self.dI = dInp
        self.X, self.y = X, y
        self.Xtrans = None
        self.encodeCatFeatures()
        self.transCatToNumpy()
        self.RFClfFit()
        print('Initiated "Pred_RFClf" base object.')

    # --- method for encoding the categorical features ------------------------
    def encodeCatFeatures(self):
        self.cEnc = OneHotEncoder()
        self.cEnc.fit(self.X)
        print('Categories:', S_NEWL, self.cEnc.categories_)
        print('n_features_in:', S_NEWL, self.cEnc.n_features_in_)
        print('feature_names_in:', S_NEWL, self.cEnc.feature_names_in_)
        print('Feature names out:', S_NEWL, self.cEnc.get_feature_names_out(),
              sep='')

    # --- method for transforming categorical features to a numpy array -------
    def transCatToNumpy(self, llCat=None):
        if llCat is None:
            llCat = self.X
        self.Xtrans = self.cEnc.transform(llCat).toarray()
        print('Transformed array:', S_NEWL, self.Xtrans, S_NEWL, 'Shape:',
              self.Xtrans.shape, sep='')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def RFClfFit(self):
        self.Clf = RandomForestClassifier(max_depth=self.dI['maxDepthClfMC'],
                                          random_state=self.dI['rndState'])
        if self.Xtrans is None:
            self.Clf.fit(self.X, self.y)
        else:
            self.Clf.fit(self.Xtrans, self.y)

    def RFClfPred(self):
        ll = self.dI['llFeaturesMC']
        self.lPred = self.Clf.predict(self.cEnc.transform(ll).toarray())
        print('List of predictions:', S_NEWL, self.lPred, sep='')

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
X, y = getDataXy(dI=dInp)
print('X:', S_NEWL, X, S_NEWL, S_DS80, S_NEWL, 'y:', S_NEWL, y, sep='')
print('type(X):', type(X), '| type(y):', type(y))
print('shape of X:', X.shape, '| shape of y:', y.shape)
try:
    print('(min, max) of X:', X.min(), X.max())
except:
    print('No min. and max. of X, as values of X are not ordered.')

if dInp['doMLP_Clf']:
    TrTeData = TrainTestData(dInp=dInp, X=X, y=y)
    cMLPPred = Pred_MLPClf(dInp=dInp, TTD=TrTeData)
    cMLPPred.printFitQuality()

    Pltr = Plotter(dInp=dInp)
    Pltr.plotConfusionMatrix(prMLPC=cMLPPred)

if dInp['doRF_Clf']:
    cFittedRFClf = Fitted_RFClf(dInp=dInp, X=X, y=y)
    # cFittedRFClf.transCatToNumpy(llCat=[['A', 'C', 'D', 'A', 'D', 'A'],
    #                                     ['B', 'D', 'B', 'B', 'C', 'A']])
    cFittedRFClf.RFClfPred()

print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
