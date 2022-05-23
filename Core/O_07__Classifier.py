# -*- coding: utf-8 -*-
###############################################################################
# --- O_07__Classifier.py ----------------------------------------------------
###############################################################################
import time

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Core.O_02__SeqAnalysis import SeqAnalysis

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
class Classifier(SeqAnalysis):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=7, lITpUpd=[1, 2]):
        super().__init__(inpDat)
        self.idO = 'O_07'
        self.descO = 'Classifier for data classification'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.iniAttr()
        self.addValsToDPF()
        self.loadInpData()
        if self.dITp['encodeCatFtr'] and not self.dITp['doTrainTestSplit']:
            self.XTrans = self.encodeCatFeatures()
        elif not self.dITp['encodeCatFtr'] and self.dITp['doTrainTestSplit']:
            self.getTrainTestDS(X=self.X, y=self.y)
        elif self.dITp['encodeCatFtr'] and self.dITp['doTrainTestSplit']:
            self.getTrainTestDS(X=self.encodeCatFeatures(), y=self.y)
        print('Initiated "Classifier" base object.')

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self):
        self.X, self.XTrans = None, None
        self.XTrain, self.XTest = None, None
        self.XTransTrain, self.XTransTest = None, None
        self.y, self.yPred = None, None
        self.yTrain, self.yTest = None, None
        self.Clf = None
        self.scoreClf, self.confusMatrix = None, None

    def loadInpData(self):
        self.dfrInp = self.loadData(pF=self.dPF['InpData'], iC=0)
        self.serNmerSeq = self.dfrInp[self.dITp['sCNmer']]
        if self.dITp['usedNmerSeq'] == self.dITp['sUnqList']:
            self.serNmerSeq = GF.iniPdSer(self.serNmerSeq.unique(),
                                          nameS=self.dITp['sCNmer'])
            lSer = []
            for cSeq in self.serNmerSeq:
                cDfr = self.dfrInp[self.dfrInp[self.dITp['sCNmer']] == cSeq]
                lSer.append(cDfr.iloc[0, :])
            self.dfrInp = GF.concLSer(lSer=lSer, ignIdx=True)
        self.X = self.dfrInp[self.dITp['lSCX']]
        self.y = self.dfrInp[self.dITp['sCY']]

    # --- print methods -------------------------------------------------------
    def printX(self):
        print(GC.S_DS20, GC.S_NEWL, 'Training input samples:', sep='')
        print(self.X)
        try:
            print('Index:', self.X.index.to_list())
            print('Columns:', self.X.columns.to_list())
        except:
            pass
        print(GC.S_DS80)

    def printY(self):
        print(GC.S_DS20, GC.S_NEWL, 'Class labels:', sep='')
        print(self.y)
        try:
            print('Index:', self.y.index.to_list())
        except:
            pass
        print(GC.S_DS80)

    def printXY(self):
        self.printX()
        self.printY()

    def printEncAttr(self, XTrans):
        nFIn, nmFIn = self.cEnc.n_features_in_, self.cEnc.feature_names_in_
        print('Categories:', GC.S_NEWL, self.cEnc.categories_, sep='')
        print('n_features_in:', GC.S_NEWL, nFIn, sep='')
        print('feature_names_in:', GC.S_NEWL, nmFIn, sep='')
        print('Feature names out:', GC.S_NEWL,
              self.cEnc.get_feature_names_out(), sep='')
        print('Transformed array:', GC.S_NEWL, XTrans, GC.S_NEWL,
              'Shape: ', XTrans.shape, sep='')

    def printPredictions(self, X2Pre=None):
        print(GC.S_NEWL, GC.S_DS04, GC.S_SPACE, 'Predictions:', sep='')
        if X2Pre is not None and len(X2Pre) == len(self.yPred):
            if self.yTest is not None:
                if self.dITp['lvlOut'] > 1:
                    for xT, yT, yP in zip(X2Pre, self.yTest, self.yPred):
                        print(xT[:10], GC.S_SPACE, GC.S_ARR_LR, GC.S_SPACE,
                              '(', yT, GC.S_COMMA, GC.S_SPACE, yP, ')', sep='')
                nPred = self.yPred.shape[0]
                nCorrect = sum([1 for k in range(nPred) if
                                self.yTest.iloc[k] == self.yPred[k]])
                s1, s2 = (str(nCorrect) + ' out of ' + str(nPred) +
                          ' correctly predicted.'), ''
                if nPred > 0:
                    x = round(nCorrect/nPred, self.dITp['rndDigCorrect'])
                    sXPc = str(round(x*100., self.dITp['rndDigCorrect']))
                    s2 = ('(' + str(sXPc) + GC.S_PERC + ')')
                print((s1 if nPred == 0 else s1 + GC.S_SPACE + s2))
            else:
                if self.dITp['lvlOut'] > 1:
                    for xT, yP in zip(X2Pre, self.yPred):
                        print(xT, GC.S_ARR_LR, yP)
        else:
            if self.dITp['lvlOut'] > 1:
                print(self.yPred)

    def printFitQuality(self):
        if self.dITp['lvlOut'] > 0:
            # print('True/Test y-values:', self.yTest.to_numpy())
            # print('Predicted y-values:', self.yPred)
            if self.scoreClf is not None:
                print('Classification score for the test data:',
                      round(self.scoreClf, self.dITp['rndDigScore']))
            if self.confusMatrix is not None:
                print('Confusion matrix:', GC.S_NEWL, self.confusMatrix,
                      sep='')

    # --- methods for getting and setting X and y -----------------------------
    def getXY(self, getTrain=None):
        X, y = self.X, self.y
        if self.dITp['encodeCatFtr']:
            X = self.XTrans
        if getTrain is not None and self.dITp['doTrainTestSplit']:
            if getTrain:
                X, y = self.XTrain, self.yTrain
                if self.dITp['encodeCatFtr']:
                    X = self.XTransTrain
            else:
                X, y = self.XTest, self.yTest
                if self.dITp['encodeCatFtr']:
                    X = self.XTransTest
        return X, y

    def setXY(self, X, y, setTrain=None):
        self.X, self.y = X, y
        if self.dITp['encodeCatFtr']:
            self.XTrans = X
        if setTrain is not None and self.dITp['doTrainTestSplit']:
            if setTrain:
                self.XTrain, self.yTrain = X, y
                if self.dITp['encodeCatFtr']:
                    self.XTransTrain = X
            else:
                self.XTest, self.yTest = X, y
                if self.dITp['encodeCatFtr']:
                    self.XTransTest = X

    # --- methods for filling the result paths dictionary ---------------------
    def addValsToDPF(self):
        sFInp = self.dITp['sFInp'] + self.dITp['xtCSV']
        self.dPF['InpData'] = GF.joinToPath(self.dITp['pInp'], sFInp)
        # sFInp = self.dITp['sUSC'].join([self.dITp['sFInp'],
        #                                 self.dITp['usedNmerSeq']]) + sXt

    # --- method for encoding and transforming the categorical features -------
    def encodeCatFeatures(self, catData=None):
        if catData is None:
            catData = self.X
        self.cEnc = OneHotEncoder()
        self.cEnc.fit(self.X)
        XTrans = self.cEnc.transform(catData).toarray()
        if self.dITp['lvlOut'] > 1:
            self.printEncAttr(XTrans=XTrans)
        return XTrans

    # --- method for splitting data into training and test data ---------------
    def getTrainTestDS(self, X, y):
        tTrTe = train_test_split(X, y, random_state=self.dITp['rndState'],
                                 test_size=self.dITp['propTestData'])
        XTrain, XTest, yTrain, yTest = tTrTe
        self.setXY(X=XTrain, y=yTrain, setTrain=True)
        self.setXY(X=XTest, y=yTest, setTrain=False)

    # --- method for fitting a Classifier -------------------------------------
    def ClfFit(self, cClf):
        bTrain = (True if self.dITp['doTrainTestSplit'] else None)
        X, y = self.getXY(getTrain=bTrain)
        if cClf is not None and X is not None and y is not None:
            try:
                cClf.fit(X, y)
            except:
                print('ERROR: Cannot fit classifier to data!')
                if self.dITp['lvlOut'] > 1:
                    self.printXY()

    # --- method for predicting with a Classifier -----------------------------
    def ClfPred(self, dat2Pre=None):
        if self.Clf is not None:
            if dat2Pre is not None and self.dITp['encodeCatFtr']:
                dat2Pre = self.cEnc.transform(dat2Pre).toarray()
            if dat2Pre is None:
                dat2Pre, self.yTest = self.getXY(getTrain=False)
            self.yPred = self.Clf.predict(dat2Pre)
            if self.dITp['lvlOut'] > 0:
                self.printPredictions(X2Pre=dat2Pre)

    # --- method for calculating the confusion matrix -------------------------
    def calcConfusionMatrix(self):
        pass
        # if self.dITp['calcConfMatrix']:
        #     self.confusMatrix = confusion_matrix(y_true=self.TTD.y_Test,
        #                                           y_pred=self.y_Pred)

    # --- methods for saving data ---------------------------------------------
    def saveResData(self):
        pass

# -----------------------------------------------------------------------------
class RndForestClf(Classifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat):
        super().__init__(inpDat)
        self.descO = 'Random Forest classifier'
        if self.dITp['doRndForestClf']:
            self.getClf()
            self.ClfFit(self.Clf)
        print('Initiated "RndForestClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = RandomForestClassifier(n_estimators=self.dITp['nEstim'],
                                          max_depth=self.dITp['maxDepthRFClf'],
                                          random_state=self.dITp['rndState'])

# -----------------------------------------------------------------------------
class NNMLPClf(Classifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat):
        super().__init__(inpDat)
        self.descO = 'Neural Network MLP classifier'
        if self.dITp['doNNMLPClf']:
            self.getClf()
            self.ClfFit(self.Clf)
            self.getScoreClf()
        print('Initiated "NNMLPClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = MLPClassifier(hidden_layer_sizes=self.dITp['tHiddenLayers'],
                                 activation=self.dITp['sActivation'],
                                 random_state=self.dITp['rndState'])

    def getScoreClf(self):
        if self.dITp['doTrainTestSplit']:
            self.scoreClf = self.Clf.score(self.XTest, self.yTest)

###############################################################################
