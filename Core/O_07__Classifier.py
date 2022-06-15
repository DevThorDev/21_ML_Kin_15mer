# -*- coding: utf-8 -*-
###############################################################################
# --- O_07__Classifier.py ----------------------------------------------------
###############################################################################
import matplotlib.pyplot as plt

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF

from Core.O_00__BaseClass import BaseClass

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
class BaseClfPrC(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar='A', iTp=7, lITpUpd=[1, 2]):
        super().__init__(inpDat)
        self.idO = 'O_07'
        self.descO = 'Classifier and AAc proportions per kinase (class) calc.'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.D = D
        self.iniAttr(sKPar=sKPar)
        self.fillDPF()
        print('Initiated "BaseClfPrC" base object.')

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self, sKPar='A'):
        lAttr2None = ['serNmerSeq', 'dfrInp', 'lSCl', 'X', 'y',
                      'XTrans', 'XTrain', 'XTest', 'XTransTrain', 'XTransTest',
                      'yPred', 'yTrain', 'yTest',
                      'Clf', 'sMth', 'sMthL', 'scoreClf', 'confusMatrix']
        lAttrDict = ['d2ResClf', 'dConfMat', 'dPropAAc']
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)
        for cAttr in lAttrDict:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, {})
        self.sKPar = sKPar

    # --- methods for filling the result paths dictionary ---------------------
    def fillDPF(self):
        sFOutPrC = self.dITp['sFOutPrC'] + self.dITp['xtCSV']
        sFConfMat = self.dITp['sFConfMat'] + self.dITp['xtCSV']
        self.dPF = self.D.yieldDPF()
        self.dPF['OutDataPrC'] = GF.joinToPath(self.dITp['pOutPrC'], sFOutPrC)
        self.dPF['ConfMat'] = GF.joinToPath(self.dITp['pConfMat'], sFConfMat)

    # --- print methods -------------------------------------------------------
    def printX(self):
        print(GC.S_DS20, GC.S_NEWL, 'Training input samples:', sep='')
        print(self.X)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', self.X.index.to_list())
                print('Columns:', self.X.columns.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printY(self):
        print(GC.S_DS20, GC.S_NEWL, 'Class labels:', sep='')
        print(self.y)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', self.y.index.to_list())
            except:
                pass
        print(GC.S_DS80)

    def printXY(self):
        self.printX()
        self.printY()

# -----------------------------------------------------------------------------
class Classifier(BaseClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar='A', iTp=7, lITpUpd=[1, 2]):
        super().__init__(inpDat, D=D, sKPar=sKPar, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Classifier for data classification'
        self.getInpData()
        if self.dITp['encodeCatFtr'] and not self.dITp['doTrainTestSplit']:
            self.XTrans = self.encodeCatFeatures()
        elif not self.dITp['encodeCatFtr'] and self.dITp['doTrainTestSplit']:
            self.getTrainTestDS(X=self.X, y=self.y)
        elif self.dITp['encodeCatFtr'] and self.dITp['doTrainTestSplit']:
            self.getTrainTestDS(X=self.encodeCatFeatures(), y=self.y)
        print('Initiated "Classifier" base object.')

    # --- methods for getting input data --------------------------------------
    def getInpData(self):
        (self.serNmerSeq, self.dfrInp, self.lSCl,
         self.X, self.y) = self.D.yieldData(sMd='Clf')
        if self.dITp['usedNmerSeq'] == self.dITp['sUnqList']:
            self.saveData(self.dfrInp, pF=self.dPF['DataClfU'],
                          saveAnyway=False)

    # --- print methods -------------------------------------------------------
    def printEncAttr(self, XTrans):
        nFIn, nmFIn = self.cEnc.n_features_in_, self.cEnc.feature_names_in_
        print('Categories:', GC.S_NEWL, self.cEnc.categories_, sep='')
        print('n_features_in:', GC.S_NEWL, nFIn, sep='')
        print('feature_names_in:', GC.S_NEWL, nmFIn, sep='')
        print('Feature names out:', GC.S_NEWL,
              self.cEnc.get_feature_names_out(), sep='')
        print('Transformed array:', GC.S_NEWL, XTrans, GC.S_NEWL,
              'Shape: ', XTrans.shape, sep='')

    def printDetailedPredict(self, X2Pr=None, nPr=None, nCr=None, cSect='C'):
        if self.dITp['lvlOut'] > 0 and cSect in ['A2']:
            s1, s2 = (str(nCr) + ' out of ' + str(nPr) + ' correctly' +
                      ' predicted.'), ''
            if nPr > 0:
                x = round(nCr/nPr, self.dITp['rndDigCorrect'])
                sXPc = str(round(x*100., self.dITp['rndDigCorrect']))
                s2 = ('(' + str(sXPc) + GC.S_PERC + ')')
            print((s1 if nPr == 0 else s1 + GC.S_SPACE + s2))
        if self.dITp['lvlOut'] > 1 and cSect in ['A1', 'B', 'C']:
            print(GC.S_NEWL, GC.S_DS04, GC.S_SPACE, 'Predictions:', sep='')
            if cSect == 'A1':
                for xT, yT, yP in zip(X2Pr, self.yTest, self.yPred):
                    print(xT[:10], GC.S_SPACE, GC.S_ARR_LR, GC.S_SPACE,
                          '(', yT, GC.S_COMMA, GC.S_SPACE, yP, ')', sep='')
            elif cSect == 'B':
                for xT, yP in zip(X2Pr, self.yPred):
                    print(xT, GC.S_ARR_LR, yP)
            elif cSect == 'C':
                print(self.yPred)

    def printPredict(self, X2Pre=None):
        nPred, nCorr, _ = tuple(self.d2ResClf[self.sKPar].values())
        if X2Pre is not None and len(X2Pre) == len(self.yPred):
            if self.yTest is not None:
                self.printDetailedPredict(X2Pre, cSect='A1')
                self.printDetailedPredict(X2Pre, nPred, nCorr, cSect='A2')
            else:
                self.printDetailedPredict(X2Pre, cSect='B')
        else:
            self.printDetailedPredict(cSect='C')

    def printFitQuality(self):
        print(GC.S_DS04, ' Fit quality for the "', self.sMthL, '" method ',
              GC.S_DS04, sep='')
        if self.scoreClf is not None:
            print('Classification score for the test data:',
                  round(self.scoreClf, self.dITp['rndDigScore']))
        if self.dITp['lvlOut'] > 0:
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

    # --- method for calculating values of the classifier results dictionary --
    def calcResPredict(self, X2Pre=None):
        if (X2Pre is not None and self.yTest is not None and
            self.yPred is not None and len(X2Pre) == len(self.yPred)):
            nPred, sKPar = self.yPred.shape[0], self.sKPar
            nCorr = sum([1 for k in range(nPred) if
                         self.yTest.iloc[k] == self.yPred[k]])
            propCorr = (nCorr/nPred if nPred > 0 else None)
            lVCalc = [nPred, nCorr, propCorr]
            for sK, cV in zip(self.dITp['lSResClf'], lVCalc):
                GF.addToDictD(self.d2ResClf, cKMain=sKPar, cKSub=sK, cVSub=cV)

    # --- method for predicting with a Classifier -----------------------------
    def ClfPred(self, dat2Pre=None):
        if self.Clf is not None:
            if dat2Pre is not None and self.dITp['encodeCatFtr']:
                dat2Pre = self.cEnc.transform(dat2Pre).toarray()
            if dat2Pre is None:
                dat2Pre, self.yTest = self.getXY(getTrain=False)
            self.yPred = self.Clf.predict(dat2Pre)
            self.calcResPredict(X2Pre=dat2Pre)
            if self.dITp['lvlOut'] > 0:
                self.printPredict(X2Pre=dat2Pre)
            self.calcConfMatrix()

    # --- methods for adapting keys of the result paths dictionary ------------
    def adaptPathConfMatrix(self):
        sFConfMat = self.dITp['sFConfMat'] + self.sMth + self.dITp['xtCSV']
        self.dPF['ConfMat'] = GF.joinToPath(self.dITp['pConfMat'], sFConfMat)

    # --- method for calculating the confusion matrix -------------------------
    def calcConfMatrix(self):
        if self.dITp['calcConfMatrix']:
            t, p, lC = self.yTest, self.yPred, self.lSCl
            self.confusMatrix = confusion_matrix(y_true=t, y_pred=p, labels=lC)
            dfrCM = GF.iniPdDfr(self.confusMatrix, lSNmC=lC, lSNmR=lC)
            if self.sMth is not None:
                self.adaptPathConfMatrix()
            self.dConfMat[self.sKPar] = dfrCM

    # --- method for plotting the confusion matrix ----------------------------
    def plotConfMatrix(self):
        if self.dITp['plotConfMatrix'] and self.confusMatrix is not None:
            CM, lCl = self.confusMatrix, self.lSCl
            D = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=lCl)
            D.plot()
            supTtl = self.dITp['sSupTtlPlt']
            if self.sMthL is not None:
                supTtl += ' (method: "' + self.sMthL + '")'
            D.figure_.suptitle(supTtl)
            plt.show()

# -----------------------------------------------------------------------------
class RndForestClf(Classifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, d2Par, sKPar='A'):
        super().__init__(inpDat, D=D, sKPar=sKPar)
        self.descO = 'Random Forest classifier'
        self.sMthL, self.sMth = self.dITp['sMthRF_L'], self.dITp['sMthRF']
        self.d2Par = d2Par
        if self.dITp['doRndForestClf']:
            self.getClf()
            self.ClfFit(self.Clf)
        print('Initiated "RndForestClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = RandomForestClassifier(random_state=self.dITp['rndState'],
                                          warm_start=self.dITp['bWarmStart'],
                                          oob_score=self.dITp['estOobScore'],
                                          n_jobs=self.dITp['nJobs'],
                                          verbose=self.dITp['vVerb'],
                                          **self.d2Par[self.sKPar])

# -----------------------------------------------------------------------------
class NNMLPClf(Classifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, d2Par, sKPar='A'):
        super().__init__(inpDat, D=D, sKPar=sKPar)
        self.descO = 'Neural Network MLP classifier'
        self.sMthL, self.sMth = self.dITp['sMthMLP_L'], self.dITp['sMthMLP']
        self.d2Par = d2Par
        if self.dITp['doNNMLPClf']:
            self.getClf()
            self.ClfFit(self.Clf)
            self.getScoreClf()
        print('Initiated "NNMLPClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = MLPClassifier(random_state=self.dITp['rndState'],
                                 warm_start=self.dITp['bWarmStart'],
                                 verbose=self.dITp['bVerb'],
                                 **self.d2Par[self.sKPar])

    def getScoreClf(self):
        if self.dITp['doTrainTestSplit']:
            self.scoreClf = self.Clf.score(self.XTest, self.yTest)

# -----------------------------------------------------------------------------
class PropCalculator(BaseClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, iTp=7, lITpUpd=[1, 2]):
        super().__init__(inpDat, D=D, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Calculator: AAc proportions per kinase (class)'
        self.getInpData()
        print('Initiated "PropCalculator" base object.')

    # --- methods for getting input data --------------------------------------
    def getInpData(self):
        (self.serNmerSeq, self.dfrInp, self.lSCl,
         self.X, self.y) = self.D.yieldData(sMd='PrC')

    # --- method for adapting a key of the result paths dictionary ------------
    def adaptPathOutDataPrC(self, sCl=None):
        sFOutPrC = self.dITp['sFOutPrC'] + self.dITp['xtCSV']
        if sCl is not None:
            sFOutPrC = (self.dITp['sUSC'].join([self.dITp['sFOutPrC'], sCl]) +
                        self.dITp['xtCSV'])
        self.dPF['OutDataPrC'] = GF.joinToPath(self.dITp['pOutPrC'], sFOutPrC)

    # --- methods calculating the proportions of AAc at all Nmer positions ----
    def calcPropAAc(self):
        if self.dITp['doPropCalc']:
            for sCl in self.lSCl:
                cDfrI = self.dfrInp[self.dfrInp[self.dITp['sCY']] == sCl]
                cD2Out, nTtl = {}, cDfrI.shape[0]
                for sPos in self.dITp['lSCX']:
                    serCPos = cDfrI[sPos]
                    for sAAc in self.dITp['lFeatSrt']:
                        nCAAc = serCPos[serCPos == sAAc].count()
                        GF.addToDictD(cD2Out, cKMain=sPos, cKSub=sAAc,
                                      cVSub=(0. if nTtl == 0 else nCAAc/nTtl))
                self.dPropAAc[sCl] = GF.iniPdDfr(cD2Out)
                self.adaptPathOutDataPrC(sCl=sCl)
                self.saveData(self.dPropAAc[sCl], pF=self.dPF['OutDataPrC'])

###############################################################################
