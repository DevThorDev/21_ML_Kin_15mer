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
        self.getDPF()
        print('Initiated "BaseClfPrC" base object.')

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self, sKPar='A'):
        lAttr2None = ['serNmerSeq', 'dfrInp', 'lSCl', 'X', 'Y',
                      'XTrans', 'XTrain', 'XTest', 'XTransTrain', 'XTransTest',
                      'YPred', 'YTrain', 'YTest',
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
    def getDPF(self):
        sBClf, sBPrC = self.D.dITp['sFInpBaseClf'], self.D.dITp['sFInpBasePrC']
        self.dITp['sOutClf'] = GF.joinS([self.D.dITp['sSetClf'], sBClf],
                                        sJoin=self.dITp['sUSC'])
        self.dITp['sOutPrC'] = GF.joinS([self.D.dITp['sSetPrC'], sBPrC],
                                        sJoin=self.dITp['sUSC'])
        self.dPF = self.D.yieldDPF()

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
        print(self.Y)
        if self.dITp['lvlOut'] > 2:
            try:
                print('Index:', self.Y.index.to_list())
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
            self.getTrainTestDS(X=self.X, Y=self.Y)
        elif self.dITp['encodeCatFtr'] and self.dITp['doTrainTestSplit']:
            self.getTrainTestDS(X=self.encodeCatFeatures(), Y=self.Y)
        print('Initiated "Classifier" base object.')

    # --- methods for getting input data --------------------------------------
    def getInpData(self):
        (self.dfrInp, self.X, self.Y, self.serNmerSeq,
         self.lSCl) = self.D.yieldData(sMd='Clf')
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
                for xT, yT, yP in zip(X2Pr, self.YTest, self.YPred):
                    print(xT[:10], GC.S_SPACE, GC.S_ARR_LR, GC.S_SPACE,
                          '(', yT, GC.S_COMMA, GC.S_SPACE, yP, ')', sep='')
            elif cSect == 'B':
                for xT, yP in zip(X2Pr, self.YPred):
                    print(xT, GC.S_ARR_LR, yP)
            elif cSect == 'C':
                print(self.YPred)

    def printPredict(self, X2Pred=None):
        nPred, nOK, _ = tuple(self.d2ResClf[self.sKPar].values())
        if X2Pred is not None and X2Pred.shape[0] == self.YPred.shape[0]:
            if self.YTest is not None:
                self.printDetailedPredict(X2Pred, cSect='A1')
                self.printDetailedPredict(X2Pred, nPred, nOK, cSect='A2')
            else:
                self.printDetailedPredict(X2Pred, cSect='B')
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

    # --- methods for getting and setting X and Y -----------------------------
    def getXY(self, getTrain=None):
        X, Y = self.X, self.Y
        if self.dITp['encodeCatFtr']:
            X = self.XTrans
        if getTrain is not None and self.dITp['doTrainTestSplit']:
            if getTrain:
                X, Y = self.XTrain, self.YTrain
                if self.dITp['encodeCatFtr']:
                    X = self.XTransTrain
            else:
                X, Y = self.XTest, self.YTest
                if self.dITp['encodeCatFtr']:
                    X = self.XTransTest
        return X, Y

    def setXY(self, X, Y, setTrain=None):
        if setTrain is not None and self.dITp['doTrainTestSplit']:
            if setTrain:
                self.YTrain = Y
                if self.dITp['encodeCatFtr']:
                    self.XTransTrain = X
                else:
                    self.XTrain = X
            else:
                self.YTest = Y
                if self.dITp['encodeCatFtr']:
                    self.XTransTest = X
                else:
                    self.XTest = X
        else:
            self.Y = Y
            if self.dITp['encodeCatFtr']:
                self.XTrans = X
            else:
                self.X = X

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
    def getTrainTestDS(self, X, Y):
        tTrTe = train_test_split(X, Y, random_state=self.dITp['rndState'],
                                 test_size=self.dITp['propTestData'])
        XTrain, XTest, YTrain, YTest = tTrTe
        self.setXY(X=XTrain, Y=YTrain, setTrain=True)
        self.setXY(X=XTest, Y=YTest, setTrain=False)

    # --- method for fitting a Classifier -------------------------------------
    def ClfFit(self, cClf):
        bTrain = (True if self.dITp['doTrainTestSplit'] else None)
        X, Y = self.getXY(getTrain=bTrain)
        if cClf is not None and X is not None and Y is not None:
            try:
                cClf.fit(X, Y)
            except:
                print('ERROR: Cannot fit classifier to data!')
                if self.dITp['lvlOut'] > 1:
                    self.printXY()

    # --- method for calculating values of the classifier results dictionary --
    def calcResPredict(self, X2Pred=None):
        if (X2Pred is not None and self.YTest is not None and
            self.YPred is not None and X2Pred.shape[0] == self.YPred.shape[0]):
            nPred, sKPar = self.YPred.shape[0], self.sKPar
            nOK = sum([1 for k in range(nPred) if
                       (self.YTest.iloc[k, :] == self.YPred.iloc[k, :]).all()])
            propOK = (nOK/nPred if nPred > 0 else None)
            lVCalc = [nPred, nOK, propOK]
            for sK, cV in zip(self.dITp['lSResClf'], lVCalc):
                GF.addToDictD(self.d2ResClf, cKMain=sKPar, cKSub=sK, cVSub=cV)
            # create dfrPred, containing the YTest and YPred columns
            sTCl, sPCl = self.dITp['sTrueCl'], self.dITp['sPredCl']
            lSCTP = [GF.joinS([s, sTCl], sJoin=self.dITp['sUSC'])
                     for s in self.YTest.columns]
            lSCTP += [GF.joinS([s, sPCl], sJoin=self.dITp['sUSC'])
                      for s in self.YPred.columns]
            dfrTP = GF.concLObjAx1([self.YTest, self.YPred], ignIdx=True)
            dfrTP.columns = lSCTP
            self.dfrPred = GF.concLObjAx1(lObj=[self.dfrInp['c15mer'], dfrTP])
            self.dfrPred.dropna(axis=0, inplace=True)

    # --- method for predicting with a Classifier -----------------------------
    def ClfPred(self, dat2Pred=None):
        if self.Clf is not None:
            if dat2Pred is not None and self.dITp['encodeCatFtr']:
                dat2Pred = self.cEnc.transform(dat2Pred).toarray()
            if dat2Pred is None:
                dat2Pred, self.YTest = self.getXY(getTrain=False)
            self.YPred = GF.iniPdDfr(self.Clf.predict(dat2Pred),
                                     lSNmC=self.YTest.columns,
                                     lSNmR=self.YTest.index)
            self.calcResPredict(X2Pred=dat2Pred)
            if self.dITp['lvlOut'] > 0:
                self.printPredict(X2Pred=dat2Pred)
            self.calcConfMatrix()

    # --- method for calculating the confusion matrix -------------------------
    def calcConfMatrix(self):
        if (self.dITp['calcConfMatrix'] and self.YTest.shape[1] <= 1 and
            self.YPred.shape[1] <= 1):
            t, p, lC = self.YTest, self.YPred, self.lSCl
            self.confusMatrix = confusion_matrix(y_true=t, y_pred=p, labels=lC)
            dfrCM = GF.iniPdDfr(self.confusMatrix, lSNmC=lC, lSNmR=lC)
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
            XTest, YTest = self.getXY(getTrain=False)
            self.scoreClf = self.Clf.score(XTest, YTest)

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
        (self.dfrInp, self.X, self.Y, self.serNmerSeq,
         self.lSCl) = self.D.yieldData(sMd='PrC')

    # --- method for adapting a key of the result paths dictionary ------------
    def setPathOutDataPrC(self, sCl=None):
        sFOutBase = GF.joinS([self.dITp['sProp'], self.dITp['sOutPrC']],
                             sJoin=self.dITp['sUS02'])
        sFOutPrC = sFOutBase + self.dITp['xtCSV']
        if sCl is not None:
            sFOutPrC = (GF.joinS([sFOutBase, sCl], sJoin=self.dITp['sUSC']) +
                        self.dITp['xtCSV'])
        self.dPF['OutDataPrC'] = GF.joinToPath(self.dITp['pOutPrC'], sFOutPrC)

    # --- methods calculating the proportions of AAc at all Nmer positions ----
    def calcPropAAc(self):
        if self.dITp['doPropCalc']:
            for sCl in self.lSCl:
                cDfrI = self.dfrInp[self.dfrInp[sCl] > 0]
                cD2Out, nTtl = {}, cDfrI.shape[0]
                for sPos in self.D.dITp['lSCXPrC']:
                    serCPos = cDfrI[sPos]
                    for sAAc in self.dITp['lFeatSrt']:
                        nCAAc = serCPos[serCPos == sAAc].count()
                        GF.addToDictD(cD2Out, cKMain=sPos, cKSub=sAAc,
                                      cVSub=(0. if nTtl == 0 else nCAAc/nTtl))
                self.dPropAAc[sCl] = GF.iniPdDfr(cD2Out)
                self.setPathOutDataPrC(sCl=sCl)
                self.saveData(self.dPropAAc[sCl], pF=self.dPF['OutDataPrC'])

###############################################################################
