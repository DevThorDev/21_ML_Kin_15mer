# -*- coding: utf-8 -*-
###############################################################################
# --- O_07__Classifier.py ----------------------------------------------------
###############################################################################
import matplotlib.pyplot as plt

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from imblearn.under_sampling import (ClusterCentroids, AllKNN,
                                     NeighbourhoodCleaningRule,
                                     RandomUnderSampler, TomekLinks)

# -----------------------------------------------------------------------------
class BaseSmplClfPrC(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar='A', iTp=7, lITpUpd=[1, 2, 6]):
        super().__init__(inpDat)
        self.idO = 'O_07'
        self.descO = 'Classifier and AAc proportions per kinase (class) calc.'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.D = D
        self.complDITp()
        self.iniAttr(sKPar=sKPar)
        self.getDPF()
        print('Initiated "BaseSmplClfPrC" base object.')

    # --- method for complementing the type input dictionary ------------------
    def complDITp(self):
        lLblTrain, sCLblsTrain = self.dITp['lLblTrain'], ''
        if not (lLblTrain is None or self.dITp['onlySglLbl']):
            sCLblsTrain = GC.S_USC.join([str(nLbl) for nLbl in lLblTrain])
            sCLblsTrain = GC.S_USC.join([GC.S_LBL, sCLblsTrain, GC.S_TRAIN])
        self.dITp['sCLblsTrain'] = sCLblsTrain

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self, sKPar='A'):
        lAttr2None = ['serNmerSeq', 'dfrInp', 'lSCl',
                      'X', 'XTrans', 'XTrain', 'XTest',
                      'XTransTrain', 'XTransTest',
                      'Y', 'YTrain', 'YTest', 'YPred', 'YProba',
                      'Clf', 'sMth', 'sMthL', 'scoreClf', 'confusMatrix',
                      'dfrPred', 'dfrProba']
        lAttrDict = ['d2ResClf', 'dConfMat', 'dPropAAc']
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)
        for cAttr in lAttrDict:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, {})
        self.sKPar = sKPar

    # --- methods for getting input data --------------------------------------
    def getInpData(self, sMd=None):
        (self.dfrInp, self.X, self.Y, self.serNmerSeq,
         self.lSCl) = self.D.yieldData(sMd=sMd)

    # --- methods for filling the result paths dictionary ---------------------
    def getDPF(self):
        dITp = self.dITp
        sBClf, sBPrC = dITp['sFInpBaseClf'], dITp['sFInpBasePrC']
        dITp['sParClf'] = GF.joinS([sBClf], cJ=dITp['sUSC'])
        dITp['sOutClf'] = GF.joinS([dITp['sSet'], sBClf], cJ=dITp['sUSC'])
        dITp['sOutPrC'] = GF.joinS([dITp['sSet'], sBPrC], cJ=dITp['sUSC'])
        self.dPF = self.D.yieldDPF()

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

    # --- methods for setting data --------------------------------------------
    def setData(self, tData):
        lSAttr = ['dfrInp', 'serNmerSeq', 'lSCl', 'X', 'XTrans', 'XTrain',
                  'XTest', 'XTransTrain', 'XTransTest', 'Y', 'YTrain', 'YTest',
                  'YPred', 'YProba']
        assert len(tData) == len(lSAttr)
        for sAttr, cV in zip(lSAttr, tData):
            if cV is not None:
                setattr(self, sAttr, cV)

    # --- methods for yielding data -------------------------------------------
    def yieldData(self):
        return (self.dfrInp, self.serNmerSeq, self.lSCl,
                self.X, self.XTrans, self.XTrain, self.XTest,
                self.XTransTrain, self.XTransTest,
                self.Y, self.YTrain, self.YTest, self.YPred, self.YProba)

    # --- method for encoding and transforming the categorical features -------
    def encodeCatFeatures(self, catData=None):
        if catData is None:
            catData = self.X
        self.cEnc = OneHotEncoder()
        self.cEnc.fit(self.X)
        XTrans = self.cEnc.transform(catData).toarray()
        if self.dITp['lvlOut'] > 1:
            self.printEncAttr(XTrans=XTrans)
        return GF.iniPdDfr(XTrans, lSNmR=self.Y.index)

    # --- method for splitting data into training and test data ---------------
    def getTrainTestDS(self, X, Y):
        tTrTe = train_test_split(X, Y, random_state=self.dITp['rndState'],
                                 test_size=self.dITp['propTestData'])
        XTrain, XTest, YTrain, YTest = tTrTe
        if not (self.dITp['onlySglLbl'] or self.dITp['lLblTrain'] is None):
            lB = [(serR.sum() in self.dITp['lLblTrain']) for _, serR in
                  YTrain.iterrows()]
            XTrain, YTrain = XTrain[lB], YTrain[lB]
        self.setXY(X=XTrain, Y=YTrain, setTrain=True)
        self.setXY(X=XTest, Y=YTest, setTrain=False)

    # --- method for splitting data into training and test data ---------------
    def encodeSplitData(self):
        if self.dITp['encodeCatFtr'] and not self.dITp['doTrainTestSplit']:
            self.XTrans = self.encodeCatFeatures()
        elif not self.dITp['encodeCatFtr'] and self.dITp['doTrainTestSplit']:
            self.getTrainTestDS(X=self.X, Y=self.Y)
        elif self.dITp['encodeCatFtr'] and self.dITp['doTrainTestSplit']:
            self.getTrainTestDS(X=self.encodeCatFeatures(), Y=self.Y)

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

    def printEncAttr(self, XTrans):
        nFIn, nmFIn = self.cEnc.n_features_in_, self.cEnc.feature_names_in_
        print('Categories:', GC.S_NEWL, self.cEnc.categories_, sep='')
        print('n_features_in:', GC.S_NEWL, nFIn, sep='')
        print('feature_names_in:', GC.S_NEWL, nmFIn, sep='')
        print('Feature names out:', GC.S_NEWL,
              self.cEnc.get_feature_names_out(), sep='')
        print('Transformed array:', GC.S_NEWL, XTrans, GC.S_NEWL,
              'Shape: ', XTrans.shape, sep='')

# -----------------------------------------------------------------------------
class ImbSampler(BaseSmplClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar='A', iTp=7, lITpUpd=[1, 2, 6]):
        super().__init__(inpDat, D=D, sKPar=sKPar, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Sampler for imbalanced learning'
        self.getInpData(sMd=GC.S_CLF)
        self.encodeSplitData()
        self.getSampler()
        print('Initiated "ImbSampler" base object.')

    # --- Function obtaining the imbalanced sampler ("imblearn") --------------
    def getSampler(self):
        self.imbSmp, dITp = None, self.dITp
        if dITp['sSampler'] == 'ClusterCentroids':
            self.imbSmp = ClusterCentroids(sampling_strategy=dITp['sStrat'],
                                           random_state=dITp['rndState'],
                                           estimator=dITp['estimator'],
                                           voting=dITp['voting'])
        elif dITp['sSampler'] == 'AllKNN':
            self.imbSmp = AllKNN(sampling_strategy=dITp['sStrat'],
                                 n_neighbors=dITp['n_neighbors_AllKNN'],
                                 kind_sel=dITp['kind_sel_AllKNN'],
                                 allow_minority=dITp['allow_minority'])
        elif dITp['sSampler'] == 'NeighbourhoodCleaningRule':
            sStrat, nNbr = dITp['sStrat'], dITp['n_neighbors_NCR']
            kindSel, thrCln = dITp['kind_sel_NCR'], dITp['threshold_cleaning']
            self.imbSmp = NeighbourhoodCleaningRule(sampling_strategy=sStrat,
                                                    n_neighbors=nNbr,
                                                    kind_sel=kindSel,
                                                    threshold_cleaning=thrCln)
        elif dITp['sSampler'] == 'RandomUnderSampler':
            self.imbSmp = RandomUnderSampler(sampling_strategy=dITp['sStrat'],
                                             random_state=dITp['rndState'],
                                             replacement=dITp['wReplacement'])
        elif dITp['sSampler'] == 'TomekLinks':
            self.imbSmp = TomekLinks(sampling_strategy=dITp['sStrat'])

    # --- Function performing the random sampling ("imblearn") ----------------
    def fitResampleImbalanced(self):
        print('Resampling data using resampler "', self.dITp['sSampler'],
              '" with sampling strategy "', self.dITp['sStrat'], '".', sep='')
        bTrain = (True if self.dITp['doTrainTestSplit'] else None)
        X, Y = self.getXY(getTrain=bTrain)
        # if not self.dITp['onlySglLbl']:
        #     Y = SF.toSglLbl(self.dITp, dfrY=Y)
        print('Initial shape of Y:', Y.shape)
        X, YRes = self.imbSmp.fit_resample(X, Y)
        print('Final shape of YRes after resampling:', YRes.shape)
        for cY in YRes.unique():
            print(cY, self.dITp['sTab'], YRes[YRes == cY].size, sep='')
        if not self.dITp['onlySglLbl']:
            YRes = SF.toMultiLbl(self.dITp, serY=YRes, lXCl=self.lSCl)
        self.setXY(X=X, Y=YRes, setTrain=bTrain)

# -----------------------------------------------------------------------------
class Classifier(BaseSmplClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar='A', iTp=7, lITpUpd=[1, 2, 6]):
        super().__init__(inpDat, D=D, sKPar=sKPar, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Classifier for data classification'
        self.Smp = ImbSampler(inpDat, D, sKPar=sKPar, iTp=iTp, lITpUpd=lITpUpd)
        self.setData(self.Smp.yieldData())
        if self.dITp['doImbSampling'] and (self.sMth == self.dITp['sMthRF'] or
                                           self.dITp['nItPartialFit'] is None):
            self.Smp.fitResampleImbalanced()
            self.setData(self.Smp.yieldData())
        print('Initiated "Classifier" base object.')

    # --- print methods -------------------------------------------------------
    def printClfFitRes(self, X):
        if self.dITp['lvlOut'] > 0:
            print('Fitted classifier to data of shape', X.shape)
            if self.optClf is not None:
                nL, oC, R04 = GC.S_NEWL, self.optClf, self.dIG['R04']
                print(GC.S_DS80, 'Grid search results:', nL,
                      'Best estimator:', nL, oC.best_estimator_, nL,
                      'Best parameters:', nL, oC.best_params_, nL,
                      'Best score: ', round(oC.best_score_, R04), sep='')
                dfrCVRes = GF.iniPdDfr(oC.cv_results_)
                print('CV results:', nL, dfrCVRes, sep='')

    def printClfFitError(self):
        print('ERROR: Cannot fit classifier to data!')
        if self.dITp['lvlOut'] > 1:
            self.printXY()

    def printClfPartialFit(self, X):
        if self.dITp['lvlOut'] > 0:
            print('Partially fitted classifier to data of shape', X.shape)

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

    # --- method for obtaining an optimised Classifier from a grid search -----
    def getOptClfGS(self):
        self.optClf, retClf = None, self.Clf
        if self.lParG is not None:
            optClf = GridSearchCV(estimator=self.Clf, param_grid=self.lParG,
                                  scoring='accuracy')
            self.optClf, retClf = optClf, optClf
        return retClf

    # --- method for fitting a Classifier -------------------------------------
    def ClfFit(self, cClf):
        bTrain = (True if self.dITp['doTrainTestSplit'] else None)
        X, Y = self.getXY(getTrain=bTrain)
        if cClf is not None and X is not None and Y is not None:
            try:
                cClf.fit(X, Y)
                self.printClfFitRes(X)
            except:
                self.printClfFitError()

    def ClfPartialFit(self, cClf, XInp, YInp):
        X, Y = XInp, YInp
        if cClf is not None and XInp is not None and YInp is not None:
            X, Y = self.Smp.imbSmp.fit_resample(XInp, YInp)
            cClf.partial_fit(X, Y, classes=self.lSCl)
            self.printClfPartialFit(X)
        return X, Y

    def fitOrPartialFitClf(self, cClf):
        if self.dITp['nItPartialFit'] is not None:
            assert type(self.dITp['nItPartialFit']) in [int, float]
            bTrain = (True if self.dITp['doTrainTestSplit'] else None)
            XIni, YIni = self.getXY(getTrain=bTrain)
            # repeat resampling and partial fit nItPartialFit times
            for _ in range(round(abs(self.dITp['nItPartialFit']))):
                X, Y = self.ClfPartialFit(cClf, XInp=XIni, YInp=YIni)
            self.setXY(X=X, Y=Y, setTrain=bTrain)
        else:
            self.ClfFit(cClf)

    # --- method for calculating values of the classifier results dictionary --
    def assembleDfrPredProba(self, lSCTP):
        lDfr = [self.dfrPred, self.dfrProba]
        for k, cYP in enumerate([self.YPred, self.YProba]):
            lDfr[k] = GF.concLObjAx1([self.YTest, cYP], ignIdx=True)
            lDfr[k].columns = lSCTP
            lDfr[k] = GF.concLObjAx1(lObj=[self.dfrInp['c15mer'], lDfr[k]])
            lDfr[k].dropna(axis=0, inplace=True)
            lDfr[k] = lDfr[k].convert_dtypes()
        [self.dfrPred, self.dfrProba] = lDfr

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
            # create dfrPred/dfrProba, containing YTest and YPred/YProba
            sTCl, sPCl = self.dITp['sTrueCl'], self.dITp['sPredCl']
            lSCTP = [GF.joinS([s, sTCl], cJ=self.dITp['sUSC'])
                     for s in self.YTest.columns]
            lSCTP += [GF.joinS([s, sPCl], cJ=self.dITp['sUSC'])
                      for s in self.YPred.columns]
            self.assembleDfrPredProba(lSCTP=lSCTP)

    # --- method for predicting with a Classifier -----------------------------
    def ClfPred(self, dat2Pred=None):
        cClf = (self.Clf if self.optClf is None else self.optClf)
        if cClf is not None:
            dITp, lSXCl = self.dITp, self.lSCl
            if dat2Pred is not None and dITp['encodeCatFtr']:
                dat2Pred = self.cEnc.transform(dat2Pred).toarray()
            if dat2Pred is None:
                dat2Pred, self.YTest = self.getXY(getTrain=False)
            if self.dITp['onlySglLbl']:
                self.YTest = SF.toMultiLbl(dITp, serY=self.YTest, lXCl=lSXCl)
            lSC, lSR = self.YTest.columns, self.YTest.index
            if self.dITp['onlySglLbl']:
                YPred = GF.iniPdSer(cClf.predict(dat2Pred), lSNmI=lSR,
                                    nameS=dITp['sEffFam'])
                self.YPred = SF.toMultiLbl(dITp, serY=YPred, lXCl=lSXCl)
            else:
                self.YPred = GF.iniPdDfr(cClf.predict(dat2Pred), lSNmC=lSC,
                                         lSNmR=lSR)
            self.YProba = GF.getYProba(cClf, dat2Pred, lSC=lSC, lSR=lSR)
            assert self.YProba.shape == self.YPred.shape
            self.calcResPredict(X2Pred=dat2Pred)
            if dITp['lvlOut'] > 0:
                print('Predicted for data of shape', dat2Pred.shape)
                print('Shape of predicted Y', self.YPred.shape)
                print('Shape of probs of Y (classes)', self.YProba.shape)
                self.printPredict(X2Pred=dat2Pred)
            self.calcConfMatrix()

    # --- method for calculating the confusion matrix -------------------------
    def calcConfMatrix(self):
        if self.dITp['calcConfMatrix']:
            YTest, YPred, lC = self.YTest, self.YPred, self.lSCl
            if len(YTest.shape) > 1:
                YTest = SF.toSglLbl(self.dITp, dfrY=self.YTest)
            if len(YPred.shape) > 1:
                YPred = SF.toSglLbl(self.dITp, dfrY=self.YPred)
            self.confusMatrix = confusion_matrix(y_true=YTest, y_pred=YPred,
                                                 labels=lC)
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
class RFClf(Classifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, sKPar='A'):
        super().__init__(inpDat, D=D, sKPar=sKPar)
        self.descO = 'Random Forest classifier'
        self.sMthL, self.sMth = self.dITp['sMthRF_L'], self.dITp['sMthRF']
        self.lParG = lG
        self.d2Par = d2Par
        if self.dITp['doRFClf']:
            cClf = self.getClf()
            self.ClfFit(cClf)
        print('Initiated "RFClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = RandomForestClassifier(random_state=self.dITp['rndState'],
                                          warm_start=self.dITp['bWarmStart'],
                                          oob_score=self.dITp['estOobScore'],
                                          n_jobs=self.dITp['nJobs'],
                                          verbose=self.dITp['vVerb'],
                                          **self.d2Par[self.sKPar])
        return self.getOptClfGS()

# -----------------------------------------------------------------------------
class MLPClf(Classifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, sKPar='A'):
        super().__init__(inpDat, D=D, sKPar=sKPar)
        self.descO = 'Neural Network MLP classifier'
        self.sMthL, self.sMth = self.dITp['sMthMLP_L'], self.dITp['sMthMLP']
        self.lParG = lG
        self.d2Par = d2Par
        if self.dITp['doMLPClf']:
            cClf = self.getClf()
            self.fitOrPartialFitClf(cClf)
            self.getScoreClf(cClf)
        print('Initiated "MLPClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = MLPClassifier(random_state=self.dITp['rndState'],
                                 warm_start=self.dITp['bWarmStart'],
                                 verbose=self.dITp['bVerb'],
                                 **self.d2Par[self.sKPar])
        return self.getOptClfGS()

    def getScoreClf(self, cClf):
        if self.dITp['doTrainTestSplit']:
            XTest, YTest = self.getXY(getTrain=False)
            self.scoreClf = cClf.score(XTest, YTest)

# -----------------------------------------------------------------------------
class PropCalculator(BaseSmplClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, iTp=7, lITpUpd=[1, 2, 6]):
        super().__init__(inpDat, D=D, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Calculator: AAc proportions per kinase (class)'
        self.getInpData(sMd=self.dITp['sPrC'])
        print('Initiated "PropCalculator" base object.')

    # --- method for adapting a key of the result paths dictionary ------------
    def setPathOutDataPrC(self, sCl=None):
        sJ1, sJ2, xt = self.dITp['sUSC'], self.dITp['sUS02'], self.dIG['xtCSV']
        sFOutBase = GF.joinS([self.dITp['sProp'], self.dITp['sOutPrC']],
                             cJ=sJ2)
        # sFE = GF.joinS([self.dITp['sMaxLenNmer'], self.dITp['sRestr']],
        #                cJ=sJ1)
        sFE = GF.joinS([self.dITp['sMaxLenNmer'], self.dITp['sRestr'],
                        self.dITp['sSglMltLbl'], self.dITp['sXclEffFam']],
                       cJ=sJ1)
        sFOutPrC = GF.joinS([sFOutBase, sFE], cJ=sJ1) + xt
        if sCl is not None:
            sFOutPrC = GF.joinS([sFOutBase, sFE, sCl], cJ=sJ1) + xt
        self.dPF['OutDataPrC'] = GF.joinToPath(self.dITp['pOutPrC'], sFOutPrC)

    # --- methods calculating the proportions of AAc at all Nmer positions ----
    def calcPropCDfr(self, cDfrI, sCl):
        cD2Out, nTtl = {}, cDfrI.shape[0]
        for sPos in self.dITp['lSCXPrC']:
            serCPos = cDfrI[sPos]
            for sAAc in self.dITp['lFeatSrt']:
                nCAAc = serCPos[serCPos == sAAc].count()
                GF.addToDictD(cD2Out, cKMain=sPos, cKSub=sAAc,
                              cVSub=(0. if nTtl == 0 else nCAAc/nTtl))
        self.dPropAAc[sCl] = GF.iniPdDfr(cD2Out)
        self.setPathOutDataPrC(sCl=sCl)
        self.saveData(self.dPropAAc[sCl], pF=self.dPF['OutDataPrC'])

    def calcPropAAc(self):
        if self.dITp['doPropCalc']:
            dfrInp, dITp = self.dfrInp, self.dITp
            for sCl in self.lSCl:
                if self.dITp['onlySglLbl']:
                    cDfrI = dfrInp[dfrInp[dITp['sEffFam']] == sCl]
                else:
                    cDfrI = dfrInp[dfrInp[sCl] > 0]
                self.calcPropCDfr(cDfrI, sCl=sCl)
            # any XCl
            if self.dITp['onlySglLbl']:
                cDfrI = dfrInp[dfrInp[dITp['sEffFam']].isin(self.lSCl)]
            else:
                cDfrI = dfrInp[(dfrInp[self.lSCl] > 0).any(axis=1)]
            self.calcPropCDfr(cDfrI, sCl=dITp['sX'])
            # entire dataset
            self.calcPropCDfr(dfrInp, sCl=dITp['sAllSeq'])

###############################################################################