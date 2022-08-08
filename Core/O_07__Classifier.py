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
        self.iniAttr(sKPar=sKPar)
        self.fillFPs()
        print('Initiated "BaseSmplClfPrC" base object.')

    # --- method for complementing the type input dictionary ------------------
    def getCLblsTrain(self):
        sLbl, sTrain = self.dITp['sLbl'], self.dITp['sTrain']
        lLblTrain, self.CLblsTrain = self.dITp['lLblTrain'], ''
        if not (lLblTrain is None or self.dITp['onlySglLbl']):
            self.CLblsTrain = GF.joinS([str(nLbl) for nLbl in lLblTrain])
            self.CLblsTrain = GF.joinS([sLbl, self.CLblsTrain, sTrain])

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self, sKPar='A'):
        lAttr2None = ['serNmerSeq', 'dfrInp', 'dClMap', 'dMltSt', 'lSXCl',
                      'X', 'XTrans', 'XTrain', 'XTest',
                      'XTransTrain', 'XTransTest',
                      'Y', 'YTrain', 'YTest', 'YPred', 'YProba',
                      'Clf', 'sMth', 'sMthL', 'scoreClf', 'confusMatrix',
                      'dfrPred', 'dfrProba']
        lAttrDict = ['d2ResClf', 'dCnfMat', 'dPropAAc']
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)
        for cAttr in lAttrDict:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, {})
        self.sKPar = sKPar
        self.getCLblsTrain()
        self.lIFE = self.D.lIFE + [self.CLblsTrain]

    # --- methods for filling the file paths ----------------------------------
    def fillFPs(self):
        self.FPs = self.D.yieldFPs()
        d2PI, dIG, dITp = {}, self.dIG, self.dITp
        lSFS = dITp['sProp']
        lSFC = dITp['sFInpBPrC']
        d2PI['OutDataPrC'] = {dIG['sPath']: dITp['pOutPrC'],
                              dIG['sLFS']: lSFS,
                              dIG['sLFC']: lSFC,
                              dIG['sLFE']: self.lIFE,
                              dIG['sLFJSC']: dITp['sUS02'],
                              dIG['sLFJCE']: dITp['sUS02'],
                              dIG['sFXt']: dIG['xtCSV']}
        self.FPs.addFPs(d2PI)
        self.d2PInf = d2PI

    # --- methods for getting (and possibly modifying) input data -------------
    def modInpDataMltSt(self, iSt=None):
        if self.dITp['doMultiSteps'] and iSt is not None:
            self.Y = self.dMltSt['dYSt'][iSt]
            self.X = self.X.loc[self.Y.index, :]
            self.dfrInp = self.dfrInp.loc[self.Y.index, :]
            self.dfrInp[self.dITp['sEffFam']] = self.Y
            self.lSXCl = sorted(list(self.Y.unique()))

    def getInpData(self, sMd=None, iSt=None):
        (self.dfrInp, self.X, self.Y, self.serNmerSeq, self.dClMap,
         self.dMltSt, self.lSXCl) = self.D.yieldData(sMd=sMd)
        self.modInpDataMltSt(iSt=iSt)

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
        lSAttr = ['dfrInp', 'serNmerSeq', 'dClMap', 'dMltSt', 'lSXCl', 'X',
                  'XTrans', 'XTrain', 'XTest', 'XTransTrain', 'XTransTest',
                  'Y', 'YTrain', 'YTest', 'YPred', 'YProba']
        assert len(tData) == len(lSAttr)
        for sAttr, cV in zip(lSAttr, tData):
            if cV is not None:
                setattr(self, sAttr, cV)

    # --- methods for yielding data -------------------------------------------
    def yieldData(self):
        return (self.dfrInp, self.serNmerSeq, self.dClMap, self.dMltSt,
                self.lSXCl, self.X, self.XTrans, self.XTrain, self.XTest,
                self.XTransTrain, self.XTransTest, self.Y, self.YTrain,
                self.YTest, self.YPred, self.YProba)

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

    def printResResampleImb(self, X, Y, YRes):
        print(GC.S_DS04, ' Size of Y:', GC.S_NEWL, 'Initial: ', Y.size,
              GC.S_VBAR_SEP, 'after resampling: ', YRes.size, sep='')
        YUnq, YResUnq = Y.unique(), YRes.unique()
        print('Y.unique():', YUnq, '| YRes.unique():', YResUnq)
        assert set(YUnq) == set(YResUnq)
        print('Sizes of classes before and after resampling:')
        for cY in YUnq:
            print(cY, self.dITp['sColon'], self.dITp['sTab'], Y[Y == cY].size,
                  self.dITp['sVBarSep'], YRes[YRes == cY].size, sep='')

# -----------------------------------------------------------------------------
class ImbSampler(BaseSmplClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar='A', iSt=None, iTp=7,
                 lITpUpd=[1, 2, 6]):
        super().__init__(inpDat, D=D, sKPar=sKPar, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Sampler for imbalanced learning'
        self.getInpData(sMd=self.dITp['sClf'], iSt=iSt)
        self.encodeSplitData()
        self.getSampler()
        print('Initiated "ImbSampler" base object.')

    # --- Function obtaining the imbalanced sampler ("imblearn") --------------
    def getStrat(self):
        sStrat = self.dITp['sStrat']
        # implement the "RealMajo" strategy
        if sStrat == self.dITp['sStratRealMajo']:
            doSpl = self.dITp['doTrainTestSplit']
            _, Y = self.getXY(getTrain=GF.isTrain(doSplit=doSpl))
            sStrat = GF.smplStratRealMajo(Y=Y)
        return sStrat

    def getSampler(self):
        self.imbSmp, dITp, sStrat = None, self.dITp, self.getStrat()
        if dITp['sSampler'] == 'ClusterCentroids':
            self.imbSmp = ClusterCentroids(sampling_strategy=sStrat,
                                           random_state=dITp['rndState'],
                                           estimator=dITp['estimator'],
                                           voting=dITp['voting'])
        elif dITp['sSampler'] == 'AllKNN':
            self.imbSmp = AllKNN(sampling_strategy=sStrat,
                                 n_neighbors=dITp['n_neighbors_AllKNN'],
                                 kind_sel=dITp['kind_sel_AllKNN'],
                                 allow_minority=dITp['allow_minority'])
        elif dITp['sSampler'] == 'NeighbourhoodCleaningRule':
            nNbr = dITp['n_neighbors_NCR']
            kindSel, thrCln = dITp['kind_sel_NCR'], dITp['threshold_cleaning']
            self.imbSmp = NeighbourhoodCleaningRule(sampling_strategy=sStrat,
                                                    n_neighbors=nNbr,
                                                    kind_sel=kindSel,
                                                    threshold_cleaning=thrCln)
        elif dITp['sSampler'] == 'RandomUnderSampler':
            self.imbSmp = RandomUnderSampler(sampling_strategy=sStrat,
                                             random_state=dITp['rndState'],
                                             replacement=dITp['wReplacement'])
        elif dITp['sSampler'] == 'TomekLinks':
            self.imbSmp = TomekLinks(sampling_strategy=sStrat)

    # --- Function performing the random sampling ("imblearn") ----------------
    def fitResampleImbalanced(self):
        print('Resampling data using resampler "', self.dITp['sSampler'],
              '" with sampling strategy "', self.dITp['sStrat'], '".', sep='')
        doSpl = self.dITp['doTrainTestSplit']
        X, Y = self.getXY(getTrain=GF.isTrain(doSplit=doSpl))
        X, YResImb = self.imbSmp.fit_resample(X, Y)
        self.printResResampleImb(X=X, Y=Y, YRes=YResImb)
        if not self.dITp['onlySglLbl']:
            YResImb = SF.toMultiLbl(self.dITp, serY=YResImb, lXCl=self.lSXCl)
        self.setXY(X=X, Y=YResImb, setTrain=GF.isTrain(doSplit=doSpl))

# -----------------------------------------------------------------------------
class Classifier(BaseSmplClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar='A', iSt=None, iTp=7,
                 lITpUpd=[1, 2, 6]):
        super().__init__(inpDat, D=D, sKPar=sKPar, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Classifier for data classification'
        self.Smp = ImbSampler(inpDat, D, sKPar=sKPar, iSt=iSt, iTp=iTp,
                              lITpUpd=lITpUpd)
        self.setData(self.Smp.yieldData())
        if self.dITp['doImbSampling']:
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
        if self.dITp['lvlOut'] > 1:
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
        if self.dITp['lvlOut'] > 0:
            print('Predicted for data of shape', X2Pred.shape)
            print('Shape of predicted Y', self.YPred.shape)
            print('Shape of probs of Y (classes)', self.YProba.shape)
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
    def getOptClfGridSearch(self):
        self.optClf, retClf = None, self.Clf
        if self.lParG is not None:
            optClf = GridSearchCV(estimator=self.Clf, param_grid=self.lParG,
                                  scoring='accuracy')
            self.optClf, retClf = optClf, optClf
        return retClf

    # --- method for selecting the desired Classifier -------------------------
    def selClf(self):
        return (self.Clf if self.optClf is None else self.optClf)

    # --- method for fitting a Classifier -------------------------------------
    def ClfFit(self, cClf):
        X, Y = self.getXY(getTrain=GF.isTrain(self.dITp['doTrainTestSplit']))
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
            cClf.partial_fit(X, Y, classes=self.lSXCl)
            self.printClfPartialFit(X)
        return X, Y

    def fitOrPartialFitClf(self, cClf):
        if (self.sMth == self.dITp['sMthMLP'] and
            self.dITp['nItPartialFit'] is not None):
            assert type(self.dITp['nItPartialFit']) in [int, float]
            doSpl = self.dITp['doTrainTestSplit']
            XIni, YIni = self.getXY(getTrain=GF.isTrain(doSplit=doSpl))
            # repeat resampling and partial fit nItPartialFit times
            for _ in range(round(abs(self.dITp['nItPartialFit']))):
                X, Y = self.ClfPartialFit(cClf, XInp=XIni, YInp=YIni)
            self.setXY(X=X, Y=Y, setTrain=GF.isTrain(doSplit=doSpl))
        else:
            self.ClfFit(cClf)
        # calculate the mean accuracy on the given test data and labels
        if self.dITp['doTrainTestSplit']:
            XTest, YTest = self.getXY(getTrain=False)
            self.scoreClf = cClf.score(XTest, YTest)

    # --- method for selecting the appropriate XTest and YTest vlues ----------
    def getXYTest(self, dat2Pred=None):
        if dat2Pred is not None and self.dITp['encodeCatFtr']:
            dat2Pred = self.cEnc.transform(dat2Pred).toarray()
        if dat2Pred is None:
            dat2Pred, self.YTest = self.getXY(getTrain=False)
        if self.dITp['onlySglLbl']:
            self.YTest = SF.toMultiLbl(self.dITp, serY=self.YTest,
                                       lXCl=self.lSXCl)
        return dat2Pred

    # --- method for calculating the predicted y classes, and their probs -----
    def getYPredProba(self, cClf, X2Pred=None):
        lSC, lSR = self.YTest.columns, self.YTest.index
        if self.dITp['onlySglLbl']:
            YPred = GF.iniPdSer(cClf.predict(X2Pred), lSNmI=lSR,
                                nameS=self.dITp['sEffFam'])
            self.YPred = SF.toMultiLbl(self.dITp, serY=YPred, lXCl=self.lSXCl)
        else:
            self.YPred = GF.iniPdDfr(cClf.predict(X2Pred), lSNmC=lSC,
                                     lSNmR=lSR)
        self.YProba = GF.getYProba(cClf, dat2Pr=X2Pred, lSC=lSC, lSR=lSR)
        assert self.YProba.shape == self.YPred.shape

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
        cClf = self.selClf()
        if cClf is not None:
            XTest = self.getXYTest(dat2Pred=dat2Pred)
            self.getYPredProba(cClf, X2Pred=XTest)
            self.calcResPredict(X2Pred=XTest)
            self.printPredict(X2Pred=XTest)
            self.calcCnfMatrix()

    # --- method for calculating the confusion matrix -------------------------
    def calcCnfMatrix(self):
        if self.dITp['calcCnfMatrix']:
            YTest, YPred, lC = self.YTest, self.YPred, self.lSXCl
            if len(YTest.shape) > 1:
                YTest = SF.toSglLbl(self.dITp, dfrY=self.YTest)
            if len(YPred.shape) > 1:
                YPred = SF.toSglLbl(self.dITp, dfrY=self.YPred)
            self.confusMatrix = confusion_matrix(y_true=YTest, y_pred=YPred,
                                                 labels=lC)
            dfrCM = GF.iniPdDfr(self.confusMatrix, lSNmC=lC, lSNmR=lC)
            self.dCnfMat[self.sKPar] = dfrCM

    # --- method for plotting the confusion matrix ----------------------------
    def plotCnfMatrix(self):
        if self.dITp['plotCnfMatrix'] and self.confusMatrix is not None:
            CM, lCl = self.confusMatrix, self.lSXCl
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
    def __init__(self, inpDat, D, lG, d2Par, sKPar='A', iSt=None):
        super().__init__(inpDat, D=D, sKPar=sKPar, iSt=iSt)
        self.descO = 'Random Forest classifier'
        self.sMthL, self.sMth = self.dITp['sMthRF_L'], self.dITp['sMthRF']
        self.lParG = lG
        self.d2Par = d2Par
        if self.dITp['doRFClf']:
            cClf = self.getClf()
            self.fitOrPartialFitClf(cClf)
        print('Initiated "RFClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = RandomForestClassifier(random_state=self.dITp['rndState'],
                                          warm_start=self.dITp['bWarmStart'],
                                          oob_score=self.dITp['estOobScore'],
                                          n_jobs=self.dITp['nJobs'],
                                          verbose=self.dITp['vVerb'],
                                          **self.d2Par[self.sKPar])
        return self.getOptClfGridSearch()

# -----------------------------------------------------------------------------
class MLPClf(Classifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, sKPar='A', iSt=None):
        super().__init__(inpDat, D=D, sKPar=sKPar, iSt=iSt)
        self.descO = 'Neural Network MLP classifier'
        self.sMthL, self.sMth = self.dITp['sMthMLP_L'], self.dITp['sMthMLP']
        self.lParG = lG
        self.d2Par = d2Par
        if self.dITp['doMLPClf']:
            cClf = self.getClf()
            self.fitOrPartialFitClf(cClf)
        print('Initiated "MLPClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        self.Clf = MLPClassifier(random_state=self.dITp['rndState'],
                                 warm_start=self.dITp['bWarmStart'],
                                 verbose=self.dITp['bVerb'],
                                 **self.d2Par[self.sKPar])
        return self.getOptClfGridSearch()

# -----------------------------------------------------------------------------
class PropCalculator(BaseSmplClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, iTp=7, lITpUpd=[1, 2, 6]):
        super().__init__(inpDat, D=D, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'Calculator: AAc proportions per kinase (class)'
        self.getInpData(sMd=self.dITp['sPrC'])
        print('Initiated "PropCalculator" base object.')

    # --- methods calculating the proportions of AAc at all Nmer positions ----
    def calcPropCDfr(self, cDfrI, sCl=None):
        cD2Out, nTtl = {}, cDfrI.shape[0]
        for sPos in self.dITp['lSCXPrC']:
            serCPos = cDfrI[sPos]
            for sAAc in self.dITp['lFeatSrt']:
                nCAAc = serCPos[serCPos == sAAc].count()
                GF.addToDictD(cD2Out, cKMain=sPos, cKSub=sAAc,
                              cVSub=(0. if nTtl == 0 else nCAAc/nTtl))
        self.dPropAAc[sCl] = GF.iniPdDfr(cD2Out)
        self.FPs.modFP(d2PI=self.d2PInf, kMn='OutDataPrC', kPos='sLFE', cS=sCl)
        self.saveData(self.dPropAAc[sCl], pF=self.FPs.dPF['OutDataPrC'])

    def calcPropAAc(self):
        if self.dITp['doPropCalc']:
            dfrInp, dITp = self.dfrInp, self.dITp
            for sCl in self.lSXCl:
                if self.dITp['onlySglLbl']:
                    cDfrI = dfrInp[dfrInp[dITp['sEffFam']] == sCl]
                else:
                    cDfrI = dfrInp[dfrInp[sCl] > 0]
                self.calcPropCDfr(cDfrI, sCl=sCl)
            # any XCl
            if self.dITp['onlySglLbl']:
                cDfrI = dfrInp[dfrInp[dITp['sEffFam']].isin(self.lSXCl)]
            else:
                cDfrI = dfrInp[(dfrInp[self.lSXCl] > 0).any(axis=1)]
            self.calcPropCDfr(cDfrI, sCl=dITp['sX'])
            # entire dataset
            self.calcPropCDfr(dfrInp, sCl=dITp['sAllSeq'])

###############################################################################