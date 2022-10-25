# -*- coding: utf-8 -*-
###############################################################################
# --- O_07__Classifier.py ----------------------------------------------------
###############################################################################
import matplotlib.pyplot as plt

import Core.C_00__GenConstants as GC
import Core.F_00__GenFunctions as GF
import Core.F_01__SpcFunctions as SF

from Core.O_00__BaseClass import BaseClass

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (GridSearchCV, HalvingGridSearchCV,
                                     RandomizedSearchCV,
                                     HalvingRandomSearchCV, KFold, GroupKFold,
                                     StratifiedKFold, StratifiedGroupKFold,
                                     RepeatedStratifiedKFold)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (PassiveAggressiveClassifier, Perceptron,
                                  SGDClassifier)
from sklearn.naive_bayes import CategoricalNB, ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from imblearn.under_sampling import (ClusterCentroids, AllKNN,
                                     NeighbourhoodCleaningRule,
                                     RandomUnderSampler, TomekLinks)

# -----------------------------------------------------------------------------
class XYData(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, iTp=7, lITpUpd=[]):
        super().__init__(inpDat)
        self.descO = 'All X- and Y-data'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.iniDataStruct()
        print('Initiated "XYData" base object.')

    def iniDataStruct(self):
        self.X, self.Y = {}, {}

    def isEnc(self, sMth=None):
        return (False if sMth is None else (self.dITp['dEncCatFtr'][sMth] in
                                            self.dITp['lSEnc']))

    def setX(self, X, sMth=None, sTp=None, iSt=None, j=None):
        tI = (self.isEnc(sMth=sMth), sTp, iSt, j)
        self.X[tI] = X

    def setY(self, Y, sMth=None, sTp=None, iSt=None, j=None):
        tI = (self.isEnc(sMth=sMth), sTp, iSt, j)
        self.Y[tI] = Y

    def setXY(self, X, Y, sMth=None, sTp=None, iSt=None, j=None):
        tI = (self.isEnc(sMth=sMth), sTp, iSt, j)
        self.X[tI], self.Y[tI] = X, Y

    def getX(self, sMth=None, sTp=None, iSt=None, j=None):
        tI, X = (self.isEnc(sMth=sMth), sTp, iSt, j), None
        if tI in self.X:
            X = self.X[tI]
        else:
            print('ERROR: Info tuple', tI , 'not in X dictionary!')
            print('Keys of X dictionary:', list(self.X))
        return X

    def getY(self, sMth=None, sTp=None, iSt=None, j=None):
        tI, Y = (self.isEnc(sMth=sMth), sTp, iSt, j), None
        if tI in self.Y:
            Y = self.Y[tI]
        else:
            print('ERROR: Info tuple', tI , 'not in Y dictionary!')
            print('Keys of Y dictionary:', list(self.Y))
        return Y

    def getXY(self, sMth=None, sTp=None, iSt=None, j=None):
        tI, X, Y = (self.isEnc(sMth=sMth), sTp, iSt, j), None, None
        if tI in self.X and tI in self.Y:
            X, Y = self.X[tI], self.Y[tI]
        else:
            if tI not in self.X:
                print('ERROR: Info tuple', tI , 'not in X dictionary!')
                print('Keys of X dictionary:', list(self.X))
                if tI in self.Y:
                    Y = self.Y[tI]
            if tI not in self.Y:
                print('ERROR: Info tuple', tI , 'not in Y dictionary!')
                print('Keys of Y dictionary:', list(self.Y))
                if tI in self.X:
                    X = self.X[tI]
        return X, Y

# -----------------------------------------------------------------------------
class ImbSampler(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, X, Y, iSt=None, sMthL=GC.S_MTH_NONE_L,
                 sMth=GC.S_MTH_NONE, iTp=7, lITpUpd=[6]):
        super().__init__(inpDat)
        self.X, self.Y = X, Y
        self.iSt, self.sMthL, self.sMth = iSt, sMthL, sMth
        self.descO = 'Sampler for imbalanced learning'
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.getImbSampler()
        print('Initiated "ImbSampler" base object.')

    # --- print methods -------------------------------------------------------
    def printStrat(self):
        if self.dITp['lvlOut'] > 0:
            if self.iSt is not None:
                print(GC.S_DS04, 'Current step index:', self.iSt)
            if self.dITp['doImbSampling']:
                print(GC.S_DS04, 'Sampling strategy: ', self.sStrat, sep='')
            else:
                print(GC.S_DS04, 'No imbalanced sampling.')

    def printResResampleImb(self, YIni, YRes, doPrt=True):
        if doPrt:
            print(GC.S_DS04, ' Size of Y:', GC.S_NEWL, 'Initial: ', YIni.size,
                  GC.S_VBAR_SEP, 'after resampling: ', YRes.size, sep='')
            YIniUnq, YResUnq = YIni.unique(), YRes.unique()
            print('Unique values of initial Y:', YIniUnq)
            print('Unique values of resampled Y:', YResUnq)
            assert set(YIniUnq) == set(YResUnq)
            print('Sizes of classes before and after resampling:')
            for cY in YResUnq:
                print(cY, self.dITp['sColon'], self.dITp['sTab'],
                      YIni[YIni == cY].size, self.dITp['sVBarSep'],
                      YRes[YRes == cY].size, sep='')

    # --- Function obtaining a custom (imbalanced) sampling strategy ----------
    def getStrat(self):
        # get default strat., or strat. of the current step index (MultiSteps)
        self.sStrat = self.dITp['sStrat']
        if self.dITp['doMultiSteps'] and self.iSt in self.dITp['dSStrat']:
            self.sStrat = self.dITp['dSStrat'][self.iSt]
        self.printStrat()
        # in case of a custom sampling strategy, calculate the dictionary
        if self.sStrat in self.dITp['lSmplStratCustom']:
            if self.sStrat == self.dITp['sStratRealMajo']:
                # implement the "RealMajo" strategy
                self.sStrat = GF.smplStratRealMajo(self.Y)
            elif self.sStrat == self.dITp['sStratShareMino']:
                # implement the "ShareMino" strategy
                dIStrat = self.dITp['dIStrat']
                self.sStrat = GF.smplStratShareMino(self.Y, dI=dIStrat)

    # --- Function obtaining the desired imbalanced sampler ("imblearn") ------
    def getImbSampler(self):
        self.getStrat()
        self.imbSmp, dITp, sStrat = None, self.dITp, self.sStrat
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
        X, YResImb = self.imbSmp.fit_resample(self.X, self.Y)
        self.printResResampleImb(YIni=self.Y, YRes=YResImb)
        if not self.dITp['onlySglLbl']:
            YResImb = SF.toMultiLbl(self.dITp, serY=YResImb, lXCl=self.lSXCl)
        return X, YResImb

# -----------------------------------------------------------------------------
class BaseClfPrC(BaseClass):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, sKPar=GC.S_0, iTp=7, lITpUpd=[1, 2, 6]):
        super().__init__(inpDat)
        self.idO = 'O_07'
        self.descO = 'Classifier and AAc proportions per kinase (class) calc.'
        self.D, self.iTp = D, iTp
        self.getDITp(iTp=iTp, lITpUpd=lITpUpd)
        self.iniAttr(sKPar=sKPar)
        self.XY = XYData(inpDat=inpDat)
        self.fillFPs()
        print('Initiated "BaseClfPrC" base object.')

    # --- method for complementing the type input dictionary ------------------
    def getCLblsTrain(self):
        sLbl, sTrain = self.dITp['sLbl'], self.dITp['sTrain']
        lLblTrain, self.CLblsTrain = self.dITp['lLblTrain'], ''
        if not (lLblTrain is None or self.dITp['onlySglLbl']):
            self.CLblsTrain = GF.joinS([str(nLbl) for nLbl in lLblTrain])
            self.CLblsTrain = GF.joinS([sLbl, self.CLblsTrain, sTrain])

    # --- methods for initialising class attributes and loading input data ----
    def iniAttr(self, sKPar=GC.S_0):
        lAttr2None = ['serNmerSeq', 'dfrInp', 'dClMap', 'dMltSt', 'lSXCl',
                      'Clf', 'optClf', 'sMth', 'sMthL', 'dfrPred', 'dfrProba']
        lAttrDict = ['d3ResClf', 'd2DfrCnfMat', 'd2ResClf', 'dDfrCnfMat',
                     'dDfrPred', 'dDfrProba', 'dPropAAc']
        lAttrDictF = ['dClf', 'dScoreClf', 'dConfusMatrix']
        for cAttr in lAttr2None:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, None)
        for cAttr in lAttrDict:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, {})
        nSplKF = self.dITp['nSplitsKF']
        for cAttr in lAttrDictF:
            if not hasattr(self, cAttr):
                setattr(self, cAttr, {j: None for j in range(nSplKF)})
        self.doPartFit = (self.sMth in self.dITp['lSMthPartFit'] and
                          self.dITp['nItPtFit'] is not None)
        self.sKPar = sKPar
        self.getCLblsTrain()
        self.lIFE = self.D.lIFE + [self.CLblsTrain]

    # --- methods for filling the file paths ----------------------------------
    def fillFPs(self):
        self.FPs = self.D.yieldFPs()
        d2PI, dIG, dITp = {}, self.dIG, self.dITp
        d2PI['OutGSRS'] = {dIG['sPath']: dITp['pOutPar'],
                           dIG['sLFS']: dITp['sGSRS'],
                           dIG['sLFC']: dITp['sFInpSzClf'],
                           dIG['sLFE']: self.lIFE,
                           dIG['sLFJSC']: dITp['sUS02'],
                           dIG['sLFJCE']: dITp['sUS02'],
                           dIG['sFXt']: dIG['xtCSV']}
        d2PI['OutDataPrC'] = {dIG['sPath']: dITp['pOutPrC'],
                              dIG['sLFS']: dITp['sProp'],
                              dIG['sLFC']: dITp['sFInpBPrC'],
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

    def printEncAttr(self, XEnc):
        nFIn, nmFIn = self.cEnc.n_features_in_, self.cEnc.feature_names_in_
        print('Categories:', GC.S_NEWL, self.cEnc.categories_, sep='')
        print('n_features_in:', GC.S_NEWL, nFIn, sep='')
        print('feature_names_in:', GC.S_NEWL, nmFIn, sep='')
        print('Feature names out:', GC.S_NEWL,
              self.cEnc.get_feature_names_out(), sep='')
        print('Encoded array:', GC.S_NEWL, XEnc, GC.S_NEWL,
              'Shape: ', XEnc.shape, sep='')

# -----------------------------------------------------------------------------
class GeneralClassifier(BaseClfPrC):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, iSt=None, sKPar=GC.S_0, cRep=0,
                 sMthL=GC.S_MTH_NONE_L, sMth=GC.S_MTH_NONE, iTp=7,
                 lITpUpd=[1, 2, 6]):
        self.sMthL, self.sMth = sMthL, sMth
        super().__init__(inpDat, D=D, sKPar=sKPar, iTp=iTp, lITpUpd=lITpUpd)
        self.descO = 'General Classifier for data classification'
        self.iSt, self.cRep, self.dSmp = iSt, cRep, {}
        self.getInpData(sMd=self.dITp['sClf'], iSt=self.iSt)
        self.doAllKFolds(inpDat, lITpUpd=lITpUpd)
        print('Initiated "GeneralClassifier" base object.')

    # --- print methods -------------------------------------------------------
    def printInfoImbSampling(self):
        print('Resampling data using resampler "', self.dITp['sSampler'],
              '" with sampling strategy "', self.dITp['sStrat'], '" for fold ',
              self.j + 1, '.', sep='')

    def printResNoResample(self, Y, doPrt=True):
        if doPrt:
            print(GC.S_DS04, ' Size of Y:', Y.size)
            YUnq = Y.unique()
            print('Unique values of Y:', YUnq)
            print('Sizes of classes:')
            for cY in YUnq:
                print(cY, self.dITp['sColon'], self.dITp['sTab'],
                      Y[Y == cY].size, sep='')

    # --- method for getting the defined kFold-Splitter -----------------------
    def getKFoldSplitter(self):
        self.kF, cTp, nSpl = None, self.dITp['tpKF'], self.dITp['nSplitsKF']
        mdShf, rndSt = self.dITp['shuffleKF'], self.dITp['rndStateKF']
        if cTp == GC.S_K_FOLD:
            self.kF = KFold(n_splits=nSpl, shuffle=mdShf, random_state=rndSt)
        elif cTp == GC.S_GROUP_K_FOLD:
            self.kF = GroupKFold(n_splits=nSpl)
        elif cTp == GC.S_STRAT_K_FOLD:
            self.kF = StratifiedKFold(n_splits=nSpl, shuffle=mdShf,
                                      random_state=rndSt)
        elif cTp == GC.S_STRAT_GROUP_K_FOLD:
            self.kF = StratifiedGroupKFold(n_splits=nSpl, shuffle=mdShf,
                                           random_state=rndSt)

    # --- method for encoding and transforming the categorical features -------
    def encodeCatFeatures(self, tpEnc=GC.S_ONE_HOT, catData=None):
        if catData is None:
            catData = self.X
        self.cEnc, XEnc = None, catData
        if tpEnc in self.dITp['lSEnc']:    # encoders implemented so far
            if tpEnc == self.dITp['sOneHot']:
                self.cEnc = OneHotEncoder()
                XEnc = self.cEnc.fit_transform(catData).toarray()
            else:
                self.cEnc = OrdinalEncoder(dtype=int, encoded_missing_value=-1)
                XEnc = self.cEnc.fit_transform(catData)
            if self.dITp['lvlOut'] > 1:
                self.printEncAttr(XEnc=XEnc)
        return GF.iniPdDfr(XEnc, lSNmR=self.Y.index)

    # --- method for splitting data into training and test data ---------------
    def splitTrainTest(self, X, Y):
        XTrain, XTest = X.loc[self.lITrain, :], X.loc[self.lITest, :]
        YTrain, YTest = Y.loc[self.lITrain], Y.loc[self.lITest]
        if not (self.dITp['onlySglLbl'] or self.dITp['lLblTrain'] is None):
            lB = [(serR.sum() in self.dITp['lLblTrain']) for _, serR in
                  YTrain.iterrows()]
            XTrain, YTrain = XTrain[lB], YTrain[lB]
        self.XY.setXY(X=XTrain, Y=YTrain, sMth=self.sMth,
                      sTp=self.dITp['sTrain'], iSt=self.iSt, j=self.j)
        self.XY.setXY(X=XTest, Y=YTest, sMth=self.sMth,
                      sTp=self.dITp['sTest'], iSt=self.iSt, j=self.j)

    # --- method for splitting data into training and test data ---------------
    def encodeSplitData(self, j=None):
        cEncM = self.dITp['dEncCatFtr'][self.sMth]
        XEnc = self.encodeCatFeatures(tpEnc=cEncM)
        if self.dITp['tpKF'] is None:
            self.XY.setX(X=XEnc, sMth=self.sMth, iSt=self.iSt, j=self.j)
        else:
            self.splitTrainTest(X=XEnc, Y=self.Y)

    # --- method for looping over all k folds ---------------------------------
    def doAllKFolds(self, inpDat, lITpUpd):
        sM, sTr, iSt = self.sMth, self.dITp['sTrain'], self.iSt
        self.getKFoldSplitter()
        for j, (lITr, lITe) in enumerate(self.kF.split(X=self.D.yieldXClf(),
                                                       y=self.D.yieldYClf())):
            self.j, self.lITrain, self.lITest = j, lITr, lITe
            self.encodeSplitData()
            X, Y = self.XY.getXY(sMth=sM, sTp=sTr, iSt=iSt, j=self.j)
            self.dSmp[self.j] = ImbSampler(inpDat, X=X, Y=Y, iSt=iSt,
                                           sMthL=self.sMthL, sMth=sM)
            if not self.doPartFit:
                if self.dITp['doImbSampling']:
                    self.printInfoImbSampling()
                    X, Y = self.dSmp[self.j].fitResampleImbalanced()
                    self.XY.setXY(X=X, Y=Y, sMth=sM, sTp=sTr, iSt=iSt,
                                  j=self.j)
                else:       # imbalanced sampling done before each partial fit
                    self.printResNoResample(Y=self.XY.getY(sMth=sM, sTp=sTr,
                                                           iSt=iSt, j=self.j))

    # --- print methods -------------------------------------------------------
    def printStatusPartFit(self, k=0):
        if (k + 1)%round(self.dITp['nItPrintPtFit']) == 0:
            print(GC.S_ARR_LR, ' Performed ', k + 1, ' partial fits using ',
                  'method "', self.sMth, '" in fold ', self.j + 1, '.', sep='')

    def printClfFitRes(self, X):
        if self.dITp['lvlOut'] > 0:
            print('Fitted Classifier to data of shape', X.shape)
            if self.optClf is not None:
                dfrRes = SF.formatDfrCVRes(self.dIG, self.dITp, self.optClf)
                sKMn, sKP, sS = 'OutGSRS', 'sLFE', GC.S_S
                if self.sMth is not None:
                    self.FPs.modFP(d2PI=self.d2PInf, kMn=sKMn, kPos=sKP,
                                   cS=self.sMth, sPos=sS, modPI=True)
                self.saveData(dfrRes, pF=self.FPs.dPF[sKMn])

    def printClfFitError(self):
        print('ERROR: Cannot fit Classifier to data!')
        if self.dITp['lvlOut'] > 1:
            self.printXY()

    def printClfPartialFit(self, X):
        if self.dITp['lvlOut'] > 1:
            print('Partially fitted Classifier to data of shape', X.shape)

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
            sM, sTe, sPd = self.sMth, self.dITp['sTest'], self.dITp['sPred']
            print(GC.S_NEWL, GC.S_DS04, GC.S_SPACE, 'Predictions:', sep='')
            YTest = self.XY.getY(sMth=sM, sTp=sTe, iSt=self.iSt, j=self.j)
            YPred = self.XY.getY(sMth=sM, sTp=sPd, iSt=self.iSt, j=self.j)
            if cSect == 'A1':
                for xT, yT, yP in zip(X2Pr, YTest, YPred):
                    print(xT[:10], GC.S_SPACE, GC.S_ARR_LR, GC.S_SPACE,
                          '(', yT, GC.S_COMMA, GC.S_SPACE, yP, ')', sep='')
            elif cSect == 'B':
                for xT, yP in zip(X2Pr, YPred):
                    print(xT, GC.S_ARR_LR, yP)
            elif cSect == 'C':
                print(YPred)

    def printPredict(self, X2Pred=None, YTest=None):
        if self.dITp['lvlOut'] > 0:
            sM, sPd = self.sMth, self.dITp['sPred']
            YPred = self.XY.getY(sMth=sM, sTp=sPd, iSt=self.iSt, j=self.j)
            print(GC.S_DS08, 'Predictions for fold', self.j + 1, GC.S_DS08)
            print('Shape of predicted Y:', YPred.shape)
            nPred, nOK, _ = tuple(self.d3ResClf[self.sKPar][self.j].values())
            if X2Pred is not None and X2Pred.shape[0] == YPred.shape[0]:
                if YTest is not None:
                    self.printDetailedPredict(X2Pred, cSect='A1')
                    self.printDetailedPredict(X2Pred, nPred, nOK, cSect='A2')
                else:
                    self.printDetailedPredict(X2Pred, cSect='B')
            else:
                self.printDetailedPredict(cSect='C')

    def printFitQuality(self):
        for j, scoreClf in self.dScoreClf.items():
            print(GC.S_DS04, ' Fit quality for the "', self.sMthL, '" method ',
                  'and fold ', j + 1, GC.S_SPACE, GC.S_DS04, sep='')
            if scoreClf is not None:
                print('Classification score for the test data:',
                      round(scoreClf, self.dITp['rndDigScore']))
            if self.dITp['lvlOut'] > 0:
                if self.dConfusMatrix is not None:
                    print('Confusion matrix of fold ', j + 1, ':', GC.S_NEWL,
                          self.dConfusMatrix[j], sep='')

    # --- method for obtaining an optimised Classifier from a grid search -----
    def getOptClf(self, cCV):
        dITp, cClf = self.dITp, self.Clf
        aggrElim, retTrScore = dITp['aggrElimHvS'], dITp['retTrScoreS']
        if dITp['typeS'] == 'GridSearchCV' and not dITp['halvingS']:
            oClf = GridSearchCV(estimator=cClf, param_grid=self.lParG,
                                scoring=dITp['scoringS'],
                                verbose=dITp['verboseS'],
                                return_train_score=retTrScore, cv=cCV)
        elif dITp['typeS'] == 'GridSearchCV' and dITp['halvingS']:
            oClf = HalvingGridSearchCV(estimator=cClf, param_grid=self.lParG,
                                       factor=dITp['factorHvS'],
                                       aggressive_elimination=aggrElim,
                                       scoring=dITp['scoringS'],
                                       verbose=dITp['verboseS'],
                                       return_train_score=retTrScore, cv=cCV)
        elif dITp['typeS'] == 'RandomizedSearchCV' and not dITp['halvingS']:
            oClf = RandomizedSearchCV(estimator=cClf,
                                      param_distributions=self.lParG,
                                      n_iter=dITp['nIterRS'],
                                      scoring=dITp['scoringS'],
                                      verbose=dITp['verboseS'],
                                      return_train_score=retTrScore, cv=cCV)
        elif dITp['typeS'] == 'RandomizedSearchCV' and dITp['halvingS']:
            oClf = HalvingRandomSearchCV(estimator=cClf,
                                         param_distributions=self.lParG,
                                         n_candidates=dITp['nCandidatesHvRS'],
                                         factor=dITp['factorHvS'],
                                         aggressive_elimination=aggrElim,
                                         scoring=dITp['scoringS'],
                                         verbose=dITp['verboseS'],
                                         return_train_score=retTrScore, cv=cCV)
        return oClf

    def setClfOptClfGridSearch(self, cClf=None):
        dITp, self.optClf, retClf = self.dITp, None, cClf
        if self.lParG is not None:
            cCV = RepeatedStratifiedKFold(n_splits=dITp['nSplitsCV'],
                                          n_repeats=dITp['nRepeatsCV'],
                                          random_state=dITp['rndState'])
            optClf = self.getOptClf(cCV)
            self.optClf, retClf = optClf, optClf
        return retClf

    # --- method for selecting the desired Classifier -------------------------
    def selClf(self, j=None):
        if j is None:
            return (self.Clf if self.optClf is None else self.optClf)
        else:
            return (self.dClf[j] if self.optClf is None else self.optClf)


    # --- method for fitting a Classifier -------------------------------------
    def ClfFit(self, cClf):
        X, Y = self.XY.getXY(sMth=self.sMth, sTp=self.dITp['sTrain'],
                             iSt=self.iSt, j=self.j)
        if cClf is not None and X is not None and Y is not None:
            try:
                cClf.fit(X, Y)
                self.printClfFitRes(X)
            except:
                self.printClfFitError()

    def ClfPartialFit(self, cClf, XInp, YInp, k=0):
        X, Y = XInp, YInp
        if cClf is not None and XInp is not None and YInp is not None:
            if self.dITp['doImbSampling']:
                cSmp = self.dSmp[self.j]
                X, Y = cSmp.imbSmp.fit_resample(XInp, YInp)
                cSmp.printResResampleImb(YIni=YInp, YRes=Y, doPrt=(k==0))
            else:
                self.printResNoResample(Y=Y, doPrt=(k==0))
            cClf.partial_fit(X, Y, classes=self.lSXCl)
            self.printClfPartialFit(X)
        return X, Y

    # --- method for selecting the appropriate XTest and YTest values ---------
    def getXYTest(self, cClf, dat2Pred=None, calcScore=False):
        sM, sTe, iSt = self.sMth, self.dITp['sTest'], self.iSt
        YTest = self.XY.getY(sMth=sM, sTp=sTe, iSt=iSt, j=self.j)
        if dat2Pred is not None:
            cEncM = self.dITp['dEncCatFtr'][sM]
            dat2Pred = self.encodeCatFeatures(tpEnc=cEncM, catData=dat2Pred)
        else:
            dat2Pred = self.XY.getX(sMth=sM, sTp=sTe, iSt=iSt, j=self.j)
        if calcScore and self.dScoreClf[self.j] is None:
            if len(YTest.shape) == 1:
                try:
                    self.dScoreClf[self.j] = cClf.score(dat2Pred, YTest)
                except:
                    print('ERROR: Cannot calculate score of classifier "',
                          self.sMthL, '"!', sep='')
        if self.dITp['onlySglLbl']:
            YTest = SF.toMultiLbl(self.dITp, serY=YTest, lXCl=self.lSXCl)
        return dat2Pred, YTest

    def fitOrPartialFitClf(self):
        sM, sTr, iSt = self.sMth, self.dITp['sTrain'], self.iSt
        for self.j in range(self.dITp['nSplitsKF']):
            cClf = self.getClf()
            if self.doPartFit and hasattr(cClf, 'partial_fit'):
                XIni, YIni = self.XY.getXY(sMth=sM, sTp=sTr, iSt=iSt, j=self.j)
                # repeat resampling and partial fit nItPtFit times
                for k in range(round(abs(self.dITp['nItPtFit']))):
                    X, Y = self.ClfPartialFit(cClf, XInp=XIni, YInp=YIni, k=k)
                    self.printStatusPartFit(k=k)
                # self.XY.setXY(X=X, Y=Y, sMth=sM, sTp=sTr, iSt=iSt, j=self.j)
            else:
                self.ClfFit(cClf)
            self.dClf[self.j] = cClf
            # calculate the mean accuracy on the given test data and labels
            if self.dITp['tpKF'] is not None:
                self.getXYTest(cClf=self.dClf[self.j], calcScore=True)

    # --- method for calculating the predicted y classes, and their probs -----
    def getYPredProba(self, cClf, X2Pred=None, YTest=None):
        sM, sPd, sPa = self.sMth, self.dITp['sPred'], self.dITp['sProba']
        sML, lSC, lSR = self.sMthL, YTest.columns, YTest.index
        if self.dITp['onlySglLbl']:
            YPred = GF.iniPdSer(cClf.predict(X2Pred), lSNmI=lSR,
                                nameS=self.dITp['sEffFam'])
            YPred = SF.toMultiLbl(self.dITp, serY=YPred, lXCl=self.lSXCl)
        else:
            YPred = GF.iniPdDfr(cClf.predict(X2Pred), lSNmC=lSC, lSNmR=lSR)
        YProba = GF.getYProba(cClf, dat2Pr=X2Pred, lSC=lSC, lSR=lSR, sMthL=sML)
        if YProba is None:
            YProba = GF.iniWShape(tmplDfr=YPred)
        assert YProba.shape == YPred.shape
        self.XY.setY(Y=YPred, sMth=sM, sTp=sPd, iSt=self.iSt, j=self.j)
        self.XY.setY(Y=YProba, sMth=sM, sTp=sPa, iSt=self.iSt, j=self.j)

    # --- method for calculating values of the Classifier results dictionary --
    def assembleDDfrPredProba(self, lSCTP, YTest=None, YPred=None):
        lDDfr, j = [self.dDfrPred, self.dDfrProba], self.j
        YProba = self.XY.getY(sMth=self.sMth, sTp=self.dITp['sProba'],
                              iSt=self.iSt, j=j)
        for k, cYP in enumerate([YPred, YProba]):
            cDfr = GF.concLOAx1([YTest, cYP], ignIdx=True)
            cDfr.columns = lSCTP
            cDfr = GF.concLOAx1(lObj=[self.dfrInp[self.dITp['sCNmer']], cDfr])
            cDfr.dropna(axis=0, inplace=True)
            cDfr = cDfr.convert_dtypes()
            lDDfr[k][j] = cDfr
        self.dDfrPred[j], self.dDfrProba[j] = lDDfr[0][j], lDDfr[1][j]

    def calcResPredict(self, X2Pred=None, YTest=None):
        YPred = self.XY.getY(sMth=self.sMth, sTp=self.dITp['sPred'],
                             iSt=self.iSt, j=self.j)
        if (X2Pred is not None and YTest is not None and
            YPred is not None and X2Pred.shape[0] == YPred.shape[0]):
            nPred = YPred.shape[0]
            nOK = sum([1 for k in range(nPred) if
                       (YTest.iloc[k, :] == YPred.iloc[k, :]).all()])
            propOK = (nOK/nPred if nPred > 0 else None)
            lVCalc = [nPred, nOK, propOK]
            for sK, cV in zip(self.dITp['lSResClf'], lVCalc):
                GF.addToD3(self.d3ResClf, cKL1=self.sKPar, cKL2=self.j,
                           cKL3=sK, cVL3=cV)
            # create dDfrPred/dDfrProba, containing YTest and YPred/YProba
            sTCl, sPCl = self.dITp['sTrueCl'], self.dITp['sPredCl']
            lSCTP = [GF.joinS([s, sTCl], cJ=self.dITp['sUSC'])
                     for s in YTest.columns]
            lSCTP += [GF.joinS([s, sPCl], cJ=self.dITp['sUSC'])
                      for s in YPred.columns]
            self.assembleDDfrPredProba(lSCTP=lSCTP, YTest=YTest, YPred=YPred)

    # --- method for calculating the confusion matrix -------------------------
    def calcCnfMatrix(self, YTest=None):
        if self.dITp['calcCnfMatrix']:
            sM, sPd, lC = self.sMth, self.dITp['sPred'], self.lSXCl
            YPred = self.XY.getY(sMth=sM, sTp=sPd, iSt=self.iSt, j=self.j)
            if len(YTest.shape) > 1:
                YTest = SF.toSglLbl(self.dITp, dfrY=YTest)
            if len(YPred.shape) > 1:
                YPred = SF.toSglLbl(self.dITp, dfrY=YPred)
            cnfMat = confusion_matrix(y_true=YTest, y_pred=YPred, labels=lC)
            self.dConfusMatrix[self.j] = cnfMat
            dfrCM = GF.iniPdDfr(cnfMat, lSNmC=lC, lSNmR=lC)
            GF.addToDictD(self.d2DfrCnfMat, cKMain=self.sKPar, cKSub=self.j,
                          cVSub=dfrCM)

    # --- method for calculating the mean values of dDfrPred/dDfrProba --------
    def calcFoldsMnPredProba(self):
        lAttrFold = ['dDfrPred', 'dDfrProba']
        lAttrFold2Set = ['dfrPred', 'dfrProba']
        for sAttr, sAttr2Set in zip(lAttrFold, lAttrFold2Set):
            dDfr = getattr(self, sAttr)
            lHd = GF.extractLHdNumColDDfr(dDfr)
            setattr(self, sAttr2Set, GF.calcMeanItO(itO=dDfr, lKMn=lHd))
        dDfr = self.d2DfrCnfMat[self.sKPar]
        lHd = GF.extractLHdNumColDDfr(dDfr)
        self.dDfrCnfMat[self.sKPar] = GF.calcMeanItO(itO=dDfr, lKMn=lHd)
        self.d2ResClf[self.sKPar] = GF.getMeansD2Val(self.d3ResClf[self.sKPar])

    # --- method for predicting with a Classifier -----------------------------
    def ClfPred(self, dat2Pred=None):
        for self.j in range(self.dITp['nSplitsKF']):
            cClf = self.selClf(j=self.j)
            if cClf is not None:
                XTest, YTest = self.getXYTest(cClf=cClf, dat2Pred=dat2Pred)
                self.getYPredProba(cClf, X2Pred=XTest, YTest=YTest)
                self.calcResPredict(X2Pred=XTest, YTest=YTest)
                self.printPredict(X2Pred=XTest, YTest=YTest)
                self.calcCnfMatrix(YTest=YTest)
        self.calcFoldsMnPredProba()

    # --- method for plotting the confusion matrix ----------------------------
    def plotCnfMatrix(self, j=None):
        if self.dITp['plotCnfMatrix'] and self.dConfusMatrix[j] is not None:
            CM, lCl = self.dConfusMatrix[j], self.lSXCl
            D = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=lCl)
            D.plot()
            supTtl = self.dITp['sSupTtlPlt']
            if self.sMthL is not None:
                supTtl += ' (method: "' + self.sMthL + '")'
            D.figure_.suptitle(supTtl)
            plt.show()

# -----------------------------------------------------------------------------
class SpecificClassifier(GeneralClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0,
                 sDesc=GC.S_DESC_NONE, sMthL=GC.S_MTH_NONE_L,
                 sMth=GC.S_MTH_NONE):
        # Classifier method is needed before "super" is initialised
        super().__init__(inpDat, D=D, iSt=iSt, sKPar=sKPar, cRep=cRep,
                         sMthL=sMthL, sMth=sMth)
        self.descO = sDesc
        self.lParG = lG
        assert self.sKPar in d2Par
        self.d2Par = d2Par
        print('Initiated "SpecificClassifier" base object.')

# -----------------------------------------------------------------------------
class DummyClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDDy, sMDyL, sMDy = GC.S_DESC_DUMMY, GC.S_MTH_DUMMY_L, GC.S_MTH_DUMMY
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDDy, sMthL=sMDyL, sMth=sMDy)
        if self.dITp['doDummyClf']:
            self.fitOrPartialFitClf()
        print('Initiated "DummyClf" base object.')

    # --- methods for fitting and predicting with a Dummy Classifier ----------
    def getClf(self):
        Clf = DummyClassifier(random_state=self.dITp['rndState'],
                              **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class AdaClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDAda, sMAdaL, sMAda = GC.S_DESC_ADA, GC.S_MTH_ADA_L, GC.S_MTH_ADA
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDAda, sMthL=sMAdaL, sMth=sMAda)
        if self.dITp['doAdaClf']:
            self.fitOrPartialFitClf()
        print('Initiated "AdaClf" base object.')

    # --- methods for fitting and predicting with an AdaBoost Classifier ------
    def getClf(self):
        Clf = AdaBoostClassifier(random_state=self.dITp['rndState'],
                                 **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class RFClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDRF, sMRFL, sMRF = GC.S_DESC_RF, GC.S_MTH_RF_L, GC.S_MTH_RF
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDRF, sMthL=sMRFL, sMth=sMRF)
        if self.dITp['doRFClf']:
            self.fitOrPartialFitClf()
        print('Initiated "RFClf" base object.')

    # --- methods for fitting and predicting with a Random Forest Classifier --
    def getClf(self):
        Clf = RandomForestClassifier(random_state=self.dITp['rndState'],
                                     warm_start=self.dITp['bWarmStart'],
                                     verbose=self.dITp['vVerbRF'],
                                     oob_score=self.dITp['estOobScoreRF'],
                                     n_jobs=self.dITp['nJobs'],
                                     **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class XTrClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDXTr, sMXTrL, sMXTr = GC.S_DESC_X_TR, GC.S_MTH_X_TR_L, GC.S_MTH_X_TR
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDXTr, sMthL=sMXTrL, sMth=sMXTr)
        if self.dITp['doXTrClf']:
            self.fitOrPartialFitClf()
        print('Initiated "XTrClf" base object.')

    # --- methods for fitting and predicting with an Extra Trees Classifier ---
    def getClf(self):
        Clf = ExtraTreesClassifier(random_state=self.dITp['rndState'],
                                   warm_start=self.dITp['bWarmStart'],
                                   verbose=self.dITp['vVerbXTr'],
                                   oob_score=self.dITp['estOobScoreXTr'],
                                   n_jobs=self.dITp['nJobs'],
                                   **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class GrBClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDGrB, sMGrBL, sMGrB = GC.S_DESC_GR_B, GC.S_MTH_GR_B_L, GC.S_MTH_GR_B
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDGrB, sMthL=sMGrBL, sMth=sMGrB)
        if self.dITp['doGrBClf']:
            self.fitOrPartialFitClf()
        print('Initiated "GrBClf" base object.')

    # --- methods for fitting and predicting with a Gradient Boosting Clf. ----
    def getClf(self):
        Clf = GradientBoostingClassifier(random_state=self.dITp['rndState'],
                                         warm_start=self.dITp['bWarmStart'],
                                         verbose=self.dITp['vVerbGrB'],
                                         **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class HGrBClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDH, sMHL, sMH = GC.S_DESC_H_GR_B, GC.S_MTH_H_GR_B_L, GC.S_MTH_H_GR_B
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDH, sMthL=sMHL, sMth=sMH)
        if self.dITp['doHGrBClf']:
            self.fitOrPartialFitClf()
        print('Initiated "HGrBClf" base object.')

    # --- methods for fitting and predicting with a Hist Gradient Boosting Clf.
    def getClf(self):
        rndSt, bWarmStart = self.dITp['rndState'], self.dITp['bWarmStart']
        Clf = HistGradientBoostingClassifier(random_state=rndSt,
                                             warm_start=bWarmStart,
                                             verbose=self.dITp['vVerbHGrB'],
                                             **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class GPClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDGP, sMGPL, sMGP = GC.S_DESC_GP, GC.S_MTH_GP_L, GC.S_MTH_GP
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDGP, sMthL=sMGPL, sMth=sMGP)
        if self.dITp['doGPClf']:
            self.fitOrPartialFitClf()
        print('Initiated "GPClf" base object.')

    # --- methods for fitting and predicting with a Gaussian Process Classifier
    def getClf(self):
        Clf = GaussianProcessClassifier(random_state=self.dITp['rndState'],
                                        warm_start=self.dITp['bWarmStart'],
                                        n_jobs=self.dITp['nJobs'],
                                        **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class PaAggClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDPaA, sMPaAL, sMPaA = GC.S_DESC_PA_A, GC.S_MTH_PA_A_L, GC.S_MTH_PA_A
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDPaA, sMthL=sMPaAL, sMth=sMPaA)
        if self.dITp['doPaAggClf']:
            self.fitOrPartialFitClf()
        print('Initiated "PaAggClf" base object.')

    # --- methods for fitting and predicting with a Passive Aggressive Clf. ---
    def getClf(self):
        Clf = PassiveAggressiveClassifier(random_state=self.dITp['rndState'],
                                          warm_start=self.dITp['bWarmStart'],
                                          verbose=self.dITp['vVerbPaA'],
                                          n_jobs=self.dITp['nJobs'],
                                          **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class PctClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDPct, sMPctL, sMPct = GC.S_DESC_PCT, GC.S_MTH_PCT_L, GC.S_MTH_PCT
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDPct, sMthL=sMPctL, sMth=sMPct)
        if self.dITp['doPctClf']:
            self.fitOrPartialFitClf()
        print('Initiated "PctClf" base object.')

    # --- methods for fitting and predicting with a Perceptron Classifier -----
    def getClf(self):
        Clf = Perceptron(random_state=self.dITp['rndState'],
                         warm_start=self.dITp['bWarmStart'],
                         verbose=self.dITp['vVerbPct'],
                         n_jobs=self.dITp['nJobs'],
                         **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class SGDClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDSGD, sMSGDL, sMSGD = GC.S_DESC_SGD, GC.S_MTH_SGD_L, GC.S_MTH_SGD
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDSGD, sMthL=sMSGDL, sMth=sMSGD)
        if self.dITp['doSGDClf']:
            self.fitOrPartialFitClf()
        print('Initiated "SGDClf" base object.')

    # --- methods for fitting and predicting with a SGD Classifier ------------
    def getClf(self):
        Clf = SGDClassifier(random_state=self.dITp['rndState'],
                            warm_start=self.dITp['bWarmStart'],
                            verbose=self.dITp['vVerbSGD'],
                            n_jobs=self.dITp['nJobs'],
                            **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class CtNBClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDCt, sMCtL, sMCt = GC.S_DESC_CT_NB, GC.S_MTH_CT_NB_L, GC.S_MTH_CT_NB
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDCt, sMthL=sMCtL, sMth=sMCt)
        if self.dITp['doCtNBClf']:
            self.fitOrPartialFitClf()
        print('Initiated "CtNBClf" base object.')

    # --- methods for fitting and predicting with a Categorical NB Classifier -
    def getClf(self):
        Clf = CategoricalNB(**self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class CpNBClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDCp, sMCpL, sMCp = GC.S_DESC_CP_NB, GC.S_MTH_CP_NB_L, GC.S_MTH_CP_NB
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDCp, sMthL=sMCpL, sMth=sMCp)
        if self.dITp['doCpNBClf']:
            self.fitOrPartialFitClf()
        print('Initiated "CpNBClf" base object.')

    # --- methods for fitting and predicting with a Complement NB Classifier --
    def getClf(self):
        Clf = ComplementNB(**self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class GsNBClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        sDGs, sMGsL, sMGs = GC.S_DESC_GS_NB, GC.S_MTH_GS_NB_L, GC.S_MTH_GS_NB
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDGs, sMthL=sMGsL, sMth=sMGs)
        if self.dITp['doGsNBClf']:
            self.fitOrPartialFitClf()
        print('Initiated "GsNBClf" base object.')

    # --- methods for fitting and predicting with a Gaussian NB Classifier ----
    def getClf(self):
        Clf = GaussianNB(**self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class MLPClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        # Classifier method is needed before "super" is initialised
        sDMLP, sMMLPL, sMMLP = GC.S_DESC_MLP, GC.S_MTH_MLP_L, GC.S_MTH_MLP
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDMLP, sMthL=sMMLPL, sMth=sMMLP)
        if self.dITp['doMLPClf']:
            self.fitOrPartialFitClf()
        print('Initiated "MLPClf" base object.')

    # --- methods for fitting and predicting with a neural network MLP Clf. ---
    def getClf(self):
        Clf = MLPClassifier(random_state=self.dITp['rndState'],
                            warm_start=self.dITp['bWarmStart'],
                            verbose=self.dITp['bVerbMLP'],
                            **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class LinSVClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        # Classifier method is needed before "super" is initialised
        sDLSV, sMLSVL, sMLSV = GC.S_DESC_LSV, GC.S_MTH_LSV_L, GC.S_MTH_LSV
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDLSV, sMthL=sMLSVL, sMth=sMLSV)
        if self.dITp['doLinSVClf']:
            self.fitOrPartialFitClf()
        print('Initiated "LinSVClf" base object.')

    # --- methods for fitting and predicting with a Linear SV Classifier ------
    def getClf(self):
        Clf = LinearSVC(random_state=self.dITp['rndState'],
                        verbose=self.dITp['vVerbLSV'],
                        **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class NuSVClf(SpecificClassifier):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, inpDat, D, lG, d2Par, iSt=None, sKPar=GC.S_0, cRep=0):
        # Classifier method is needed before "super" is initialised
        sDNSV, sMNSVL, sMNSV = GC.S_DESC_NSV, GC.S_MTH_NSV_L, GC.S_MTH_NSV
        super().__init__(inpDat, D=D, lG=lG, d2Par=d2Par, iSt=iSt, sKPar=sKPar,
                         cRep=cRep, sDesc=sDNSV, sMthL=sMNSVL, sMth=sMNSV)
        if self.dITp['doNuSVClf']:
            self.fitOrPartialFitClf()
        print('Initiated "NuSVClf" base object.')

    # --- methods for fitting and predicting with a Nu-Support SV Classifier --
    def getClf(self):
        Clf = NuSVC(random_state=self.dITp['rndState'],
                    verbose=self.dITp['vVerbNSV'],
                    **self.d2Par[self.sKPar])
        return self.setClfOptClfGridSearch(cClf=Clf)

# -----------------------------------------------------------------------------
class PropCalculator(BaseClfPrC):
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