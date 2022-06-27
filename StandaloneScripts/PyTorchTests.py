# -*- coding: utf-8 -*-
###############################################################################
# --- PyTorchTests.py ---------------------------------------------------------
###############################################################################
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor

# ### CONSTANTS ###############################################################
# --- files, directories and paths --------------------------------------------
P_TEMP = os.path.join('..', '..', '..', '13_Sysbio03_Phospho15mer',
                      '98_TEMP_CSV')

S_F_INP_TEST = 'Test_AAcPolar_Combined_S_KinasesPho15mer_202202'
S_F_OUT_TEST = S_F_INP_TEST + '_Results'

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
S_TP_PRED_RESPONSE = S_TYPE + S_USC + S_PRED + S_RESPONSE
S_TP_TRAIN_TEST = S_TYPE + S_USC + S_TRAIN + S_TEST

# --- file name extensions ----------------------------------------------------
XT_CSV = S_DOT + S_CSV

# --- numbers -----------------------------------------------------------------
propTestData = .33

R08 = 8

# --- sets --------------------------------------------------------------------

# --- lists -------------------------------------------------------------------

# ### INPUT ###################################################################
# --- flow control ------------------------------------------------------------
doTest = True

# --- boolean variables -------------------------------------------------------

# --- numbers -----------------------------------------------------------------
iMax = 3

# --- strings -----------------------------------------------------------------
sDevice = 'cpu'
lSHdCX = [str(k) for k in range(-iMax, iMax + 1)]
sHdCY = 'EffectorCl'

# --- lists -------------------------------------------------------------------

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

# === derived values and input processing =====================================

# --- fill input dictionary ---------------------------------------------------
dInp = {# --- flow control ----------------------------------------------------
        'doTest': doTest,
        # --- boolean variables -----------------------------------------------
        # --- files, directories and paths ------------------------------------
        'pTemp': P_TEMP,
        'sFInpTest': S_F_INP_TEST + XT_CSV,
        'sFOutTest': S_F_OUT_TEST + XT_CSV,
        # --- file name extensions --------------------------------------------
        'xtCSV': XT_CSV,
        # --- numbers
        'propTestData': propTestData,
        'R08': R08,
        # --- sets
        # --- lists
        # --- strings
        'sDot': S_DOT,
        'sSemicol': S_SEMICOL,
        'sCSV': S_CSV,
        'sSpecies': S_SPECIES,
        'sDevice': sDevice,
        'lSHdCX': lSHdCX,
        'sHdCY': sHdCY,
        'sDevice': sDevice,
        # --- numbers ---------------------------------------------------------
        # --- lists -----------------------------------------------------------
        # --- dictionaries ----------------------------------------------------
        # === derived values and input processing =============================
        }
dInp['pInpTest'] = os.path.join(dInp['pTemp'], dInp['sFInpTest'])
dInp['pOutTest'] = os.path.join(dInp['pTemp'], dInp['sFOutTest'])

# === INPUT. ==================================================================
batchSize = 64       # 64

epochs = 5
classes = ['T-shirt/top',
           'Trouser',
           'Pullover',
           'Dress',
           'Coat',
           'Sandal',
           'Shirt',
           'Sneaker',
           'Bag',
           'Ankle boot']

iTestData = 1

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
        pass

# -----------------------------------------------------------------------------
class Predictor:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dInp, TTD):
        self.idO = 'Pr'
        self.descO = 'Predictor based on the MLP classifier'
        self.dI = dInp
        self.TTD = TTD
        self.predictMLP()
        self.calcConfusionMatrix()
        print('Initiated "Predictor" base object.')

    # --- method for getting classifier score and predictions -----------------
    def predictMLP(self):
        pass

    # --- method for calculating the confusion matrix -------------------------
    def calcConfusionMatrix(self):
        pass

    # --- print methods -------------------------------------------------------
    def printFitQuality(self):
        pass

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
        pass

# -----------------------------------------------------------------------------
class DataFromCSV(data.Dataset):
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dInp, iCol=None):
        self.idO = 'DatCSV'
        self.descO = 'Create a dataset with data from a CSV file'
        self.dI = dInp
        self.loadInpData(iCol=iCol)
        print('Initiated "DataFromCSV" base object.')

    # --- "len" method --------------------------------------------------------
    def __len__(self):
        return self.nRec

    # --- "getitem" method ----------------------------------------------------
    def __getitem__(self, cIdx):
        X, y = self.Xy[self.dI['lSHdCX']], self.Xy[self.dI['sHdCY']]
        return X.iloc[cIdx, :].to_numpy(), y.iat[cIdx]
        # return self.X.loc[cIdx, :], self.y.loc[cIdx]
        # return (self.XyTrain[self.dI['lSHdCX']].to_numpy()[cIdx],
        #         self.XyTrain[self.dI['sHdCY']].to_numpy()[cIdx])

    # --- Functions loading input DataFrames ----------------------------------
    def loadInpData(self, iCol=None):
        dfrInp = readCSV(pF=self.dI['pInpTest'], iCol=iCol)
        # print('Input DataFrame:', S_NEWL, dfrInp, sep='')
        for sHdCX in self.dI['lSHdCX']:
            assert sHdCX in dfrInp.columns
        assert self.dI['sHdCY'] in dfrInp.columns
        self.Xy = dfrInp[self.dI['lSHdCX'] + [self.dI['sHdCY']]]
        self.nRec = self.Xy.shape[0]

    # # --- Functions splitting the data into training and test sets ------------
    # def splitTrainTest(self, Xy):
    #     nTest = round(self.dI['propTestData']*self.Xy.shape[0])
    #     nTrain = self.Xy.shape[0] - nTest
    #     trainData, testData = data.random_split(self.Xy, (nTrain, nTest))
    #     self.XyTrain, self.XyTest = trainData, testData
    #     # self.XTrain, self.yTrain = self.getXY(trainData)
    #     # self.XTest, self.yTest = self.getXY(testData)

    # # --- Functions creating the DataLoaders ----------------------------------
    # def getDataLoaders(self):
    #     self.LoaderTrain = data.DataLoader(self.XyTrain, batch_size=batchSize)
    #     self.LoaderTest = data.DataLoader(self.XyTest, batch_size=batchSize)

# -----------------------------------------------------------------------------
class DataSet:
    # --- initialisation of the class -----------------------------------------
    def __init__(self, dInp, X, y):
        self.idO = 'DatSet'
        self.descO = 'A dataset to be used in a DataLoader'
        self.dI = dInp
        self.X, self.y = X, y
        assert self.X.shape[0] == self.y.shape[0]
        self.lenDS = self.y.shape[0]
        print('Initiated "DataSet" base object.')

    # --- "len" method --------------------------------------------------------
    def __len__(self):
        return self.lenDS

    # --- "getitem" method ----------------------------------------------------
    def __getitem__(self, cIdx):
        return self.X.to_numpy()[cIdx], self.y.to_numpy()[cIdx]

# -----------------------------------------------------------------------------
class DataSetTS(data.Dataset):
    def __init__(self, inpDfr, lSCX, sCY):
        self.Xy = inpDfr
        self.lSCX = lSCX
        self.sCY = sCY
        self.nRec = self.Xy.shape[0]

    def __len__(self):
        return self.nRec

    def __getitem__(self, cIdx):
        return (self.Xy[self.lSCX].iloc[cIdx, :].to_numpy(),
                self.Xy[self.sCY].iat[cIdx])

# =============================================================================

# --- Download training data from open datasets. ------------------------------
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor())

# --- Download test data from open datasets. ----------------------------------
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor())

# --- Create data loaders. ----------------------------------------------------


# train_dataloader = data.DataLoader(training_data, batch_size=batchSize)
# test_dataloader = data.DataLoader(test_data, batch_size=batchSize)

# for X, y in test_dataloader:
#     print('Shape of X [N, C, H, W]:', X.shape)
#     for k, cEl in enumerate(X.shape):
#         print('Length of dimension', k, 'of X:', cEl)
#     print('Shape of y:', y.shape, '/ data type:', y.dtype)
#     for k, cEl in enumerate(y.shape):
#         print('Length of dimension', k, 'of y:', cEl)
#     break

# --- Define neural network model. --------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        # super(NeuralNetwork, self).__init__()
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28*28, 512),
                                               nn.ReLU(),
                                               nn.Linear(512, 512),
                                               nn.ReLU(),
                                               nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# --- Instantiate neural network model. ---------------------------------------
# model = NeuralNetwork().to(sDevice)
# print(model)

# --- Optimizing the Model Parameters. ----------------------------------------
# --- Define loss function and optimizer. -------------------------------------
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# --- Define the training function. -------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(sDevice), y.to(sDevice)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), batch*len(X)
            print('loss:', round(loss, 7), '[', current, '/', size, 'this is',
                  round(current/size*100., 2), '% ]')

# --- Define the test function. -----------------------------------------------
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(sDevice), y.to(sDevice)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print('Test Error: \n Accuracy:', round(100*correct, 2), '/ Avg loss:',
          round(test_loss, 6), '\n')

# --- Training process: look through 'epochs'. --------------------------------
# for t in range(epochs):
#     print('Epoch', t + 1, '\n-------------------------------')
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print('Done!')

# --- Save the model: serialize the internal state dictionary. ----------------
# torch.save(model.state_dict(), 'model.pth')
# print('Saved PyTorch Model State to "model.pth"')

# --- Load the model from the serialized internal state dictionary. -----------
# model = NeuralNetwork()
# model.load_state_dict(torch.load('model.pth'))
# print('Loaded PyTorch Model State from "model.pth"')

# --- Use the model to make predictions. --------------------------------------
# model.eval()
# x, y = test_data[iTestData][0], test_data[iTestData][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     # print(f'Predicted: "{predicted}", Actual: "{actual}"')
#     print('Predicted: "', predicted, '"; Actual: "', actual, '"', sep='')

# === END =====================================================================

# ### MAIN ####################################################################
print(S_EQ80, S_NEWL, S_DS30, ' PyTorchTests.py ', S_DS20, S_NEWL, sep='')

# df1 = pd.DataFrame(np.random.randn(6, 4), columns=list('ABCD'))
# ser1 = pd.Series(list('XZYZZX'), name='EffCl')
# cSlimData = DataSetTS(inpDfr=pd.concat([df1, ser1], axis=1), lSCX=list('ABCD'),
#                       sCY='EffCl')

# df2 = pd.DataFrame([list('XZYZZX'), list('KLHMJK'), list('HGFSII')],
#                    index=['A', 'B', 'EffCl']).T
df2 = pd.DataFrame([list('XZYZZX'), list('KLHMJK'), [.1, .33, .29, .89, .51, .22]],
                   index=['A', 'B', 'EffCl']).T
cSlimData = DataSetTS(inpDfr=df2.infer_objects(), lSCX=list('AB'), sCY='EffCl')

cInpData = DataFromCSV(dInp=dInp, iCol=0)

cData = cSlimData
# cData = cInpData

nTest = round(dInp['propTestData']*cData.Xy.shape[0])
nTrain = cData.Xy.shape[0] - nTest
trainData, testData = data.random_split(cData, (nTrain, nTest))

print('Xy:', S_NEWL, cData.Xy, S_NEWL, sep='')
print('type(Xy):', type(cData.Xy))
print('shape of Xy:', cData.Xy.shape)
print('(nTrain, nTest) =', (nTrain, nTest))
print('trainData =', trainData)
print('testData =', testData)

trainLoader = data.DataLoader(trainData.dataset)
for k, (cX, cLbl) in enumerate(trainLoader):
    if k < 10:
        print(k, ': Data =', cX, '| label =', cLbl)

# print('Dataset of subdata:\n', cData.XyTrain.dataset)
# lInd = cData.XyTrain.indices
# print('Indices of subdata:', lInd[:8], '...', lInd[-8:])
# print('Lengths: (', len(cData.XyTrain), ',', len(lInd), ')')

# print('First line of DataSet:', cData.XyTrain.loc[lInd[0], :])
# for k, cDS in enumerate(cData.XyTrain.to_numpy()):
#     print(k, ':', cDS)


# cData.getDataLoaders()
# print(cData.LoaderTrain)
# nTrain = len(cData.LoaderTrain.dataset)
# print('sizeTrainDS:', nTrain, '/ DS type:', type(cData.LoaderTrain.dataset))
# for k, (cX, cY) in enumerate(cData.LoaderTrain):
#     print(k, ':\n', (cX, cY), '\n')

# print(testDataloader)

# print(list(testDataloader))
# for X, y in testDataloader:
#     print('Shape of X [N, C, H, W]:', X.shape)
#     for k, cEl in enumerate(X.shape):
#         print('Length of dimension', k, 'of X:', cEl)
#     print('Shape of y:', y.shape, '/ data type:', y.dtype)
#     for k, cEl in enumerate(y.shape):
#         print('Length of dimension', k, 'of y:', cEl)
#     break

print(S_DS80, S_NEWL, S_DS30, ' DONE ', S_DS44, sep='')

###############################################################################
