# -*- coding: utf-8 -*-
###############################################################################
# --- D_80__Looper.py ---------------------------------------------------------
###############################################################################
from scipy import stats

import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Data for looper over parameter sets and repetitions (D_80__Looper)'
sNmSpec = 'Input data for the Looper class in O_80__Looper'

# --- flow control ------------------------------------------------------------
# use dNumRep for both parameter grid calculations and parameter dictionary
# evaluations. Parameter grid calculations for a particular classification
# method are performed if the number of repetitions is positve. Otherwise,
# a parameter set is evaluated number of repetitions times.
nRp0, nRpDef = GC.N_REP_0, 5
dNumRep = {GC.S_MTH_DUMMY: nRp0,
           GC.S_MTH_ADA: nRp0,
           GC.S_MTH_RF: nRp0,
           GC.S_MTH_X_TR: nRp0,
           GC.S_MTH_GR_B: nRp0,
           GC.S_MTH_H_GR_B: nRp0,
           GC.S_MTH_GP: nRp0,
           GC.S_MTH_PA_A: nRp0,
           GC.S_MTH_PCT: nRp0,
           GC.S_MTH_SGD: nRpDef,
           GC.S_MTH_CT_NB: nRp0,
           GC.S_MTH_CP_NB: nRp0,
           GC.S_MTH_GS_NB: nRp0,
           GC.S_MTH_MLP: nRp0,
           GC.S_MTH_LSV: nRp0,
           GC.S_MTH_NSV: nRp0}

doParGrid = True                   # do parameter grid calculations?

useKey0 = False                     # use the parameter key GC.S_0?
                                    # (in case doParGrid == False)

# === Dummy Classifier ========================================================
# --- list of parameter grids for Dummy Classifier grid search ----------------
lParGrid_Dy = [{'strategy': ['uniform', 'stratified']}]

# --- parameter dictionary for Dummy Classifier -------------------------------
d2Par_Dy = {GC.S_0: {'strategy': 'uniform',
                     'constant': None},
            'A': {'strategy': 'stratified',
                  'constant': None}}

# === AdaBoost Classifier =====================================================
# --- list of parameter grids for AdaBoost Classifier grid search -------------
lParGrid_Ada = [{'n_estimators': [10, 50, 100, 1000],
                 'learning_rate': [0.5, 1.0, 2.0],
                 'algorithm': ['SAMME', 'SAMME.R']}]
# lParGrid_Ada = [{'n_estimators': [1000],
#                  'learning_rate': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#                  'algorithm': ['SAMME']}]
lParGrid_Ada = [{'n_estimators': [1000],
                  'learning_rate': stats.uniform(loc=0.5, scale=0.3),
                  'algorithm': ['SAMME']}]

# --- parameter dictionary for AdaBoost Classifier ----------------------------
d2Par_Ada = {GC.S_0: {'n_estimators': 50,
                      'learning_rate': 1.,
                      'algorithm': 'SAMME.R'},
             'A': {'n_estimators': 1000,
                   'learning_rate': 0.55,
                   'algorithm': 'SAMME'}}

# === Random Forest Classifier ================================================
# --- list of parameter grids for Random Forest Classifier grid search --------
# lParGrid_RF = [{'n_estimators': [100, 1000],
#                 'criterion': ['gini', 'entropy', 'log_loss'],
#                 'bootstrap': [True, False]},
#                {'max_features': [None, 'sqrt'],
#                 'ccp_alpha': [0.0, 0.1]}]
lParGrid_RF = [{'criterion': ['gini', 'entropy', 'log_loss'],
                'ccp_alpha': [0.0, 0.1]},
               {'max_features': [None, 'sqrt']}]
lParGrid_RF = [{'criterion': ['gini', 'entropy', 'log_loss'],
                'bootstrap': [True, False],
                'ccp_alpha': stats.uniform(loc=0.0, scale=0.5)},
               {'max_features': [None, 'sqrt']}]

# --- parameter dictionary for Random Forest Classifier -----------------------
d2Par_RF = {GC.S_0: {'n_estimators': 100,
                     'criterion': 'gini',
                     'max_depth': None,
                     'min_samples_split': 2,
                     'min_samples_leaf': 1,
                     'min_weight_fraction_leaf': 0.0,
                     'max_features': 'sqrt',
                     'max_leaf_nodes': None,
                     'min_impurity_decrease': 0.0,
                     'bootstrap': True,
                     'ccp_alpha': 0.0,
                     'max_samples': None},
            'B': {'n_estimators': 100,
                  'criterion': 'entropy',
                  'max_depth': None,
                  'min_samples_split': 2,
                  'min_samples_leaf': 1,
                  'min_weight_fraction_leaf': 0.0,
                  'max_features': 'sqrt',
                  'max_leaf_nodes': None,
                  'min_impurity_decrease': 0.0,
                  'bootstrap': True,
                  'ccp_alpha': 0.0,
                  'max_samples': None}}

# === Extra Trees Classifier ==================================================
# --- list of parameter grids for Extra Trees Classifier grid search ----------
# lParGrid_XTr = [{'n_estimators': [100, 1000],
#                  'criterion': ['gini', 'entropy', 'log_loss'],
#                  'bootstrap': [True, False]},
#                 {'max_features': [None, 'sqrt'],
#                  'ccp_alpha': [0.0, 0.1]}]
lParGrid_XTr = [{'criterion': ['gini', 'entropy', 'log_loss'],
                 'ccp_alpha': [0.0, 0.1]},
                {'max_features': [None, 'sqrt']}]
lParGrid_XTr = [{'criterion': ['gini', 'entropy', 'log_loss'],
                 'bootstrap': [True, False],
                 'ccp_alpha': stats.uniform(loc=0.0, scale=0.5)},
                {'max_features': [None, 'sqrt']}]

# --- parameter dictionary for Extra Trees Classifier -------------------------
d2Par_XTr = {GC.S_0: {'n_estimators': 100,
                      'criterion': 'gini',
                      'max_depth': None,
                      'min_samples_split': 2,
                      'min_samples_leaf': 1,
                      'min_weight_fraction_leaf': 0.0,
                      'max_features': 'sqrt',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'bootstrap': False,
                      'ccp_alpha': 0.0,
                      'max_samples': None},
             'A': {'n_estimators': 100,
                   'criterion': 'entropy',
                   'max_depth': None,
                   'min_samples_split': 2,
                   'min_samples_leaf': 1,
                   'min_weight_fraction_leaf': 0.0,
                   'max_features': 'sqrt',
                   'max_leaf_nodes': None,
                   'min_impurity_decrease': 0.0,
                   'bootstrap': True,
                   'ccp_alpha': 0.0,
                   'max_samples': None}}

# === Gradient Boosting Classifier ============================================
# --- list of parameter grids for Gradient Boosting Classifier grid search ----
lParGrid_GrB = [{'loss': ['log_loss', 'exponential'],
                 'learning_rate': [0.1, 0.5],
                 'n_estimators': [100, 1000],
                 'subsample': [0.5, 1.0],
                 'criterion': ['friedman_mse', 'squared_error']},
                {'max_features': [None, 'sqrt'],
                 'ccp_alpha': [0.0, 0.1]}]
lParGrid_GrB = [{'loss': ['log_loss', 'exponential'],
                 'learning_rate': stats.uniform(loc=0.0, scale=1.0),
                 'ccp_alpha': stats.uniform(loc=0.0, scale=0.5),
                 'n_estimators': [100, 1000],
                 'subsample': stats.uniform(loc=0.1, scale=0.9),
                 'criterion': ['friedman_mse', 'squared_error']}]

# --- parameter dictionary for Gradient Boosting Classifier -------------------
d2Par_GrB = {GC.S_0: {'loss': 'log_loss',
                      'learning_rate': 0.1,
                      'n_estimators': 100,
                      'subsample': 1.0,
                      'criterion': 'friedman_mse',
                      'min_samples_split': 2,
                      'min_samples_leaf': 1,
                      'min_weight_fraction_leaf': 0.0,
                      'max_depth': 3,
                      'min_impurity_decrease': 0.0,
                      'init': None,
                      'max_features': None,
                      'max_leaf_nodes': None,
                      'validation_fraction': 0.1,
                      'n_iter_no_change': None,
                      'tol': 1.0e-4,
                      'ccp_alpha': 0.0}}

# === Hist Gradient Boosting Classifier =======================================
# --- list of parameter grids for Hist Gradient Boosting Classifier grid search
# lParGrid_HGrB = [{'learning_rate': [0.1, 0.5],
#                   'max_iter': [100, 500],
#                   'max_leaf_nodes': [10, 31, None],
#                   'max_depth': [5, None],
#                   'min_samples_leaf': [5, 20],
#                   'l2_regularization': [0, 0.5],
#                   'max_bins': [63, 255]}]
lParGrid_HGrB = [{'learning_rate': stats.uniform(loc=0.0, scale=1.0),
                  'max_iter': stats.randint(10, 1000 + 1),
                  'max_leaf_nodes': [10, 31, None],
                  'max_depth': [5, None],
                  'min_samples_leaf': stats.randint(5, 20 + 1),
                  'l2_regularization': stats.uniform(loc=0.0, scale=1.0),
                  'max_bins': [63, 255]}]

# --- parameter dictionary for Hist Gradient Boosting Classifier --------------
d2Par_HGrB = {GC.S_0: {'loss': 'log_loss',
                       'learning_rate': 0.1,
                       'max_iter': 100,
                       'max_leaf_nodes': 31,
                       'max_depth': None,
                       'min_samples_leaf': 20,
                       'l2_regularization': 0,
                       'max_bins': 255,
                       'categorical_features': None,
                       'monotonic_cst': None,
                       'early_stopping': 'auto',
                       'scoring': 'loss',
                       'validation_fraction': 0.1,
                       'n_iter_no_change': 10,
                       'tol': 1.0e-7}
              }

# === Gaussian Process Classifier =============================================
# --- list of parameter grids for Gaussian Process Classifier grid search -----
lParGrid_GP = [{'n_restarts_optimizer': [0, 1, 10],
                'max_iter_predict': [10, 100, 1000],
                'multi_class': ['one_vs_rest', 'one_vs_one']}]
lParGrid_GP = [{'n_restarts_optimizer': stats.randint(0, 10 + 1),
                'max_iter_predict': stats.randint(10, 1000 + 1),
                'multi_class': ['one_vs_rest', 'one_vs_one']}]

# --- parameter dictionary for Gaussian Process Classifier --------------------
d2Par_GP = {GC.S_0: {'kernel': None,
                     'optimizer': 'fmin_l_bfgs_b',
                     'n_restarts_optimizer': 0,
                     'max_iter_predict': 100,
                     'copy_X_train': True,
                     'multi_class': 'one_vs_rest'},
            # 'B': {'kernel': None,
            #       'optimizer': 'fmin_l_bfgs_b',
            #       'n_restarts_optimizer': 0,
            #       'max_iter_predict': 100,
            #       'copy_X_train': True,
            #       'multi_class': 'one_vs_one'}
            }

# === Passive Aggressive Classifier ===========================================
# --- list of parameter grids for Passive Aggressive Classifier grid search ---
lParGrid_PaA = [{'C': [0.1, 1.0, 10.0],
                 'loss': ['hinge', 'squared_hinge'],
                 'class_weight': [None, 'balanced']}]
lParGrid_PaA = [{'max_iter': [10000],
                 'tol': [1.0e-3, 1.0e-6, 1.0e-9],
                 'loss': ['hinge'],
                 'class_weight': ['balanced'],
                 'average': [False, True]}]
# lParGrid_PaA = [{'C': stats.loguniform(a=0.01, b=10.0),
#                  'max_iter': [100, 1000, 3000],
#                  'tol': [1.0e-2, 1.0e-3, 1.0e-4],
#                  'loss': ['hinge', 'squared_hinge'],
#                  'class_weight': [None, 'balanced']}]
# lParGrid_PaA = [{'C': stats.loguniform(a=0.01, b=10.0),
#                  'max_iter': [100, 1000, 3000],
#                  'tol': [1.0e-2, 1.0e-3, 1.0e-4],
#                  'loss': ['hinge'],
#                  'class_weight': ['balanced']}]

# --- parameter dictionary for Passive Aggressive Classifier ------------------
d2Par_PaA = {GC.S_0: {'C': 1.0,
                      'fit_intercept': True,
                      'max_iter': 1000,
                      'tol': 1.0e-3,
                      'early_stopping': False,
                      'validation_fraction': 0.1,
                      'n_iter_no_change': 5,
                      'shuffle': True,
                      'loss': 'hinge',
                      'class_weight': None,
                      'average': False},
             'A': {'max_iter': 10000,
                   'tol': 1.0e-6,
                   'class_weight': None,
                   'average': True}}

# === Perceptron Classifier ===================================================
# --- list of parameter grids for Perceptron Classifier grid search -----------
lParGrid_Pct = [{'penalty': ['elasticnet', None],
                 'alpha': [1.0e-8],
                 'l1_ratio': [0.5],
                  'eta0': [0.01, 0.1, 1., 10.],
                 # 'class_weight': [None, 'balanced']
                 }]
# lParGrid_Pct = [{'penalty': [None, 'l2', 'l1', 'elasticnet'],
#                  'alpha': stats.uniform(loc=0.00001, scale=(1. - 0.00001)),
#                  'l1_ratio': stats.uniform(loc=0., scale=1.),
#                  'eta0': stats.uniform(loc=0.1, scale=(10. - 0.1))}]

# --- parameter dictionary for Perceptron Classifier --------------------------
d2Par_Pct = {GC.S_0: {'penalty': None,
                      'alpha': 0.0001,
                      'l1_ratio': 0.15,
                      'fit_intercept': True,
                      'max_iter': 1000,
                      'tol': 1.0e-3,
                      'shuffle': True,
                      'eta0': 1,
                      'early_stopping': False,
                      'validation_fraction': 0.1,
                      'n_iter_no_change': 5,
                      'class_weight': None},
             'A': {'penalty': 'elasticnet',
                   'alpha': 1.0e-8,
                   'l1_ratio': 0.5,
                   'eta0': 0.01}}

# === Stochastic Gradient Descent (SGD) Classifier ============================
# --- list of parameter grids for SGD Classifier grid search ------------------
lParGrid_SGD = [{'loss': ['hinge', 'log_loss', 'modified_huber',
                          'squared_hinge', 'perceptron', 'squared_error',
                          'huber', 'epsilon_insensitive',
                          'squared_epsilon_insensitive'],
                 'penalty': ['l2', 'l1', 'elasticnet'],
                 'alpha': [0.0001, 1.],
                 'l1_ratio': [0., 0.15, 0.5, 1.],
                 'epsilon': [0.0001, 0.1, 1., 10.],
                 'learning_rate': ['constant', 'optimal', 'invscaling',
                                   'adaptive'],
                 'eta0': [0.5, 1., 2.],
                 'power_t': [0.0, 0.5, 1., 2.],
                 'class_weight': [None, 'balanced']}]
lParGrid_SGD = [{'loss': ['hinge', 'log_loss', 'modified_huber',
                          'squared_hinge', 'perceptron'],
                 'penalty': ['l2', 'l1', 'elasticnet'],
                 'alpha': stats.uniform(loc=0.00001, scale=(1. - 0.00001)),
                 'l1_ratio': stats.uniform(loc=0., scale=1.),
                 'epsilon': stats.uniform(loc=0., scale=10.),
                 'learning_rate': ['constant', 'optimal', 'invscaling',
                                   'adaptive'],
                 'eta0': stats.uniform(loc=0.1, scale=(10. - 0.1)),
                 'power_t': stats.uniform(loc=0., scale=2.),
                 'class_weight': [None, 'balanced'],
                 'average': [False, True]}]

# --- parameter dictionary for SGD Classifier ---------------------------------
d2Par_SGD = {GC.S_0: {'loss': 'hinge',
                      'penalty': 'l2',
                      'alpha': 0.0001,
                      'l1_ratio': 0.15,
                      'fit_intercept': True,
                      'max_iter': 1000,
                      'tol': 1.0e-3,
                      'shuffle': True,
                      'epsilon': 0.1,
                      'learning_rate': 'optimal',
                      'eta0': 0.0,
                      'power_t': 0.5,
                      'early_stopping': False,
                      'validation_fraction': 0.1,
                      'n_iter_no_change': 5,
                      'class_weight': None,
                      'average': False}}

# === Categorical NB Classifier ===============================================
# --- list of parameter grids for Categorical NB Classifier grid search -------
lParGrid_CtNB = [{'alpha': [0, 0.5, 1., 2.],
                  'fit_prior': [False, True]}]
lParGrid_CtNB = [{'alpha': stats.uniform(loc=0., scale=1.),
                  'fit_prior': [False, True]}]

# --- parameter dictionary for Categorical NB Classifier ----------------------
d2Par_CtNB = {GC.S_0: {'alpha': 1.,
                       'fit_prior': True,
                       'class_prior': None,
                       'min_categories': None},
              'A': {'alpha': 0.37,
                    'fit_prior': False}}

# === Complement NB Classifier ================================================
# --- list of parameter grids for Complement NB Classifier grid search --------
lParGrid_CpNB = [{'alpha': [0, 0.5, 1., 2.],
                  'fit_prior': [False, True],
                  'norm': [False, True]}]
lParGrid_CpNB = [{'alpha': [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                            0.85, 0.9, 0.95, 1., 1.05, 1.1]}]
# lParGrid_CpNB = [{'alpha': stats.uniform(loc=0., scale=1.),
#                   'fit_prior': [False, True],
#                   'norm': [False, True]}]
# lParGrid_CpNB = [{'alpha': stats.uniform(loc=0.5, scale=0.45)}]

# --- parameter dictionary for Complement NB Classifier -----------------------
d2Par_CpNB = {GC.S_0: {'alpha': 1.,
                       'fit_prior': True,
                       'class_prior': None,
                       'norm': False},
              'A': {'alpha': 1.,
                    'fit_prior': False}}

# === Gaussian NB Classifier ==================================================
# --- list of parameter grids for Gaussian NB Classifier grid search ----------
lParGrid_GsNB = [{'var_smoothing': [1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8]}]
lParGrid_GsNB = [{'var_smoothing': stats.loguniform(a=8.0e-2, b=2.0)}]

# --- parameter dictionary for Gaussian NB Classifier -------------------------
d2Par_GsNB = {GC.S_0: {'priors': None,
                       'var_smoothing': 1.0e-9},
              'A': {'var_smoothing': 0.3}}

# === neural network MLP Classifier ===========================================
# --- list of parameter grids for neural network MLP Classifier grid search ---
lParGrid_MLP = [{'hidden_layer_sizes': [(100,), (1024, 256, 64, 16)],
                 'activation': ['relu', 'identity', 'logistic', 'tanh'],
                 'solver': ['adam', 'lbfgs', 'sgd'],
                 'learning_rate': ['constant', 'adaptive'],
                 'momentum': [0.6, 0.9, 0.98]}]
lParGrid_MLP = [{'hidden_layer_sizes': [(10,), (100,), (1024, 256, 64, 16)],
                 'activation': ['relu', 'identity', 'logistic', 'tanh'],
                 'solver': ['adam', 'lbfgs', 'sgd']
                 # ,
                 # 'learning_rate': ['constant', 'adaptive'],
                 # 'momentum': stats.uniform(loc=0.1, scale=(0.99 - 0.1))
                 }]

# --- parameter dictionary for neural network MLP Classifier ------------------
d2Par_MLP = {GC.S_0: {'hidden_layer_sizes': (100,),
                      'activation': 'relu',
                      'solver': 'adam',
                      'alpha': 0.0001,
                      'batch_size': 'auto',
                      'learning_rate': 'constant',
                      'learning_rate_init': 0.001,
                      'power_t': 0.5,
                      'max_iter': 200,
                      'shuffle': True,
                      'tol': 1e-4,
                      'momentum': 0.9,
                      'nesterovs_momentum': True,
                      'early_stopping': False,
                      'validation_fraction': 0.1,
                      'beta_1': 0.9,
                      'beta_2': 0.999,
                      'epsilon': 1e-8,
                      'n_iter_no_change': 10,
                      'max_fun': 15000},
             'AH': {'hidden_layer_sizes': (100,),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'batch_size': 'auto',
                    'learning_rate': 'constant',
                    'learning_rate_init': 0.001,
                    'power_t': 0.5,
                    'max_iter': 50000,
                    'shuffle': True,
                    'tol': 1e-6,
                    'momentum': 0.8,
                    'nesterovs_momentum': True,
                    'early_stopping': False,
                    'validation_fraction': 0.1,
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'epsilon': 1e-8,
                    'n_iter_no_change': 10,
                    'max_fun': 15000}}

# === Linear SV Classifier ====================================================
# --- list of parameter grids for Linear SV Classifier grid search ------------
lParGrid_LSV = [{'penalty': ['l1', 'l2'],
                 'loss': ['hinge', 'squared_hinge'],
                 'dual': [False, True],
                 'C': [0.1, 1., 10.],
                 'multi_class': ['ovr', 'crammer_singer'],
                 'intercept_scaling': [0.1, 1., 10.],
                 'class_weight': [None, 'balanced']}]
lParGrid_LSV = [{'penalty': ['l1', 'l2'],
                 'loss': ['hinge', 'squared_hinge'],
                 'C': stats.uniform(loc=0.1, scale=(10. - 0.1)),
                 'intercept_scaling': stats.uniform(loc=0.1,
                                                    scale=(10. - 0.1)),
                 'class_weight': [None, 'balanced']}]

# --- parameter dictionary for Linear SV Classifier ---------------------------
d2Par_LSV = {GC.S_0: {'penalty': 'l2',
                      'loss': 'squared_hinge',
                      'dual': True,
                      'tol': 1e-4,
                      'C': 1.0,
                      'multi_class': 'ovr',
                      'fit_intercept': True,
                      'intercept_scaling': 1,
                      'class_weight': None,
                      'max_iter': 1000}}

# === Nu-Support SV Classifier ================================================
# --- list of parameter grids for Nu-Support SV Classifier grid search --------
lParGrid_NSV = [{'nu': [0.1, 0.5, 1.],
                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                 'degree': [2, 3, 4],
                 'gamma': ['scale', 'auto'],
                 'coef0': [0., 1.],
                 'shrinking': [False, True],
                 'probability': [False, True],
                 'class_weight': [None, 'balanced'],
                 'decision_function_shape': ['ovo', 'ovr'],
                 'break_ties': [False, True]}]
lParGrid_NSV = [{'nu': stats.uniform(loc=0.1, scale=(1. - 0.1)),
                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                 'degree': stats.randint(2, 3 + 1),
                 'gamma': ['scale', 'auto'],
                 'coef0': stats.uniform(loc=0., scale=1.),
                 'shrinking': [False, True],
                 'probability': [False, True],
                 'class_weight': [None, 'balanced'],
                 'decision_function_shape': ['ovo', 'ovr'],
                 'break_ties': [False, True]}]

# --- parameter dictionary for Nu-Support SV Classifier -----------------------
d2Par_NSV = {GC.S_0: {'nu': 0.5,
                      'kernel': 'rbf',
                      'degree': 3,
                      'gamma': 'scale',
                      'shrinking': True,
                      'probability': False,
                      'tol': 1e-3,
                      'class_weight': None,
                      'max_iter': -1,
                      'decision_function_shape': 'ovr',
                      'break_ties': False}}

# === other input =============================================================
# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sMn = GC.S_MEAN
sSD = GC.S_SD
sSEM = GC.S_SEM

# --- sets --------------------------------------------------------------------

# --- lists -------------------------------------------------------------------
lSTp = [sMn, sSD, sSEM]
lSTpOut = [sMn, sSEM]

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

# === derived values and input processing =====================================
if not doParGrid:
    lParGrid_Dy = None
    lParGrid_Ada = None
    lParGrid_RF = None
    lParGrid_XTr = None
    lParGrid_GrB = None
    lParGrid_HGrB = None
    lParGrid_GP = None
    lParGrid_PaA = None
    lParGrid_Pct = None
    lParGrid_SGD = None
    lParGrid_CtNB = None
    lParGrid_CpNB = None
    lParGrid_GsNB = None
    lParGrid_MLP = None
    lParGrid_LSV = None
    lParGrid_NSV = None

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'dNumRep': dNumRep,
       'doParGrid': doParGrid,
       'useKey0': useKey0,
       # === Dummy Classifier
       # --- parameter grid for optimising the Dummy Classifier
       'lParGrid_Dummy': lParGrid_Dy,
       # --- parameter dictionary for Dummy Classifier
       'd2Par_Dummy': d2Par_Dy,
       # === AdaBoost Classifier
       # --- parameter grid for optimising the AdaBoost Classifier
       'lParGrid_Ada': lParGrid_Ada,
       # --- parameter dictionary for AdaBoost Classifier
       'd2Par_Ada': d2Par_Ada,
       # === Random Forest Classifier
       # --- parameter grid for optimising the Random Forest Classifier
       'lParGrid_RF': lParGrid_RF,
       # --- parameter dictionary for Random Forest Classifier
       'd2Par_RF': d2Par_RF,
       # === Extra Trees Classifier
       # --- parameter grid for optimising the Extra Trees Classifier
       'lParGrid_XTr': lParGrid_XTr,
       # --- parameter dictionary for Extra Trees Classifier
       'd2Par_XTr': d2Par_XTr,
       # === Gradient Boosting Classifier
       # --- parameter grid for optimising the Gradient Boosting Classifier
       'lParGrid_GrB': lParGrid_GrB,
       # --- parameter dictionary for Gradient Boosting Classifier
       'd2Par_GrB': d2Par_GrB,
       # === Hist Gradient Boosting Classifier
       # --- parameter grid for optimising the Hist Gradient Boosting Classif.
       'lParGrid_HGrB': lParGrid_HGrB,
       # --- parameter dictionary for Hist Gradient Boosting Classifier
       'd2Par_HGrB': d2Par_HGrB,
       # === Gaussian Process Classifier
       # --- parameter grid for optimising the Gaussian Process Classifier
       'lParGrid_GP': lParGrid_GP,
       # --- parameter dictionary for Gaussian Process Classifier
       'd2Par_GP': d2Par_GP,
       # === Passive Aggressive Classifier
       # --- parameter grid for optimising the Passive Aggressive Classifier
       'lParGrid_PaA': lParGrid_PaA,
       # --- parameter dictionary for Passive Aggressive Classifier
       'd2Par_PaA': d2Par_PaA,
       # === Perceptron Classifier
       # --- parameter grid for optimising the Perceptron Classifier
       'lParGrid_Pct': lParGrid_Pct,
       # --- parameter dictionary for Perceptron Classifier
       'd2Par_Pct': d2Par_Pct,
       # === SGD Classifier
       # --- parameter grid for optimising the SGD Classifier
       'lParGrid_SGD': lParGrid_SGD,
       # --- parameter dictionary for SGD Classifier
       'd2Par_SGD': d2Par_SGD,
       # === Categorical NB Classifier
       # --- parameter grid for optimising the Categorical NB Classifier
       'lParGrid_CtNB': lParGrid_CtNB,
       # --- parameter dictionary for Categorical NB Classifier
       'd2Par_CtNB': d2Par_CtNB,
       # === Complement NB Classifier
       # --- parameter grid for optimising the Complement NB Classifier
       'lParGrid_CpNB': lParGrid_CpNB,
       # --- parameter dictionary for Complement NB Classifier
       'd2Par_CpNB': d2Par_CpNB,
       # === Gaussian NB Classifier
       # --- parameter grid for optimising the Gaussian NB Classifier
       'lParGrid_GsNB': lParGrid_GsNB,
       # --- parameter dictionary for Gaussian NB Classifier
       'd2Par_GsNB': d2Par_GsNB,
       # === neural network MLP Classifier
       # --- parameter grid for optimising the neural network MLP Classifier
       'lParGrid_MLP': lParGrid_MLP,
       # --- parameter dictionary for neural network MLP Classifier
       'd2Par_MLP': d2Par_MLP,
       # === Linear SV Classifier
       # --- parameter grid for optimising the Linear SV Classifier
       'lParGrid_LSV': lParGrid_LSV,
       # --- parameter dictionary for Linear SV Classifier
       'd2Par_LSV': d2Par_LSV,
       # === Nu-Support SV Classifier
       # --- parameter grid for optimising the Nu-Support SV Classifier
       'lParGrid_NSV': lParGrid_NSV,
       # --- parameter dictionary for Nu-Support SV Classifier
       'd2Par_NSV': d2Par_NSV,
       # --- parameter dictionary for all Classifier methods
       'd3Par': {GC.S_MTH_DUMMY: d2Par_Dy,
                 GC.S_MTH_ADA: d2Par_Ada,
                 GC.S_MTH_RF: d2Par_RF,
                 GC.S_MTH_X_TR: d2Par_XTr,
                 GC.S_MTH_GR_B: d2Par_GrB,
                 GC.S_MTH_H_GR_B: d2Par_HGrB,
                 GC.S_MTH_GP: d2Par_GP,
                 GC.S_MTH_PA_A: d2Par_PaA,
                 GC.S_MTH_PCT: d2Par_Pct,
                 GC.S_MTH_SGD: d2Par_SGD,
                 GC.S_MTH_CT_NB: d2Par_CtNB,
                 GC.S_MTH_CP_NB: d2Par_CpNB,
                 GC.S_MTH_GS_NB: d2Par_GsNB,
                 GC.S_MTH_MLP: d2Par_MLP,
                 GC.S_MTH_LSV: d2Par_LSV,
                 GC.S_MTH_NSV: d2Par_NSV},
       # === other input
       # --- numbers
       # --- strings
       'sMn': sMn,
       'sSD': sSD,
       'sSEM': sSEM,
       # --- sets
       # --- lists
       'lSTp': lSTp,
       'lSTpOut': lSTpOut
       # --- dictionaries
       # === derived values and input processing
       }

###############################################################################