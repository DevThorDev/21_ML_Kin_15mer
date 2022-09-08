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
dNumRep = {GC.S_MTH_DUMMY: 0,
           GC.S_MTH_ADA: 0,
           GC.S_MTH_RF: 0,
           GC.S_MTH_X_TR: 0,
           GC.S_MTH_GR_B: 0,
           GC.S_MTH_H_GR_B: 0,
           GC.S_MTH_GP: 0,
           GC.S_MTH_PA_A: 1,
           GC.S_MTH_PCT: 0,
           GC.S_MTH_SGD: 0,
           GC.S_MTH_CT_NB: 0,
           GC.S_MTH_CP_NB: 0,
           GC.S_MTH_GS_NB: 0,
           GC.S_MTH_MLP: 0,
           GC.S_MTH_LSV: 0,
           GC.S_MTH_NSV: 0}

# === Dummy Classifier ========================================================
# --- list of parameter grids for Dummy Classifier grid search ----------------
# or lParGrid_Dy = None if no such search should be performed
lParGrid_Dy = None

# --- parameter dictionary for Dummy Classifier -------------------------------
d2Par_Dy = {'A': {'strategy': 'uniform',
                  'constant': None}
            # 'B': {'strategy': 'stratified',
            #       'constant': None}
            }

# === AdaBoost Classifier =====================================================
# --- list of parameter grids for AdaBoost Classifier grid search -------------
# or lParGrid_Ada = None if no such search should be performed
# lParGrid_Ada = [{'n_estimators': [100, 1000],
#                  'learning_rate': [0.5, 1.0, 2.0],
#                  'algorithm': ['SAMME', 'SAMME.R']}]
lParGrid_Ada = [{'learning_rate': [0.5, 1.0, 2.0],
                 'algorithm': ['SAMME', 'SAMME.R']}]
lParGrid_Ada = [{'learning_rate': stats.uniform(0.0, 2.0),
                 'algorithm': ['SAMME', 'SAMME.R']}]
# lParGrid_Ada = None

# --- parameter dictionary for AdaBoost Classifier ----------------------------
d2Par_Ada = {'A': {'n_estimators': 100,
                   'learning_rate': 1,
                   'algorithm': 'SAMME.R'},
             # 'B': {'n_estimators': 200,
             #       'learning_rate': 0.5,
             #       'algorithm': 'SAMME.R'}
             }

# === Random Forest Classifier ================================================
# --- list of parameter grids for Random Forest Classifier grid search --------
# or lParGrid_RF = None if no such search should be performed
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
                'ccp_alpha': stats.uniform(0.0, 0.5)},
               {'max_features': [None, 'sqrt']}]
# lParGrid_RF = None

# --- parameter dictionary for Random Forest Classifier -----------------------
d2Par_RF = {'B': {'n_estimators': 100,
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
                  'max_samples': None}
            # 'S': {'n_estimators': 1000,
            #       'criterion': 'log_loss',
            #       'max_depth': None,
            #       'min_samples_split': 2,
            #       'min_samples_leaf': 1,
            #       'min_weight_fraction_leaf': 0.0,
            #       'max_features': None,
            #       'max_leaf_nodes': None,
            #       'min_impurity_decrease': 0.0,
            #       'bootstrap': True,
            #       'ccp_alpha': 0.0,
            #       'max_samples': None}
            }

# === Extra Trees Classifier ==================================================
# --- list of parameter grids for Extra Trees Classifier grid search ----------
# or lParGrid_XTr = None if no such search should be performed
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
                 'ccp_alpha': stats.uniform(0.0, 0.5)},
                {'max_features': [None, 'sqrt']}]
# lParGrid_XTr = None

# --- parameter dictionary for Extra Trees Classifier -------------------------
d2Par_XTr = {'A': {'n_estimators': 100,
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
                   'max_samples': None}
             }

# === Gradient Boosting Classifier ============================================
# --- list of parameter grids for Gradient Boosting Classifier grid search ----
# or lParGrid_GrB = None if no such search should be performed
lParGrid_GrB = [{'loss': ['log_loss', 'exponential'],
                 'learning_rate': [0.1, 0.5],
                 'n_estimators': [100, 1000],
                 'subsample': [0.5, 1.0],
                 'criterion': ['friedman_mse', 'squared_error', 'mse']},
                {'max_features': [None, 'sqrt'],
                 'ccp_alpha': [0.0, 0.1]}]
lParGrid_GrB = [{'loss': ['log_loss', 'exponential'],
                 'learning_rate': stats.uniform(0.0, 1.0),
                 'ccp_alpha': stats.uniform(0.0, 0.5),
                 'n_estimators': [100, 1000],
                 'subsample': stats.uniform(0.1, 1.0),
                 'criterion': ['friedman_mse', 'squared_error', 'mse']}]
lParGrid_GrB = None

# --- parameter dictionary for Gradient Boosting Classifier -------------------
d2Par_GrB = {'A': {'loss': 'log_loss',
                   'learning_rate': 0.1,
                   'n_estimators': 100,
                   'subsample': 1.0,
                   'criterion': 'friedman_mse',
                   'max_depth': 3,
                   'min_samples_split': 2,
                   'min_samples_leaf': 1,
                   'min_weight_fraction_leaf': 0.0,
                   'max_features': None,
                   'max_leaf_nodes': None,
                   'min_impurity_decrease': 0.0,
                   'ccp_alpha': 0.0}
             }

# === Hist Gradient Boosting Classifier =======================================
# --- list of parameter grids for Hist Gradient Boosting Classifier grid search
# or lParGrid_HGrB = None if no such search should be performed
lParGrid_HGrB = [{'learning_rate': [0.1, 0.5],
                  'max_leaf_nodes': [10, 31, None],
                  'min_samples_leaf': [5, 20],
                  'max_bins': [63, 255]}]
# lParGrid_HGrB = [{'loss': ['log_loss', 'auto'],
#                   'learning_rate': [0.1, 0.5],
#                   'max_iter': [100, 500],
#                   'max_leaf_nodes': [10, 31, None],
#                   'max_depth': [5, None],
#                   'min_samples_leaf': [5, 20],
#                   'l2_regularization': [0, 0.5],
#                   'max_bins': [63, 255]}]
lParGrid_HGrB = [{'loss': ['log_loss', 'auto'],
                  'learning_rate': stats.uniform(0.0, 1.0),
                  'max_iter': stats.randint(10, 1000 + 1),
                  'max_leaf_nodes': [10, 31, None],
                  'max_depth': [5, None],
                  'min_samples_leaf': stats.randint(5, 20 + 1),
                  'l2_regularization': stats.uniform(0.0, 1.0),
                  'max_bins': [63, 255]}]
# lParGrid_HGrB = None

# --- parameter dictionary for Hist Gradient Boosting Classifier --------------
d2Par_HGrB = {'A': {'loss': 'log_loss',
                    'learning_rate': 0.1,
                    'max_iter': 100,
                    'max_leaf_nodes': None,
                    'max_depth': None,
                    'min_samples_leaf': 1,
                    'l2_regularization': 0,
                    'max_bins': 255,
                    'categorical_features': [1],
                    'monotonic_cst': None,
                    'early_stopping': False,
                    'scoring': None,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': 10,
                    'tol': 1.0e-7}
              }

# === Gaussian Process Classifier =============================================
# --- list of parameter grids for Gaussian Process Classifier grid search -----
# or lParGrid_GP = None if no such search should be performed
lParGrid_GP = [{'n_restarts_optimizer': [0, 1, 10],
                'max_iter_predict': [10, 100, 1000],
                'multi_class': ['one_vs_rest', 'one_vs_one']}]
lParGrid_GP = [{'n_restarts_optimizer': stats.randint(0, 10 + 1),
                'max_iter_predict': stats.randint(10, 1000 + 1),
                'multi_class': ['one_vs_rest', 'one_vs_one']}]
lParGrid_GP = None

# --- parameter dictionary for Gaussian Process Classifier --------------------
d2Par_GP = {'A': {'kernel': None,
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
# or lParGrid_PaA = None if no such search should be performed
lParGrid_PaA = [{'C': [0.1, 1.0, 10.0],
                 'loss': ['hinge', 'squared_hinge'],
                 'class_weight': [None, 'balanced']}]
lParGrid_PaA = [{'C': stats.uniform(0.01, 10.0),
                 'loss': ['hinge', 'squared_hinge'],
                 'class_weight': [None, 'balanced']}]
# lParGrid_PaA = None

# --- parameter dictionary for Passive Aggressive Classifier --------------------
d2Par_PaA = {'A': {'C': 1.0,
                   'max_iter': 1000,
                   'tol': 1.0e-3,
                   'early_stopping': False,
                   'validation_fraction': 0.1,
                   'n_iter_no_change': 5,
                   'shuffle': True,
                   'loss': 'hinge',
                   'class_weight': None,
                   'average': False}}

# === neural network MLP Classifier ===========================================
# --- list of parameter grids for neural network MLP Classifier grid search ---
# or lParGrid_MLP = None if no such search should be performed
lParGrid_MLP = [{'hidden_layer_sizes': [(100,), (1024, 256, 64, 16)],
                 'activation': ['relu', 'identity', 'logistic', 'tanh'],
                 'solver': ['adam', 'lbfgs', 'sgd'],
                 'learning_rate': ['constant', 'adaptive'],
                 'momentum': [0.6, 0.9, 0.98]}]
lParGrid_MLP = [{'hidden_layer_sizes': [(100,), (1024, 256, 64, 16)],
                 'activation': ['relu', 'identity', 'logistic', 'tanh'],
                 'solver': ['adam', 'lbfgs', 'sgd'],
                 'learning_rate': ['constant', 'adaptive'],
                 'momentum': stats.uniform(0.1, 0.99)}]
lParGrid_MLP = None

# --- parameter dictionary for neural network MLP Classifier ------------------
d2Par_MLP = {'AH': {'hidden_layer_sizes': (100,),
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
                    'max_fun': 15000},
                # 'AJ': {'hidden_layer_sizes': (100,),
                #       'activation': 'tanh',
                #       'solver': 'adam',
                #       'alpha': 0.0001,
                #       'batch_size': 'auto',
                #       'learning_rate': 'constant',
                #       'learning_rate_init': 0.001,
                #       'power_t': 0.5,
                #       'max_iter': 50000,
                #       'shuffle': True,
                #       'tol': 1e-4,
                #       'momentum': 0.8,
                #       'nesterovs_momentum': True,
                #       'early_stopping': False,
                #       'validation_fraction': 0.1,
                #       'beta_1': 0.9,
                #       'beta_2': 0.999,
                #       'epsilon': 1e-8,
                #       'n_iter_no_change': 10,
                #       'max_fun': 15000}
               }

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

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'dNumRep': dNumRep,
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
       # --- parameter grid for optimising the Extra Trees Classifier
       'lParGrid_XTr': lParGrid_XTr,
       # --- parameter dictionary for Extra Trees Classifier
       'd2Par_XTr': d2Par_XTr,
       # --- parameter grid for optimising the Gradient Boosting Classifier
       'lParGrid_GrB': lParGrid_GrB,
       # --- parameter dictionary for Gradient Boosting Classifier
       'd2Par_GrB': d2Par_GrB,
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
       # === neural network MLP Classifier
       # --- parameter grid for optimising the neural network MLP Classifier
       'lParGrid_MLP': lParGrid_MLP,
       # --- parameter dictionary for neural network MLP Classifier
       'd2Par_MLP': d2Par_MLP,
       # --- parameter dictionary for all Classifier methods
       'd3Par': {GC.S_MTH_DUMMY: d2Par_Dy,
                 GC.S_MTH_ADA: d2Par_Ada,
                 GC.S_MTH_RF: d2Par_RF,
                 GC.S_MTH_X_TR: d2Par_XTr,
                 GC.S_MTH_GR_B: d2Par_GrB,
                 GC.S_MTH_H_GR_B: d2Par_HGrB,
                 GC.S_MTH_GP: d2Par_GP,
                 GC.S_MTH_PA_A: d2Par_PaA,
                 GC.S_MTH_MLP: d2Par_MLP},
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