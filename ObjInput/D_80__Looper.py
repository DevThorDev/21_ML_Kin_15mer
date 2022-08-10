# -*- coding: utf-8 -*-
###############################################################################
# --- D_80__Looper.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Data for looper over parameter sets and repetitions (D_80__Looper)'
sNmSpec = 'Input data for the Looper class in O_80__Looper'

# --- flow control ------------------------------------------------------------
dNumRep = {GC.S_MTH_DUMMY: 2,
           GC.S_MTH_ADA: 2,
           GC.S_MTH_RF: 2,
           GC.S_MTH_GP: 2,
           GC.S_MTH_MLP: 2}

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
lParGrid_Ada = [{'n_estimators': [100, 1000],
                 'learning_rate': [0.5, 1.0, 2.0],
                 'algorithm': ['SAMME', 'SAMME.R']}]
lParGrid_Ada = None

# --- parameter dictionary for AdaBoost Classifier ----------------------------
d2Par_Ada = {'A': {'n_estimators': 100,
                   'learning_rate': 1,
                   'algorithm': 'SAMME.R'},
             # 'B': {'n_estimators': 200,
             #       'learning_rate': 0.5,
             #       'algorithm': 'SAMME.R'}
             }

# === random forest Classifier ================================================
# --- list of parameter grids for random forest Classifier grid search --------
# or lParGrid_RF = None if no such search should be performed
lParGrid_RF = [{'n_estimators': [100, 1000],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'bootstrap': [True, False]},
               {'max_features': [None, 'sqrt'],
                'ccp_alpha': [0.0, 0.1]}]
lParGrid_RF = None

# --- parameter dictionary for random forest Classifier -----------------------
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

# === Gaussian Process Classifier =============================================
# --- list of parameter grids for Gaussian Process Classifier grid search -----
# or lParGrid_RF = None if no such search should be performed
lParGrid_GP = [{'n_restarts_optimizer': [0, 1, 10],
                'max_iter_predict': [10, 100, 1000],
                'multi_class': ['one_vs_rest', 'one_vs_one']}]
lParGrid_GP = None

# --- parameter dictionary for Gaussian Process Classifier -----------------------
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

# === neural network MLP Classifier ===========================================
# --- list of parameter grids for neural network MLP Classifier grid search ---
# or lParGrid_MLP = None if no such search should be performed
lParGrid_MLP = [{'hidden_layer_sizes': [(100,), (1024, 256, 64, 16)],
                 'activation': ['relu', 'identity', 'logistic', 'tanh'],
                 'solver': ['adam', 'lbfgs', 'sgd'],
                 'learning_rate': ['constant', 'adaptive'],
                 'momentum': [0.6, 0.9, 0.98]}]
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
       # === random forest Classifier
       # --- parameter grid for optimising the random forest Classifier
       'lParGrid_RF': lParGrid_RF,
       # --- parameter dictionary for random forest Classifier
       'd2Par_RF': d2Par_RF,
       # === Gaussian Process Classifier
       # --- parameter grid for optimising the Gaussian Process Classifier
       'lParGrid_GP': lParGrid_GP,
       # --- parameter dictionary for Gaussian Process Classifier
       'd2Par_GP': d2Par_GP,
       # === neural network MLP Classifier
       # --- parameter grid for optimising the neural network MLP Classifier
       'lParGrid_MLP': lParGrid_MLP,
       # --- parameter dictionary for neural network MLP Classifier
       'd2Par_MLP': d2Par_MLP,
       # --- parameter dictionary for all Classifier methods
       'd3Par': {GC.S_MTH_DUMMY: d2Par_Dy,
                 GC.S_MTH_ADA: d2Par_Ada,
                 GC.S_MTH_RF: d2Par_RF,
                 GC.S_MTH_GP: d2Par_GP,
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