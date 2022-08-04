# -*- coding: utf-8 -*-
###############################################################################
# --- D_80__Looper.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Data for looper over parameter sets and repetitions (D_80__Looper)'
sNmSpec = 'Input data for the Looper class in O_80__Looper'

# --- flow control ------------------------------------------------------------
dNumRep = {GC.S_MTH_RF: 2,
           GC.S_MTH_MLP: 0}

# --- names and paths of files and dirs ---------------------------------------

# --- list of parameter grids for random forest classifier grid search --------
# or lParGrid_RF = None if no such search should be performed
lParGrid_RF = [{'n_estimators': [100, 1000],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'bootstrap': [True, False]},
               {'max_features': [None, 'sqrt'],
                'ccp_alpha': [0.0, 0.1]}]
lParGrid_RF = None

# --- parameter dictionary for random forest classifier -----------------------
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
            # 'C': {'n_estimators': 100,
            #       'criterion': 'log_loss',
            #       'max_depth': None,
            #       'min_samples_split': 2,
            #       'min_samples_leaf': 1,
            #       'min_weight_fraction_leaf': 0.0,
            #       'max_features': 'sqrt',
            #       'max_leaf_nodes': None,
            #       'min_impurity_decrease': 0.0,
            #       'bootstrap': True,
            #       'ccp_alpha': 0.0,
            #       'max_samples': None},
            # 'I': {'n_estimators': 1000,
            #       'criterion': 'entropy',
            #       'max_depth': None,
            #       'min_samples_split': 2,
            #       'min_samples_leaf': 1,
            #       'min_weight_fraction_leaf': 0.0,
            #       'max_features': None,
            #       'max_leaf_nodes': None,
            #       'min_impurity_decrease': 0.0,
            #       'bootstrap': True,
            #       'ccp_alpha': 0.0,
            #       'max_samples': None},
            # 'P': {'n_estimators': 1000,
            #       'criterion': 'entropy',
            #       'max_depth': None,
            #       'min_samples_split': 2,
            #       'min_samples_leaf': 1,
            #       'min_weight_fraction_leaf': 0.0,
            #       'max_features': 'sqrt',
            #       'max_leaf_nodes': None,
            #       'min_impurity_decrease': 0.0,
            #       'bootstrap': False,
            #       'ccp_alpha': 0.0,
            #       'max_samples': None},
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

# --- list of parameter grids for neural network MLP classifier grid search ---
# or lParGrid_MLP = None if no such search should be performed
lParGrid_MLP = [{'hidden_layer_sizes': [(100,), (1024, 256, 64, 16)],
                 'activation': ['relu', 'identity', 'logistic', 'tanh'],
                 'solver': ['adam', 'lbfgs', 'sgd'],
                 'learning_rate': ['constant', 'adaptive'],
                 'momentum': [0.6, 0.9, 0.98]}]
lParGrid_MLP = None

# --- parameter dictionary for neural network MLP classifier ------------------
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
                    'tol': 1e-4,
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
       # --- parameter grid for optimising the random forest classifier
       'lParGrid_RF': lParGrid_RF,
       # --- parameter dictionary for random forest classifier
       'd2Par_RF': d2Par_RF,
       # --- parameter grid for optimising the neural network MLP classifier
       'lParGrid_MLP': lParGrid_MLP,
       # --- parameter dictionary for neural network MLP classifier
       'd2Par_MLP': d2Par_MLP,
       # --- parameter dictionary for all classifier methods
       'd3Par': {GC.S_MTH_RF: d2Par_RF,
                 GC.S_MTH_MLP: d2Par_MLP},
       # --- names and paths of files and dirs
       # --- input for random forest classifier
       # --- input for neural network MLP classifier
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