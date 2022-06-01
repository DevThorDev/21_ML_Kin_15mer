# -*- coding: utf-8 -*-
###############################################################################
# --- D_80__Looper.py ---------------------------------------------------------
###############################################################################
import Core.C_00__GenConstants as GC

# --- general -----------------------------------------------------------------
sOType = 'Data for looper over parameter sets and repetitions (D_80__Looper)'
sNmSpec = 'Input data for the Looper class in O_80__Looper'

# --- flow control ------------------------------------------------------------
dNumRep = {GC.S_MTH_RF: 3,
           GC.S_MTH_MLP: 3}

# --- names and paths of files and dirs ---------------------------------------

# --- parameter dictionary for random forest classifier -----------------------
d2Par_RF = {'A': {'n_estimators': 100,
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
                  'max_samples': None},
            'C': {'n_estimators': 100,
                  'criterion': 'log_loss',
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

# --- parameter dictionary for neural network MLP classifier ------------------
d2Par_NNMLP = {'A': {'hidden_layer_sizes': (256, 128, 64, 32),
                     'activation': 'relu',
                     'solver': 'adam',
                     'alpha': 0.0001,
                     'batch_size': 'auto',
                     'learning_rate': 'constant',
                     'learning_rate_init': 0.001,
                     'power_t': 0.5,
                     'max_iter': 500,
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
               'B': {'hidden_layer_sizes': (1024, 256, 64, 16),
                     'activation': 'relu',
                     'solver': 'adam',
                     'alpha': 0.0001,
                     'batch_size': 'auto',
                     'learning_rate': 'constant',
                     'learning_rate_init': 0.001,
                     'power_t': 0.5,
                     'max_iter': 500,
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
               'C': {'hidden_layer_sizes': (100,),
                     'activation': 'relu',
                     'solver': 'adam',
                     'alpha': 0.0001,
                     'batch_size': 'auto',
                     'learning_rate': 'constant',
                     'learning_rate_init': 0.001,
                     'power_t': 0.5,
                     'max_iter': 500,
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
                     'max_fun': 15000}}

# --- numbers -----------------------------------------------------------------

# --- strings -----------------------------------------------------------------
sMn = GC.S_MEAN
sSD = GC.S_SD
sSEM = GC.S_SEM

# --- sets --------------------------------------------------------------------

# --- lists -------------------------------------------------------------------
lSTp = [sMn, sSD, sSEM]

# --- dictionaries ------------------------------------------------------------

# === assertions ==============================================================

# === derived values and input processing =====================================

# === create input dictionary =================================================
dIO = {# --- general
       'sOType': sOType,
       'sNmSpec': sNmSpec,
       # --- flow control
       'dNumRep': dNumRep,
       # --- parameter dictionary for random forest classifier
       'd2Par_RF': d2Par_RF,
       # --- parameter dictionary for neural network MLP classifier
       'd2Par_NNMLP': d2Par_NNMLP,
       # --- parameter dictionary for all classifier methods
       'd3Par': {GC.S_MTH_RF: d2Par_RF,
                 GC.S_MTH_MLP: d2Par_NNMLP},
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
       'lSTp': lSTp
       # --- dictionaries
       # === derived values and input processing
       }

###############################################################################
