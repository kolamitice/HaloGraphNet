#----------------------------------------------------------------------
# List of constants and some common functions
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import numpy as np
import torch
import os
import random

# Random seeds
torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

#--- PARAMETERS AND CONSTANTS ---#

# Reduced Hubble constant
hred = 0.7

# Root path for simulations
simpathroot = "/projects/QUIJOTE/CAMELS/Sims/"

# Box size in comoving kpc/h
boxsize = 25.e3

# Validation and test size
valid_size, test_size = 0.15, 0.15

# Batch size
batch_size = 128

# 1 if train for performing symbolic regression later, 0 otherwise
sym_reg = 0

# 1 if use L1 regularization with messages. Needed for symbolic regression
use_l1 = 0

# Weight of the message L1 regularization in the total loss respect to the standard loss (used for symbolic regression)
l1_reg = 0.01

#--- FUNCTIONS ---#

# Name of the model and hyperparameters
def namemodel(params):
    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims = params
    return simsuite+"_"+simset+"_model_"+use_model+"_lr_{:.2e}_weightdecay_{:.2e}_layers_{:d}_knn_{:.2e}_epochs_{:d}".format(learning_rate, weight_decay, n_layers, k_nn, n_epochs)

# Change to the other CAMELS simulation suite
def changesuite(suite):
    if suite=="IllustrisTNG":
        newsuite = "SIMBA"
    elif suite=="SIMBA":
        newsuite = "IllustrisTNG"
    return newsuite

# Choose color depending on the CAMELS simulation suite
def colorsuite(suite):
    if suite=="IllustrisTNG":   return "purple"
    elif suite=="SIMBA":            return "deepskyblue"
