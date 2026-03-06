# %%
import torch
%matplotlib inline
from matplotlib import pyplot as plt
import math
import torch.nn.functional as F
from torch.nn.functional import relu
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np
import gc
from pylab import *
import os
import random
import json
import pandas as pd
from scipy.stats import norm
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import sys
import matplotlib.pyplot as plt
import time

from linear_transformer import Transformer_F, attention, generate_data, in_context_loss, generate_data_inplace

np.set_printoptions(precision = 4, suppress = True)
torch.set_printoptions(precision=2)
device = torch.device("cuda")
torch.cuda.set_device(0)

# %%
# Set Hyperparameters

# Fixed
n_head = 1
d = 5
B = 1000
var = 0.05
shape_k = 0.1

# Number of Iterations to run
max_iters = 10000
hist_stride = 1

# We vary the following parameters
n_layer = 3
mode = 'normal'
N = 20
seeds = [0,1,2,3,4,5]
algos = ['sgd','adam']
lrs = [0.02]


# %%
# pipe output to log file
log_dir = 'log' 
os.makedirs(log_dir, exist_ok=True)
f = open(log_dir + '/train.log', "a", 1)
sys.stdout = f
filename_format = log_dir + '/train_layer{}_N{}_{}_{}_{}_lr{}_sd{}.pth'

# %%
# one-step update of (non-)clipping algotirthm
def clip_and_step(allparam, optimizer, toclip, clip_threshold = 1.):
    grad_all = allparam.grad
    grad_p = grad_all
    norm_p = grad_p.norm()
    if toclip and norm_p > clip_threshold:
            grad_all.mul_(clip_threshold/norm_p)
    optimizer.step()
    return norm_p.item()

# %%

## Train linear transformer

for alg in algos:
    for toclip in [True]: # True means with clipping, False means without clipping
        for lr in lrs:
            for sd in seeds:
                filename = filename_format.format(n_layer, N, mode, alg, toclip, lr, sd)
                print(filename)
                np.random.seed(sd)
                torch.manual_seed(sd)
                hist_list = list()

                # initialize model paramter
                model = Transformer_F(n_layer, n_head, d, var)
                model.to(device)

                # create optimizer
                if alg == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
                elif alg == 'adam':
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=0)
                else: assert False

                for t in range(max_iters):
                    start = time.time()
                    # save model parameters
                    if t%hist_stride ==0:
                        hist_list.append(model.allparam.clone().detach())

                    #  generate a new batch of training set
                    Z, y = generate_data(mode,N,d,B,shape_k)
                    Z = Z.to(device)
                    y = y.to(device)

                    loss = in_context_loss(model, Z, y)
                    loss_value = loss.item()
                    loss.backward()

                    if mode == 'sphere':
                        clip_threshold = 0.1
                    else:
                        clip_threshold = 1.0

                    # take optimizer step
                    norms = clip_and_step(model.allparam, optimizer,toclip,clip_threshold)
                    optimizer.zero_grad()
                    
                    end=time.time()
                    if t%100 ==0 or t<5:
                        print('iter {} | Loss: {}  time: {}  gradnorm: {}'.format(t,loss_value, end-start, norms))
                
                torch.save({'hist_list':hist_list}, filename)

# %%
sys.stdout = sys.__stdout__
f.close()


