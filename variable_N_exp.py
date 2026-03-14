# %%
import torch
from matplotlib import pyplot as plt
import sys
import time
import os
import numpy as np
import math

#####################################################
# This is almost identical to simple demonstration 
# -- except covariates have a skewed covariance matrix
#
# In this notebook, we train a 3-layer linear transformer with
# - context-length 20
# - covariate dimension 5, standard Gaussian distribution
# We plot
# - test loss against number of iterations
# - imshow of each parameter matrix at end of training
# - distance-to-identity of each parameter matrix
#####################################################

#use cuda if available, else use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
# import the model and some useful functions
from linear_transformer import Transformer_F, attention, generate_data, in_context_loss, generate_data_inplace

# set up some print options
np.set_printoptions(precision = 2, suppress = True)
torch.set_printoptions(precision=2)

#begin logging
cur_dir = 'log' 
os.makedirs(cur_dir, exist_ok=True)
#f = open(cur_dir + '/rotation.log', "a", 1)
#sys.stdout = f

# %%
# Set up problem parameters

lr = 0
clip_r = 0.001
alg = 'adam'
mode = 'normal'

n_layer = 3  # number of layers of transformer
d = 5        # dimension


n_head = 1  # 1-headed attention
B = 40000  # 1000 minibatch size
var = 0.0001  # initializations scale of transformer parameter
shape_k = 0.1  # shape_k: parameter for Gamma distributed covariates
max_iters = 8000  # Number of Iterations to run
hist_stride = 1  # stride for saved model paramters in `train.ipynb'
stride = 100

# a convenience function for taking a step and clipping
def clip_and_step(allparam, optimizer, clip_r = None):
    norm_p=None
    grad_all = allparam.grad
    if clip_r is not None:
        for l in range(grad_all.shape[0]):
            for h in range(grad_all.shape[1]):
                for t in range(grad_all.shape[2]):
                    norm_p = grad_all[l,h,t,:,:].norm().item()
                    if norm_p > clip_r:
                        grad_all[l,h,t,:,:].mul_(clip_r/norm_p)
    optimizer.step()
    return norm_p

filename_format = '/variable_N_hist_{}_{}_{}.pth'
Ns = range(20,21,2)     # context length
seeds=[0,1,2,3,4]
keys = []
for s in seeds:
    for N in Ns:
        keys.append((s,N,))

# %%
for key in keys:
    sd = key[0]
    N = key[1]
    filename = cur_dir + filename_format.format(n_layer, N, sd)
    print(key)
    
    prob_seed = sd
    opt_seed = sd
    
    hist = []
    
    #set seed and initialize model
    torch.manual_seed(opt_seed)
    
    model = Transformer_F(n_layer, 1, d, var)
    model.to(device)
    #initialize algorithm. Important: set beta = 0.9 for adam, 0.999 is very slow
    
    
    if N < 5:
        lr = 0.001
    elif N < 15:
        lr = 0.01
    else:
        lr = 0.01
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.99, 0.99), weight_decay=0)
    
    # set seed
    # sample random rotation matrix
    # initialize initial training batch
    np.random.seed(prob_seed)
    torch.manual_seed(prob_seed)
    gaus = torch.FloatTensor(5,5).uniform_(-1,1).to(device)
    U = torch.linalg.svd (gaus)[0].to(device)
    D = torch.diag(torch.FloatTensor([1,1,1/2,1/4,1])).to(device)
    
    #  generate a SINGLE BATCH of training set USED FOREVER
    Z, y = generate_data(mode,N,d,B,shape_k, U, D)
    Z = Z.to(device)
    y = y.to(device)
    for t in range(max_iters):
        start = time.time()
        if t%2000==0 and t>1:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] *0.5
        Z,y = generate_data_inplace(Z, U=U, D=D)
        start = time.time()
        # save model parameters
        if t%stride ==0:
            hist.append(model.allparam.clone().detach())
        loss = in_context_loss(model, Z, y)
        # compute gradient, take step
        loss.backward()
        norms = clip_and_step(model.allparam, optimizer, clip_r=clip_r)
        optimizer.zero_grad()
        end=time.time()
        if t%500 ==0 or t<5:
            print('iter {} | Loss: {}  time: {}  gradnorm: {}'.format(t,loss.item(), end-start, norms))
    torch.save({'hist':hist, 'U':U, 'D':D}, filename)

# %%
####################################
# compute test loss
####################################
#hist_dict = torch.load(filename)['hist_dict']
keys = []
seeds = [0,1,2,3,4]
Ns = range(2,21,2)
for s in seeds:
    for N in Ns:
        keys.append((s,N,))
loss_dict = {}
for sd in seeds:
    key = (sd,)
    loss_dict[key] = torch.zeros(len(Ns))
    for N in Ns:
        filename = cur_dir + filename_format.format(n_layer, N, sd)
        hist = torch.load(filename)['hist']
        U = torch.load(filename)['U']
        D = torch.load(filename)['D']
        np.random.seed(999)
        torch.manual_seed(999)
        Z, y = generate_data(mode,N,d,B,shape_k,U,D)
        Z = Z.to(device)
        y = y.to(device)
        model = Transformer_F(n_layer, n_head, d, var).to(device)
        loss = 100
        bestmodel = None
        for t in range(len(hist)-10, len(hist)):
            with torch.no_grad():
                model.allparam.copy_(hist[t])
            newloss = in_context_loss(model, Z, y).item()
            if (newloss < loss):
                loss=newloss
                bestmodel = hist[t]
        with torch.no_grad():
            model.allparam.copy_(bestmodel)
        np.random.seed(99)
        torch.manual_seed(99)
        Z, y = generate_data(mode,N,d,B,shape_k,U,D)
        Z = Z.to(device)
        y = y.to(device)    
        loss_dict[key][N//2-1] = in_context_loss(model, Z, y).item()

# %%
########################################################
# plot log final test loss against N, for sanity check
########################################################

fig_dir = 'figures' 
os.makedirs(fig_dir, exist_ok=True)

fig, ax = plt.subplots(1, 1,figsize = (9, 9))

losses = torch.zeros(len(seeds), len(Ns))
keys = loss_dict.keys()
for idx, key in enumerate(keys):
    losses[idx,:] = np.log(loss_dict[key])
losses_mean = torch.mean(losses, axis=0)
losses_std = torch.std(losses, axis=0)
ax.plot(Ns, losses_mean, color = 'red', lw = 3, label='3-Layer Linear Transformer')
ax.fill_between(Ns, losses_mean-losses_std, losses_mean+losses_std, color = 'red', alpha = 0.2)

# %%
def do_gd(Z,eta,numstep):
    N = Z.shape[0]-1
    X = Z[0:N-1,0:5]
    Y = Z[0:N-1,5]
    w = torch.zeros(X.shape[1]).to(device)
    for k in range(numstep):
        XTXw = torch.einsum('ik,ij,j->k',X,X,w)
        XTY = torch.einsum('ik,i->k',X,Y)
        grad = XTXw - XTY
        w = w - eta * grad
    return w

def eval_w_instance(Z, Ytest, w):
    N = Z.shape[0]-1
    Xtest = Z[N,0:5]
    prediction = torch.einsum('i,i->',w,Xtest)
    return (Ytest - prediction)**2, prediction


## code for running 3-step GD loss
gd_loss_matrix = torch.zeros(len(seeds),10)
#for seed in seeds:
#    gd_loss_matrix.append([None]*10)
    
for N in Ns:
    #find best eta
    #load seed 1 just to find eta
    sd = 1
    best_loss = 10000
    best_eta = 0
    numstep = 3
    
    filename = cur_dir + filename_format.format(n_layer, N, sd)
    U = torch.load(filename)['U']
    D = torch.load(filename)['D']
    #generate test data
    np.random.seed(999)
    torch.manual_seed(999)
    Z, y = generate_data(mode,N,d,B,shape_k,U,D)
    Z = Z.to(device)
    y = y.to(device)
    #done generating data 
    
    for eta in [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.16]:
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(5000):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_gd(Zi,eta,numstep)
            gd_loss, gd_pred = eval_w_instance(Zi, Ytesti, w)
            total_loss = total_loss + gd_loss
        mean_loss = total_loss / 5000
        ### end of evaluate mean loss ###
        print('eta: {}, loss: {}'.format(eta, mean_loss))
        if (mean_loss < best_loss):
            best_eta = eta
            best_loss = mean_loss
    print('best eta: {} for N={}'.format(best_eta, N))
    
    #now do actual evaluation
    for sd in seeds:
        opt_seed = sd
        
        filename = cur_dir + filename_format.format(n_layer, N, sd)
        U = torch.load(filename)['U']
        D = torch.load(filename)['D']
        #generate test data
        torch.manual_seed(sd)
        Z, y = generate_data(mode,N,d,B,shape_k,U,D)
        Z = Z.to(device)
        y = y.to(device)
        #done generating data 
        eta = best_eta
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(5000):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_gd(Zi,eta,numstep)
            gd_loss, gd_pred = eval_w_instance(Zi, Ytesti, w)
            total_loss = total_loss + gd_loss
        mean_loss = total_loss / 5000
        gd_loss_matrix[sd,int(N/2-1)] = mean_loss
        
gd_loss_mean = gd_loss_matrix.mean(dim=0)
gd_loss_std = gd_loss_matrix.var(dim=0)**0.5        

# %%
def do_preconditioned_gd(Z,eta,numstep,U,D):
    N = Z.shape[0]-1
    X = Z[0:N-1,0:5]
    Y = Z[0:N-1,5]
    w = torch.zeros(X.shape[1]).to(device)
    X = torch.einsum('ij, jk, Nk -> Ni', (torch.inverse(D),U.t(),X))
    for k in range(numstep):
        XTXw = torch.einsum('ik,ij,j->k',X,X,w)
        XTY = torch.einsum('ik,i->k',X,Y)
        grad = XTXw - XTY
        w = w - eta * grad
    return w

def eval_w_instance_precon(Z, Ytest, w, U, D):
    N = Z.shape[0]-1
    Xtest = Z[N,0:5]
    Xtest = torch.einsum('ij, jk, k -> i', (torch.inverse(D),U.t(),Xtest))
    prediction = torch.einsum('i,i->',w,Xtest)
    return (Ytest - prediction)**2, prediction


## code for running 3-step GD loss
pgd_loss_matrix = torch.zeros(len(seeds),10)
#for seed in seeds:
#    gd_loss_matrix.append([None]*10)
    
for N in Ns:
    #find best eta
    #load seed 1 just to find eta
    best_loss = 10000
    best_eta = 0
    numstep = 3
    
    filename = cur_dir + filename_format.format(n_layer, N, sd)
    U = torch.load(filename)['U'].to(device)
    D = torch.load(filename)['D'].to(device)
    #generate test data
    np.random.seed(999)
    torch.manual_seed(999)
    Z, y = generate_data(mode,N,d,B,shape_k,U,D)
    Z = Z.to(device)
    y = y.to(device)
    #done generating data 
    
    for eta in [0.008, 0.01, 0.02, 0.04, 0.08, 0.16]:
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(5000):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_preconditioned_gd(Zi,eta,numstep,U,D)
            pgd_loss, pgd_pred = eval_w_instance_precon(Zi, Ytesti, w, U, D)
            total_loss = total_loss + pgd_loss
        mean_loss = total_loss / 5000
        ### end of evaluate mean loss ###
        print('eta: {}, loss: {}'.format(eta, mean_loss))
        if (mean_loss < best_loss):
            best_eta = eta
            best_loss = mean_loss
    print('best eta: {} for N={}'.format(best_eta, N))
    
    #now do actual evaluation
    for sd in seeds:
        opt_seed = sd
        
        filename = cur_dir + filename_format.format(n_layer, N, sd)
        U = torch.load(filename)['U'].to(device)
        D = torch.load(filename)['D'].to(device)
        #generate test data
        torch.manual_seed(sd)
        Z, y = generate_data(mode,N,d,B,shape_k,U,D)
        Z = Z.to(device)
        y = y.to(device)
        #done generating data 
        eta = best_eta
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(5000):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_preconditioned_gd(Zi,eta,numstep,U,D)
            pgd_loss, pgd_pred = eval_w_instance_precon(Zi, Ytesti, w, U, D)
            total_loss = total_loss + pgd_loss
        mean_loss = total_loss / 5000
        pgd_loss_matrix[sd,int(N/2-1)] = mean_loss
        
pgd_loss_mean = pgd_loss_matrix.mean(dim=0)
pgd_loss_std = pgd_loss_matrix.var(dim=0)**0.5

# %%
def do_OLS(Z,eta,numstep,U,D):
    N = Z.shape[0]-1
    X = Z[0:N-1,0:5]
    Y = Z[0:N-1,5]
    w = torch.zeros(X.shape[1])
    X = torch.einsum('ij, jk, Nk -> Ni', (torch.inverse(D),U.t(),X))
    for k in range(numstep):
        XTX = torch.einsum('ik,ij->kj',X,X)
        XTY = torch.einsum('ik,i->k',X,Y)
        w = torch.einsum('ik,k->i', torch.linalg.pinv(XTX) , XTY)
    return w

def eval_w_instance_OLS(Z, Ytest, w,U,D):
    N = Z.shape[0]-1
    Xtest = Z[N,0:5]
    Xtest = torch.einsum('ij, jk, k -> i', (torch.inverse(D),U.t(),Xtest))
    prediction = torch.einsum('i,i->',w,Xtest)
    return (Ytest - prediction)**2, prediction


## code for running 3-step GD loss
OLS_loss_matrix = torch.zeros(len(seeds),10)
#for seed in seeds:
#    gd_loss_matrix.append([None]*10)
    
for N in Ns:
    #now do actual evaluation
    for sd in seeds:
        opt_seed = sd
        filename = cur_dir + filename_format.format(n_layer, N, sd)
        U = torch.load(filename)['U'].to(device)
        D = torch.load(filename)['D'].to(device)
        #generate test data
        torch.manual_seed(sd)
        Z, y = generate_data(mode,N,d,B,shape_k,U,D)
        Z = Z.to(device)
        y = y.to(device)
        #done generating data 
        eta = best_eta
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(5000):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_OLS(Zi,eta,numstep,U,D)
            OLS_loss, OLS_pred = eval_w_instance_OLS(Zi, Ytesti, w, U, D)
            total_loss = total_loss + OLS_loss
        mean_loss = total_loss / 5000
        OLS_loss_matrix[sd,int(N/2-1)] = mean_loss
        print('N={}, loss={}'.format(N,mean_loss))
        
ols_loss_mean = OLS_loss_matrix.mean(dim=0)
ols_loss_std = OLS_loss_matrix.var(dim=0)**0.5

# %%
####################################
# plot final test loss against N
####################################

fig_dir = 'figures' 
os.makedirs(fig_dir, exist_ok=True)

fig, ax = plt.subplots(1, 1,figsize = (9, 9))

losses = torch.zeros(len(seeds), len(Ns))
keys = loss_dict.keys()
for idx, key in enumerate(keys):
    losses[idx,:] = loss_dict[key]
losses_mean = torch.mean(losses, axis=0)
losses_std = torch.std(losses, axis=0)

plt.plot(Ns, gd_loss_mean, color='blue', label='3-Step GD')
plt.fill_between(Ns, gd_loss_mean - gd_loss_std, gd_loss_mean + gd_loss_std, color='blue', alpha=0.2)
plt.plot(Ns, pgd_loss_mean, color='green', label='3-Step Preconditioned GD')
plt.fill_between(Ns, pgd_loss_mean - pgd_loss_std, pgd_loss_mean + pgd_loss_std, color='green', alpha=0.2)
ax.plot(Ns, losses_mean, color = 'red', lw = 3, label='3-Layer Linear Transformer')
ax.fill_between(Ns, losses_mean-losses_std, losses_mean+losses_std, color = 'red', alpha = 0.2)
plt.plot(Ns, ols_loss_mean, color='purple', label='OLS')
plt.fill_between(Ns, ols_loss_mean - ols_loss_std, ols_loss_mean + ols_loss_std, color='purple', alpha=0.2)

plt.ylabel('Loss',fontsize=30)
plt.xlabel('Number of ICL Examples',fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30, width = 3, length = 10)
ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
plt.xticks(np.arange(2,24, 4))
ax.legend(fontsize=24)
#ax.set_yscale('log')


plt.tight_layout()
plt.savefig(fig_dir + '/3-step-variable-N-plot.pdf', dpi=600)


