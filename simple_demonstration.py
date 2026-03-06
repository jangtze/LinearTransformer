# %%
import torch
from matplotlib import pyplot as plt
import sys
import time
import os
import numpy as np


#####################################################
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
#torch.cuda.set_device(1)
# import the model and some useful functions
from linear_transformer import Transformer_F, attention, generate_data, in_context_loss

# set up some print options
np.set_printoptions(precision = 2, suppress = True)
torch.set_printoptions(precision=2)

#begin logging
log_dir = 'log' 
#exp_dir = 'simple_demonstration' 
cur_dir = log_dir #os.path.join(log_dir, exp_dir)
os.makedirs(cur_dir, exist_ok=True)
#f = open(cur_dir + '/train.log', "a", 1)
#sys.stdout = f

# %%
# Set up problem parameters

lr = 0.001
clip_r = 1000
alg = 'adam'
mode = 'normal'

n_layer = 3  # number of layers of transformer
N = 20     # context length
d = 5        # dimension


n_head = 1  # 1-headed attention
B = 4000  # 1000 minibatch size
var = 0.0001  # initializations scale of transformer parameter
shape_k = 0.1  # shape_k: parameter for Gamma distributed covariates
max_iters = 10000  # Number of Iterations to run
hist_stride = 1  # stride for saved model paramters in `train.ipynb'
stride = 100

# a convenience function for taking a step and clipping
def clip_and_step(allparam, optimizer, clip_r = None):
    norm_p=None
    grad_all = allparam.grad
    if clip_r is not None:
        norm_p = grad_all.norm().item()
        if norm_p > clip_r:
            grad_all.mul_(clip_r/norm_p)
    optimizer.step()
    return norm_p

# %%
filename_format = '/linearTF_exp_{}_{}_{}.pth'
filename = filename_format.format(n_layer, N, d)
filename = (cur_dir + filename)
hist_dict = {}


seeds = [0,1,2] #for demonstration purpose, just use 3 seeds
keys = [(s,) for s in seeds]
for key in keys:
    sd = key[0]
    
    prob_seed = sd
    opt_seed = sd
    
    hist_dict[key] = []
    
    #set seed and initialize model
    torch.manual_seed(opt_seed)
    model = Transformer_F(n_layer, 1, d, var)
    model.to(device)
    #initialize algorithm. Important: set beta = 0.9 for adam, 0.999 is very slow
    if alg == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    elif alg == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=0)
    else: assert False
    
    #set seed and initialize initial training batch
    np.random.seed(prob_seed)
    torch.manual_seed(prob_seed)
    
    for t in range(max_iters):
        start = time.time()
        # save model parameters
        if t%hist_stride ==0:
            hist_dict[key].append(model.allparam.clone().detach())
        #  generate a new batch of training set
        Z, y = generate_data(mode,N,d,B,shape_k)
        Z = Z.to(device)
        y = y.to(device)
        loss = in_context_loss(model, Z, y)
        
        # compute gradient, take step
        loss.backward()
        norms = clip_and_step(model.allparam, optimizer, clip_r=clip_r)
        optimizer.zero_grad()
        end=time.time()
        if t%100 ==0 or t<5:
            print('iter {} | Loss: {}  time: {}  gradnorm: {}'.format(t,loss.item(), end-start, norms))
    #save to 
torch.save({'hist_dict':hist_dict}, filename)

# %%
####################################
# compute test loss
####################################
hist_dict = torch.load(filename)['hist_dict']
loss_dict = {}
for key in hist_dict:
    sd = key[0]
    
    loss_dict[key] = torch.zeros(max_iters//stride)
    
    np.random.seed(99)
    torch.manual_seed(99)
    Z, y = generate_data(mode,N,d,B,shape_k)
    Z = Z.to(device)
    y = y.to(device)
    model = Transformer_F(n_layer, n_head, d, var).to(device)
    for t in range(0,max_iters,stride):
        with torch.no_grad():
            model.allparam.copy_(hist_dict[key][t])
        loss_dict[key][t//stride] = in_context_loss(model, Z, y).item()

# %%
####################################
# plot the test loss with error bars
####################################

fig_dir = 'figures' 
os.makedirs(fig_dir, exist_ok=True)

fig, ax = plt.subplots(1, 1,figsize = (7, 6))

losses = torch.zeros(len(seeds), max_iters//stride)
keys = loss_dict.keys()
for idx, key in enumerate(keys):
    losses[idx,:] = loss_dict[key]
losses_mean = torch.mean(losses, axis=0)
losses_std = torch.std(losses, axis=0)
ax.plot(range(0,max_iters,stride), losses_mean, color = 'red', lw = 3)#, label='Adam')
ax.fill_between(range(0,max_iters,stride), losses_mean-losses_std, losses_mean+losses_std, color = 'red', alpha = 0.2)
ax.set_xlabel('Iteration',fontsize=40)
ax.set_ylabel('ICL Test Loss',fontsize=40)
ax.tick_params(axis='both', which='major', labelsize=30, width = 3, length = 10)
ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
#ax.legend(fontsize=30)
ax.set_yscale('log')


plt.tight_layout()
plt.savefig(fig_dir + '/simple_demonstration_loss_plot.pdf', dpi=600)

# %%
####################################
# display the parameter matrices
# image/font setting assumes d=5
####################################

key = (0,)
for l in range(n_layer-1):
    for h in range(n_head):
        fig, ax = plt.subplots(1, 1,figsize = (6, 6))
        matrix = hist_dict[key][9999][l,h,0,:,:]
        # Create a heatmap using imshow
        im = ax.imshow(matrix.cpu(), cmap='gray_r')
        # Add the matrix values as text
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], '.2f'), ha='center', va='center', color='r')
        # Add a colorbar for reference
        fig.colorbar(im)
        ax.set_title('$B_{}$'.format(l),fontsize=20)
        
        plt.savefig(fig_dir + '/simple_demonstration_B{}.pdf'.format(l), dpi=600)
for l in range(n_layer):
    for h in range(n_head):
        fig, ax = plt.subplots(1, 1,figsize = (6, 6))
        matrix = hist_dict[key][9999][l,h,1,:,:]
        # Create a heatmap using imshow
        im = ax.imshow(matrix.cpu(), cmap='gray_r')
        # Add the matrix values as text
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], '.2f'), ha='center', va='center', color='r')
        # Add a colorbar for reference
        fig.colorbar(im)
        ax.set_title('$A_{}$'.format(l),fontsize=20)
        plt.savefig(fig_dir + '/simple_demonstration_A{}.pdf'.format(l), dpi=600)
    

# %%
########################################################
# plot the distance-to-identity of each matrix with time
########################################################

# function for computing distance to identity
def compute_dist_identity(M):
    scale = torch.sum(torch.diagonal(M))/M.shape[0]
    ideal_identity = scale* torch.eye(M.shape[0]).to(device)
    difference = M - ideal_identity
    err = (torch.norm(difference,p='fro')/torch.norm(M,p='fro')).item()
    return err

########################################
# compute distances (assume n_head = 1)
########################################
dist_dict = {}
            
for key in hist_dict:
    (sd,) = key
    dist_dict[key] = torch.zeros(n_layer, 2, max_iters//stride)

    for t in range(0,max_iters,stride):
        with torch.no_grad():
            allparam = hist_dict[key][t]
        for i in range(n_layer):
            for j in range(2):
                dist_dict[key][i,j,t//stride] = compute_dist_identity(allparam[i,0,j,:,:])

# %%

####################################
# plot distances
####################################

fig_dir = 'figures' 
os.makedirs(fig_dir, exist_ok=True)

fig, axs = plt.subplots(3, 2,figsize = (14, 18))

labels = ['B0', 'B1', None, 'A0', 'A1', 'A2']
colors = ['red','orange',None, 'green','blue','black']

labels = ['B0', 'B1', None, 'A0', 'A1', 'A2']
colors = ['red','orange',None, 'green','blue','black']

#make P plots
for l in range(n_layer):
    for pq in range(2):
        if l==n_layer-1 and pq==0:
            continue
        ax = axs[l,pq]
        dist_p = torch.zeros(len(seeds), max_iters//stride)
        for idx, sd in enumerate(seeds):
            losses[idx,:] = dist_dict[(sd,)][l,pq,:]
        dist_mean = torch.mean(losses, axis=0)
        dist_std = torch.std(losses, axis=0)
        
        style_id = l + 3*pq
        
        ax.plot(range(0,max_iters,stride), dist_mean, color = colors[style_id], lw = 3, label=labels[style_id])
        ax.fill_between(range(0,max_iters,stride), dist_mean-dist_std, dist_mean+dist_std, color = colors[style_id], alpha = 0.2)
        #ax.set_xlabel('Iteration',fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=20, width = 3, length = 10)
        ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
        ax.legend(fontsize=30)
        ax.set_yscale('log')

plt.savefig(fig_dir + '/simple_demonstration_dist_to_id.pdf', dpi=600)


# %%
print((hist_dict[(0,)][100]- hist_dict[(1,)][100]).norm())
print((compute_dist_identity(hist_dict[(0,)][100][i,0,j,:,:]) - compute_dist_identity(hist_dict[(1,)][100][i,0,j,:,:])))
print((dist_dict[(0,)][0,0,10]- dist_dict[(1,)][0,0,10]).norm())


# %%



