# %%
# get header to colab server

# ! for file in \
# 'linear_transformer.py'\
# ; do\
#     echo "downloading ${file} ... ";\
#   curl \
#   -o "${file}"\
#   -L "https://raw.githubusercontent.com/jangtze/LinearTransformer/refs/heads/padding/${file}";\
# done

# %%
# ### use matplot2tikz
# ! pip install matplot2tikz

# %%
from matplotlib import pyplot as plt
import matplot2tikz as tikzplotlib
import sys
import time
import os
import numpy as np
import math
import itertools # for different contexts

import torch

import pandas as pd

##############################################################################################################
# Trains a linear Transformer with 1,2,3,4 layers
# Plots the test loss of trained Transformer against 1,2,3,4 steps of gradient descent (with and without preconditioning)
##############################################################################################################

# use intel arc ipex
import intel_extension_for_pytorch as ipex
# # Move to XPU
device = 'xpu' # intel arc ipex
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# import the model and some useful functions
from linear_transformer import Transformer_F, attention, generate_data, in_context_loss2, generate_data_inplace

# %%
# set up some print options
np.set_printoptions(precision = 2, suppress = True)
torch.set_printoptions(precision=2)

#begin logging
log_dir = 'log' 
fig_dir = 'figures_gd_l_20_10' 
os.makedirs(fig_dir, exist_ok=True)
cur_dir = log_dir #os.path.join(log_dir, exp_dir)
os.makedirs(cur_dir, exist_ok=True)
#f = open(cur_dir + '/rotation.log', "a", 1)
#sys.stdout = f

# %%
# Set up problem parameters

# data
mode = 'normal'
B = 20000  # 1000 minibatch size
var = 0.0001  # initializations scale of transformer parameter
shape_k = 0.1  # shape_k: parameter for Gamma distributed covariates

# model
n_layer = 4  # number of layers of transformer
n_head = 1  # 1-headed attention
N = 20     # context length
d = 5        # dimension

# optimizaton / learning
alg = 'adam'
max_iters = 10000 # 20000  # Number of Iterations to run
lr = 0.01
half_lr_each_nth_step = 4000
gen_new_inplace_each = 100
clip_r = 0.01

hist_stride = 100  # stride for saved model paramters in `train.ipynb'
stride = 100

possible_combinations =  list(itertools.product([0,1], repeat=N))


# a convenience function for taking a step and clipping
def clip_and_step(allparam, optim, clip_r = None):
    norm_p=None
    grad_all = allparam.grad
    if clip_r is not None:
        for l in range(grad_all.shape[0]):
            for h in range(grad_all.shape[1]):
                for t in range(grad_all.shape[2]):
                    norm_p = grad_all[l,h,t,:,:].norm().item()
                    if norm_p > clip_r:
                        grad_all[l,h,t,:,:].mul_(clip_r/norm_p)
    optim.step()
    return norm_p

# %%
# chose train mask / amount of examples
######################################


# ####
# # if we want to loop all instead -- bad idea, takes too long
# for option in possible_combinations:
#     if not all(option):
#         continue

# chosen_combinations = possible_combinations[1:] # all ~ 1M
# chosen_combinations = [p for p in possible_combinations if sum(p)==10] # all with context of 10 examples ~200k
# chosen_combinations = [p for p in possible_combinations if sum(p)>=10] # all with context of at least 10 examples ~ 600k
# chosen_combinations = [p for p in possible_combinations if sum(p)>=12] # all with context of at least 12 examples ~ 260k
# chosen_combinations = [p for p in possible_combinations if sum(p)>=15] # all with context of at least 15 examples ~ 21k
# chosen_combinations = [p for p in possible_combinations if sum(p)>=18] # all with context of at least 18 examples = 211 --> ~ ...min
# chosen_combinations = [p for p in possible_combinations if sum(p)>=19] # all with context of at least 19 examples = 21 --> ~ ...min

# ####
# # choose random example from context
# # start from index 1 to avoid all zero mask
# selection = np.random.randint(1,len(possible_combinations),1)[0]
# option = possible_combinations[selection]
# chosen_combinations = [option] # to keep the rest the same
# print(option,'len ', sum(option))

# train_mask_specs = '_1_opt_1outof'+str(N)

# ####
# # choose random subset from context
# subset_size         = 5
# amount_of_examples  = 10
# all_with_amount_of_examples = [p for p in possible_combinations if sum(p)==amount_of_examples]
# # option = all_with_amount_of_examples[np.random.randint(1,len(selection),size=1)[0]]
# selection = np.random.randint(1,len(all_with_amount_of_examples),size=subset_size)
# chosen_combinations = [all_with_amount_of_examples[index] for index in selection] 

# train_mask_specs = '_' + str(subset_size)+'opt_'+str(amount_of_examples)+'outof'+str(N)

####
# train full
option = torch.ones((N,))
chosen_combinations = [option] # to keep the rest the same

train_mask_specs = '_trainfull_'+str(N)




# %%
#format for saving run data
filename_format = '/variable_L_hist2010_{}_{}_{}.pth'
# filename = (cur_dir + filename)
hist_dict = {}
train_loss_dict = {}

n_layers = [1,2,3,4]  # number of layers of transformer
seeds=[0,1,2]#,3,4]
keys = []
for s in seeds:
    for n_layer in n_layers:
        keys.append((s,n_layer,))

# %%
os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = "1"
# %%
print("start whole training ...")
start_full_training = time.time()
for key in keys:
    sd = key[0]
    n_layer = key[1]
    filename = cur_dir + filename_format.format(n_layer, N, sd)
    print(key)
    
    prob_seed = sd
    opt_seed = sd
    
    hist = []
    hist_dict[key] = []
    train_loss_dict[key] = torch.zeros(max_iters//stride, len(chosen_combinations))
    
    #set seed and initialize model
    torch.manual_seed(opt_seed)
    model = Transformer_F(n_layer, 1, d, var)
    model.to(device)
    #initialize algorithm. Important: set beta = 0.9 for adam, 0.999 is very slow
    if alg == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    elif alg == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.99, 0.9), weight_decay=0)
    else: assert False

    # Optimize the model and optimizer objects
    # https://christianjmills.com/posts/intel-pytorch-extension-tutorial/native-ubuntu/
    if device == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

    # set seed
    # sample random rotation matrix
    # initialize initial training batch
    np.random.seed(prob_seed)
    torch.manual_seed(prob_seed)
    gaus = torch.FloatTensor(5,5).uniform_(-1,1).to(device)
    U = torch.linalg.svd (gaus)[0].to(device)
    D = torch.diag(torch.FloatTensor([1,1,1/2,1/4,1])).to(device)
    Z, y, Z_train = generate_data(mode,N,d,B,shape_k, U, D)
    Z[:,-1,-1] = y # need to write y back in to keep the rest of the treatment the same

    Z = Z.to(device)
    y = y.to(device)

    print("start training for key ", key)
    start_training = time.time()
    for t in range(max_iters):
        if t%half_lr_each_nth_step==0 and t>1:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] *0.5
        if t%gen_new_inplace_each==0:
            Z,y = generate_data_inplace(Z, U=U, D=D)
            Z[:,-1,-1] = y # need to write y back in to keep the rest of the treatment the same
        start = time.time()
        # save model parameters
        if t%stride ==0:
            hist.append(model.allparam.clone().detach())
            hist_dict[key].append(model.allparam.clone().detach())
        # loss = in_context_loss(model, Z, y)

        loss_list = torch.zeros((len(chosen_combinations),))
        for opt_idx, option in enumerate(chosen_combinations):
        # for _ in range(1): # to have same indentation, when not looping all but choosing random option    
            
            ##################
            # for all options

            mask = torch.tensor( option, dtype=bool )
            curr_context    = Z_train[:,mask,:]
            # curr_y          = Z_train[:,mask,-1]
            # curr_y          = Z_train[:,mask,-1]#.detach().clone()
            curr_y          = Z_train[:,mask,-1].detach().clone()
            # print(curr_y[...,-3:])
            # curr_context[:,-1,-1] = 0
            curr_context[:,-1,-1].zero_()
            # print(curr_context[...,-1][...,-3:])
            # print(curr_y[...,-3:]) # with these prints => should work without detach.clone

            output = model(curr_context) # full length with 0 padding after first sum(mask) (= amount of non-zeros in mask)
            predicitons = output[:,:sum(mask),:]
            loss = in_context_loss2(predicitons, curr_y)
            loss_list[opt_idx] = loss.item()
            
            # # save training loss
            # if t%stride == 0:
            #     train_loss_dict[key][t//stride] = loss
                
            # compute gradient, take step
            loss.backward()
            norms = clip_and_step(model.allparam, optimizer, clip_r=clip_r)
            optimizer.zero_grad()
        

        end=time.time()
        if t%100 ==0 or t<5:
            print('iter {} | Loss: {}  time: {}s  gradnorm: {}'.format(t,loss.item(), end-start, norms))

        # save training loss
        if t%stride == 0:
            non_zero_loss_mask = torch.tensor(loss_list, dtype=bool)
            train_loss_dict[key][t//stride] = loss_list[non_zero_loss_mask]#.mean()

    torch.save({'hist':hist, 'U':U, 'D':D}, filename)
    #save to 
    end_training = time.time()
    print("end training for key ", key, " time ", end_training-start_training, "s")
end_full_training = time.time()
train_time = end_full_training-start_full_training
print("end full training: time ", train_time, "s")
torch.save({'hist':hist, 'U':U, 'D':D}, filename+'_final')

# %%
# plot the train loss with error bars
####################################


fig, ax = plt.subplots(1, 1,figsize = (7, 6))

train_losses = torch.zeros(len(seeds), len(n_layers), max_iters//stride, len(chosen_combinations))
keys = train_loss_dict.keys()

# for idlen in len(keys[0]):
#     set([a[idlen] for a in keys])

# # key 0 = seed , key 1 = layer num
# seed_rg = list(set([a[0] for a in keys]))
# layr_rg = list(set([a[1] for a in keys]))

# # for idx, key in enumerate(keys):
# for sx in enumerate(seed_rg):
#     for lx in enumerate(layr_rg):
for isx, sx in enumerate(seeds):
    for ilx, lx in enumerate(n_layers):
        curr_key = (sx,lx)
        train_losses[isx,ilx,:] = train_loss_dict[curr_key]
train_losses_mean = torch.mean(train_losses, axis=(0,1,-1)).detach().numpy()
train_losses_std = torch.std(train_losses, axis=(0,1,-1)).detach().numpy()

train_loss_mean_final   = train_losses_mean[-1].item()
train_loss_std_final    = train_losses_std[-1].item()
train_loss_min          = min(train_losses_mean).item()

ax.plot(range(0,max_iters,stride), train_losses_mean, color = 'red', lw = 3)#, label='Adam')
ax.fill_between(range(0,max_iters,stride), train_losses_mean-train_losses_std, train_losses_mean+train_losses_std, color = 'red', alpha = 0.2)
ax.set_xlabel('Iteration',fontsize=30)
ax.set_ylabel('Train Loss',fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30, width = 3, length = 10)
ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
#ax.legend(fontsize=30)
# ax.set_yscale('log')

plt.tight_layout()
output_file_name = fig_dir + '/varL_gd_train_loss_plot' + train_mask_specs
plt.savefig(output_file_name + '.pdf', dpi=600)

tikzplotlib.save(output_file_name + '.tex')

# %%
# chose test mask / amount of examples
######################################


# ####
# # if we want to loop all instead -- bad idea, takes too long
# for option in possible_combinations:
#     if not all(option):
#         continue

# chosen_combinations = possible_combinations[1:] # all except zero context ~ 1M
# chosen_combinations = [p for p in possible_combinations if sum(p)==10] # all with context of 10 examples ~200k
# chosen_combinations = [p for p in possible_combinations if sum(p)>=10] # all with context of at least 10 examples ~ 600k
# chosen_combinations = [p for p in possible_combinations if sum(p)>=12] # all with context of at least 12 examples ~ 260k
# chosen_combinations = [p for p in possible_combinations if sum(p)>=15] # all with context of at least 15 examples ~ 21k --> ~ 15h
# chosen_combinations = [p for p in possible_combinations if sum(p)>=18] # all with context of at least 18 examples = 211 --> ~ 10min
# chosen_combinations = [p for p in possible_combinations if sum(p)>=19] # all with context of at least 19 examples = 21 --> ~ 1min

# ####
# # choose random example from context
# # start from index 1 to avoid all zero mask
# selection = np.random.randint(1,len(possible_combinations),1)[0]
# option = possible_combinations[selection]
# chosen_combinations = [option] # to keep the rest the same
# # print(option,'len ', sum(option))

# test_mask_specs = '_1_opt_1outof'+str(N)

####
# choose random subset from context
subset_size         = 3
amount_of_examples  = 10
all_with_amount_of_examples = [p for p in possible_combinations if sum(p)==amount_of_examples]
# option = all_with_amount_of_examples[np.random.randint(1,len(selection),size=1)[0]]
selection = np.random.randint(1,len(all_with_amount_of_examples),size=subset_size)
chosen_combinations = [all_with_amount_of_examples[index] for index in selection] 

test_mask_specs = '_'+str(subset_size)+'opt_'+str(amount_of_examples)+'outof'+str(N)

# ####
# # test full
# option = torch.ones((N,))
# chosen_combinations = [option] # to keep the rest the same

# test_mask_specs = '_testfull_'+str(N)


# %%
# compute test loss for trained linear Transformers
########################################################
loss_dict = {}
print("start whole testing ...")
start_full_testing = time.time()
for sd in seeds:
    key = (sd,)
    loss_list = torch.zeros((len(chosen_combinations),len(n_layers)))
    for opt_idx, option in enumerate(chosen_combinations):

    # for _ in range(1): # to have same indentation, when not looping all but choosing random option    
        # for all options
        mask = torch.tensor( option, dtype=bool )
            
        for layr_idx, n_layer in enumerate(n_layers):
            
            # loss_dict[key] = torch.zeros(4) # HACK where does that come from?
            # load parameters for given n_layer and seed
            filename = cur_dir + filename_format.format(n_layer, N, sd)
            hist = torch.load(filename)['hist']
            U = torch.load(filename)['U']
            D = torch.load(filename)['D']
            
            # given short(er) training steps, may have some unstable solutions
            # on a validation set of (seed=999), find the solution with best validation
            # loss from the last 20 runs
            np.random.seed(999)
            torch.manual_seed(999)
            Z, y, Z_train   = generate_data(mode,N,d,B,shape_k,U,D)
            Z[:,-1,-1]      = y # need to write y back in to keep the rest of the treatment the same
            # Z = Z.to(device)
            # y = y.to(device)

            curr_context    = Z[:,mask,:] # need to leave one out for last test one
            # curr_y          = Z[:,mask,-1]#.detach().clone()
            # curr_context[:,-1,-1] = 0
            curr_y          = Z[:,mask,-1].detach().clone()
            curr_context[:,-1,-1].zero_()

            model = Transformer_F(n_layer, n_head, d, var).to(device)
            best_loss = 100
            bestmodel = None
            for t in range(len(hist)-20, len(hist)): # only the last 20

                with torch.no_grad():
                    model.allparam.copy_(hist[t])

                output = model(curr_context) # full length with 0 padding after first sum(mask) (= amount of non-zeros in mask)
                predicitons = output[:,:sum(mask),:]
                newloss = in_context_loss2(predicitons, curr_y).item()
                if (newloss < best_loss):
                    best_loss= newloss
                    bestmodel = hist[t]

            # HACK why reeval when we can save best loss ?
            # with torch.no_grad():
            #     model.allparam.copy_(bestmodel)

            # np.random.seed(99)
            # torch.manual_seed(99)
            # Z, y, Z_train = generate_data(mode,N,d,B,shape_k,U,D)
            # Z[:,-1,-1]      = y # need to write y back in to keep the rest of the treatment the same

            # curr_context    = Z[:,mask,:] # need to leave one out for last test one
            # # curr_y          = Z[:,mask,-1]#.detach().clone()
            # # curr_context[:,-1,-1] = 0
            # curr_y          = Z[:,mask,-1].detach().clone()
            # curr_context[:,-1,-1].zero_()
            
            # output = model(Z) # full length with 0 padding after first sum(mask) (= amount of non-zeros in mask)
            # predicitons = output[:,:sum(mask),:]
            # nloss = in_context_loss2(predicitons, curr_y).item()

            # #compute loss
            # # loss_dict[key][n_layer-1] = in_context_loss2(Z, y).log().item()
            # # loss_dict[key][layr_idx] = nloss

            loss_list[opt_idx,layr_idx] = best_loss

    non_zero_loss_mask = torch.tensor(loss_list, dtype=bool)
    if len(loss_list[non_zero_loss_mask].shape) == 1:
        loss_list[non_zero_loss_mask]
        loss_list = loss_list[:,torch.newaxis]
        # loss_list = torch.expand_dims(loss_list, axis=-1) # only in numpy
    else:
        loss_dict[key] = loss_list[non_zero_loss_mask]#.mean()

end_full_testing = time.time()
test_time = end_full_testing-start_full_testing
print("end full testing: time ", test_time, "s")
# %%
# plot log final test loss against N, for sanity check
########################################################

fig, ax = plt.subplots(1, 1,figsize = (9, 9))

losses = torch.zeros(len(seeds), len(n_layers), len(chosen_combinations))
keys = loss_dict.keys()
for idx, key in enumerate(keys):
    # losses[idx,:, :] = np.log(loss_dict[key])
    losses[idx,:,:] = loss_dict[key]
losses_mean = torch.mean(losses, axis=(0,-1)) # over seed and different combinations of context examples
losses_std = torch.std(losses, axis=(0,-1))

test_loss_mean_final    = losses_mean[-1].item()
test_loss_std_final     = losses_std[-1].item()
test_loss_min           = min(losses_mean).item()

ax.plot(n_layers, losses_mean, color = 'red', lw = 3, label='3-Layer Linear Transformer')
ax.fill_between(n_layers, losses_mean-losses_std, losses_mean+losses_std, color = 'red', alpha = 0.2)
ax.set_xlabel('Iteration',fontsize=30)
ax.set_ylabel('Test Loss',fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20, width = 3, length = 10)
ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
#ax.legend(fontsize=30)
# ax.set_yscale('log')


plt.tight_layout()
output_file_name = fig_dir + '/rotation_demonstration_adam_loss_plot' + test_mask_specs
plt.savefig(output_file_name + '.pdf', dpi=600)

tikzplotlib.save(output_file_name + '.tex')

# %% 
# evaluate the performance of x steps of Gradient Descent
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


# %%
gd_loss_matrix = torch.zeros(len(seeds),4)
gd_sample_size = min(B,5000)
# gd_sample_size = min(B,1000)

print("start whole gd ...")
start_full_gd = time.time()
for n_layer in n_layers:
    #first find best eta
    #load seed 1 for U,D matrices
    sd = 1
    best_loss = 10000
    best_eta = 0
    numstep = n_layer
    # load UD matrices
    filename = cur_dir + filename_format.format(n_layer, N, sd)
    U = torch.load(filename)['U']
    D = torch.load(filename)['D']
    #generate test data using seed 999
    np.random.seed(999)
    torch.manual_seed(999)
    Z, y, Z_train = generate_data(mode,N,d,B,shape_k,U,D)
    # Z[:,-1,-1] = y # need to write y back in to keep the rest of the treatment the same
    # Z = Z.to(device)
    # y = y.to(device)
    #done generating data 
    
    for eta in [0.008, 0.01, 0.02, 0.04, 0.08, 0.16]:
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(gd_sample_size): # HACK wtf ...
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_gd(Zi,eta,numstep)
            gd_loss, gd_pred = eval_w_instance(Zi, Ytesti, w)
            total_loss = total_loss + gd_loss
        mean_loss = total_loss / gd_sample_size
        ### end of evaluate mean loss ###
        print('eta: {}, loss: {}'.format(eta, mean_loss))
        if (mean_loss < best_loss):
            best_eta = eta
            best_loss = mean_loss
    print('best eta: {} for n_layer={}'.format(best_eta, n_layer))
    
    #now do actual evaluation
    for sd in seeds:
        opt_seed = sd
        
        filename = cur_dir + filename_format.format(n_layer, N, sd)
        U = torch.load(filename)['U']
        D = torch.load(filename)['D']
        #generate test data
        torch.manual_seed(sd)
        Z, y, Z_train = generate_data(mode,N,d,B,shape_k,U,D)
        Z[:,-1,-1] = y # need to write y back in to keep the rest of the treatment the same
        # Z = Z.to(device)
        # y = y.to(device)
        #done generating data 
        eta = best_eta
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(Z.shape[0]):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_gd(Zi,eta,numstep)
            gd_loss, gd_pred = eval_w_instance(Zi, Ytesti, w)
            total_loss = total_loss + gd_loss
        mean_loss = total_loss / Z.shape[0]
        gd_loss_matrix[sd,n_layer-1] = mean_loss
        
end_full_gd = time.time()
gd_time = end_full_gd-start_full_gd
print("end full testing: time ", gd_time, "s")

#compute mean and std of log test loss for plotting
# gd_loss_mean = gd_loss_matrix.log().mean(dim=0)
# gd_loss_std = gd_loss_matrix.log().var(dim=0)**0.5
gd_loss_mean = gd_loss_matrix.mean(dim=0)
gd_loss_std = gd_loss_matrix.std(dim=0)

gd_loss_mean_final   = gd_loss_mean[-1].item()
gd_loss_std_final    = gd_loss_std[-1].item()
gd_loss_min          = min(gd_loss_mean).item()


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



pgd_loss_matrix = torch.zeros(len(seeds),4)

print("start whole pgd ...")
start_full_pgd = time.time()
for n_layer in n_layers:
    #first find best eta
    #load seed 1 for U,D matrices
    sd = 1
    best_loss = 10000
    best_eta = 0
    numstep = n_layer
    # load UD matrices
    filename = cur_dir + filename_format.format(n_layer, N, sd)
    U = torch.load(filename)['U'].to(device)
    D = torch.load(filename)['D'].to(device)
    #generate test data using seed 999
    np.random.seed(999)
    torch.manual_seed(999)
    Z, y, Z_train = generate_data(mode,N,d,B,shape_k,U,D)
    # Z = Z.to(device)
    # y = y.to(device)
    #done generating data 
    
    for eta in [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.16]:
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(gd_sample_size):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_preconditioned_gd(Zi,eta,numstep,U,D)
            pgd_loss, pgd_pred = eval_w_instance_precon(Zi, Ytesti, w, U, D)
            total_loss = total_loss + pgd_loss
        mean_loss = total_loss / gd_sample_size
        ### end of evaluate mean loss ###
        print('eta: {}, loss: {}'.format(eta, mean_loss))
        if (mean_loss < best_loss):
            best_eta = eta
            best_loss = mean_loss
    print('best eta: {} for n_layer={}'.format(best_eta, n_layer))
    
    #now do actual evaluation
    for sd in seeds:
        opt_seed = sd
        
        filename = cur_dir + filename_format.format(n_layer, N, sd)
        U = torch.load(filename)['U'].to(device)
        D = torch.load(filename)['D'].to(device)
        #generate test data
        torch.manual_seed(sd)
        Z, y, Z_train = generate_data(mode,N,d,B,shape_k,U,D)
        Z = Z.to(device)
        y = y.to(device)
        #done generating data 
        eta = best_eta
        ### start of evaluate mean loss ###
        total_loss = 0
        for i in range(gd_sample_size):
            Zi = Z[i,:,:]
            Ytesti = y[i]
            w = do_preconditioned_gd(Zi,eta,numstep,U,D)
            pgd_loss, pgd_pred = eval_w_instance_precon(Zi, Ytesti, w, U, D)
            total_loss = total_loss + pgd_loss
        mean_loss = total_loss / gd_sample_size
        pgd_loss_matrix[sd,n_layer-1] = mean_loss

end_full_pgd = time.time()
pgd_time = end_full_pgd-start_full_pgd
print("end full testing: time ", pgd_time, "s")

#compute mean and std of log test loss for plotting
# pgd_loss_mean = pgd_loss_matrix.log().mean(dim=0)
# pgd_loss_std = pgd_loss_matrix.log().var(dim=0)**0.5
pgd_loss_mean = pgd_loss_matrix.mean(dim=0)
pgd_loss_std = pgd_loss_matrix.std(dim=0)
    
pgd_loss_mean_final   = pgd_loss_mean[-1].item()
pgd_loss_std_final    = pgd_loss_std[-1].item()
pgd_loss_min          = min(pgd_loss_mean).item()


# %%
####################################
# plot final test loss against N
####################################


fig, ax = plt.subplots(1, 1,figsize = (9, 9))

losses = torch.zeros(len(seeds), len(n_layers))
keys = loss_dict.keys()
for idx, key in enumerate(keys):
    losses[idx,:] = loss_dict[key]
losses_mean = torch.mean(losses, axis=0)
losses_std = torch.std(losses, axis=0)

plt.plot(n_layers, gd_loss_mean, color='blue', label='GD')
plt.fill_between(n_layers, gd_loss_mean - gd_loss_std, gd_loss_mean + gd_loss_std, color='blue', alpha=0.2)
plt.plot(n_layers, pgd_loss_mean, color='green', label='Preconditioned GD')
plt.fill_between(n_layers, pgd_loss_mean - pgd_loss_std, pgd_loss_mean + pgd_loss_std, color='green', alpha=0.2)
ax.plot(n_layers, losses_mean, color = 'red', lw = 3, label='Linear Transformer')
ax.fill_between(n_layers, losses_mean-losses_std, losses_mean+losses_std, color = 'red', alpha = 0.2)

plt.ylabel('log(Loss)',fontsize=30)
plt.xlabel('Number of Layers/Steps',fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30, width = 3, length = 10)
ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
ax.legend(fontsize=24)
#ax.set_yscale('log')


plt.tight_layout()
# plt.savefig(fig_dir + '/variable-L-plot.pdf', dpi=600)

output_file_name = fig_dir + '/variable-L-plot' + test_mask_specs
plt.savefig(output_file_name + '.pdf', dpi=600)

tikzplotlib.save(output_file_name + '.tex')

# %%


# %%

# Sources
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
# https://stackoverflow.com/questions/73975135/list-comprehension-using-f-strings
# https://stackoverflow.com/questions/68518600/add-a-title-to-a-dataframe
# https://stackoverflow.com/questions/4152963/get-name-of-current-script-in-python
# https://stackoverflow.com/questions/14380371/export-a-latex-table-from-pandas-dataframe
# import inspect

# def retrieve_var(var):
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     return [var_name for var_name, var_val in callers_local_vars if var_val is var]

# todf = []
# for i in [lr, mode, alg, clip_r, N, d, n_layer, n_head]:
#     todf.append(
#         (retrieve_var(i)[0], i)
#     )

variables_list = [
    (i, globals()[i]) if i in globals().keys()
    else (i, 'n/a')
    for i in [
        'mode', 'B', 'var', 'shape_k', # data
        'd', 'N', 'n_layer', 'n_head', # model
        'alg', 'clip_r', 'max_iters',  # training
        'lr', 'half_lr_each_nth_step',
        'gen_new_inplace_each',
        # 'subset_size', 'amount_of_examples', # is in specs
        'hist_stride', 'stride', 
        'train_mask_specs', 'test_mask_specs',
        'train_time', 'test_time', 'gd_time', 'pgd_time',
        'train_loss_mean_final', 'train_loss_std_final', 'train_loss_min',
        'test_loss_mean_final', 'test_loss_std_final', 'test_loss_min',
        'gd_sample_size',
        'gd_loss_mean_final', 'gd_loss_std_final', 'gd_loss_min',
        'pgd_loss_mean_final', 'pgd_loss_std_final', 'pgd_loss_min'
        ]
    ]

variables_df = pd.DataFrame(variables_list, columns=['variable name', 'value'])
variables_df.columns=pd.MultiIndex.from_product([[os.path.basename(__file__)],variables_df.columns])

variables_df.to_latex(fig_dir + '/' + fig_dir.removeprefix('figures_') + '_vars.tex', index=False)
variables_df
# %%
