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

# use intel arc ipex
import intel_extension_for_pytorch as ipex
# # Move to XPU
device = 'xpu' # intel arc ipex
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# import the model and some useful functions
from linear_transformer import Transformer_F, attention, generate_data, in_context_loss2

# %%
# set up some print options
np.set_printoptions(precision = 2, suppress = True)
torch.set_printoptions(precision=2)

#begin logging
log_dir = 'log' 
fig_dir = 'figures_adam_10_15' 
os.makedirs(fig_dir, exist_ok=True)
cur_dir = log_dir #os.path.join(log_dir, exp_dir)
os.makedirs(cur_dir, exist_ok=True)

# %%
# Set up problem parameters

# data
mode = 'normal'
B = 20000  # 1000 minibatch size
var = 0.0001  # initializations scale of transformer parameter
shape_k = 0.1  # shape_k: parameter for Gamma distributed covariates

# model
n_layer = 3  # number of layers of transformer
n_head = 1  # 1-headed attention
N = 20     # context length
d = 5        # dimension

# optimizaton / learning
alg = 'adam'
max_iters = 10000  # Number of Iterations to run
lr = 0.1
half_lr_each_nth_step = 2000
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

####
# choose random subset from context
subset_size         = 5
amount_of_examples  = 10
all_with_amount_of_examples = [p for p in possible_combinations if sum(p)==amount_of_examples]
# option = all_with_amount_of_examples[np.random.randint(1,len(selection),size=1)[0]]
selection = np.random.randint(1,len(all_with_amount_of_examples),size=subset_size)
chosen_combinations = [all_with_amount_of_examples[index] for index in selection] 

train_mask_specs = '_' + str(subset_size)+'opt_'+str(amount_of_examples)+'outof'+str(N)

# ####
# # train full
# option = torch.ones((N,))
# chosen_combinations = [option] # to keep the rest the same

# train_mask_specs = '_trainfull_'+str(N)




# %%
# training
filename_format = '/rotation_hist_adam_{}_{}_{}.pth'
filename = filename_format.format(n_layer, N, d)
filename = (cur_dir + filename)
hist_dict = {}
train_loss_dict = {}
U_dict = {}
D_dict = {}

seeds = [0,1,2] #for demonstration purpose, just use 3 seeds
keys = [(s,) for s in seeds]
print("start whole training ...")
start_full_training = time.time()
for key in keys:
    sd = key[0]
    
    prob_seed = sd
    opt_seed = sd
    
    hist_dict[key] = []
    train_loss_dict[key] = torch.zeros(max_iters//stride, len(chosen_combinations))
    
    #set seed and initialize model
    torch.manual_seed(opt_seed)
    model = Transformer_F(n_layer, n_head, d, var)
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
    U_dict[key]=U
    D_dict[key]=D
    Z, y, Z_train = generate_data(mode,N,d,B,shape_k, U, D)
    Z = Z.to(device)
    y = y.to(device)

    print("start training for key ", key)
    start_training = time.time()
    for t in range(max_iters):
        if t%half_lr_each_nth_step==0 and t>1:# and t < 200001:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] *0.5
        # if t%100==0:
        #     Z,y = generate_data_inplace(Z, shape_k=0.1, U=U, D=D)
        #if t==6000:
        #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] *0.2
        #if t==12000:
        #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] *0.2
        #if t==16000:
        #    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] *0.2
        start = time.time()
        # save model parameters
        if t%hist_stride == 0:
            hist_dict[key].append(model.allparam.clone().detach())
        #  generate a new batch of training set
        Z, y, Z_train = generate_data(mode,N,d,B,shape_k)
        Z[:,-1,-1]      = y # need to write y back in to keep the rest of the treatment the same

        # Z = Z.to(device) # type: ignore
        # y = y.to(device)

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

    #save to 
    end_training = time.time()
    print("end training for key ", key, " time ", end_training-start_training, "s")
end_full_training = time.time()
train_time = end_full_training-start_full_training
print("end full training: time ", train_time, "s")
torch.save({'hist_dict':hist_dict, 'U_dict':U_dict, 'D_dict':D_dict}, filename)

# %%
# plot the train loss with error bars
####################################


fig, ax = plt.subplots(1, 1,figsize = (7, 6))

train_losses = torch.zeros(len(seeds), max_iters//stride, len(chosen_combinations))
keys = train_loss_dict.keys()
for idx, key in enumerate(keys):
    train_losses[idx,:,:] = train_loss_dict[key]
train_losses_mean = torch.mean(train_losses, axis=(0,-1)).detach().numpy()
train_losses_std = torch.std(train_losses, axis=(0,-1)).detach().numpy()

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
ax.set_yscale('log')


plt.tight_layout()
output_file_name = fig_dir + '/rotation_demonstration_adam_train_loss_plot' + train_mask_specs
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
subset_size         = 5
amount_of_examples  = 15
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
# compute test loss
####################################
#hist_dict = torch.load(filename)['hist_dict']
loss_dict = {}
print("start whole testing ...")
start_full_testing = time.time()
for key in hist_dict:
    sd = key[0]
    
    U = U_dict[key]
    D = D_dict[key]
    
    loss_dict[key] = torch.zeros(max_iters//stride, len(chosen_combinations))
    
    np.random.seed(99)
    torch.manual_seed(99)
    Z, y, Z_train   = generate_data(mode,N,d,B,shape_k)
    Z[:,-1,-1]      = y # need to write y back in to keep the rest of the treatment the same
    # Z = Z.to(device)
    # y = y.to(device)

    model = Transformer_F(n_layer, n_head, d, var).to(device)

    for t in range(0,max_iters,stride):

        loss_list = torch.zeros((len(chosen_combinations),))
        for opt_idx, option in enumerate(chosen_combinations):

        # for _ in range(1): # to have same indentation, when not looping all but choosing random option    
            # for all options
            mask = torch.tensor( option, dtype=bool )
            curr_context    = Z[:,mask,:] # need to leave one out for last test one
            # curr_y          = Z[:,mask,-1]#.detach().clone()
            # curr_context[:,-1,-1] = 0
            curr_y          = Z[:,mask,-1].detach().clone()
            curr_context[:,-1,-1].zero_()

            with torch.no_grad():
                model.allparam.copy_(hist_dict[key][t//stride])
                # model.allparam.copy_(hist_dict[key][t//hist_stride])

            output = model(curr_context) # full length with 0 padding after first sum(mask) (= amount of non-zeros in mask)
            predicitons = output[:,:sum(mask),:]
            loss_list[opt_idx] = in_context_loss2(predicitons, curr_y).item()
        non_zero_loss_mask = torch.tensor(loss_list, dtype=bool)
        loss_dict[key][t//stride] = loss_list[non_zero_loss_mask]#.mean()

end_full_testing = time.time()
test_time = end_full_testing-start_full_testing
print("end full testing: time ", test_time, "s")
# %%
# plot the test loss with error bars
####################################

fig, ax = plt.subplots(1, 1,figsize = (7, 6))

losses = torch.zeros(len(seeds), max_iters//stride, len(chosen_combinations))
keys = loss_dict.keys()
for idx, key in enumerate(keys):
    losses[idx,:,:] = loss_dict[key]#.log()
losses_mean = torch.mean(losses, axis=(0,-1)) # over seed and different combinations of context examples
losses_std = torch.std(losses, axis=(0,-1))

test_loss_mean_final    = losses_mean[-1].item()
test_loss_std_final     = losses_std[-1].item()
test_loss_min           = min(losses_mean).item()

ax.plot(range(0,max_iters,stride), losses_mean, color = 'blue', lw = 3)#, label='Adam')
ax.fill_between(range(0,max_iters,stride), losses_mean-losses_std, losses_mean+losses_std, color = 'black', alpha = 0.2)
ax.set_xlabel('Iteration',fontsize=30)
ax.set_ylabel('Test Loss',fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20, width = 3, length = 10)
ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
#ax.legend(fontsize=30)
ax.set_yscale('log')


plt.tight_layout()
output_file_name = fig_dir + '/rotation_demonstration_adam_loss_plot' + test_mask_specs
plt.savefig(output_file_name + '.pdf', dpi=600)

tikzplotlib.save(output_file_name + '.tex')

# %%
####################################
# display the parameter matrices
# image/font setting assumes d=5
####################################

key = (0,)

U = U_dict[(0,)]
D = D_dict[(0,)]
UD = torch.mm(U,D)     

for l in range(n_layer-1):
    for h in range(n_head):
        fig, ax = plt.subplots(1, 1,figsize = (6, 6))
        matrix = hist_dict[key][-1][l,h,0,:,:]
        # Create a heatmap using imshow
        im = ax.imshow(matrix.cpu(), cmap='gray_r')
        # Add the matrix values as text
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], '.2f'), ha='center', va='center', color='r')
        # Add a colorbar for reference
        fig.colorbar(im)
        ax.set_title('$B_{}$'.format(l),fontsize=20)
        
        output_file_name = fig_dir + '/rotation_demonstration_B{}'.format(l) + test_mask_specs
        plt.savefig(output_file_name + '.pdf', dpi=600)

        tikzplotlib.save(output_file_name + '.tex')

for l in range(n_layer):
    for h in range(n_head):
        fig, ax = plt.subplots(1, 1,figsize = (6, 6))
        matrix = hist_dict[key][-1][l,h,1,:,:]
        #rotate matrix by inverse of UD
        matrix = torch.mm(torch.mm(UD.t(), matrix), UD)
        # Create a heatmap using imshow
        im = ax.imshow(matrix.cpu(), cmap='gray_r')
        # Add the matrix values as text
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], '.2f'), ha='center', va='center', color='r')
        # Add a colorbar for reference
        fig.colorbar(im)
        ax.set_title('$A_{}$'.format(l),fontsize=20)

        output_file_name = fig_dir + '/rotation_demonstration_A{}'.format(l) + test_mask_specs
        plt.savefig(output_file_name + '.pdf', dpi=600)

        tikzplotlib.save(output_file_name + '.tex')
    

# %%
# plot the distance-to-identity of each matrix with time
########################################################

# function for computing distance to identity
def compute_dist_identity(M):
    scale = torch.sum(torch.diagonal(M))/M.shape[0]
    ideal_identity = scale* torch.eye(M.shape[0]).to(device)
    difference = M - ideal_identity
    err = (torch.norm(difference,p='fro')/torch.norm(M,p='fro'))
    return err

########################################
# compute distances (assume n_head = 1)
########################################
dist_dict = {}

id_dist_dict={}
            
for key in hist_dict:
    (sd,) = key
    dist_dict[key] = torch.zeros(n_layer, 2, max_iters//stride)
    id_dist_dict[key] = torch.zeros(n_layer, 2, max_iters//stride)
    U = U_dict[key]
    D = D_dict[key]
    UD = torch.mm(U,D)        
    for t in range(0,max_iters,stride):
        with torch.no_grad():
            allparam = hist_dict[key][t//stride]
        for i in range(n_layer):
            for j in range(2):
                matrix = allparam[i,0,j,:,:]
                if j ==1:
                    id_dist_dict[key][i,j,t//stride] = compute_dist_identity(matrix).item()
                    matrix = torch.mm(torch.mm(UD.t(), matrix), UD)
                dist_dict[key][i,j,t//stride] = compute_dist_identity(matrix).item()

# %%
# plot distances
####################################

labels = ['$B_0$', '$B_1$', None, 
          '$\Sigma^{1/2} A_0 \Sigma^{1/2}$', 
          '$\Sigma^{1/2} A_1 \Sigma^{1/2}$', 
          '$\Sigma^{1/2} A_2 \Sigma^{1/2}$']
names = ['B0', 'B1', None, 
          'A0', 
          'A1', 
          'A2']
colors = ['red','orange',None, 'green','blue','black']

distances: torch.Tensor = torch.zeros(len(seeds), max_iters//stride)

#make P plots
for l in range(n_layer):
    for pq in range(2):
        if l==n_layer-1 and pq==0:
            continue
        if pq ==0:
            continue
        fig, ax = plt.subplots(1, 1,figsize = (9, 7))
        if pq==1:
            id_dist_p = torch.zeros(len(seeds), max_iters//stride)
            for idx, sd in enumerate(seeds):
                distances[idx,:] = id_dist_dict[(sd,)][l,pq,:]
            dist_mean = torch.mean(distances, axis=0)
            dist_std = torch.std(distances, axis=0)
            ax.plot(range(0,max_iters,stride), dist_mean, color = 'red', lw = 3, label='$A_{}$'.format(l))
            ax.fill_between(range(0,max_iters,stride), dist_mean-dist_std, dist_mean+dist_std, color = 'red', alpha = 0.2)
        
        dist_p = torch.zeros(len(seeds), max_iters//stride)
        for idx, sd in enumerate(seeds):
            distances[idx,:] = dist_dict[(sd,)][l,pq,:]
        dist_mean = torch.mean(distances, axis=0)
        dist_std = torch.std(distances, axis=0)
        
        style_id = l + 3*pq
        
        ax.plot(range(0,max_iters,stride), dist_mean, color = colors[style_id], lw = 3, label=labels[style_id])
        ax.fill_between(range(0,max_iters,stride), dist_mean-dist_std, dist_mean+dist_std, color = colors[style_id], alpha = 0.2)
        ax.tick_params(axis='both', which='major', labelsize=20, width = 3, length = 10)
        ax.tick_params(axis='both', which='minor', labelsize=20, width = 3, length = 5)
        
        ax.set_ylim([0,1])
        ax.set_xlabel('Iteration',fontsize=30)
        ax.set_ylabel('Distance to Id',fontsize=30)
        ax.legend(fontsize=30)
        # ax.set_yscale('log')
        
        output_file_name = fig_dir + '/rotation_demonstration_dist_to_id_adam_{}'.format(l) + test_mask_specs
        plt.savefig(output_file_name + '.pdf', dpi=600)

        tikzplotlib.save(output_file_name + '.tex')
    


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
        # 'subset_size', 'amount_of_examples', # is in specs
        'hist_stride', 'stride', 
        'train_mask_specs', 'test_mask_specs',
        'train_time', 'test_time',
        'train_loss_mean_final', 'train_loss_std_final', 'train_loss_min',
        'test_loss_mean_final', 'test_loss_std_final', 'test_loss_min'
        ]
    ]

variables_df = pd.DataFrame(variables_list, columns=['variable name', 'value'])
variables_df.columns=pd.MultiIndex.from_product([[os.path.basename(__file__)],variables_df.columns])

variables_df.to_latex(fig_dir + '/' + fig_dir.removeprefix('figures_') + '_vars.tex', index=False)
variables_df
# %%
