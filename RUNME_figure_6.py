# -*- coding: utf-8 -*-
"""

Script for creating the subfigures of Figure 6 in the paper 
"On Optimal Regularization Parameters via Bilevel Learning" (Ehrhardt, Gazzola, Scott, 2023)

The experiment illustrates how in a realistic denoising setting, zero is not a
solution to the bilevel problem almost surely. We demonstrate that the assumption 
of zero mean noise is tight in deducing this.

@author: Sebastian J. Scott (ss2767@bath.ac.uk)
"""

setup = {"centre" : [0,0]}      # Fig 6a,6b (30 sec)
setup = {"centre" : [-.1,0] }   # Fig 6c,6d (30 sec)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###  You do not need to modify any of the below code to create the figures  ###
###############################################################################
#%% Import relevant modules
import numpy as np
from numpy.random import default_rng

import scipy.ndimage
import matplotlib.pyplot as plt

from tqdm import tqdm

import odl

from bilevel import *


#%% Useful functions
def axis_fun(ax,col='black'):
    ax.spines['left'].set_position('zero')# set the x-spine
    ax.spines['right'].set_color('none')# turn off the right spine/ticks
    ax.yaxis.tick_left()
    ax.spines['bottom'].set_position('zero')# set the y-spine
    ax.spines['top'].set_color('none')# turn off the top spine/ticks
    ax.xaxis.tick_bottom()
    
    #color xaxes
    ax.spines['bottom'].set_color(col)
    ax.xaxis.label.set_color(col)
    ax.tick_params(axis='x', colors=col)
    
    ax.spines['left'].set_color(col)
    ax.yaxis.label.set_color(col)
    ax.tick_params(axis='y', colors=col)
 

#%% Generate data

use_odl = False
rng = default_rng(4)# reproducability

dat_num = 1000  # int : number of data samples
n = 2      # int: length of signal

ll_linesearch = "backtracking" # BFGS step-size method

#############################################
#######   Parameter search space   ########## 
param_num = 50 # int : number of regularisation parameters to be considered
param_space = [np.linspace(0,0.1,param_num)]


botleft = np.array([-1.6,-1.6])
topright = np.array([1.6,1.6])

#############################################
#######     Generate measurements   ######### 

if use_odl:     
    space = odl.rn(n)
    A, show_old_theory = odl.IdentityOperator(space) ,True
    xexact = [space.element([1,.0]) for _ in range(dat_num)] 
else: 
    A, show_old_theory = np.eye(2) ,True
    xexact = [np.array([1,.0]) for _ in range(dat_num)] 

mean_centre = np.array(setup["centre"])

yns = [xexact[ind] + rng.normal(size=[n], scale=.1, loc=mean_centre ) for ind in range(dat_num)] 
verbose =  True

#%% Plot of ground truth and associated measurements

fig, ax = plt.subplots()
ax.plot(np.array(yns)[:,0],np.array(yns)[:,1],'.',color='crimson',markersize=12,markeredgecolor="black", markeredgewidth=.5,zorder=2)
ax.plot(np.array(xexact)[:,0],np.array(xexact)[:,1],'*',color='gold',markersize=18 ,markeredgecolor="black", markeredgewidth=1.8, zorder=3)

ax.set_xlim([-.1,1.3])
ax.set_ylim([-.7,.7])
axis_fun(ax)
plt.tight_layout()

# %% Specify upper and lower level problems

# Default arguments
L,  gamma = None, 0.01

reg = 'strict-huber'

datafit_fun,  reg_fun, breg_fun , reg_grad, ll_analytic_soln = create_ll_problem(A, reg, n, L=L, gamma=gamma,use_odl=use_odl)

def sum_over_samples(x,y,fun,dat_num):
    if not isinstance(x,list):
        return fun(x,y)
    out = 0
    for ind in range(dat_num):
        out += fun(x[ind],y[ind])
    return out/dat_num

# ## Mean squared error
if use_odl: 
    ul_fun_summand = lambda x , xtrue : 0.5 * odl.solvers.L2NormSquared(space)(x - xtrue)
    ll_fun = lambda params, y : datafit_fun(y) + params[0]*reg_fun
    AtAinv = (A.adjoint@A).inverse
else:
    ul_fun_summand = lambda x,xtrue : 0.5*np.linalg.norm(x-xtrue)**2
    ll_fun = lambda params, y : lambda x : datafit_fun(y)(x) + params[0]*reg_fun(x)
    ll_grad = lambda params, y : lambda x: A.T@A@x - A.T@y + params[0]*reg_grad(x)
    AtAinv = lambda x : np.linalg.inv(A.T@A) @ x

ul_fun = lambda x, xtrue : sum_over_samples(x, xtrue, ul_fun_summand, dat_num)

# Functions that evaluate the theoretical conditions
theory_old_fun = lambda xtrue, x0 : reg_fun( x0) > reg_fun(  xtrue)
theory_new_fun = lambda xtrue, x0 : reg_fun(AtAinv ( x0)) - breg_fun(AtAinv (x0),x0) > reg_fun(AtAinv (xtrue)) - breg_fun(AtAinv( xtrue),x0)


#%% Evaluate upper level for different parameter values
out_bfs = bilevel(xtrue=xexact,  ynoise=yns, A=A, ul_fun=ul_fun, ll_fun = ll_fun, ll_grad=ll_grad, method='BFS', param_space=param_space, analytic = ll_analytic_soln, verbose=verbose, ll_linesearch = ll_linesearch, use_odl=use_odl) 

#%%  Plot upper level cost

ul_vs_alpha = out_bfs.get('ul_cost')
fs = 20
plt.rcParams.update({'font.size': 15})
ul_cost = out_bfs.get('ul_cost')
fig, ax = plt.subplots(figsize=(6,5) )
ax.plot(param_space[0],ul_vs_alpha , lw=3)
ax.plot(param_space[0][np.argmin(ul_vs_alpha)] , ul_vs_alpha[np.argmin(ul_vs_alpha)], '*', markersize=15,markeredgecolor="black", markeredgewidth=1)
ax.set_xlabel(r'$\alpha$', fontsize=fs)
plt.tight_layout()


# %% Calculate proportions of samples that satify old and new theory

satisfied_old = 0
satisfied_new = 0
for ind in range(dat_num):
    if theory_old_fun(xexact[ind],yns[ind]): satisfied_old += 1
    if theory_new_fun(xexact[ind],yns[ind]): satisfied_new += 1
print('\n#######################\nPoints satisfying:')
if show_old_theory:
    print('Old Theory: ' +str(satisfied_old)+'/'+str(dat_num)+', '+str(satisfied_old/dat_num*100)+'%')
print('New Theory: ' +str(satisfied_new)+'/'+str(dat_num)+', '+str(satisfied_new/dat_num*100)+'%')

