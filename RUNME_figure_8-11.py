# -*- coding: utf-8 -*-
"""

Script for creating the subfigures of Figures 8 through to 11 in the paper 
"On Optimal Regularization Parameters via Bilevel Learning" (Ehrhardt, Gazzola, Scott, 2023)


@author: Sebastian J. Scott (ss2767@bath.ac.uk)
"""
#%% Specify subfigure to be generated

# To create some of the subfigures in Figures 8-11, uncomment the relevant line below.
# Running the script will then generate the relevant subfigure. The approximate time to 
#   generate said subfigures is also indicated


setup = {"ul":"MSE", "reg":"hub", "A_choice":"id", "pmax":1e-1}           # Fig 8,9a,9c (1 sec)
# setup = {"ul":"MSE", "reg":"hubTV", "A_choice":"id", "pmax":5e+0}         # Fig 8,9b,9d (3 hrs)

# setup = {"ul":"MSE", "reg":"hubTV", "A_choice":"blur", "pmax":3e+0}       # Fig 10,11a,11c (4.5hrs)
# setup = {"ul":"pred_risk", "reg":"hubTV", "A_choice":"blur", "pmax":3e+0} # Fig 10,11b,11d (4.5hrs)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###  You do not need to modify any of the below code to create the figures  ###
###############################################################################
#%% Import relevant modules
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as image

from bilevel import *
import odl
        
import time

#%% Useful functions

def axis_fun(ax):
    ax.spines['left'].set_position('zero')# set the x-spine
    ax.spines['right'].set_color('none')# turn off the right spine/ticks
    ax.yaxis.tick_left()
    ax.spines['bottom'].set_position('zero')# set the y-spine
    ax.spines['top'].set_color('none')# turn off the top spine/ticks
    ax.xaxis.tick_bottom()


#  Display function
def display_ele(odl_ele, cmap='gray' , clip=True,clip_low=0,clip_max=255):
    fig, ax = plt.subplots(figsize=[8,8])
    if clip:
        plt.imshow(np.clip(odl_ele,clip_low,clip_max) , cmap=cmap)
    else:
        im = plt.imshow(odl_ele, cmap=cmap)
        cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)
    ax.set_axis_off()
    plt.tight_layout()   
    
def sum_over_samples(x,y,fun,num):
    if not isinstance(x,list):
        return fun(x,y)
    out = 0
    for ind in range(dat_num):
        out += fun(x[ind],y[ind])
    return out    

#%% Generate data

dat_num = 1   # Number of data samples
n = 256 # Width of (square) image

# Default arguments
space = odl.uniform_discr(min_pt=[0,0], max_pt=[n,n], shape=[n,n])
L,  gamma = None, None
use_old_theory = False
cold_start = True
plot_extra_info = False

verbose =  2

noiselevel = 0.05

ll_max_iter = 10000 # Maximum number of iterations for BFGS


#############################################
#######   Parameter search space   ########## 

param_max = setup["pmax"]
param_num = 50 # Number of regularisation parameters to be considered
param_space = [np.flip(np.linspace(0,param_max,param_num))]

###################################################
#######   Specify the forward operator   ########## 

A_choice = setup["A_choice"]
if A_choice == "id": A, use_old_theory = odl.IdentityOperator(space), True
else:
    filter_width = 1.3  # Standard deviation of the Gaussian filter
    ft = odl.trafos.FourierTransform(space)
    c = filter_width ** 2 / 4.0 ** 2
    gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * c))
    A = ft.inverse * gaussian * ft


#########################################################
###   Specify upper level loss and regualariser    ######

ul = setup["ul"]
reg_type = setup["reg"]
if reg_type =="hub":  reg, gamma ='huber', 0.01
else: reg, gamma , L ='L-huber', 0.01, odl.Gradient(space)

problem_detail_string = "A: "+setup["A_choice"] +  " |  Upper level: "+ul+"  |  Regulariser: "+reg_type
print("\n"+problem_detail_string+"\n")
#%% Load image

np.random.seed(4) # Reproducability

img = image.open("./test_image.png")

xexact = space.element(img.resize([n,n]).convert('L') )

yns = A(xexact)
noise = odl.phantom.white_noise(A.range)
yns += noise * yns.norm() / noise.norm() * noiselevel

#%% Calculate unreguarised reconstruction
xnaive = space.zero()
x0_ll_eval, x0_grad_norms, x0_stopit, x0_step_lengths, x0_backtrack_its, x0_convg_flag = bfgs_method(f=0.5*odl.solvers.L2NormSquared(space)*(A-yns), x=xnaive, maxiter=ll_max_iter, tol=1e-8, verbose=1)

#%% Display information of unregularized reconstruction
if plot_extra_info:
    fig,ax = plt.subplot_mosaic([['ll_eval', 'll_gnorm'],
                        ['step_length','backtrack_its']
                        ],figsize=(10,6))
    
    
    ax['ll_eval'].plot(x0_ll_eval)
    ax['ll_eval'].set_title('Lower level evaluation')
    ax['ll_eval'].set_yscale('log')
            
    ax['ll_gnorm'].plot(x0_grad_norms)
    ax['ll_gnorm'].set_title('Log gradient norm of lower level')
    ax['ll_gnorm'].set_yscale('log')
    
    ax['step_length'].plot(x0_step_lengths)
    ax['step_length'].set_title('Step length')
    ax['step_length'].set_yscale('log')        
    
    ax['backtrack_its'].plot(x0_backtrack_its)
    ax['backtrack_its'].set_title('Number of backtracking iterations')
    
    plt.suptitle( 'BFGS info of unregularized reconstruction\n'+problem_detail_string+'\nStopit:'+str(x0_stopit)+
                  ' Convg flag:'+str(x0_convg_flag))
    
    plt.tight_layout()


#%% Display ground truth, measurement, and unregularized reconstruction

display_ele(xexact)
display_ele(yns)
if A_choice == "blur":
    display_ele(xnaive)

# %% Specify upper and lower level problems

datafit_fun,  reg_fun, breg_fun , reg_grad, ll_analytic_soln= create_ll_problem(A,reg, n, L=L, gamma=gamma)
if use_old_theory and reg=='huber':
     def _huber_prox(param,y):
         out = ( 1 - param/(param+1))*(y/gamma)
         inds = np.abs(y/gamma) > param + 1
         out[inds] = ( 1 - param/(np.abs(y[inds]/gamma)))*y[inds]/gamma
         return gamma * out
             
     ll_analytic_soln = lambda params, y : _huber_prox(params[0] , y)

# ## Predictive risk
if ul == 'pred_risk': 
    use_old_theory = False
    ul_fun_summand = lambda x , xtrue : 0.5 * (odl.solvers.L2NormSquared(A.range)* A)(x - xtrue)
    
    # functions that evaluate our condition
    theory_big = lambda xtrue, x0 : reg_fun( x0)
    theory_sml = lambda xtrue, x0 : reg_fun(xtrue) - breg_fun( xtrue,x0)
    theory_new_fun = lambda xtrue, x0 : theory_big(xtrue,x0) > theory_sml(xtrue,x0)
    sml,big = 'R(x*) - D(x*,x0): ' , 'R(x0)           : '

# ## Mean squared error
elif ul == 'MSE':
    ul_fun_summand = lambda x , xtrue : 0.5 * odl.solvers.L2NormSquared(space)(x - xtrue)
    
    # functions that evaluate our condition
    AtAinv = (A.adjoint@A).inverse

    theory_big = lambda xtrue, x0 : reg_fun(AtAinv ( x0)) - breg_fun(AtAinv (x0),x0)
    theory_sml = lambda xtrue, x0 : reg_fun(AtAinv (xtrue)) - breg_fun(AtAinv( xtrue),x0)
    theory_new_fun = lambda xtrue, x0 : theory_big(xtrue,x0) > theory_sml(xtrue,x0)
    sml,big = 'R(Bx*) - D(Bx*,x0): ' , 'R(Bx0) - D(Bx0,x0): '
    
if use_old_theory:
    theory_old_fun = lambda xtrue, x0 : reg_fun( x0) > reg_fun(  xtrue)
else:
    theory_old_fun = lambda xtrue, x0 : 'N/A'
    
ul_fun = lambda x, xtrue : sum_over_samples(x, xtrue, ul_fun_summand, dat_num)
ll_fun = lambda params, y : datafit_fun(y) + params[0]*reg_fun

print('################################')
if use_old_theory:
    print('R(x*)             : ' +str(reg_fun(xexact)))
    print('R(y)              : ' +str(reg_fun(yns)))
    print('Old Theory satisfied : '+str(theory_old_fun(xexact,xnaive))+"\n")
print(sml+str(theory_sml(xexact,xnaive)))
print(big+str(theory_big(xexact,xnaive)))
print('New Theory satisfied: '+str(theory_new_fun(xexact,xnaive)))
print('################################\n')

del sml,big, theory_big,theory_sml

#%% Brute Force Search method
t0 = time.time()
out_bfs = bilevel(xtrue=xexact, ynoise=yns, A=A, ul_fun=ul_fun, ll_fun = ll_fun,method='BFS',  param_space=param_space, analytic = ll_analytic_soln, verbose=verbose,
                  cold_start=cold_start,ll_max_iter=ll_max_iter) 
t1 = time.time()
#%% Solution to bilevel learning problem and associated reconstruction
display_ele(out_bfs.soln.xrecon)
print('\n######################################\nUpper level:       '+ul)
print('Optimal parameter: '+str(out_bfs.soln.param[0])+'    ['+str(param_space[0][0])+','+str(param_space[0][-1])+']\nRun time:        '+time.strftime("%H:%M:%S", time.gmtime(t1-t0))+'\n######################################\n')

#%% Plot of upper level cost

fs = 20
plt.rcParams.update({'font.size': 15})
ul_cost = out_bfs.get('ul_cost')
fig, ax = plt.subplots(figsize=(6,5) )

ax.plot(param_space[0],(np.array(ul_cost)) , lw=3)
ax.plot(param_space[0][np.argmin(ul_cost)] , ul_cost[np.argmin(ul_cost)], '*', markersize=15,markeredgecolor="black", markeredgewidth=1)
ax.set_xlabel(r'$\alpha$', fontsize=fs)

top = max(np.array(ul_cost)[np.array(ul_cost) < 1e+300])
bot = min(np.array(ul_cost)[np.array(ul_cost) < 1e+300])
ax.set_ylim([bot - (top-bot)*0.1,top])
ax.set_xlim([(param_space[0].min() - param_space[0].max())*.05,param_space[0].max()*1.05])
if A_choice =='blur': ax.ticklabel_format(axis='both',style='sci', scilimits=(0,0))
plt.tight_layout()

#%% Optional convergence plots
if plot_extra_info:
    if not( A_choice == 'id' and reg == 'huber') : # Huber norm denosing utilises analytic solution
        if param_num > 1:
            fig,ax = plt.subplot_mosaic([#['final_grad_norms','final_cost'],
                                ['it_num','stop_flag']], figsize=(12,5))
            all_info = out_bfs.get('info')
            final_gnorms = np.array([info['ll_gnorm'][-1] for info in all_info])
            final_costs = np.array([info['ll_eval'][-1] for info in all_info])
            it_nums = np.array([info['ll_its'] for info in all_info])
            stopping_flags = np.array([info['flag'] for info in all_info])
            
            ax['it_num'].plot(param_space[0],it_nums)
            ax['it_num'].set_xlabel(r'$\alpha$')
            ax['it_num'].set_title('Stopping iteration')
            
            ax['stop_flag'].plot(param_space[0],stopping_flags)
            ax['stop_flag'].set_xlabel(r'$\alpha$')
            ax['stop_flag'].set_title('Convergence flag\n0: grad tol reached\n1: max iterations reached\n2: step size sufficiently small')
            
            plt.suptitle(problem_detail_string+'\n'+r'BFGS results for the considered $\alpha$')
            
            plt.tight_layout()
            
        fig,ax = plt.subplot_mosaic([['ll_gnorm','ll_eval'],
                            ['steplength','backtrack_its']], figsize=(12,6))
        
        ax['ll_gnorm'].plot(out_bfs.soln.info['ll_gnorm'])
        ax['ll_gnorm'].set_title('Gradient norm of LL')
        ax['ll_gnorm'].set_yscale('log')
        
        ax['ll_eval'].plot(out_bfs.soln.info['ll_eval'])
        ax['ll_eval'].set_title('Evaluation of LL')
        ax['ll_eval'].set_yscale('log')
        
        ax['steplength'].plot(out_bfs.soln.info['step_length'])
        ax['steplength'].set_title('Determined step length')
        ax['steplength'].set_yscale('log')
        
        ax['backtrack_its'].plot(out_bfs.soln.info['backtrack_its'])
        ax['backtrack_its'].set_title('Number of backtracking iterations')
        plt.suptitle(problem_detail_string+'\nBFGS results for the optimal reconstruction\n'+r'$\alpha=$'+str(out_bfs.soln.param[0])
                     + '\nStopping it:'+str(out_bfs.soln.info['ll_its'])+'  Convergence flag:'+str(out_bfs.soln.info['flag']) )
        plt.tight_layout()