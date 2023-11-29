# -*- coding: utf-8 -*-
"""

Script for creating the subfigures of Figures 4 and 5, and creating Figure 7 in 
the paper "On Optimal Regularization Parameters via Bilevel Learning"
(Ehrhardt, Gazzola, Scott, 2023)

These Figures consider a 2-dimensional problem and illustrate the regions where 
our new condition is satisfied, and (if applicable) the old condition is satisfied, and 
also where 0 is a solution to the bilevel learning problem.

@author: Sebastian J. Scott (ss2767@bath.ac.uk)
"""

#%% Specify subfigure to be generated

# To create one of the subfigures in Figure 4-5 and Figure 7, uncomment the relevant line below.
# Running the script will then generate the relevant subfigure. The approximate time to 
#   generate said subfigure is also indicated


# setup = {"ul":"MSE", "reg":"tikh", "A_choice":"id"}           # Fig 4a (1 minute)
# setup = {"ul":"MSE", "reg":"H1", "A_choice":"id"}             # Fig 4b (20 minutes)
setup = {"ul":"MSE", "reg":"hub", "A_choice":"id"}            # Fig 4c (2 minutes)
# setup = {"ul":"MSE", "reg":"hubTV", "A_choice":"id"}          # Fig 4b (2 hours)

# setup = {"ul":"MSE", "reg":"tikh", "A_choice":"blur"}         # Fig 5a (22 mins)
# setup = {"ul":"MSE", "reg":"H1", "A_choice":"blur"}           # Fig 5b (25 mins)
# setup = {"ul":"MSE", "reg":"hub", "A_choice":"blur"}          # Fig 5c (31 mins)
# setup = {"ul":"MSE", "reg":"hubTV", "A_choice":"blur"}        # Fig 5d (36 mins)

# setup = {"ul":"pred_risk", "reg":"tikh", "A_choice":"deblur"} # Fig 7 (30 mins)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###  You do not need to modify any of the below code to create the figures  ###
###############################################################################
#%% Import relevant modules

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import colors 
from matplotlib.lines import Line2D

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
    
def gaussian_1d_forward_op(sigma,n):
    Afull = np.empty([n,n])
    for i in range(n):
        vec = np.zeros(n)
        vec[i] = 1
        Afull[:,i] = scipy.ndimage.gaussian_filter(vec,sigma=sigma)
    return Afull

def sum_over_samples(x,y,fun,num):
    if not isinstance(x,list):
        return fun(x,y)
    out = 0
    for ind in range(dat_num):
        out += fun(x[ind],y[ind])
    return out

def spin(arr):
    return np.flip(np.flip(arr, axis=0),1).T

def bool_contour(bool_array):
    # Return boolean array that is a "gradient" with reflective boundary conditions
    bool_contour = bool_array.copy()
    bool_contour2 = bool_array.copy()
    for ind in range(len(bool_array)):
        vert_tmp = bool_array[ind].copy()
        vert_tmp[0:-2] = vert_tmp[1:-1]
        vert_tmp[-1] = vert_tmp[-2] # boundary condition
        bool_contour[ind] =  vert_tmp ^ bool_array[ind]
        
        horiz_tmp = np.array([b[ind] for b in bool_array])
        bool_vert_tmp = horiz_tmp . copy()
        horiz_tmp[0:-2] = horiz_tmp[1:-1]
        horiz_tmp[-1] = horiz_tmp[-2]
        bool_contour2[ind] = horiz_tmp ^ bool_vert_tmp
    out = bool_contour | bool_contour2.T
    out[out] = np.NaN
    return out


#%% Specification of the bilevel problem

dat_num = 1   # int : number of data samples
n = 2         # int: length of signal

# Default arguments
L,  gamma = None, None
space = odl.rn(n)
show_old_theory = False # Old theory is only for denoising


use_odl = False # Whether ODL should be utilised or if scipy should be used instead

verbose = False
display_bfgs_info = False # Display extra information regarding the solution

############################################
###### Options for lower level solver   ####
ll_max_iter = 10000 # Maximum number of iterations for BFGS to solve lower level

############################################
#######   Parameter search space   ######### 
param_num = 100 # Number of regularisation parameters to be considered
param_space = [np.concatenate(([0] ,10**np.linspace(-8,3,param_num-2),  [1e+7])) ]

##################################################
#########   Region of search variables   ######### 
mesh_size = 150

# Boundary of the reconstruction space to be considered
botleft = np.array([-1.6,-1.6])
topright = np.array([1.6,1.6])

#################################################
########     Forward operator    ################ 

if setup["A_choice"]=="id":
    A, show_old_theory = np.eye(n) ,True
else:
    A = gaussian_1d_forward_op(0.8,n)

if use_odl: A = odl.MatrixOperator(A)

#######################################################
###  Specify upper level loss and regualariser   ######
ul = setup["ul"]
reg_type = setup["reg"]

if reg_type == "tikh": reg="l2"
elif reg_type == "H1":
    reg,L = "L-l2", np.array([[1,-1]],dtype='float') 
    if use_odl: L = odl.MatrixOperator(L)
elif reg_type =="hub": reg,gamma = "huber",0.01
else: # Huberised TV
    reg,gamma,L="L-huber",0.01, np.array([[1,-1]],dtype='float') 
    if use_odl: L = odl.MatrixOperator(L)

###################################################
########   Ground truth and noisy measurement   ###
if use_odl:
    xexact = [space.element([1,.5])]
else:
    xexact = [np.array([1,.5])]
    
problem_detail_string = "A: "+setup["A_choice"] +  " |  Upper level: "+ul+"  |  Regulariser: "+reg_type
print("\n"+problem_detail_string+"\n")
# %% Specify lower level problems

datafit_fun,  reg_fun, breg_fun , reg_grad, ll_analytic_soln = create_ll_problem(A, reg, n, L=L, gamma=gamma, use_odl=use_odl)

if reg=='l2':
    if show_old_theory:
        ll_analytic_soln = lambda params, y :  y / (1+params[0]) 
elif reg == 'huber':
    if show_old_theory: # solution to denoising
        def _huber_prox(param,y):
            out = ( 1 - param/(param+1))*(y/gamma)
            inds = np.abs(y/gamma) > param + 1
            out[inds] = ( 1 - param/(np.abs(y[inds]/gamma)))*y[inds]/gamma
            return gamma * out
                
        ll_analytic_soln = lambda params, y : _huber_prox(params[0] , y)
        

#%% Create upper level cost function and the theory conditions to check

# ## Predictive risk
if ul == 'pred_risk':
    if use_odl:
        ul_fun_summand = lambda x , xtrue : 0.5 * (odl.solvers.L2NormSquared(A.range)* A)(x - xtrue)
    else:
        ul_fun_summand = lambda x , xtrue : 0.5 * np.linalg.norm(A@(x - xtrue))**2
        
    # functions that evaluate our conditions
    theory_old_fun = lambda xtrue, x0 : reg_fun( x0) > reg_fun( xtrue)
    theory_new_fun = lambda xtrue, x0 : reg_fun( x0) > reg_fun(xtrue) - breg_fun( xtrue,x0)


# ## Mean squared error
elif ul == 'MSE':
    if use_odl:
        ul_fun_summand = lambda x , xtrue : 0.5 * odl.solvers.L2NormSquared(space)(x - xtrue)
        AtAinv = lambda x : np.linalg.pinv((A.adjoint.matrix)@A.matrix) @ x
        theory_new_fun = lambda xtrue, x0 : reg_fun(AtAinv ( x0)) - breg_fun(AtAinv (x0),x0) > reg_fun(AtAinv (xtrue)) - breg_fun(AtAinv( xtrue),x0)
    else:
        ul_fun_summand = lambda x , xtrue : 0.5 * np.linalg.norm(x - xtrue)**2
        AtAinv =  np.linalg.pinv(A.T@A)
        theory_new_fun = lambda xtrue, x0 : reg_fun(AtAinv @( x0)) - breg_fun(AtAinv@ (x0),x0) > reg_fun(AtAinv @(xtrue)) - breg_fun(AtAinv@( xtrue),x0)

    theory_old_fun = lambda xtrue, x0 : reg_fun( x0) > reg_fun(  xtrue)
    if show_old_theory:
        theory_new_fun = lambda xtrue, x0 : reg_fun(x0) > reg_fun( (xtrue)) - breg_fun(( xtrue),x0)


ul_fun = lambda x, xtrue : sum_over_samples(x, xtrue, ul_fun_summand, dat_num)

if use_odl:
    ll_fun = lambda params, y:  datafit_fun(y) + params[0]*reg_fun
    ll_grad = None # gradient is a property of ll_fun in this case
else:
    ll_fun = lambda params, y: lambda x:  datafit_fun(y)(x) + params[0]*reg_fun(x)
    ll_grad = lambda params, y : lambda x: A.T@A@x - A.T@y + params[0]*reg_grad(x)



# %% Compute theory condition boundaries for positivity: 
   
reg_xtrue = reg_fun(xexact[0])
plt_mesh_horiz = np.linspace(topright[0],botleft[0],mesh_size) # discretise grid
plt_mesh_vert  = np.linspace(botleft[1],topright[1],mesh_size) # discretise grid

reg_flag = np.ndarray([mesh_size,mesh_size], dtype='bool')      # old theory
theory_flag = np.ndarray([mesh_size,mesh_size], dtype='bool')   # our condition - Bregman

# This is performed with respect to the reconstruction space
for total_ind in tqdm(range(mesh_size**2), desc="Theory"):
        ind2 = total_ind%mesh_size
        ind1 = int((total_ind-ind2)/mesh_size)
        
        # Determine point in reconstruction space to consider
        if use_odl: point = space.element(np.array([plt_mesh_horiz[ind1],plt_mesh_vert[ind2]]))
        else: point = np.array([plt_mesh_horiz[ind1],plt_mesh_vert[ind2]])
        
        # Update relevant quantities of interest
        reg_flag[ind1][ind2] =  theory_old_fun(xexact[0],point)
        theory_flag[ind1][ind2] = theory_new_fun(xexact[0],point)
        
del point, total_ind, ind1, ind2

# Correct orientation
reg_flag = spin(reg_flag)
theory_flag = spin(theory_flag)

reg_contour = np.full((mesh_size,mesh_size) ,1. )
reg_contour[reg_flag] = np.NaN 
theory_contour = np.full((mesh_size,mesh_size) ,4. )
theory_contour[theory_flag] = np.NaN 


# %%  Calculate heat map of optimal parameters

param_mesh = np.ndarray([mesh_size,mesh_size])  # optimal parameter
ul_evald = np.ndarray([mesh_size,mesh_size])    # evaluation of upper level cost
dini_deriv = np.ndarray([mesh_size,mesh_size])  # dini derivative
soln_its = np.ndarray([mesh_size,mesh_size])  # number of its of LL its used at soln
soln_grad = np.ndarray([mesh_size,mesh_size])  # Grad tol of LL at solution
soln_flag = np.ndarray([mesh_size,mesh_size])  # Grad tol of LL at solution
        
for total_ind in tqdm(range(mesh_size**2), desc="Numerics"):
        ind2= total_ind%mesh_size
        ind1 = int((total_ind-ind2)/mesh_size)
        
        # Determine point in reconstruction space to consider
        if use_odl:
            point = space.element([plt_mesh_horiz[ind1],plt_mesh_vert[ind2]])
            yns_tmp = [A( point)] # Assuming A is invertible
        else:
            point = np.array([plt_mesh_horiz[ind1],plt_mesh_vert[ind2]])
            yns_tmp = [A@( point)] # Assuming A is invertible
            
        # Calculate solution to bilevel problem
        out_bfs_tmp = bilevel(xtrue=xexact,  ynoise=yns_tmp, A=A, ul_fun=ul_fun, ll_fun = ll_fun, method='BFS', param_space=param_space, analytic = ll_analytic_soln, verbose=verbose, ll_max_iter = ll_max_iter, ll_grad=ll_grad, use_odl=use_odl) 
        
        # Update relevant quantities of interest
        param_mesh[ind1][ind2]= out_bfs_tmp.soln.param      
        ul_evald[ind1][ind2] = ul_fun([point], xexact) # evaluation of upper level cost at reconstruction point  
        dini_deriv[ind1][ind2] = (out_bfs_tmp.recons[1].ul_cost  - out_bfs_tmp.recons[0].ul_cost)/out_bfs_tmp.recons[1].param[0]
        soln_its[ind1][ind2] = out_bfs_tmp.soln.info['ll_its']
        soln_grad[ind1][ind2] = out_bfs_tmp.soln.info['ll_gnorm']
        soln_flag[ind1][ind2] = out_bfs_tmp.soln.info['flag']
        
del yns_tmp, out_bfs_tmp, point, total_ind, ind1, ind2

# Correct orientation
param_mesh = spin(param_mesh)
ul_evald = spin(ul_evald)
dini_deriv = spin(dini_deriv)
soln_grad = spin(soln_grad)
soln_its = spin(soln_its)
soln_flag = spin(soln_flag)



#%% Plot convergence results of BFGS

if display_bfgs_info:
    fig,ax = plt.subplot_mosaic([['stopit', 'gnorm','flag']],figsize=(18,7))
    
    tmp = ax['stopit'].imshow(soln_its)
    plt.colorbar(tmp, ax=ax['stopit'],fraction=0.046, pad=0.04)
    ax['stopit'].set_title('Stopping iteration')
    ax['stopit'].axis('off')
    
    soln_grad_trunc = soln_grad.copy()
    soln_grad_trunc[soln_grad_trunc<1e-16] = 1e-16
    tmp = ax['gnorm'].imshow(np.log10(soln_grad_trunc))
    plt.colorbar(tmp, ax=ax['gnorm'],fraction=0.046, pad=0.04)
    ax['gnorm'].set_title('Log gradient norm')
    ax['gnorm'].axis('off')
    
    tmp = ax['flag'].imshow(soln_flag)
    plt.colorbar(tmp, ax=ax['flag'],fraction=0.046, pad=0.04)
    ax['flag'].set_title('Convergence flag\n0 - Grad tol reached\n1 - Max iterations reached\n2 - Step size sufficently small')
    ax['flag'].axis('off')
    
    plt.suptitle(problem_detail_string+'\nBFGS info at solutions')

#%% Compute condition boundaries and useful variables for trajectory plots

extent=[plt_mesh_horiz.min(),plt_mesh_horiz.max(),plt_mesh_vert.min(),plt_mesh_vert.max()]
# colors to plot the contours
col0 = "gold"       # gold
col1 = "black"      # black - numerically computed
col2 = "tab:red"    # red   - old theory
col3 = "royalblue"  # blue  - new theory

cmap0 = colors.LinearSegmentedColormap.from_list("", [col0,col0])
cmap1 = colors.LinearSegmentedColormap.from_list("", [col1,col1])
cmap2 = colors.LinearSegmentedColormap.from_list("", [col2,col2])
cmap3 = colors.LinearSegmentedColormap.from_list("", [col3,col3])

empty_display = np.empty([mesh_size,mesh_size])
empty_display[:] = np.nan

theory_display = np.empty([mesh_size,mesh_size])
theory_display[:] = np.nan
theory_display[bool_contour(theory_flag)] = 1

heur_display = np.empty([mesh_size,mesh_size])
heur_display[:] = np.nan
heur_display[bool_contour(reg_flag)] = 1

numeric_region = np.ones([mesh_size,mesh_size])
numeric_region[param_mesh>1e-12] = np.nan

numeric_display = np.empty([mesh_size,mesh_size])
numeric_display[:] = np.nan
numeric_display[bool_contour(param_mesh>1e-12)] = 1

#%% Calculate ratio of area increase of theory compared to numeric

count_zero = mesh_size**2 - sum(sum(param_mesh>1e-12))
count_new = mesh_size**2 - sum(sum(theory_flag))
count_old = mesh_size**2 - sum(sum(reg_flag))

print("\n###############################################")
print("CONDITION   | NO. POINTS | AREA INCREASE RATIO")
print("Zero optimal|    "+str(count_zero)+"    | n/a")
print("New theory  |    "+str(count_new)+"    | "+str(count_new/count_zero))
if show_old_theory:
    print("Old theory  |    "+str(count_old)+"    | "+str(count_old/count_zero))
print("###############################################\n")

#%% Plot of theory boundaries

fs=15 # Font size
lw = 5 # Line width

fig, ax = plt.subplots(figsize=(7,7))
ax.contour( ul_evald, levels=15, cmap='binary',alpha=1, extent=extent, origin='upper')

ax.plot(xexact[0][0],xexact[0][1],'*',color='gold',markersize=18 ,markeredgecolor="black", markeredgewidth=1.8, zorder=10) 
legend_elements = [Line2D([0],[0],color=col3,lw=3,label='New'),
                   Line2D([0],[0],color=col1,lw=3,label='Numerical')]
if show_old_theory:
    ax.imshow(heur_display , extent=extent,zorder=3,cmap=cmap2)
    legend_elements = [Line2D([0],[0],color=col2,lw=3,label='Old'),
                       Line2D([0],[0],color=col3,lw=3,label='New'),
                       Line2D([0],[0],color=col1,lw=3,label='Numerical')]
ax.imshow(theory_display , extent=extent,zorder=3,cmap=cmap3)
ax.imshow(numeric_region, extent=extent, alpha=0.3, cmap=cmap0,zorder=2)
ax.imshow(numeric_display, extent=extent, cmap=cmap1,zorder=2)

ax.legend(handles=legend_elements,fontsize=fs,loc='lower right')

axis_fun(ax)
plt.tight_layout()
