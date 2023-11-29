# -*- coding: utf-8 -*-
"""
Script of useful functions and class definitions for reproducing results in the paper 
"On Optimal Regularization Parameters via Bilevel Learning". 


@author: Sebastian J. Scott (ss2767@bath.ac.uk)
"""
#%%
import numpy as np
# import scipy

import odl
from tqdm.contrib.itertools import product
from scipy.optimize import fmin_bfgs

from odl.solvers.smooth.newton import _bfgs_direction

from tqdm import tqdm

#%% Construction lower level functionals    

def create_ll_problem( A, reg, n, **kwargs):
    
    # default values
    ll_analytic_soln = None
    reg_grad = None
    
    use_odl = kwargs.pop('use_odl', True)
    
    if use_odl:
        datafit_fun = lambda y : 0.5 * odl.solvers.L2NormSquared(A.range) * (A - y)
    else:
        datafit_fun = lambda y : lambda x : 0.5 * np.linalg.norm(A@x - y)**2
        
        
    ##########  Tikhonov
    if reg == 'l2': 
        if use_odl:
        
            reg_fun = 0.5*odl.solvers.L2NormSquared(A.domain)
        else:
            reg_fun = lambda x: 0.5*np.linalg.norm(x)**2
            reg_grad = lambda x : x
        # Bregman distance
        breg_fun = lambda a, b : reg_fun(a-b)
        
        
    ##########  Generalised Tihkonov
    elif reg == 'L-l2': 
        if use_odl:
            L = kwargs.pop('L' , odl.IdentityOperator(A.domain))
            
            reg_fun = 0.5 * odl.solvers.L2NormSquared(L.range) * L
            try:
                tmp1,tmp2 = type(A.matrix),type(L.matrix)
                if tmp1 is np.ndarray and tmp2 is np.ndarray:
                    Amat, Lmat = A.matrix, L.matrix
                    ll_analytic_soln = lambda params, y :   np.linalg.solve( Amat.T @ Amat + params[0]*Lmat.T@ Lmat , Amat.T @ y) 
                del tmp1,tmp2
            except:
                pass
        else:
            L = kwargs.pop('L' , np.eye(A.shape[1]))
            reg_fun = lambda x :  0.5 * np.linalg.norm(L@x)**2
            reg_grad = lambda x : L.T@L@x
            
        # Bregman distance
        breg_fun = lambda a, b : reg_fun(a-b)
        
    
    ##########  Huber norm
    elif reg == 'huber':
        gamma = kwargs.pop('gamma', 0.01) # Smoothing constant
        if gamma is None: gamma= 0.01 # Account for empty input
        
        if use_odl: # Utilise ODL
            reg_fun = odl.solvers.Huber(A.domain , gamma=gamma)
            breg_fun = lambda a,b : reg_fun(a) - reg_fun(b) - A.domain.inner(reg_fun.gradient(b) , a-b )
            
        else: # Do not utilise ODL
            def _huber_fun(x):
                out = x**2 / (2*gamma)
                inds = np.abs(x) > gamma
                out[inds] = np.abs(x[inds]) - gamma/2
                return sum(out)
            reg_fun = _huber_fun
            
            def _huber_grad(x):
                out = x/gamma
                inds = np.abs(x) > gamma
                out[inds] =  np.sign(x[inds])
                return out
            reg_grad = lambda x:  _huber_grad(x)
            
            breg_fun = lambda  a,b : reg_fun(a) - reg_fun(b) - np.dot(reg_grad(b) , a-b)
        
            if A.shape[0]==A.shape[1]: # See if in denoising case
                if (np.eye(A.shape[0]) ==A).all():
                    def _huber_prox(param,y):
                        out = ( 1 - param/(param+1))*(y/gamma)
                        inds = np.abs(y/gamma) > param + 1
                        out[inds] = ( 1 - param/(np.abs(y[inds]/gamma)))*y[inds]/gamma
                        return gamma * out
                            
                    ll_analytic_soln = lambda params, y : _huber_prox(params[0] , y)


    ##########  Generalised Huber norm
    elif reg == 'L-huber':
        gamma = kwargs.pop('gamma', 0.01) # Smoothing constant
        if gamma is None: gamma= 0.01
        
        if use_odl:
             L = kwargs.pop('L' , odl.IdentityOperator(A.domain))
             reg_fun = odl.solvers.Huber(L.range , gamma=gamma) * L
             breg_fun = lambda a,b : reg_fun(a) - reg_fun(b) - reg_fun.gradient.range.inner(reg_fun.gradient(b) , a-b )
             
        else: # Do not utilise ODL
            L = kwargs.pop('L' , np.eye(A.shape[0]))
            def _huber_fun(x):
                out = (x)**2 / (2*gamma)
                inds = np.abs(x) > gamma
                out[inds] = np.abs(x[inds]) - gamma/2
                return sum(out)
            reg_fun = lambda x: _huber_fun(L@x)
            def _huber_grad(x):
                out = x/gamma
                inds = np.abs(x) > gamma
                out[inds] =  np.sign(x[inds])
                return out
            reg_grad = lambda x : L.T@ _huber_grad(L@x)
            breg_fun = lambda  a,b : reg_fun(a) - reg_fun(b) - np.dot(reg_grad(b) , a-b)
        
        
    ##########  Perturbed Huber norm (ensure strict convexity)
    elif reg == 'strict-huber':
        # ----  l1 penality regularisation ----
        gamma = kwargs.pop('gamma', 0.01) # Smoothing constant
        if gamma is None: gamma= 0.01
        beta = kwargs.pop('beta', 0.01) # Quadratic perturbation
        if beta is None: beta=0.01
        
        if use_odl:
            reg_fun = odl.solvers.Huber(A.domain , gamma=gamma) + beta * 0.5 * odl.solvers.L2NormSquared(A.domain)        
            breg_fun = lambda a,b : reg_fun(a) - reg_fun(b) - A.domain.inner(reg_fun.gradient(b) , a-b )
            
        else: # Do not utilise ODL
            def _huber_fun(x):
                out = (x)**2 / (2*gamma)
                inds = np.abs(x) > gamma
                out[inds] = np.abs(x[inds]) - gamma/2
                return sum(out)
            def _huber_grad(x):
                out = x/gamma
                inds = np.abs(x) > gamma
                out[inds] =  np.sign(x[inds])
                return out
            reg_fun = lambda x : _huber_fun(x) + beta/2*np.linalg.norm(x)**2
            reg_grad = lambda x : _huber_grad(x) + beta*x
            breg_fun = lambda  a,b : reg_fun(a) - reg_fun(b) - np.dot(reg_grad(b) , a-b)
            
    else:
        raise Exception("Choice of regulariser not recognised")      
    
    return datafit_fun, reg_fun, breg_fun , reg_grad, ll_analytic_soln


#%% BFGS


def bfgs_method(f, x, maxiter=1000, tol=1e-15, num_store=None,
                hessinv_estimate=None, callback=None, verbose=False):
    r"""
    The following is modified version of the BFGS method as implemented in ODL.
    https://odlgroup.github.io/odl/_modules/odl/solvers/smooth/newton.html#bfgs_method
    
    Quasi-Newton BFGS method to minimize a differentiable function.

    Can use either the regular BFGS method, or the limited memory BFGS method.

    Notes
    -----
    This is a general and optimized implementation of a quasi-Newton
    method with BFGS update for solving a general unconstrained
    optimization problem

    .. math::
        \min f(x)

    for a differentiable function
    :math:`f: \mathcal{X}\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

    .. math::
        \nabla f: \mathcal{X} \to \mathcal{X}.

    The QN method is an approximate Newton method, where the Hessian
    is approximated and gradually updated in each step. This
    implementation uses the rank-one BFGS update schema where the
    inverse of the Hessian is recalculated in each iteration.

    The algorithm is described in [GNS2009], Section 12.3 and in the
    `BFGS Wikipedia article
    <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93\
Goldfarb%E2%80%93Shanno_algorithm>`_

    Parameters
    ----------
    f : `Functional`
        Functional with ``f.gradient``.
    x : ``f.domain`` element
        Starting point of the iteration
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    num_store : int, optional
        Maximum number of correction factors to store. For ``None``, the method
        is the regular BFGS method. For an integer, the method becomes the
        Limited Memory BFGS method.
    hessinv_estimate : `Operator`, optional
        Initial estimate of the inverse of the Hessian operator. Needs to be an
        operator from ``f.domain`` to ``f.domain``.
        Default: Identity on ``f.domain``
    callback : callable, optional
        Object executing code per iteration, e.g. plotting each iterate.

    References
    ----------
    [GNS2009] Griva, I, Nash, S G, and Sofer, A. *Linear and nonlinear
    optimization*. Siam, 2009.
    """
    grad = f.gradient
    if x not in grad.domain:
        raise TypeError('`x` {!r} is not in the domain of `grad` {!r}'
                        ''.format(x, grad.domain))


    ys = []
    ss = []

    grad_x = grad(x)
    grad_norms = np.empty(maxiter+1)
    grad_norms[0] = grad_x.norm()
    f_evals = np.empty(maxiter+1)
    f_evals[0] = f(x)
    step_lengths = np.empty(maxiter)
    backtrack_its = np.empty(maxiter)
    
    # Starting step length for backtracking algorithm for first iteration of BFGS
    # step_next = f_evals[0] + grad_norms[0]/2 
    step_next = 1.

    convg_flag = 0 # Convergence flag: 
        #   0 - Gradient sufficiently small, 
        #   1 - Max number of iterations reached
        #   2 - Determined step size sufficiently small, 
    for i in tqdm(range(maxiter),disable=not(verbose)):
        # Determine a stepsize using line search
        search_dir = -_bfgs_direction(ss, ys, grad_x, hessinv_estimate)
        # dir_deriv = search_dir.inner(grad_x)
        # if np.abs(dir_deriv) == 0:
        #     return f_evals[:i], grad_norms[:i], i, step_lengths[:i], backtrack_its[:i], convg_flag # we found an optimum
        
        # step = line_search(x, direction=search_dir, dir_derivative=dir_deriv)
        step_length, step_next, armijo_satisfied, backtrack_it = backtracking(f, grad_x, x,search_dir=search_dir, step_size=step_next)
        step_lengths[i] = step_length
        backtrack_its[i] = backtrack_it
        
        
        if not armijo_satisfied:
            convg_flag = 2
            return f_evals[:i], grad_norms[:i], i, step_lengths[:i], backtrack_its[:i], convg_flag
        
        # Update x
        x_update = search_dir
        x_update *= step_length
        x += x_update

        grad_x, grad_diff = grad(x), grad_x
        grad_diff.lincomb(-1, grad_diff, 1, grad_x)

        grad_norms[i+1] = grad_x.norm()  
        f_evals[i+1] = f(x)

        y_inner_s = grad_diff.inner(x_update)

        # Test for convergence
        if np.abs(y_inner_s) < tol:
            if grad_x.norm() < tol:
                return f_evals[:i], grad_norms[:i], i+1, step_lengths[:i], backtrack_its[:i], convg_flag
            else:
                # Reset if needed
                ys = []
                ss = []
                continue

        # Update Hessian
        ys.append(grad_diff)
        ss.append(x_update)
        if num_store is not None:
            # Throw away factors if they are too many.
            ss = ss[-num_store:]
            ys = ys[-num_store:]

        if callback is not None:
            callback(x)
            
    convg_flag = 1 # Maximum number of iterations reached
    return f_evals, grad_norms, i+1, step_lengths, backtrack_its, convg_flag

def backtracking(fun, grad_s, s, search_dir, step_size=1):
    """
    Implementation of backtracking linesearch
    fun(s_{k+1}) < f(s_k) - tau* step_size * \| grad(s_k)\|**2
    step_size is the initial step size to consider
    
    INPUTS:
    fun : function of single input that returns a float
    grad_s : Evaluation of gradient at current iterate
    s : current iterate that we want to improve upon
    step_size : Initial step size to be considered
    
    OUTPUTS:
    step_size - step size that should be utilised
    flag - Flag for if the Armijo condition was satisfied
    """
    # Specify relevant quantities for the Armijo condition
    fun_s = fun(s)
    try: grad_search_inner = grad_s.inner(search_dir)
    except: grad_search_inner = np.dot(grad_s,search_dir)
    discount = 1e-2
    armijo = lambda step_size : fun(s - step_size*grad_s) < fun_s - discount*step_size*grad_search_inner
    
    factor_increase = 2.
    factor_decrease = 0.4
    
    it = 0
    
    # Terminate only if the Armijo condition is satisfied, or if the step size is too small. 
    while True:
        if armijo(step_size):
            step_next = factor_increase * step_size
            return step_size, step_next, True, it
        
        elif step_size <= 1e-14:
            # Step size is too small
            return step_size, 0, False, it
            
        else: 
            # Consider a different parameter
            step_size *= factor_decrease
            it += 1


#%% Main bilevel class

class reconstruction():
    # Class to store various useful information related to a single reconstruction
    def __init__(self,xrecon,param,ul_cost=None,ll_cost=None,alg=None,info=None):
        self.xrecon = xrecon
        self.param = param
        self.ul_cost = ul_cost
        self.ll_cost = ll_cost
        self.alg = alg
        
        self.info = info 
        """Extra information stored as a dictionary. Can include
              stop_it: stopping iteration of algorithm
              ll_gnorm: Gradient norm of lower level cost function
              convg_flag: flag regarding convergence of algorithm
              step_lengths: array of step lengths used for each iteration of algorithm
                 """
        



class bilevel():
    # Class to solve the bilevel learning problem
    def __init__(self,xtrue,ynoise,A,**kwargs):
        # ----------------------------------------------
        # -------      Problem inputs          --------
        
        # Ground truth and observations
        self.xtrue = xtrue      # Ground truth
        self.ynoise = ynoise    # Noisy measurement
        self.A = A              # Forward operator
        self.y = kwargs.pop('y',None) # Noiseless measurement
        
        # Specification of noise - optional
        self.delta = kwargs.pop('delta',None)
        self.gauss_vec = kwargs.pop('gauss_vec',None)
        self.noiselevel = kwargs.pop('noiselevel',None)
        
        # ----------------------------------------------
        # -----    Lower level solver inputs     -------
        self.use_odl = kwargs.pop('use_odl' , True) # Whether ODL or scipy should be utilsied
        
        # Analytic expression for solution to lower level problem is supplied
        self.analytic = kwargs.pop('analytic',None) 
        
        # Cost functions to be minimised - lower level requires f and g functions for PDHG
        self.upper_level_cost = kwargs.pop('ul_fun',None)
        
        if self.analytic is None: # No analytic solution is provided - use numerical solver instead
            self.lower_level_fun = kwargs.pop('ll_fun',None) 
            if not self.use_odl:
                self.lower_level_grad_fun = kwargs.pop('ll_grad' , None)
            
        self._gtol = kwargs.pop('gtol',1e-8) # Gradient tolerance for lower level problem
        self.cold_start = kwargs.pop('cold_start' , True)
        
        self.ll_max_iter = kwargs.pop('ll_max_iter' , 10000) # Maximum number of lower level solver iterations
        self.ll_linesearch = kwargs.pop('ll_linesearch' , 'backtracking')
        self._bfgs_store = kwargs.pop('bfgs_store',None) # Number of vectors to store in L-BFGS. By default all are stored i.e.e BFGS is performed
        
        if self.use_odl:
            self.Anorm = kwargs.pop('Anorm' , odl.power_method_opnorm(A) )
        else:
            self.Anorm = kwargs.pop('Anorm' , np.linalg.norm(A))
            
        # Determine whether numerous data pairs have been supplied
        self.single_data = True # Default options
        if isinstance(xtrue,list):
            if len(xtrue) != len(ynoise):
                raise ValueError('The number of groundtruths '+str(len(xtrue))+' does not match the number of measurements '+str(len(ynoise)))
            self.single_data = False
            
            # Use cold start with warm restarts for iterative algorithm unless stated otherwise
            if self.use_odl:
                self._x_start = kwargs.pop('x_start', [self.A.domain.zero() for _ in range(len(xtrue))] )
            else:
                self._x_start = kwargs.pop('x_start', [np.zeros(A.shape[1]) for _ in range(len(xtrue))] )
        else:
            # Use cold start for iterative algorithm unless stated otherwise
            if self.use_odl:
                self._x_start = kwargs.pop('x_start',self.A.domain.zero())
            else:
                self._x_start = kwargs.pop('x_start', np.zeros(A.shape[1]))

        # ---------------------------------------------
        # ----------    Optional inputs   -------------
        
        # Parameter space - p-list of array values where p is the dimension of the parameter space
        self.param_space = kwargs.pop('param_space', [np.array([1e-6,1e-0])])
             
        self.method = kwargs.pop('method','BFS')
        self.verbose = kwargs.pop('verbose', True)
 
        # ---------------------------------------------
        # --------      Solve the problem    ----------
        
        # Allocate memory for all iterates and optimal solution
        self.recons = [] # - List of variables of class reconstruction
        self.soln = None # - Variable of class reconstruction
        
        # Solve the bilevel problem
        self.solve_bilevel()
        
    def solve_bilevel(self):
        cost_fun = lambda x : self.upper_level_cost(x , self.xtrue) 
        self.brute_force_search(cost_fun)
    
    
    def brute_force_search(self,cost_fun):
        """ Consider each parameter in self.param_space, and find associated reconstruction that achieves smallest
            upper level cost."""
        
        costOpt = np.inf # no current best param, assume infinite cost initially
                
        self.soln = reconstruction(self._x_start.copy(),None,costOpt,None)
        paramInd = 0
        
        # ---------  Find optimal parameter for given training data  ------ # 
        for param in product(*self.param_space, disable=not(self.verbose==1), desc='Parameter Combination'):
                
            # Obtain solution to lower level problem
            alg, ll_eval, ll_gnorm, ll_its, step_length, backtrack_its, flag = self.solve_lower_level(param)
            
            
            try:
                self._x_start = alg.x
                tmpCost = cost_fun(alg.x)
                    
            except: # If an analytic solution is provided, alg will be an array of the reconstruction
                self._x_start = alg
                tmpCost = cost_fun(alg)
                
            
            # Update best paramater
            info = {'ll_eval':ll_eval, 'll_gnorm':ll_gnorm, 'll_its':ll_its, 'flag':flag, 'step_length':step_length, 'backtrack_its':backtrack_its}
            param_float = np.array([float(p) for p in param]) # convert tuple to array
            if tmpCost < costOpt:
                self.soln = reconstruction(self._x_start.copy(),param_float, ul_cost = tmpCost, ll_cost = ll_eval, info=info)
                costOpt = tmpCost
            
            # --- Store all reconstructions and costs
            tmp_recon = reconstruction(xrecon=self._x_start.copy(),param=param_float,ul_cost=tmpCost,ll_cost=ll_eval, info=info)
            tmp_recon.alg = alg
            self.recons.append(tmp_recon)
            
            paramInd += 1 # Increment index    
    
    def solve_lower_level(self,param):
        # Return a list of reconstructions corresponding to each data pair provided
        
        # Data is a single pair
        if self.single_data:            
            
            # Utilise closed form solution to the lower level problem if provided
            if self.analytic is not None:
                return self.analytic( param, self.ynoise), None, None, None,None, None, 0
            
            # Numerically solve lower level problem
            else:
                if self.use_odl:
                    bfgs_fun  = self.lower_level_fun(param , self.ynoise)     
                    if self.cold_start: self._x_start = self.A.domain.zero()  # cold start   
                    
                    ll_eval, ll_gnorm, ll_its, step_length, backtrack_its, flag = bfgs_method(bfgs_fun , self._x_start , maxiter=self.ll_max_iter,tol=self._gtol, num_store=self._bfgs_store, verbose=self.verbose>1) # Inplace update to variable
                    
                    # odl.solvers.bfgs_method(bfgs_fun , self._x_start , maxiter=self.ll_max_iter,tol=1e-9,line_search=line_search, callback=callback, num_store=20) # inplace update
                    
                    return self._x_start.copy(), ll_eval, ll_gnorm, ll_its, step_length, backtrack_its, flag
                
                else: # Do not use ODL
                    bfgs_fun  = self.lower_level_fun(param , self.ynoise)  
                    bfgs_grad_fun  = self.lower_level_grad_fun(param , self.ynoise)     
                    if self.cold_start: self._x_start = np.zeros(self.A.shape[1])
        
                    
                    self._x_start, ll_eval, g, _, ll_its,_,flag  =fmin_bfgs(f=bfgs_fun, x0=self._x_start, fprime = bfgs_grad_fun, full_output=True, disp=False, maxiter=self.ll_max_iter, gtol=self._gtol)
                    ll_gnorm = np.linalg.norm(g)
                    
                    return self._x_start, ll_eval, ll_gnorm, ll_its, None, None, flag
              
        # Numerous training data is supplied
        else:
            dat_num = len(self.xtrue)
        
            # Utilise closed form solution to the lower level problem if provided
            if self.analytic is not None:
                return [self.analytic( param, self.ynoise[ind]) for ind in range(dat_num)], None, None, None, None, None, 0
            
            # Numerically solve lower level problem
            else:
                # Allocate memory
                ll_gnorm,ll_its,ll_eval,step_length, backtrack_its, flags = np.empty(dat_num), np.empty(dat_num), np.empty(dat_num), np.empty(dat_num), np.empty(dat_num), np.empty(dat_num)
                
                if self.use_odl:
                    if self.cold_start: self._x_start = [self.A.domain.zero() for _ in range(dat_num)] # cold start  
                    
                    for ind in range(dat_num):
                        # Solve using BFGS'                    
                        bfgs_fun  = self.lower_level_fun(param , self.ynoise[ind])   
                        
                        ll_eval[ind], ll_gnorm[ind], ll_its[ind],step_length[ind], backtrack_its[ind], flag[ind] = bfgs_method(bfgs_fun , self._x_start , maxiter=self.ll_max_iter,tol=self._gtol, num_store=self._bfgs_store) # Inplace update to variable
                        
                        return self._x_start.copy(), ll_eval, ll_gnorm, ll_its, step_length, backtrack_its, flag
                        
                else:
                    if self.cold_start: self._x_start = [np.zeros(self.A.shape[1]) for _ in range(dat_num)] # cold start  
                    
                    for ind in range(dat_num):
                        # Solve using BFGS'
                        bfgs_fun  = self.lower_level_fun(param , self.ynoise[ind])  
                        bfgs_grad_fun  = self.lower_level_grad_fun(param , self.ynoise[ind])  
                        
                        self._x_start[ind], ll_eval[ind], g, _, ll_its[ind],_,flags[ind]  =  fmin_bfgs(f=bfgs_fun, x0=self._x_start[ind], fprime = bfgs_grad_fun, full_output=True, disp=False, maxiter=self.ll_max_iter, gtol=self._gtol)
                        ll_gnorm[ind] = np.linalg.norm(g)
                    return self._x_start, ll_eval, ll_gnorm, ll_its, None, None, flags
    
    def get(self,attr_name):
        """ Extracts list containing data of property attr_name from all reconstructions in the bilevel class variable"""
        
        return [getattr(single_recon,attr_name) for single_recon in self.recons]
