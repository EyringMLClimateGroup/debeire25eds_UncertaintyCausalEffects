from mpi4py import MPI
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.stats as scist
import math
from tigramite import data_processing as pp
import pandas as pd# 
import matplotlib.pyplot as plt
from savar.model_generator import SavarGenerator
from savar.savar import SAVAR
import numpy as np
from abc import abstractmethod
import copy
from functools import partial
from all_functions import *
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
if rank ==0 : print("Running on %d procs (one for each bootstrap realization)" %nprocs)
                    
# class DimensionReduction:
#     def __init__(self,data: np.ma):
#         """
#         :param data: np.ma of shape (Time,Variables)
#         """
#         self.data= data
#         self.loadings=None
#     @abstractmethod
#     def fit(self,ncomp:int):
#         """
#         :param ncomp: int, number of components to keep of the DimensionReduction method
#         """
#         pass
    
# class PCA_nipals(DimensionReduction):
    
#     def fit(self,ncomp:int =5, cv:bool =False,maxiter:int=5000,scale:bool=False,center:bool=True):
#         PCA_ = nipals.Nipals(self.data)
#         PCA_.fit(ncomp=ncomp,cv=cv,maxiter=maxiter,scale=scale,center=center)
#         self.R2cum=PCA_.R2cum
#         self.R2=PCA_.R2
#         self.loadings=PCA_.loadings.values
#         self.scores=PCA_.scores
#         self.eig=PCA_.eig
                    
# class Varimax(PCA_nipals):
    
#     def fit(self,ncomp:int =5, cv:bool =False,maxiter:int=5000,scale:bool=False,center:bool=True):
#         PCA_nipals.fit(self,ncomp=ncomp,cv=cv,maxiter=maxiter,scale=scale,center=center)
#         Vr, Rot = Varimax.varimax(self.loadings, verbosity=0)
#         Vr= Varimax.svd_flip(Vr)
#         s2 = np.diag(self.eig)**2 / (self.data.shape[0] - 1.)
#         # matrix with diagonal containing variances of rotated components
#         S2r = np.dot(np.dot(np.transpose(Rot), s2), Rot)
#         expvar = np.diag(S2r)
#         sorted_expvar = np.sort(expvar)[::-1]
#         # reorder all elements according to explained variance (descending)
#         nord = np.argsort(expvar)[::-1]
#         Vr = Vr[:, nord]
#         total_var = np.sum(np.var(self.data, axis = 0))
#         #save Varimax output
#         self.loadings=Vr
#         self.R2 = sorted_expvar/total_var
#         self.R2cum = np.cumsum(self.R2)
        
#     def varimax(Phi, gamma = 1, q = 10, 
#         rtol = np.finfo(np.float32).eps ** 0.5,
#         verbosity=0):
#         """Varimax rotation of Phi."""
#         p,k = Phi.shape
#         R = np.eye(k)
#         d=0
#         # print Phi
#         for i in range(q):
#             if verbosity > 1:
#                 if i % 10 == 0.:
#                     print("\t\tVarimax iteration %d" % i)
#             d_old = d
#             Lambda = np.dot(Phi, R)
#             u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 
#                        - (gamma/float(p)) * np.dot(Lambda, 
#                         np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
#             R = np.dot(u,vh)
#             d = np.sum(s)
#             if d_old!=0 and abs(d - d_old) / d < rtol: break
#         # print i
#         return np.dot(Phi, R), R
    
#     def svd_flip(u, v=None, u_based_decision=True):
#         """Sign correction to ensure deterministic output from SVD.
#         Adjusts the columns of u and the rows of v such that the loadings in the
#         columns in u that are largest in absolute value are always positive.
#         Parameters
#         ----------
#         u, v : ndarray
#             u and v are the output of `linalg.svd` or
#             `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
#             so one can compute `np.dot(u * s, v)`.
#         u_based_decision : boolean, (default=True)
#             If True, use the columns of u as the basis for sign flipping.
#             Otherwise, use the rows of v. The choice of which variable to base the
#             decision on is generally algorithm dependent.
#         Returns
#         -------
#         u_adjusted, v_adjusted : arrays with the same dimensions as the input.
#         """
#         if v is None:
#              # rows of v, columns of u
#              max_abs_rows = np.argmax(np.abs(u), axis=0)
#              signs = np.sign(u[max_abs_rows, range(u.shape[1])])
#              u *= signs
#              return u

#         if u_based_decision:
#             # columns of u, rows of v
#             max_abs_cols = np.argmax(np.abs(u), axis=0)
#             signs = np.sign(u[max_abs_cols, range(u.shape[1])])
#             u *= signs
#             v *= signs[:, np.newaxis]
#         else:
#             # rows of v, columns of u
#             max_abs_rows = np.argmax(np.abs(v), axis=1)
#             signs = np.sign(v[range(v.shape[0]), max_abs_rows])
#             u *= signs
#             v *= signs[:, np.newaxis]

#         return u, v
    
# class CausalDiscovery:
#     def __init__(self,data,var_names,mask=None):
#         """
#         :param data: np.ma or np.ndarray of shape (Time,Variables)
#         :param var_names: list of string containing name of Variables
#         :param mask: mask, default is None
#         """
#         self.data = data
#         self.var_names = var_names
#         self.mask = mask
#         self.dataframe= pp.DataFrame(data,mask=mask,var_names=var_names)
#     @abstractmethod
#     def get_parents(self,tau_max: int, tau_min: int = 1):
#         """
#         :param tau_max: int,
#         :param tau_min: int, default is 1
#         """
#         pass
       
# class PCMCI_(CausalDiscovery):
#     def get_parents(self,tau_max: int, tau_min: int = 1,include_lagzero_parents=False):
#         cond_int_test= RobustParCorr()
#         pc_alpha = [0.05,0.1,0.2]
#         self.tau_min=tau_min
#         self.tau_max=tau_max
#         pcmci_ = PCMCI(dataframe=self.dataframe, cond_ind_test=cond_int_test,verbosity=0)
#         results_ = pcmci_.run_pcmci(tau_min=tau_min,tau_max=tau_max,pc_alpha=pc_alpha,
#                                         selected_links=None)
#         self.parents = pcmci_.return_parents_dict(results_["graph"],results_["val_matrix"],
#                                    include_lagzero_parents=include_lagzero_parents)
#         return self.parents
    
# # class AdaptiveLasso(CausalDiscovery):
# #     pass
# class AllParents(CausalDiscovery):
#     def get_parents(self,tau_max: int, tau_min: int = 1,include_lagzero_parents=False):
#         self.tau_min=tau_min
#         self.tau_max=tau_max
#         parents={}
#         T,N = self.dataframe.values[0].shape
#         for var in range(N):
#             parents[var]= []
#             for par in range(N):
#                 for tau in range(tau_min,tau_max):
#                     parents[var].append((par,-tau))
#                 if include_lagzero_parents and par!=var:
#                     parents[var].append((par,0))
#         self.parents = parents
#         return self.parents

# class LinCoeffEstimation:
#     @abstractmethod
#     def __init__(self):
#         pass
#     @abstractmethod
#     def fit(self,data, parents,tau_min,tau_max):
#         """
#         Returns estimated coefficient in a (tau_max(or tau_max+1),components,components) array
#         residuals, and variance of residuals
#         """
#         pass
    
# class LinMed(LinCoeffEstimation):
#     def __init__(self,data,var_names,mask=None):
#         """
#         :param data: np.ma or np.ndarray of shape (Time,Variables)
#         :param var_names: list of string containing name of Variables
#         :param mask: mask, default is None
#         """
#         self.data = data
#         self.var_names = var_names
#         self.mask = mask
#         self.dataframe= pp.DataFrame(data,mask=mask,var_names=var_names)
    
#     def fit(self, parents: dict,tau_max: int,tau_min: int = 1):
#         """
#         :parents: dictionary of parents for each variable used as predictors
#         :targets: list of variable to use as targets
#         :param tau_max: int,
#         :param tau_min: int, default is 1
#         """
#         self.tau_min=tau_min
#         self.tau_max=tau_max
#         self.parents=parents
#         dataframe= self.dataframe
#         data=self.data
#         med = LinearMediation(dataframe,
#                  data_transform=None,
#                  verbosity=0)
#         med.fit_model(all_parents=parents,tau_max=tau_max,return_data= True)
#         lin_coeff = med.phi
#         #Compute residuals and estimate covariance of residuals        
#         T,N=data.shape
#         res=data[tau_max:,:]
#         for var in med.fit_results.keys():
#             sub_model = med.fit_results[var]
#             if sub_model is not None:
#                 model=sub_model["model"]
#                 saved_data = med.fit_results[var]["data"].T
#                 y=saved_data[:,1]
#                 x=saved_data[:,2:]
#                 y_pred=model.predict(x)
#                 res[:,var]=y-y_pred
#             else:
#                 continue
#         self.fitted_coef=lin_coeff
#         cov_res= res.T@res/(T-tau_max)
#         self.res=res
#         self.model=med
#         return lin_coeff,cov_res,res
    
# class VARmodel(LinCoeffEstimation):
#     def __init__(self,data,var_names=None,mask=None):
#         self.data= data
#         self.var_names=var_names
#         self.mask= mask
#         self.dataframe= pp.DataFrame(data,mask=mask,var_names=var_names)
#     def fit(self, parents,tau_max,tau_min):
#         VAR_model = VAR(self.data)
#         self.tau_max= tau_max
#         self.tau_min=tau_min
#         VAR_results = VAR_model.fit(maxlags=tau_max,trend="n")
#         self.model=VAR_model
#         fitted_coef = VAR_results.coefs
#         res= VAR_results.resid
#         cov_res= VAR_results.sigma_u
#         lin_coeff_=np.zeros((tau_max+1,fitted_coef.shape[1],fitted_coef.shape[2]))
#         lin_coeff_[1:,...]=fitted_coef
#         self.res=res
#         self.fitted_coef=lin_coeff_
#         return self.fitted_coef,res,cov_res

# def generate_ts_from_coefs_and_residuals(VAR,res,array,min_lag=0):
#         tau_max,N_var,_=VAR.shape
#         tau_max += -1+min_lag
#         T=array.shape[0]
#         T_transient = res.shape[0] #total sample size with transient fraction
#         y_gen= np.zeros((T_transient,N_var))
#         y_gen+= res #add noise
#         y_gen[:tau_max+1,...]= array[:tau_max+1,...] #initial values
#         for i in range(tau_max+1,T_transient):
#             for t in reversed(range(min_lag,tau_max+1)):
#                 y_gen[i,...]+= y_gen[i-t,...].dot(VAR[t,...].T)
#         new_array= np.zeros((T,N_var))
#         new_array=y_gen[T_transient-T:,...]
#         #std_scaler = StandardScaler()
#         #new_array = std_scaler.fit_transform(new_array)
#         return new_array
    
# def get_G(K,W,W_pinv,Psi_inf):
#     return np.kron((K@W@Psi_inf).T,Psi_inf@W_pinv)

# def get_Z_from_dataframe(data,tau_min,tau_max):
#     T_data,N_data= data.shape
#     Z_t = np.zeros((N_data*(tau_max-tau_min+1),1))
#     Z = np.zeros((N_data*(tau_max-tau_min+1),T_data-tau_max))
#     for t in range(tau_max,T_data):
#         tuple_Z_t=(data[t-tau,:] for tau in range(tau_min,tau_max+1))
#         Z_t= np.hstack(tuple_Z_t).T
#         Z[:,t-tau_max]=Z_t
#     return Z
# def get_Gamma_from_Z(Z):
#     return Z@Z.T/Z.shape[1]
    
# def quantity_of_interest(coeff,W_,W_pinv_,forcing_weights):
#     #function defining the quantity of interest, for example sensitivity
#     return (compute_LTE(coeff,W_,W_pinv_)@forcing_weights).mean()

# def calculate_q_for_boot(CausalMethod_,med,W_,W_pinv_,b,new_data):
#     #estimate parents from new_data
#     new_CausalMethod_ = globals()[CausalMethod_.__class__.__name__](new_data,CausalMethod_.var_names, CausalMethod_.mask)
#     new_parents = new_CausalMethod_.get_parents(med.tau_max,med.tau_min)
#     #estimate linear coefficients from new_parents
#     new_LinMed_model = globals()[med.__class__.__name__](new_data,med.var_names,med.mask)
#     new_lin_coeff,_,_= new_LinMed_model.fit(new_parents,med.tau_max,med.tau_min)
#     new_q = quantity_of_interest(new_lin_coeff,W_,W_pinv_,b)
#     return new_q
    

class ConfidenceIntervalMethod:
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def compute_CI(data,sig_level):
        pass
    
class Standard_LinMedBS(ConfidenceIntervalMethod):
    def compute_CI(self,CausalMethod_,LinMed_model,sig_level: float, realizations: int, 
                   W_: np.ndarray,W_pinv_: np.ndarray,b: np.ndarray):        
        """
        Perform standard bootstrapping from the cov of the residuals and estimate LTE Confidence interval
        :param CausalMethod_: Causal Method (PCMCI, AllParents, ...)
        :param LinMed_model: Linear Mediation Model
        :param sig_level: float between 0 and 1, significance level of confidence interval
        :param realizations: int, number of bootstrap realizations,
        :param W: np.ndarray, weights,
        :param W_pinv: pseudo-inverse of W,
        :param b: forcing weights vector,
        """
        # global calculate_q_for_boot
        transient_frac=0.3
        lin_coeff= LinMed_model.fitted_coef
        med = copy.deepcopy(LinMed_model)
        estimated_q = quantity_of_interest(lin_coeff,W_,W_pinv_,b)
        boot_res=np.empty((realizations,)+estimated_q.shape)
        res_ = med.res
        cov_res = np.cov(res_, rowvar=0)
        mean_res = np.mean(res_, axis=0)
        data = med.dataframe.values[0] 
        T_data= data.shape[0]
        T_transient =  int(math.floor(transient_frac*T_data))
        data_boots = []
        calc_q_for_boot = partial(calculate_q_for_boot, CausalMethod_, med, W_,W_pinv_,b)

        if rank == 0:
            for r in range(realizations):
                #draw new residuals from estimated covariance of res
                new_res = np.random.multivariate_normal(mean=mean_res, cov=cov_res, size=T_data+T_transient)
                new_data= generate_ts_from_coefs_and_residuals(lin_coeff,new_res,data)
                data_boots.append(new_data)
        data_boots = comm.scatter(data_boots, root=0)
        results = calc_q_for_boot(data_boots)
        boot_res = comm.allgather(results)#,root=0)
        boot_res = np.array(boot_res)
        self.boot_res = boot_res
        confidence_interval = np.quantile(boot_res, axis=0,q = [sig_level/2,1-sig_level/2])
        return confidence_interval

class Hall_LinMedBS(ConfidenceIntervalMethod):
    def compute_CI(self,CausalMethod_,LinMed_model,sig_level: float, realizations: int, 
                   W_: np.ndarray,W_pinv_: np.ndarray,b: np.ndarray):        
        """
        Perform Hall correction bootstrapping from the cov of the residuals and estimate LTE Confidence interval
        :param CausalMethod_: Causal Method (PCMCI, AllParents, ...)
        :param LinMed_model: Linear Mediation Model
        :param sig_level: float between 0 and 1, significance level of confidence interval
        :param realizations: int, number of bootstrap realizations,
        :param W: np.ndarray, weights,
        :param W_pinv: pseudo-inverse of W,
        :param b: forcing weights vector,
        """
        transient_frac=0.3
        lin_coeff= LinMed_model.fitted_coef
        med = copy.deepcopy(LinMed_model)
        estimated_q = quantity_of_interest(lin_coeff,W_,W_pinv_,b)
        boot_res=np.empty((realizations,)+estimated_q.shape)
        res_ = med.res
        cov_res = np.cov(res_, rowvar=0)
        mean_res = np.mean(res_, axis=0)
        data = med.dataframe.values[0] 
        T_data= data.shape[0]
        T_transient =  int(math.floor(transient_frac*T_data))
        data_boots = []
        calc_q_for_boot = partial(calculate_q_for_boot, CausalMethod_, med, W_,W_pinv_,b)

        if rank == 0:
            for r in range(realizations):
                #draw new residuals from estimated covariance of res
                new_res = np.random.multivariate_normal(mean=mean_res, cov=cov_res, size=T_data+T_transient)
                new_data= generate_ts_from_coefs_and_residuals(lin_coeff,new_res,data)
                data_boots.append(new_data)
        data_boots = comm.scatter(data_boots, root=0)
        results = calc_q_for_boot(data_boots)- estimated_q

        boot_res = comm.allgather(results)
        boot_res= np.array(boot_res)
        self.boot_res = boot_res
        confidence_interval = -np.quantile(boot_res, axis=0,q = [1-sig_level/2,sig_level/2])+estimated_q
        return confidence_interval
    
class Standard_LinMedBS_onresiduals(ConfidenceIntervalMethod):
    def compute_CI(self,CausalMethod_,LinMed_model,sig_level: float, realizations: int, 
                   W_: np.ndarray,W_pinv_: np.ndarray,b: np.ndarray):        
        """
        Perform normal bootstrapping by resampling the residuals and estimate LTE Confidence interval
        :param CausalMethod_: Causal Method (PCMCI, AllParents, ...)
        :param LinMed_model: Linear Mediation Model
        :param sig_level: float between 0 and 1, significance level of confidence interval
        :param realizations: int, number of bootstrap realizations,
        :param W: np.ndarray, weights,
        :param W_pinv: pseudo-inverse of W,
        :param b: forcing weights vector,
        """
        transient_frac=0.3
        lin_coeff= LinMed_model.fitted_coef
        med = copy.deepcopy(LinMed_model)
        estimated_q = quantity_of_interest(lin_coeff,W_,W_pinv_,b)
        boot_res=np.empty((realizations,)+estimated_q.shape)
        res_ = med.res
        data = med.dataframe.values[0] 
        T_data= data.shape[0]
        T_transient =  int(math.floor(transient_frac*T_data))
        data_boots = []
        calc_q_for_boot = partial(calculate_q_for_boot, CausalMethod_, med, W_,W_pinv_,b)

        if rank == 0:
            for r in range(realizations):
                #draw new residuals from estimated covariance of res
                res_df=pd.DataFrame(res_)
                new_res_df = res_df.sample(n=T_data+T_transient,replace=True,axis=0)
                new_res=new_res_df.values
                new_res-= new_res.mean(axis=0) #center the residuals
                new_data= generate_ts_from_coefs_and_residuals(lin_coeff,new_res,data)
                data_boots.append(new_data)

        data_boots = comm.scatter(data_boots, root=0)
        results = calc_q_for_boot(data_boots)
        boot_res = comm.allgather(results)#,root=0)

        boot_res = np.array(boot_res)
        self.boot_res = boot_res
        confidence_interval = np.quantile(boot_res, axis=0,q = [sig_level/2,1-sig_level/2])
        return confidence_interval
    
class Hall_LinMedBS_onresiduals(ConfidenceIntervalMethod):
    def compute_CI(self,CausalMethod_,LinMed_model,sig_level: float, realizations: int, 
                   W_: np.ndarray,W_pinv_: np.ndarray,b: np.ndarray):        
        """
        Perform Hall correction bootstrapping by resampling the residuals and estimate LTE Confidence interval
        :param CausalMethod_: Causal Method (PCMCI, AllParents, ...)
        :param LinMed_model: Linear Mediation Model
        :param sig_level: float between 0 and 1, significance level of confidence interval
        :param realizations: int, number of bootstrap realizations,
        :param W: np.ndarray, weights,
        :param W_pinv: pseudo-inverse of W,
        :param b: forcing weights vector,
        """
        transient_frac=0.3
        lin_coeff= LinMed_model.fitted_coef
        med = copy.deepcopy(LinMed_model)
        estimated_q = quantity_of_interest(lin_coeff,W_,W_pinv_,b)
        phi_boots= np.empty((realizations,)+lin_coeff.shape)
        boot_res=np.empty((realizations,)+estimated_q.shape)
        res_ = med.res
        data = med.dataframe.values[0] 
        T_data= data.shape[0]
        T_transient =  int(math.floor(transient_frac*T_data))
        data_boots = []
        calc_q_for_boot = partial(calculate_q_for_boot, CausalMethod_, med, W_,W_pinv_,b)

        if rank == 0:
            for r in range(realizations):
                #draw new residuals from estimated covariance of res
                res_df=pd.DataFrame(res_)
                new_res_df = res_df.sample(n=T_data+T_transient,replace=True,axis=0)
                new_res=new_res_df.values
                new_res-= new_res.mean(axis=0) #center the residuals
                new_data= generate_ts_from_coefs_and_residuals(lin_coeff,new_res,data)
                data_boots.append(new_data)

        data_boots = comm.scatter(data_boots, root=0)
        results = calc_q_for_boot(data_boots) -estimated_q
        boot_res = comm.allgather(results)#,root=0)
        boot_res = np.array(boot_res)
        self.boot_res = boot_res

        confidence_interval = -np.quantile(boot_res, axis=0,q = [1-sig_level/2,sig_level/2])+estimated_q
        return confidence_interval

class AsymptoticCI(ConfidenceIntervalMethod):
    def compute_CI(self,LinMed_model,sig_level: float, 
                   W_: np.ndarray,W_pinv_: np.ndarray,b: np.ndarray):
        """
        Apply asymptotic distribution formula and derive an asymptotic confidence interval
                
        :param LinMed_model: Linear Mediation Model
        :param sig_level: float between 0 and 1, significance level of confidence interval
        :param W: np.ndarray, weights,
        :param W_pinv: pseudo-inverse of W,
        :param b: forcing weights vector,
        """
        lin_coeff= LinMed_model.fitted_coef
        med = copy.deepcopy(LinMed_model)
        LTE_estimate= compute_LTE(lin_coeff,W_,W_pinv_) 
        estimated_q= quantity_of_interest(lin_coeff,W_,W_pinv_,b)
        data = med.dataframe.values[0] 
        res_ = med.res
        cov_res = np.cov(res_.T)
        quantile_norm_dist = scist.norm.ppf(1-sig_level/2)
        min_lag,max_lag=med.tau_min,med.tau_max
        Z = get_Z_from_dataframe(data,min_lag,max_lag)
        Gamma = get_Gamma_from_Z(Z)
        Gamma_pinv = np.linalg.pinv(Gamma)
        tau_dim = max_lag-min_lag+1
        T_res = res_.shape[0]
        N=data.shape[1]
        L=W_.shape[1]
        K=np.zeros((tau_dim*N,N))
        for i in range(tau_dim):
            K[i*N:(i+1)*N,:]= np.eye(N,N)
        G_LTE=get_G(K,W_,W_pinv_,LTE_estimate)
        cov_coeff = np.kron(Gamma_pinv,cov_res)
        alpha_cov=((np.ones((1,L*L))@G_LTE)@cov_coeff@(G_LTE.T@np.ones((L*L,1)))/T_res/L**2)[0][0]
        LTE_upper_bound=estimated_q+quantile_norm_dist*np.sqrt(alpha_cov)
        LTE_lower_bound=estimated_q-quantile_norm_dist*np.sqrt(alpha_cov)
        return [LTE_lower_bound,LTE_upper_bound]
 
# def get_links_coeffs_matrix(link_coeff_dict,tau_max):
#     N=len(list(link_coeff_dict))
#     link_coeff_mat = np.zeros((tau_max + 1,N, N))
#     for i in list(link_coeff_dict):
#         for par in list(link_coeff_dict[i]):
#                 j, tau,val = par[0][0],par[0][1], par[1]
#                 link_coeff_mat[abs(tau),i,j] = val
#     return link_coeff_mat

# def sort_pca_comp_by_correlation(true_weights,estimated_weights):
#     N_comp,L=true_weights.shape
#     U= [i for i in range(N_comp)]
#     index_=[]
#     for i in range(N_comp):
#         j_max = np.argmax([np.abs(np.corrcoef(estimated_weights[i,...].T,true_weights[j,...].T))[0,1] 
#                    for j in U])
#         index_.append(U[j_max])
#         del U[j_max]
#     index_=[index_.index(i) for i in range(N_comp)]
#     return index_
# def get_timeseries_from_loadings(data,weights):
#         return data.dot(weights.T)

def compare_confidence_interval_allseeds(sig_level=0.1,realizations=200,varying_parameter="T",value_range=[100],seed_list=[34],verbosity=0,causal_method="PCMCI_",fit_method="LinMed"):
    if verbosity and rank==0: 
        print("Causal Method: %s\n Linear fit: %s"%(causal_method,fit_method))
    global T
    T= 500 #number of samples
    mode_strength=0.5 #Strength of modes
    components=5 #number of PCA components
    nx,ny=20,30 #resolution
    max_lag=3 #Max time lag
    #not varying param:
    b_forcing = np.ones((nx*ny,1))
    min_lag=1
    res_sig_level = 0.05
    df_results_spread=pd.DataFrame(columns=["COV-StandardBS","COV-HallBS","RES-StandardBS","RES-HallBS","Asymp-CI"])
    df_results_est=pd.DataFrame(columns=["MAE"])
    df_results_ci=pd.DataFrame(columns=["COV-StandardBS","COV-HallBS","RES-StandardBS","RES-HallBS","Asymp-CI"])
    df_results_spread_std=pd.DataFrame(columns=["COV-StandardBS","COV-HallBS","RES-StandardBS","RES-HallBS","Asymp-CI"])
    df_results_est_std=pd.DataFrame(columns=["MAE"])
    for param_value in value_range:
        globals()[varying_parameter]=param_value
        df_results_spread_seed=pd.DataFrame(columns=["COV-StandardBS","COV-HallBS","RES-StandardBS","RES-HallBS","Asymp-CI"])
        df_results_est_seed=pd.DataFrame(columns=["MAE"])
        df_results_ci_seed=pd.DataFrame(columns=["COV-StandardBS","COV-HallBS","RES-StandardBS","RES-HallBS","Asymp-CI"])
        if verbosity and rank==0: print("Computing confidence intervals for %s = %d"%(varying_parameter,param_value))
        for seed in seed_list:
            modelseed= seed
            if verbosity and rank==0: print("Seed #%d:"%seed)
            random_savar= SavarGenerator(n_variables= components, time_length=T,
                         # Noise
                         noise_strength= mode_strength, noise_variance= None, noise_weights= None,
                         resolution= (nx,ny), noise_cov= None,
                         latent_noise_cov= None, fast_cov= None,
                         # Fields
                         data_field= None, noise_data_field= None,
                         seasonal_data_field= None, forcing_data_field= None,
                         # Weights
                         mode_weights= None,  gaussian_shape= True,
                         random_mode=True, dipole= False, mask_variables= None, norm_weight= True,
                         # links
                         n_cross_links=components, auto_coeffs_mean= 0.3, auto_coffs_std= 0.2,
                         auto_links= True,auto_strength_threshold= 0.2, auto_random_sign= 0.5,
                         cross_mean= 0.3, cross_std= 0.2, cross_threshold= 0.2,
                         cross_random_sign= 0.2, tau_max = max_lag, tau_min= min_lag,
                         n_trial= 1000, model_seed= modelseed,
                         # external forcings
                         forcing_dict=None, season_dict= None,
                         # Ornstein
                         ornstein_sigma=None, n_var_ornstein=None,
                         # Linearity
                         linearity="linear",
                         verbose=False)
            while True:
                try:
                    if verbosity and rank==0:
                        print("Seed value = %d"%random_savar.model_seed)
                    random_savar.generate_savar()
                except RecursionError:
                    modelseed += 1312414
                    random_savar.model_seed= modelseed
                else:
                    break
            savar_dict = {
            "links_coeffs": random_savar.links_coeffs,
            "time_length": random_savar.time_length,
            "transient": random_savar.transient,
            "mode_weights": random_savar.mode_weights,
            "noise_weights": random_savar.noise_weights,
            "noise_strength": random_savar.noise_strength,
            "noise_variance": random_savar.noise_variance,
            "noise_cov": random_savar.noise_cov,
            "fast_cov": random_savar.fast_cov,
            "latent_noise_cov": random_savar.latent_noise_cov,
            "forcing_dict": random_savar.forcing_dict,
            "season_dict": random_savar.season_dict,
            "data_field": random_savar.data_field,
            "noise_data_field": random_savar.noise_data_field,
            "seasonal_data_field": random_savar.seasonal_data_field,
            "forcing_data_field": random_savar.forcing_data_field,
            "linearity": random_savar.linearity,
            "verbose": random_savar.verbose,
            "model_seed": random_savar.model_seed
            }
            savar= SAVAR(**savar_dict)
            savar.generate_data()
            data = savar.data_field.T
            links_dict = savar.links_coeffs
            links_coeffs = get_links_coeffs_matrix(links_dict,max_lag)
            W = savar.mode_weights.reshape((components,nx*ny))
            W_pinv= np.linalg.pinv(W)
            ###Estimation of modes
            DimRedMethod = Varimax(data)
            DimRedMethod.fit(ncomp=components, cv=False,maxiter=5000,scale=False,center=True)
            DimRed_loadings = DimRedMethod.loadings
            W_estimate = DimRedMethod.loadings.T
            pca_idx_order = sort_pca_comp_by_correlation(W,W_estimate)
            W_estimate = W_estimate[pca_idx_order,...] #reorder components from left to right
            W_estimate_pinv = np.linalg.pinv(W_estimate)
            ts = get_timeseries_from_loadings(data,W)
            ##Estimation of parents
            var_names= [str(i) for i in range(components)]
            CausalMethod= globals()[causal_method](ts,var_names=var_names, mask=None)
            parents = CausalMethod.get_parents(tau_min=min_lag,tau_max=max_lag)
            ###Estimation of coefficients
            LinCoeffMethod = globals()[fit_method](ts,var_names=var_names, mask=None)
            links_coeffs_estimate,_,_ = LinCoeffMethod.fit(parents=parents,tau_max=max_lag,tau_min=min_lag)
            true_q = quantity_of_interest(links_coeffs,W,W_pinv,b_forcing)
            if true_q>2.6: 
                if rank ==0: print("!!! Large values")
                continue
            estimated_q = quantity_of_interest(links_coeffs_estimate,W,W_pinv,b_forcing)
            #Cov-res BS
            Standard_BS=Standard_LinMedBS()
            CI_linmed_bs = Standard_BS.compute_CI(CausalMethod,LinCoeffMethod,sig_level,realizations,W,W_pinv,b_forcing)
            #Cov-res Hall BS
            Hall_BS=Hall_LinMedBS()
            CI_linmed_hall=Hall_BS.compute_CI(CausalMethod,LinCoeffMethod,sig_level,realizations,W,W_pinv,b_forcing)
            #Res BS
            Standard_res_BS = Standard_LinMedBS_onresiduals()
            CI_res_bs = Standard_res_BS.compute_CI(CausalMethod,LinCoeffMethod,sig_level,realizations,W,W_pinv,b_forcing)
            #Res Hall BS
            Hall_res_BS = Hall_LinMedBS_onresiduals()
            CI_res_hall = Hall_res_BS.compute_CI(CausalMethod,LinCoeffMethod,sig_level,realizations,W,W_pinv,b_forcing)
            #Asymp CI
            Asymp_CI_class=AsymptoticCI()
            Asymp_CI=Asymp_CI_class.compute_CI(LinCoeffMethod,sig_level,W,W_pinv,b_forcing)
            df_results_spread_seed.loc[seed,:] = [CI_linmed_bs[1]-CI_linmed_bs[0],CI_linmed_hall[1]-CI_linmed_hall[0],CI_res_bs[1]-CI_res_bs[0],CI_res_hall[1]-CI_res_hall[0],Asymp_CI[1]-Asymp_CI[0]]
            absolute_error=np.abs(estimated_q-true_q)
            df_results_est_seed.loc[seed,:]= [absolute_error]
            df_results_ci_seed.loc[seed,:]= [int(2*np.abs(estimated_q-true_q)<=(CI_linmed_bs[1]-CI_linmed_bs[0])),
                                            int(2*np.abs(estimated_q-true_q)<=(CI_linmed_hall[1]-CI_linmed_hall[0])),
                                            int(2*np.abs(estimated_q-true_q)<=(CI_res_bs[1]-CI_res_bs[0])),
                                            int(2*np.abs(estimated_q-true_q)<=(CI_res_hall[1]-CI_res_hall[0])),
                                            int(2*np.abs(estimated_q-true_q)<=(Asymp_CI[1]-Asymp_CI[0]))]
        df_results_spread.loc[param_value,:] = df_results_spread_seed.mean(axis=0)
        df_results_est.loc[param_value,:]= df_results_est_seed.mean(axis=0)
        df_results_ci.loc[param_value,:]=df_results_ci_seed.sum(axis=0)/len(df_results_ci_seed.index)
        df_results_spread_std.loc[param_value,:] = df_results_spread_seed.std(axis=0)
        df_results_est_std.loc[param_value,:]= df_results_est_seed.std(axis=0)
        if rank ==0 and verbosity:
            print(len(df_results_ci_seed.index))
            print(df_results_spread)
            print(df_results_est)
            print(df_results_ci)
            print(df_results_spread_std)
            print(df_results_est_std)
    return df_results_spread,df_results_est,df_results_ci,df_results_spread_std,df_results_est_std
if __name__=='__main__':
    # adapt by changing n_seed, value_range and Causal method and Linear coeff estimation method
    varying_parameter="T"
    n_seed,seed_start= 100,3 #103

    causalmet = "PCMCI_"
    #causalmet = "AllParents"

    linmed = "LinMed"
    #linmed = "VARmodel"

    #value_range=[100,500,1000,5000]
    #value_range=[10000]
    value_range=[20000]
    
    seed_list= [i for i in range(seed_start,seed_start+n_seed) if i not in [30,58,79]] #those seeds are nonstationary

    df_res_spread,df_res_est,df_res_ci,df_res_spread_std, df_res_est_std  = compare_confidence_interval_allseeds(0.1,nprocs,"T",value_range,seed_list,1,causalmet,linmed)
    if rank ==0:
        df_res_spread.to_csv("./results/confidence_interval_spread_%s_%s_%d-%d.csv"%(causalmet,linmed,value_range[0],value_range[-1]))
        df_res_est.to_csv("./results/confidence_interval_mae_%s_%s_%d-%d.csv"%(causalmet,linmed,value_range[0],value_range[-1]))
        df_res_ci.to_csv("./results/confidence_interval_cirate_%s_%s_%d-%d.csv"%(causalmet,linmed,value_range[0],value_range[-1]))
        df_res_spread_std.to_csv("./results/confidence_interval_spread_std_%s_%s_%d-%d.csv"%(causalmet,linmed,value_range[0],value_range[-1]))
        df_res_est_std.to_csv("./results/confidence_interval_mae_std_%s_%s_%d-%d.csv"%(causalmet,linmed,value_range[0],value_range[-1]))