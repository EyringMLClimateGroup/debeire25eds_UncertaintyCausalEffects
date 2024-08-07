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