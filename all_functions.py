import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from savar.model_generator import SavarGenerator
from savar.savar import SAVAR
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

def generate_simulation(N_sim,varying_parameter,value_range,verbosity=0):
    #Used to generate SAVAR data for run_benchmark_sens_err_*.py experiments
    #Param that can be varied
    global T,mode_strength,components,nx,ny,auto_coef_mean,n_cross_links
    T= 500 #number of samples
    mode_strength=0.5 #Strength of modes
    components=5 #number of PCA components
    n_cross_links=5
    nx,ny=20,30 #resolution
    auto_coef_mean = 0.3 #
    max_lag=3 #Max time lag
    #not varying param:
    min_lag=1
    res_sig_level = 0.05
    start_seed= 23
    all_DimRed = all_subclasses(DimensionReduction)
    all_CausalMethod = all_subclasses(CausalDiscovery)
    all_LinCoeffEstimation = all_subclasses(LinCoeffEstimation)
    if verbosity>0:
        print("Dimension Reduction methods found: %s"%", ".join(all_DimRed))
        print("Causal Discovery methods found: %s"%", ".join(all_CausalMethod))
        print("Linear Coefficient Estimation methods found: %s"%", ".join(all_LinCoeffEstimation))
    df_linkcoeff_metric=pd.DataFrame(columns=["method","parameter","simulation","error"])
    df_lte_mrae=pd.DataFrame(columns=["method","parameter","simulation","error"])
    df_sens_err=pd.DataFrame(columns=["method","parameter","simulation","error"])
    for param_value in value_range:
        globals()[varying_parameter]=param_value
        if verbosity>0:
            print("Generating Simulations for %s = %s:"%(varying_parameter,str(param_value)))
        for i in range(N_sim):
            if i==0: modelseed = start_seed
            else: modelseed += 1
            random_savar= SavarGenerator(n_variables= components, time_length=T,
                         # Noise
                         noise_strength= mode_strength, noise_variance= None, noise_weights= None,
                         resolution= (nx,ny), noise_cov= None,
                         latent_noise_cov= None, fast_cov= np.eye(nx*ny),
                         # Fields
                         data_field= None, noise_data_field= None,
                         seasonal_data_field= None, forcing_data_field= None,
                         # Weights
                         mode_weights= None,  gaussian_shape= True,
                         random_mode=True, dipole= False, mask_variables= None, norm_weight= True,
                         # links
                         n_cross_links=n_cross_links, auto_coeffs_mean= auto_coef_mean, auto_coffs_std= 0.2,
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
            if verbosity>1:
                print("Generating Sim#%d of SAVAR data of shape %s"%(i,str((nx,ny))))
            
            while True:
                try:
                    #print("Seed value = %d"%random_savar.model_seed)
                    random_savar.generate_savar()
                except RecursionError:
                    modelseed +=1
                    random_savar.model_seed= modelseed
                    #random_savar.generate_savar()
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
            for DimRed_name in all_DimRed:
                nextIterationFlag = False               
                DimRedMethod = globals()[DimRed_name](data)
                DimRedMethod.fit(ncomp=components, cv=False,maxiter=20000,scale=False,center=True)
                W_estimate = DimRedMethod.loadings.T
                pca_idx_order = sort_pca_comp_by_correlation(W,W_estimate)
                W_estimate = W_estimate[pca_idx_order,...] #reorder components
                W_estimate_pinv = np.linalg.pinv(W_estimate)
                ts = get_timeseries_from_loadings(data,W_estimate)
                for CausalMethod_name in all_CausalMethod:
                    var_names= [str(i) for i in range(components)]
                    CausalMethod= globals()[CausalMethod_name](ts,var_names=var_names, mask=None)
                    parents = CausalMethod.get_parents(tau_min=min_lag,tau_max=max_lag)
                    for LinCoeffMethod_name in all_LinCoeffEstimation:
                        LinCoeffMethod = globals()[LinCoeffMethod_name](ts,var_names=var_names, mask=None)
                        links_coeffs_estimate,_,_ = LinCoeffMethod.fit(parents=parents,tau_max=max_lag,tau_min=min_lag)
                        lte_matrix_estimate = compute_LTE(links_coeffs_estimate,W_estimate,W_estimate_pinv)
                        lte_matrix = compute_LTE(links_coeffs,W,W_pinv)
                        sensitivity_estimate = compute_sensitivity(links_coeffs_estimate,W_estimate,W_estimate_pinv)
                        sensitivity = compute_sensitivity(links_coeffs,W,W_pinv)
                        sensitivity_error= np.abs(sensitivity_estimate-sensitivity)
                        LTE_MRAE = compute_MRAE(lte_matrix,lte_matrix_estimate)
                        LinCoeff_metric = compute_MSE_links(links_coeffs,links_coeffs_estimate,ground_truth_link_only=False)
                        print(LTE_MRAE)
                        if LTE_MRAE>5:
                            nextIterationFlag= True #singular LTE
                            break
                        if LinCoeffMethod_name== "VARmodel":
                            method_name= "+".join([DimRed_name,LinCoeffMethod_name])
                        else:
                            method_name= "+".join([DimRed_name,CausalMethod_name,LinCoeffMethod_name])
                        df_linkcoeff_metric=df_linkcoeff_metric.append({"method":method_name,"parameter":param_value,
                                               "simulation":i,
                                               "error": LinCoeff_metric},ignore_index=True)
                        df_lte_mrae=df_lte_mrae.append({"method":method_name,"parameter":param_value,
                                               "simulation":i,
                                               "error":LTE_MRAE
                                              },ignore_index=True)
                        df_sens_err = df_sens_err.append({"method":method_name,"parameter":param_value,
                                               "simulation":i,
                                               "error":sensitivity_error
                                              },ignore_index=True)
                    if nextIterationFlag: break
                if nextIterationFlag: continue
    df_lte_mrae=df_lte_mrae.drop_duplicates()
    df_linkcoeff_metric=df_linkcoeff_metric.drop_duplicates()
    df_sens_err=df_sens_err.drop_duplicates()
    return df_lte_mrae,df_linkcoeff_metric,df_sens_err

def generate_ts_from_coefs_and_residuals(VAR,res,array,min_lag=0):
        tau_max,N_var,_=VAR.shape
        tau_max += -1+min_lag
        T=array.shape[0]
        T_transient = res.shape[0] #total sample size with transient fraction
        y_gen= np.zeros((T_transient,N_var))
        y_gen+= res #add noise
        y_gen[:tau_max+1,...]= array[:tau_max+1,...] #initial values
        for i in range(tau_max+1,T_transient):
            for t in reversed(range(min_lag,tau_max+1)):
                y_gen[i,...]+= y_gen[i-t,...].dot(VAR[t,...].T)
        new_array= np.zeros((T,N_var))
        new_array=y_gen[T_transient-T:,...]
        #std_scaler = StandardScaler()
        #new_array = std_scaler.fit_transform(new_array)
        return new_array

def compute_LTE(lin_coeff_,W_,W_pinv_):
    N,L = W_.shape
    Id_N = np.eye(N)
    Id_L = np.eye(L)
    AK = np.sum(lin_coeff_[:,:,:],axis=0)
    Psi_inf= W_pinv_@(np.linalg.inv(Id_N-AK)-Id_N)@W_+Id_L #use the inverse of N-dim matrix for faster computations
    return Psi_inf

def get_G(K,W,W_pinv,Psi_inf):
    return np.kron((K@W@Psi_inf).T,Psi_inf@W_pinv)

def get_Z_from_dataframe(data,tau_min,tau_max):
    T_data,N_data= data.shape
    Z_t = np.zeros((N_data*(tau_max-tau_min+1),1))
    Z = np.zeros((N_data*(tau_max-tau_min+1),T_data-tau_max))
    for t in range(tau_max,T_data):
        tuple_Z_t=(data[t-tau,:] for tau in range(tau_min,tau_max+1))
        Z_t= np.hstack(tuple_Z_t).T
        Z[:,t-tau_max]=Z_t
    return Z
def get_Gamma_from_Z(Z):
    return Z@Z.T/Z.shape[1]
    
def quantity_of_interest(coeff,W_,W_pinv_,forcing_weights):
    #function defining the quantity of interest, for example sensitivity
    return (compute_LTE(coeff,W_,W_pinv_)@forcing_weights).mean()

def calculate_q_for_boot(CausalMethod_,med,W_,W_pinv_,b,new_data):
    #estimate parents from new_data
    new_CausalMethod_ = globals()[CausalMethod_.__class__.__name__](new_data,CausalMethod_.var_names, CausalMethod_.mask)
    new_parents = new_CausalMethod_.get_parents(med.tau_max,med.tau_min)
    #estimate linear coefficients from new_parents
    new_LinMed_model = globals()[med.__class__.__name__](new_data,med.var_names,med.mask)
    new_lin_coeff,_,_= new_LinMed_model.fit(new_parents,med.tau_max,med.tau_min)
    new_q = quantity_of_interest(new_lin_coeff,W_,W_pinv_,b)
    return new_q

def get_links_coeffs_matrix(link_coeff_dict,tau_max):
    N=len(list(link_coeff_dict))
    link_coeff_mat = np.zeros((tau_max + 1,N, N))
    for i in list(link_coeff_dict):
        for par in list(link_coeff_dict[i]):
                j, tau,val = par[0][0],par[0][1], par[1]
                link_coeff_mat[abs(tau),i,j] = val
    return link_coeff_mat

def get_timeseries_from_loadings(data,weights):
        return data.dot(weights.T)

def all_subclasses(cls):

    if cls == type:
        raise ValueError("Invalid class - 'type' is not a class")

    subclasses = set()

    stack = []
    try:
        stack.extend(cls.__subclasses__())
    except (TypeError, AttributeError) as ex:
        raise ValueError("Invalid class" + repr(cls)) from ex  

    while stack:
        sub = stack.pop()
        subclasses.add(sub.__name__)
        try:
            stack.extend(sub.__subclasses__())
        except (TypeError, AttributeError):
           continue

    return list(subclasses)


def compute_sensitivity(lin_coeff_,W_,W_pinv_):
    Psi_inf= compute_LTE(lin_coeff_,W_,W_pinv_)
    return Psi_inf.mean()
    
def compute_MSE_links(true_links,estimated_links,ground_truth_link_only):
    sse=0
    count=0
    for i in range(true_links.shape[0]):
        for j in range(true_links.shape[1]):
            for k in range(true_links.shape[2]):
                if ground_truth_link_only: #only compute MSE on links which are in the ground truth
                    if true_links[i,j,k]!=0:
                        count+=1
                        sse+= np.square(true_links[i,j,k]-estimated_links[i,j,k])
                else:
                    count+=1
                    sse+= np.square(true_links[i,j,k]-estimated_links[i,j,k])
    return sse/count
    
def compute_MSE(true_lte,estimated_lte):
    sse=0
    count=0
    for i in range(true_lte.shape[0]):
        for j in range(true_lte.shape[1]):
                if np.abs(true_lte[i,j])>10e-4:
                    count+=1
                    sse+= np.square(true_lte[i,j]-estimated_lte[i,j])
    return sse/count
    
def compute_MRAE(true_lte,estimated_lte):
    sse=0
    count=0
    for i in range(true_lte.shape[0]):
        for j in range(true_lte.shape[1]):
                if np.abs(true_lte[i,j])>10e-4:
                    count+=1
                    sse+= np.abs(true_lte[i,j]-estimated_lte[i,j])/np.abs(true_lte[i,j])
    return sse/count

def sort_pca_comp_by_correlation(true_weights,estimated_weights):
    N_comp,L=true_weights.shape
    U= [i for i in range(N_comp)]
    index_=[]
    for i in range(N_comp):
        j_max = np.argmax([np.abs(np.corrcoef(estimated_weights[i,...].T,true_weights[j,...].T))[0,1] 
                   for j in U])
        index_.append(U[j_max])
        del U[j_max]
    index_=[index_.index(i) for i in range(N_comp)]
    return index_