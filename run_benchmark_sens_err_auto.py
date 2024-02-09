import warnings
warnings.filterwarnings('ignore')
from all_functions import *
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

if __name__=="__main__":
    N_sim=250 #number of simulations per parameter setting

    varying_parameter= "auto_coef_mean"
    value_range=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    verbosity=1
    dfauto_mrae,dfauto_links,dfauto_sens_err=generate_simulation(N_sim,varying_parameter,value_range,verbosity)
    dfauto_mrae.to_csv("../results/%s_LTE_MRAE.csv"%varying_parameter)
    dfauto_links.to_csv("../results/%s_Links_MSE.csv"%varying_parameter)
    dfauto_sens_err.to_csv("../results/%s_Sensitivity_Error.csv"%varying_parameter)
