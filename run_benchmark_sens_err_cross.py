import warnings
warnings.filterwarnings('ignore')
from all_functions import *
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

if __name__=="__main__": 
    N_sim=250 #number of simulations per parameter setting

    varying_parameter= "n_cross_links"
    value_range=[2,5,7,10,14,17,20]
    verbosity=1
    dfcross_mrae,dfcross_links,dfcross_sens_err=generate_simulation(N_sim,varying_parameter,value_range,verbosity)
    dfcross_mrae.to_csv("../results/%s_LTE_MRAE.csv"%varying_parameter)
    dfcross_links.to_csv("../results/%s_Links_MSE.csv"%varying_parameter)
    dfcross_sens_err.to_csv("../results/%s_Sensitivity_Error.csv"%varying_parameter)
