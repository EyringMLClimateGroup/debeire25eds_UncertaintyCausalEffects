import warnings
warnings.filterwarnings('ignore')
from all_functions import *
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

if __name__=="__main__":
    N_sim=250 #number of simulations per parameter setting

    varying_parameter= "components"
    value_range=[2,5,7,9,12,15]
    verbosity=1
    dfcomp_mrae,dfcomp_links,dfcomp_sens_err=generate_simulation(N_sim,varying_parameter,value_range,verbosity)
    dfcomp_mrae.to_csv("../results/%s_LTE_MRAE.csv"%varying_parameter)
    dfcomp_links.to_csv("../results/%s_Links_MSE.csv"%varying_parameter)
    dfcomp_sens_err.to_csv("../results/%s_Sensitivity_Error.csv"%varying_parameter)
