import warnings
warnings.filterwarnings('ignore')
from all_functions import *
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

if __name__=="__main__":
    N_sim=250 #number of simulations per parameter setting

    varying_parameter= "mode_strength"
    value_range=[0.01,0.1,0.25,0.5,0.75,1.]
    verbosity=1
    dfmode_mrae,dfmode_links,dfmode_sens_err=generate_simulation(N_sim,varying_parameter,value_range,verbosity)
    dfmode_mrae.to_csv("../results/%s_LTE_MRAE.csv"%varying_parameter)
    dfmode_links.to_csv("../results/%s_Links_MSE.csv"%varying_parameter)
    dfmode_sens_err.to_csv("../results/%s_Sensitivity_Error.csv"%varying_parameter)

