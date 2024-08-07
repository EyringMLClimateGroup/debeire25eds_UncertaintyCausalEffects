import warnings
warnings.filterwarnings('ignore')
from all_functions import *
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

if __name__=="__main__":
    N_sim=250 #number of simulations per parameter setting

    varying_parameter= "nx"
    value_range=[10,20,30,50,60,80]
    verbosity=1
    dfL_mrae,dfL_links,dfL_sens_err=generate_simulation(N_sim,varying_parameter,value_range,verbosity)
    dfL_mrae.to_csv("./output/%s_LTE_MRAE.csv"%varying_parameter)
    dfL_links.to_csv("./output/%s_Links_MSE.csv"%varying_parameter)
    dfL_sens_err.to_csv("./output/%s_Sensitivity_Error.csv"%varying_parameter)