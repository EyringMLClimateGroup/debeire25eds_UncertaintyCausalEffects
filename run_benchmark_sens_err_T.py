import warnings
warnings.filterwarnings('ignore')
from all_functions import *
from CausalDiscovery import *
from LinCoeffEstimation import *
from DimensionReduction import *

if __name__=="__main__":
    N_sim=250 #number of simulations per parameter setting

    varying_parameter= "T"
    value_range=[50,100,200,300,400,500,750,1000]
    verbosity=1
    dfT_mrae,dfT_links,dfT_sens_err=generate_simulation(N_sim,varying_parameter,value_range,verbosity)
    dfT_mrae.to_csv("../results/%s_LTE_MRAE.csv"%varying_parameter)
    dfT_links.to_csv("../results/%s_Links_MSE.csv"%varying_parameter)
    dfT_sens_err.to_csv("../results/%s_Sensitivity_Error.csv"%varying_parameter)
