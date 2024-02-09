import warnings
warnings.filterwarnings('ignore')
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, RobustParCorr
from abc import abstractmethod

class CausalDiscovery:
    def __init__(self,data,var_names,mask=None):
        """
        :param data: np.ma or np.ndarray of shape (Time,Variables)
        :param var_names: list of string containing name of Variables
        :param mask: mask, default is None
        """
        self.data = data
        self.var_names = var_names
        self.mask = mask
        self.dataframe= pp.DataFrame(data,mask=mask,var_names=var_names)
    @abstractmethod
    def get_parents(self,tau_max: int, tau_min: int = 1):
        """
        :param tau_max: int,
        :param tau_min: int, default is 1
        """
        pass
       
class PCMCI_(CausalDiscovery):
    def get_parents(self,tau_max: int, tau_min: int = 1,include_lagzero_parents=False):
        cond_int_test= RobustParCorr()
        pc_alpha = [0.05,0.1,0.2]
        self.tau_min=tau_min
        self.tau_max=tau_max
        pcmci_ = PCMCI(dataframe=self.dataframe, cond_ind_test=cond_int_test,verbosity=0)
        results_ = pcmci_.run_pcmci(tau_min=tau_min,tau_max=tau_max,pc_alpha=pc_alpha,
                                        selected_links=None)
        self.parents = pcmci_.return_parents_dict(results_["graph"],results_["val_matrix"],
                                   include_lagzero_parents=include_lagzero_parents)
        return self.parents
    
class AllParents(CausalDiscovery):
    def get_parents(self,tau_max: int, tau_min: int = 1,include_lagzero_parents=False):
        self.tau_min=tau_min
        self.tau_max=tau_max
        parents={}
        T,N = self.dataframe.values[0].shape
        for var in range(N):
            parents[var]= []
            for par in range(N):
                for tau in range(tau_min,tau_max):
                    parents[var].append((par,-tau))
                if include_lagzero_parents and par!=var:
                    parents[var].append((par,0))
        self.parents = parents
        return self.parents