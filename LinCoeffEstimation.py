import warnings
warnings.filterwarnings('ignore')
from tigramite import data_processing as pp
from tigramite.models import LinearMediation
from statsmodels.tsa.api import VAR
import numpy as np
from abc import abstractmethod

class LinCoeffEstimation:
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def fit(self,data, parents,tau_min,tau_max):
        """
        Returns estimated coefficient in a (tau_max(or tau_max+1),components,components) array
        residuals, and variance of residuals
        """
        pass
    
class LinMed(LinCoeffEstimation):
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
    
    def fit(self, parents: dict,tau_max: int,tau_min: int = 1):
        """
        :parents: dictionary of parents for each variable used as predictors
        :targets: list of variable to use as targets
        :param tau_max: int,
        :param tau_min: int, default is 1
        """
        self.tau_min=tau_min
        self.tau_max=tau_max
        self.parents=parents
        dataframe= self.dataframe
        data=self.data
        med = LinearMediation(dataframe,
                 data_transform=None,
                 verbosity=0)
        med.fit_model(all_parents=parents,tau_max=tau_max,return_data= True)
        lin_coeff = med.phi
        #Compute residuals and estimate covariance of residuals        
        T,N=data.shape
        res=data[tau_max:,:]
        for var in med.fit_results.keys():
            sub_model = med.fit_results[var]
            if sub_model is not None:
                model=sub_model["model"]
                saved_data = med.fit_results[var]["data"].T
                y=saved_data[:,1]
                x=saved_data[:,2:]
                y_pred=model.predict(x)
                res[:,var]=y-y_pred
            else:
                continue
        self.fitted_coef=lin_coeff
        cov_res= res.T@res/(T-tau_max)
        self.res=res
        self.model=med
        return lin_coeff,cov_res,res
    
class VARmodel(LinCoeffEstimation):
    def __init__(self,data,var_names=None,mask=None):
        self.data= data
        self.var_names=var_names
        self.mask= mask
        self.dataframe= pp.DataFrame(data,mask=mask,var_names=var_names)
    def fit(self, parents,tau_max,tau_min):
        VAR_model = VAR(self.data)
        self.tau_max= tau_max
        self.tau_min=tau_min
        VAR_results = VAR_model.fit(maxlags=tau_max,trend="n")
        self.model=VAR_model
        fitted_coef = VAR_results.coefs
        res= VAR_results.resid
        cov_res= VAR_results.sigma_u
        lin_coeff_=np.zeros((tau_max+1,fitted_coef.shape[1],fitted_coef.shape[2]))
        lin_coeff_[1:,...]=fitted_coef
        self.res=res
        self.fitted_coef=lin_coeff_
        return self.fitted_coef,res,cov_res