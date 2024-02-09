import warnings
warnings.filterwarnings('ignore')
from nipals import nipals
import numpy as np
from abc import abstractmethod

class DimensionReduction:
    def __init__(self,data: np.ma):
        """
        :param data: np.ma of shape (Time,Variables)
        """
        self.data= data
        self.loadings=None
    @abstractmethod
    def fit(self,ncomp:int):
        """
        :param ncomp: int, number of components to keep of the DimensionReduction method
        """
        pass
    
class PCA_nipals(DimensionReduction):
    
    def fit(self,ncomp:int =5, cv:bool =False,maxiter:int=5000,scale:bool=False,center:bool=True):
        PCA_ = nipals.Nipals(self.data)
        PCA_.fit(ncomp=ncomp,cv=cv,maxiter=maxiter,scale=scale,center=center)
        self.R2cum=PCA_.R2cum
        self.R2=PCA_.R2
        self.loadings=PCA_.loadings.values
        self.scores=PCA_.scores
        self.eig=PCA_.eig
                    
class Varimax(PCA_nipals):
    
    def fit(self,ncomp:int =5, cv:bool =False,maxiter:int=5000,scale:bool=False,center:bool=True):
        PCA_nipals.fit(self,ncomp=ncomp,cv=cv,maxiter=maxiter,scale=scale,center=center)
        Vr, Rot = Varimax.varimax(self.loadings, verbosity=0)
        Vr= Varimax.svd_flip(Vr)
        s2 = np.diag(self.eig)**2 / (self.data.shape[0] - 1.)
        # matrix with diagonal containing variances of rotated components
        S2r = np.dot(np.dot(np.transpose(Rot), s2), Rot)
        expvar = np.diag(S2r)
        sorted_expvar = np.sort(expvar)[::-1]
        # reorder all elements according to explained variance (descending)
        nord = np.argsort(expvar)[::-1]
        Vr = Vr[:, nord]
        total_var = np.sum(np.var(self.data, axis = 0))
        #save Varimax output
        self.loadings=Vr
        self.R2 = sorted_expvar/total_var
        self.R2cum = np.cumsum(self.R2)
        
    def varimax(Phi, gamma = 1, q = 10, 
        rtol = np.finfo(np.float32).eps ** 0.5,
        verbosity=0):
        """Varimax rotation of Phi."""
        p,k = Phi.shape
        R = np.eye(k)
        d=0
        # print Phi
        for i in range(q):
            if verbosity > 1:
                if i % 10 == 0.:
                    print("\t\tVarimax iteration %d" % i)
            d_old = d
            Lambda = np.dot(Phi, R)
            u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 
                       - (gamma/float(p)) * np.dot(Lambda, 
                        np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(u,vh)
            d = np.sum(s)
            if d_old!=0 and abs(d - d_old) / d < rtol: break
        # print i
        return np.dot(Phi, R), R
    
    def svd_flip(u, v=None, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        Parameters
        ----------
        u, v : ndarray
            u and v are the output of `linalg.svd` or
            `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
            so one can compute `np.dot(u * s, v)`.
        u_based_decision : boolean, (default=True)
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.
        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        if v is None:
             # rows of v, columns of u
             max_abs_rows = np.argmax(np.abs(u), axis=0)
             signs = np.sign(u[max_abs_rows, range(u.shape[1])])
             u *= signs
             return u

        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]

        return u, v