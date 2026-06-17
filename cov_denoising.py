# cov_denoising.py
# Adapted from Marcos M. Lòpez de Prado, "Machine Learning for Asset Managers" (2020), Section 2 

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

def mpPDF(var, q, pts):
    """Marcenko-Pastur probability density function 
       q = T/N"""
    eMin, eMax = var*(1-(1./q)**0.5)**2, var*(1+(1./q)**0.5)**2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q/(2*np.pi*var*eVal) * np.sqrt(np.maximum((eMax-eVal)*(eVal-eMin), 0.0))
    # ensure one-dimensional data for Series creation
    pdf = pd.Series(pdf.flatten(), index=eVal.flatten())
    return pdf

def getPCA(matrix):
    """perform PCA (eigen decomposition) on a symmetric matrix and return sorted eigenvalues and eigenvectors"""
    eVal, eVec = np.linalg.eigh(matrix)  # compute eigenvalues and eigenvectors
    indices = eVal.argsort()[::-1]  # sort eigenvalues in descending order
    eVal, eVec = eVal[indices], eVec[:, indices]  # reorder eigenvectors accordingly
    eVal = np.diagflat(eVal)  # convert eigenvalues to a diagonal matrix
    return eVal, eVec


def fitKDE(obs, bWidth=0.01, x=None, kernel='gaussian'):
    """ fits a kernel density estimator (KDE) to the given observations """
    if len (obs.shape) == 1:obs=obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None: 
        x=np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:x=x.reshape(-1, 1)
    logProb=kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logProb), index = x.flatten())
    return pdf

def getCovMatrix(returns):
    """
    compute the empirical covariance matrix from stock returns using the dot-product method

    parameters:
    -----------
    returns : pd.DataFrame
        DataFrame where each column represents the returns of a stock

    returns:
    --------
    np.ndarray
        covariance matrix of the stock returns.
    """
    X = returns.values  # convert DataFrame to NumPy array
    X -= X.mean(axis=0)  # demean returns
    T = X.shape[0]  # number of time periods
    cov = np.dot(X.T, X) / (T - 1)  # compute covariance matrix
    return cov

def cov2corr(cov):
    """ derive the correlation matrix from a covariance matrix """
    std = np.sqrt(np.diag(cov))
    std = np.where(std <= 0, 1.0, std)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0) # numerical error
    np.fill_diagonal(corr, 1.0)
    return corr

def errPDFs(var, eVal, q, bWidth, pts=1000):
    """
    calculate error between theoretical and empirical PDFs
    
    parameters:
    -----------
    var : float
        variance parameter for MP distribution
    eVal : array-like
        empirical eigenvalues
    q : float
        ratio of observations to variables
    bWidth : float
        bandwidth for kernel density estimation
    pts : int, default=1000
        number of points for PDF evaluation
    
    returns:
    --------
    float
        sum of squared errors between PDFs
    """
    # calculate theoretical MP distribution
    pdf0 = mpPDF(var, q, pts)
    
    # fit empirical distribution using KDE
    pdf1 = fitKDE(eVal, bWidth=bWidth, x=pdf0.index.values)
    
    # calculate sum of squared errors
    sse = np.sum((pdf1 - pdf0)**2)
    
    return sse

def findMaxEval(eVal, q, bWidth):
    """find maximum eigenvalue threshold through MP distribution fitting"""
    lam = np.asarray(eVal).ravel()
    out = minimize(lambda *x: errPDFs(*x), 1.0,
                  args=(lam, q, bWidth),
                  bounds=((0.8, 1.2),))  # slightly tighter, realistic for corr matrices
    var = out['x'][0] if out['success'] else 1.0
    eMax = var*(1+(1./q)**0.5)**2
    # fail-safe so MP bulk does not swallow the entire spectrum
    if eMax >= lam.max():
        var = 1.0
        eMax = var*(1+(1./q)**0.5)**2
    return eMax, var

def denoisedCorr(eVal, eVec, nFacts):
    # remove noise from corr by fixing random eigenvalues    
    eVal_ = np.diag(eVal).copy()
    n = eVal_.shape[0]
    if nFacts >= n:
        corr1 = eVec @ np.diag(eVal_) @ eVec.T
        return cov2corr(corr1)
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(n - nFacts)
    corr1 = eVec @ np.diag(eVal_) @ eVec.T
    return cov2corr(corr1)

def denoisedCorr2(eVal, eVec, nFacts, alpha = 0):
    # remove noise from corr through targeted shrinkage 
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]
    corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
    corr1 = np.dot(eVecR, eValR).dot(eVecR.T)
    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    return cov2corr(corr2)  # enforce unit diagonal

# define wrapper function
def denoise_cov_from_cov(
        
    cov: pd.DataFrame,
    q: float,
    bWidth: float = 0.01,
    method: str = "const_resid",  # "const_resid" -> denoisedCorr, "targeted" -> denoisedCorr2
    alpha: float = 0.0,
) -> pd.DataFrame:
    
    # guardrail for rolling window estimation
    if q <= 1.0:
        return cov.copy()
    
    # cov -> corr
    cov = pd.DataFrame(cov, index=cov.index, columns=cov.columns)
    cov_vals = cov.values
    std = np.sqrt(np.diag(cov_vals))
    std = np.where(std <= 0, 1.0, std)
    corr = cov2corr(cov_vals)
    corr = 0.5 * (corr + corr.T) # enforce symmetry 


    # pca on corr
    eVal, eVec = getPCA(corr)

    # mp threshold -> nFacts
    eMax, _ = findMaxEval(np.diag(eVal), q=q, bWidth=bWidth)
    nFacts = int(np.sum(np.diag(eVal) > eMax))
    nFacts = max(1, min(nFacts, corr.shape[0]))

    # denoise corr
    if method == "const_resid":
        corr_d = denoisedCorr(eVal, eVec, nFacts)
    elif method == "targeted":
        corr_d = denoisedCorr2(eVal, eVec, nFacts, alpha=alpha)
    else:
        raise ValueError("unknown method")
    
    np.fill_diagonal(corr_d, 1.0)

    # corr -> cov using original std
    cov_d = corr_d * np.outer(std, std)
    cov_d = 0.5 * (cov_d + cov_d.T)
    return pd.DataFrame(cov_d, index=cov.index, columns=cov.columns)
