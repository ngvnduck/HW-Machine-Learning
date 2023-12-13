import numpy as np

_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    # Compute the polynomial kernel matrix
    return (np.dot(X1, X2.T) + 1) ** _polyDegree


def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    # Compute the Gaussian kernel matrix
    pairwise_sq_dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-pairwise_sq_dists / (2 * _gaussSigma**2))


def myCosineSimilarityKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    # Compute the cosine similarity kernel matrix (CIS 519 ONLY)
    norm_X1 = np.linalg.norm(X1, axis=1, keepdims=True)
    norm_X2 = np.linalg.norm(X2, axis=1, keepdims=True)
    return np.dot(X1, X2.T) / (norm_X1 * norm_X2)
