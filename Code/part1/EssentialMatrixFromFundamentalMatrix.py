import numpy as np


def EssentialMatrixFromFundamentalMatrix(F, K):
    E = K.T @ F @ K
    U, S, V = np.linalg.svd(E)
    return U @ np.diag([1, 1, 0]) @ V
