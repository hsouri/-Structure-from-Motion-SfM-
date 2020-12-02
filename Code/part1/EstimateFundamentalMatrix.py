import numpy as np


def EstimateFundamentalMatrix(correspondences1, correspondences2):
    size = correspondences1.shape[0]
    nrm_corres1 , T1 = normalization(correspondences1, N=size)
    nrm_corres2, T2 = normalization(correspondences2, N=size)
    A = get_A(nrm_corres1, nrm_corres2, N=size)
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape((3,3)).T
    F = T1.T @ F @ T2
    U, S, V = np.linalg.svd(F.T)
    S = np.array([[S[0], 0, 0],
                  [0, S[1], 0],
                  [0, 0, 0]])
    F = U @ S @ V
    return F / F[2, 2]


def normalization(corres, N=8):
    m_x = np.mean(corres, axis=0)[0]
    m_y = np.mean(corres, axis=0)[1]
    sigma = get_sigma(corres, m_x, m_y, N)
    T = get_T(sigma, m_x, m_y)

    normalized_corres = get_narm_corres(corres, T, N=N)

    return normalized_corres, T


def get_sigma(corres, m_x, m_y, N=8):
    return np.mean(np.sqrt((corres[:, 0] - m_x) ** 2 + (corres[:, 1] - m_y) ** 2))


def get_narm_corres(corres, T, N=8):
    return (T @ np.append(corres.T, np.ones((N, 1))).reshape((-1, N))).T


def get_A(corres1, corres2, N=8):
    A = []
    for i in range(N):
        x_prime = corres1[i, 0]
        y_prime = corres1[i, 1]
        x = corres2[i, 0]
        y = corres2[i, 1]
        A.append([
            x * x_prime, x * y_prime, x, y * x_prime, y * y_prime, y, x_prime, y_prime, 1
        ])
    return A


def get_T(sigma, m_x, m_y):
    return np.dot(
        np.array([[1/sigma, 0, 0], [0, 1/sigma, 0], [0, 0, 1]]),
        np.array([[1, 0, -m_x], [0, 1, -m_y], [0, 0, 1]]))