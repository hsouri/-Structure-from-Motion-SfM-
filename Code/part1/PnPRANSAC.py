import numpy as np
import random
from tqdm import tqdm
from functions import H, p_3_2


def PnPRANSAC(X, x, K):
    C_new = np.zeros((3, 1))
    R_new = np.identity(3)
    n = 0
    epsilon = 5
    x_H = H(x)
    M = 500
    N = x.shape[0]

    for _ in tqdm(range(M)):
        random_idx = random.sample(range(x.shape[0]), 6)
        C, R = LinearPnP(X[random_idx][:], x[random_idx][:], K)
        S = []
        for j in range(N):
            re_p = p_3_2(x_H[j][:], K, C, R)
            e = np.sqrt( np.square((x_H[j, 0]) - re_p[0]) + np.square((x_H[j, 1] - re_p[1])))
            if e < epsilon:
                S.append(j)
        abs_S = len(S)

        if n < abs_S:
            n = abs_S; R_new = R; C_new = C

        if abs_S == x.shape[0]:
            break
    return C_new, R_new


def LinearPnP(X, x, K):

    I = np.ones((X.shape[0], 1))
    X = np.concatenate((X, I), axis=1)
    x = np.concatenate((x, I), axis=1)
    x = np.transpose(np.linalg.inv(K) @ x.T)
    M = []
    for i in range(X.shape[0]):

        row_1 = np.hstack((np.hstack((np.zeros((1, 4)), -X[i, :].reshape((1, 4)))), x[i, :][1] * X[i, :].reshape((1, 4))))
        row_2 = np.hstack((np.hstack((X[i, :].reshape((1, 4)), np.zeros((1, 4)))), -x[i, :][0] * X[i, :].reshape((1, 4))))
        row_3 = np.hstack((np.hstack((-x[i, :][1] * X[i, :].reshape((1, 4)),
                                   x[i, :][0] * X[i, :].reshape((1, 4)))), np.zeros((1, 4))))
        A = np.vstack((np.vstack((row_1, row_2)), row_3))
        if (i == 0):
            M = A
        else:
            M = np.concatenate((M, A), axis=0)
    U, S, V_T = np.linalg.svd(M)
    R = V_T[-1].reshape((3, 4))[:, 0: 3]
    u, s, v_T = np.linalg.svd(R)
    np.identity(3)[2][2] = np.linalg.det(np.matmul(u, v_T))
    R = np.dot(np.dot(u, np.identity(3)), v_T)
    C = -np.dot(np.linalg.inv(R), V_T[-1].reshape((3, 4))[:, 3])
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R