import numpy as np

def ExtractCameraPose(E):

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    U, S, V_T = np.linalg.svd(E)

    R = [U @ W @ V_T, U @ W @ V_T, U @ W.T @ V_T, U @ W.T @ V_T]
    C = [U[:, 2], -U[:, 2], U[:, 2], -U[:, 2]]

    for index in range(4):
        if (np.linalg.det(R[index]) < 0):
            R[index] = -R[index]
            C[index] = -C[index]

    return C, R
