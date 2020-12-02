import numpy as np


def LinearTriangulation(K, C_1, R_1, C_2, R_2, x_1, x_2):
    p_1 = K @ R_1 @ np.concatenate((np.identity(3), -C_1.reshape((3, 1))), axis=1)
    p_2 = K @ R_2 @ np.concatenate((np.identity(3), -C_2.reshape((3, 1))), axis=1)
    X_1 = np.hstack((x_1, np.ones((x_1.shape[0], 1))))
    X_2 = np.hstack((x_2, np.ones((x_1.shape[0], 1))))
    X = np.zeros((x_1.shape[0], 3))
    for i in range(x_1.shape[0]):
        X_1_i = X_1[i, :]
        X_2_i = X_2[i, :]
        s1 = np.array([[0, -X_1_i[2], X_1_i[1]], [X_1_i[2], 0, X_1_i[0]], [X_1_i[1], X_1_i[0], 0]])
        s2 = np.array([[0, -X_2_i[2], X_2_i[1]], [X_2_i[2], 0, X_2_i[0]], [X_2_i[1], X_2_i[0], 0]])
        A = np.concatenate((np.dot(s1, p_1), np.dot(s2, p_2)), axis=0)
        S, U, v_T = np.linalg.svd(A)
        x = v_T[-1] / v_T[-1, -1]
        x = np.reshape(x, (len(x), -1))
        X[i, :] = x[0: 3].T
    return X
