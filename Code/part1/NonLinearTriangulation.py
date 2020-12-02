import numpy as np
import scipy.optimize as opt


def NonLinearTriangulation(K, x_1, x_2, x_0, R_1, C_1, R_2, C_2):
    x_0 = x_0.flatten()
    optimized_params = opt.least_squares(fun=function, x0=x_0, method="dogbox", args=[K, x_1, x_2, R_1, C_1, R_2, C_2])
    return optimized_params.x.reshape((x_1.shape[0], 3))


def function(x_0, K, x_1, x_2, R_1, C_1, R_2, C_2):
    X = x_0.reshape((x_1.shape[0], 3))
    C_2 = C_2.reshape((3, -1))
    X = np.concatenate((X, np.ones((x_1.shape[0], 1))), axis=1)

    t_1 = np.concatenate((np.identity(3), -C_1), axis=1)
    t_2 = np.concatenate((np.identity(3), -C_2), axis=1)
    P1 = K @ R_1 @ t_1
    P2 = K @ R_2 @ t_2

    term_1 = ((x_1[:, 0] - np.divide(((P1[0, :] @ X.T).T), ((P1[2, :] @ X.T).T))) +
              (x_1[:, 1] - np.divide(((P1[1, :] @ X.T).T), ((P1[2, :] @ X.T).T))))
    term_2 = ((x_2[:, 0] - np.divide(((P2[0, :] @ X.T).T), ((P2[2, :] @ X.T).T))) +
              (x_2[:, 1] - np.divide(((P2[1, :] @ X.T).T), ((P2[2, :] @ X.T).T))))

    return sum(sum(term_1, term_2))
