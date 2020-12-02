import numpy as np
from scipy.spatial.transform import Rotation
from scipy import optimize


def NonLinearPnP(X, x, K, C_0, R_0):
    q = Rotation.from_dcm(R_0).as_quat()
    optimized_param = optimize.least_squares(fun=error, method="dogbox", args=[K, X, x],
                                             x0=[C_0[0], C_0[1], C_0[2], q[0], q[1], q[2], q[3]])
    R = optimized_param.x[3:7]
    return optimized_param.x[0:3], Rotation.from_quat([R[0], R[1], R[2], R[3]]).as_dcm()


def error(x0, K, X, x):
    R = Rotation.from_quat([x0[3:7][0], x0[3:7][1], x0[3:7][2], x0[3:7][3]]).as_dcm()
    P = K @ R @ np.concatenate((np.identity(3), -x0[0:3].reshape(-1, 1)), axis=1)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    error_vec = x[:, 0] - (P[0, :] @ X.T).T / (P[2, :] @ X.T).T + x[:, 1] - (P[1, :] @ X.T).T / (P[2, :] @ X.T).T
    return sum(error_vec)
