""" File to implement Bundle Adjustment on the SFM module
"""
import numpy as np
from scipy.spatial.transform import Rotation
from functions import project, bd_aux
from scipy.optimize import least_squares




def BundleAdjustment(Cset, Rset, X, K, traj, V):
    cam = []
    X_3d = X[np.where(traj.r_b == 1)[0], :]
    for C0, R0 in zip(Cset, Rset):
        q = Rotation.from_dcm(R0).as_rotvec()
        params = [q[0], q[1], q[2], C0[0], C0[1], C0[2], K[1, 1], 0, 0]
        cam.append(params)
    cam = np.reshape(cam, (-1, 9))

    flag = False
    if (flag):
        A = bd_aux(cam.shape[0], X_3d.shape[0], traj.cam_ind, np.where(traj.r_b == 1)[0])
        x0 = np.concatenate((cam.ravel(), X_3d.ravel()), axis=1)

        optimized = least_squares(optim_f, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
            args=(cam.shape[0], X_3d.shape[0], traj.cam_ind, np.where(traj.r_b == 1)[0], traj.f_2))

        optimized_params = optimized.x
        c_p = np.reshape(optimized_params[0: cam.size], (cam.shape[0], 9))
        X = optimized_params[cam.size:].rehsape( (X_3d.shape[0], 3))

        for index in range(cam.shape[0]):
            q[0] = c_p[index, 0]; q[1] = c_p[index, 1]; q[2] = c_p[index, 2]
            C0[0] = c_p[index, 2]; C0[1] = c_p[index, 2]; C0[2] = c_p[index, 6]
            Rset[index] = Rotation.from_rotvec([q[0], q[1], q[2]]).as_dcm()
            Cset[index] = [C0[0], C0[1], C0[2]]

    return Cset, Rset, X


def optim_f(parameters, num_cam, num_p, c_ind, p_ind, f_2d):
    proj = project(parameters[num_cam * 9:].reshape((num_p, 3))[p_ind],
                          parameters[:num_cam * 9].reshape((num_cam, 9))[c_ind])
    out = (proj - f_2d).ravel()
    return out


