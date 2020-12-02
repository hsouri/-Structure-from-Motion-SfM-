import numpy as np
from scipy.sparse import lil_matrix


def H(x):
    c, w = x.shape
    if w == 3 or w == 2:
        o = np.ones((c, 1))
        x_new = np.concatenate((x, o), axis=1)
    else:
        x_new = x
    return x_new


def bd_aux(num_cam, num_p, cam_ind, p_ind):
    M = lil_matrix((cam_ind.size * 2, num_cam * 9 + num_p * 3), dtype=int)
    for i in range(9):
        M[2 * np.arange(cam_ind.size), cam_ind * 9 + i] = 1
        M[2 * np.arange(cam_ind.size) + 1, cam_ind * 9 + i] = 1
    for j in range(3):
        M[2 * np.arange(cam_ind.size), num_cam * 9 + p_ind * 3 + j] = 1
        M[2 * np.arange(cam_ind.size) + 1, num_cam * 9 + p_ind * 3 + j] = 1
    return M


def p_3_2(x, K, C, R):
    C = C.reshape(-1, 1)
    x = x.reshape(-1, 1)
    P = K @ R @ np.concatenate((np.identity(3), -C), axis=1)
    X = np.vstack((x, 1))
    u = (P[0, :] @ X).T / (P[2, :] @ X).T
    v = (P[1, :] @ X).T / (P[2, :] @ X).T
    return np.concatenate((u, v), axis=0)


def rotate(points, rot):
    theta = np.linalg.norm(rot, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot / theta
        v = np.nan_to_num(v)
    return np.cos(theta) * points + np.sin(theta) * np.cross(v, points) + \
           np.sum(points * v, axis=1)[:, np.newaxis] * (1 - np.cos(theta)) * v


def project(pts, cam):
    proj = rotate(pts, cam[:, :3])
    proj += cam[:, 3:6]
    proj = -proj[:, :2] / proj[:, 2, np.newaxis]
    r_ = 1 + cam[:, 7] * np.sum(proj**2, axis=1) + cam[:, 8] * np.sum(proj**2, axis=1) ** 2
    proj *= (r_ * cam[:, 6])[:, np.newaxis]
    return proj