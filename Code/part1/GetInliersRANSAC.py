import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix


def GetInliersRANSAC(features1, features2, features_ind):
    M = 500
    size = 8
    N = features1.shape[0]
    n = 0
    epsilon = 0.005

    for _ in range(M):
        inliers_list_1 = []; inliers_list_2 = []; iniler_lidices_list = []
        x_hat_1, x_hat_2 = get_random_correspondences(features1, features2, N, size)
        F = EstimateFundamentalMatrix(x_hat_1, x_hat_2)
        S = 0
        for i in range(N):
            x_1 = np.append(features1[i, :], 1)
            x_2 = np.append(features2[i, :], 1)
            if abs(x_2.T @ F @ x_1) < epsilon:
                inliers_list_1.append(features1[i, :])
                inliers_list_2.append(features2[i, :])
                iniler_lidices_list.append(features_ind[i])
                S += 1
        if n < S:
            n = S
            final_F = F
            final_inl_1 = np.array(inliers_list_1)
            final_inl_2 = np.array(inliers_list_2)
            final_ind = np.array(iniler_lidices_list)

    return final_ind, final_F, final_inl_1, final_inl_2


def get_random_correspondences(vec1, vec2, N, size):
    random = np.random.randint(0, N, size=size)
    return vec1[random, :], vec2[random, :]