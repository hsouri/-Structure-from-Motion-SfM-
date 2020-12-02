
def DisambiguateCameraPose(C_set, R_set, X_set):
    max_so_far = 0
    for i in range(4):
        satisfied_points = 0
        for j in range(X_set[i].shape[0]):
            if R_set[i][2, :] @ (X_set[i][j, :] - C_set[i]) > 0 and X_set[i][j, 2] >= 0:
                satisfied_points += 1
        if satisfied_points > max_so_far:
            C = C_set[i]
            R = R_set[i]
            X = X_set[i]
            max_so_far = satisfied_points

    return C, R, X