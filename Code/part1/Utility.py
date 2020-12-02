import numpy as np
import cv2
import matplotlib.pyplot as plt
from GetInliersRANSAC import GetInliersRANSAC
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from NonLinearTriangulation import NonLinearTriangulation
from NonLinearPnP import NonLinearPnP
from PnPRANSAC import PnPRANSAC
from BuildVisibilityMatrix import BuildVisibilityMatrix
from BundleAdjustment import BundleAdjustment
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math


class pair_images_environment:

    def __init__(self, index1, index2):
        self.image1_index = index1
        self.image2_index = index2
        # self.q = [int(index1) - 1, int(index2) - 1]
        self.q = [int(index1) , int(index2) ]
        self.image1_path = 'Data/' + str(index1) + '.jpg'
        self.image2_path = 'Data/' + str(index2) + '.jpg'
        self.features_indices = None
        self.outliers_indices = None
        self.featues_points1 = None
        self.featues_points2 = None
        self.outliers_points1 = None
        self.outliers_points2 = None
        self.F = None
        self.E = None
        self.C_set = None
        self.R_set = None
        self.X_set = []
        self.C = []
        self.R = []
        self.L_X = []
        self.NL_X = []
        self.f_3 = []
        self.f_2 = []
        self. vis = []
        self.r_b = []
        self.Cset = []
        self.Rset = []
        self.re_indices = []
        self.cam_ind = []
        self.color = []

    def set_parameters(self, outliers, B, U, V):
        self.features_indices = np.where(np.logical_and(B[:, self.image1_index - 1],
                                                        B[:, self.image2_index - 1]) == True)
        self.outliers_indices = np.where(np.logical_and(outliers[:, self.image1_index - 1],
                                                        outliers[:, self.image2_index - 1]) == True)

        self.featues_points1 = self.set_feature_points(U, V, image_index=self.image1_index)
        self.outliers_points1 = self.set_outlier_points(U, V, image_index=self.image1_index)
        self.featues_points2 = self.set_feature_points(U, V, image_index=self.image2_index)
        self.outliers_points2 = self.set_outlier_points(U, V, image_index=self.image2_index)
        self.F = EstimateFundamentalMatrix(self.featues_points1, self.featues_points2)

    def set_feature_points(self, U, V, image_index):
        return np.concatenate((U[self.features_indices, image_index - 1].reshape((-1, 1)),
                                          V[self.features_indices, image_index - 1].reshape((-1, 1))), axis=1)

    def set_outlier_points(self, U, V, image_index):
        return np.concatenate((U[self.outliers_indices, image_index - 1].reshape((-1, 1)),
                              V[self.outliers_indices, image_index - 1].reshape((-1, 1))), axis=1)


class utility:

    def __init__(self):
        self.K = np.array([[568.996140852, 0, 643.21055941],
                           [0, 568.988362396, 477.982801038],
                           [0,          0,                1]])
        self.num_imgs = 6
        self.img1 = 1
        self.img2 = 4
        self.corres_threshold = 8
        self.data_root = "Data/"
        self.U = []
        self.V = []
        self.B = []
        self.RGB = []
        self.outliers = []
        self.PIE = None

    def initialize(self):
        for first_index in range(1, self.num_imgs):
            u = []; v = []; b = []
            for second_index in range(first_index + 1, self.num_imgs + 1):
                new_u, new_v, new_b, new_rgb = self.find_match(first_index, second_index)

                if second_index == first_index + 1:
                    u = new_u; v = new_v; b = new_b
                else:
                    u, v, b = self.concat_list(u, v, b, new_u[:, 1].reshape((-1, 1)), new_v[:, 1].reshape((-1, 1)),
                                               new_b[:, 1].reshape((-1, 1)), axis=1)
            if first_index == 1:
                self.U = u; self.V = v; self.B = b; self.RGB = new_rgb
            else:
                u, v, b = self.concat_list(np.zeros((u.shape[0], first_index - 1)), np.zeros((v.shape[0], first_index - 1)),
                                           np.zeros((b.shape[0], first_index - 1)), u, v, b, axis=1)
                self.U, self.V, self.B = self.concat_list(self.U, self.V, self.B, u, v, b, axis=0)
                self.RGB = np.concatenate((self.RGB, new_rgb), axis=0)

    def set_environmnet(self, image_index1=2, image_index2=3):
        self.PIE = pair_images_environment(image_index1, image_index2)

    def save_class(self):
        np.save('U.npy', self.U)
        np.save('V.npy', self.V)
        np.save('B.npy', self.B)
        np.save('RGB.npy', self.RGB)
        np.save('outliers.npy', self.outliers)

    def plot_points(self, d_3=True, d_2=False):
        if d_3:
            self.plot_3_d()
        if d_2:
            self.plot_2_d()



    def plot_matchings(self, inliers=False, outliers=False):
        self.PIE.set_parameters(self.outliers, self.B, self.U, self.V)
        self.plot_features(plot_inliers=inliers, plot_outliers=outliers)

    def rand_color(self):
        return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

    def plot_features(self, plot_inliers=False, plot_outliers=False):
        name = '.png'
        big_pic, shift = self.get_bigpic_frame()
        radius = 4
        thickness = 1
        if plot_inliers:
            name = "inliers_" + name
            big_pic = self.insert_points(big_pic, self.PIE.featues_points1,
                                         self.PIE.featues_points2, shift, radius, thickness)
        if plot_outliers:
            name = "outliers_" + name
            big_pic = self.insert_points(big_pic, self.PIE.outliers_points1,
                                         self.PIE.outliers_points2, shift, radius, thickness)
        # cv2.resizeWindow(name, 1000, 600)
        name = str(self.PIE.image1_index) + "_" + str(self.PIE.image2_index) + "_" + name
        cv2.imwrite(name, big_pic)


    def insert_points(self, big_pic, points1, points2, shift, radius, thickness):
        for index, point in enumerate(points1):
            point1 = (int(point[0]), int(point[1]))
            point2 = (int(points2[index][0]) + shift, int(points2[index][1]))
            cv2.circle(big_pic, point1, radius, self.rand_color(), -1)
            cv2.circle(big_pic, point2, radius, self.rand_color(), -1)
            cv2.line(big_pic, point1, point2, self.rand_color(), thickness=thickness)
        return big_pic

    def plot_3_d(self):
        points = self.PIE.f_3
        name = "3_d.png"
        fig = plt.figure()
        fig = plt.axes(projection='3d')
        fig.scatter3D(
            points[:, 0], points[:, 1], points[:, 2], c=self.PIE.color / 255.0, s=1)
        fig.set_xlabel('x')
        fig.set_ylabel('y')
        fig.set_zlabel('z')
        fig.set_xlim([-0.5, 1])
        fig.set_ylim([-0.5, 1])
        fig.set_zlim([0, 1.5])
        plt.savefig(name)

    def get_bigpic_frame(self):

        image1 = cv2.imread(self.PIE.image1_path)
        image2 = cv2.imread(self.PIE.image2_path)
        u1, v1 = image1.shape[:2]; u2, v2 = image2.shape[:2]
        frame = np.zeros((max([u1, u2]), v1 + v2, 3), dtype='uint8')
        frame[:u1, :v1, :] = image1
        frame[:u2, v1: v1 + v2, :] = image2
        return frame, v1

    def load_class(self):
        self.U = np.load('U.npy')
        self.V = np.load('V.npy')
        self.B = np.load('B.npy')
        self.RGB = np.load('RGB.npy')
        self.outliers = np.load('outliers.npy')

    def concat_list(self, u, v, b, new_u, new_v, new_b, axis=1):
        u_ = np.concatenate((u, new_u), axis=axis)
        v_ = np.concatenate((v, new_v), axis=axis)
        b_ = np.concatenate((b, new_b), axis=axis)
        return u_, v_, b_

    def append_second_features(self, row, image_index, u, v, b, rgb):

        if (len(image_index[0]) != 0):
            index = image_index[0][0]
            u.append(row[index + 1]); v.append(row[index + 2]); b.append(1)
        else:
            u.append(0); v.append(0); b.append(0)

    def plot_2_d(self):
        points = self.PIE.f_3
        name = "2_d.png"
        plt.scatter(points[:, 0], points[:, 2], c=self.PIE.color / 255.0, s=1)
        self.cam_plot(self.PIE.C_set, self.PIE.R_set)
        ax1 = plt.gca()
        ax1.set_xlabel('x')
        ax1.set_ylabel('z')
        # ax.set_xlim([-0.5, 1])
        # ax.set_ylim([-0.5, 2])
        plt.savefig(name)


    def update_outlier_mat(self, index, first_img_index, second_img_index):
        self.outliers[index, first_img_index - 1] = 1
        self.outliers[index, second_img_index - 1] = 1

    def read_matching(self, image_path):
        output = []
        with open(image_path) as matching_file:
            for index, row in enumerate(matching_file):
                if index != 0:
                    output.append(np.transpose(row.rstrip('\n').split()).astype(float))
        return output

    def append_first_features(self, row, u, v, b, rgb):
        u.append(row[3])
        v.append(row[4])
        b.append(1)
        rgb.append(row[0])
        rgb.append(row[1])
        rgb.append(row[2])

    def get_angles(self, R):

        if not math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]) < 1e-6:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
            z = 0

        return np.array([x, y, z])

    def cam_plot(self, C, R):
        for index, cam in enumerate(C):
            R = np.rad2deg(self.get_angles(R[index]))
            plt.plot(C[index][0], C[index][2], marker=(3, 0, int(R[1])), markersize=10, linestyle='None')

    def update_global_params(self):

        self.PIE.f_3 = self.PIE.f_3[np.where(self.PIE.r_b == 1)[0], :]
        self.PIE.color = self.RGB[np.where(self.PIE.r_b == 1)[0], :]

    def update_B_mat(self, index, image_index):
        self.B[index, image_index - 1] = 0

    def add_matching(self, new_u, new_v, new_b, new_rgb, u, v, b, rgb, flag):
        if flag:
            new_u.append(np.transpose(u)); new_v.append(np.transpose(v))
            new_b.append(np.transpose(b)); new_rgb.append(np.transpose(rgb))

    def get_image_path(self, index):
        return self.data_root + "matching" + str(index) + ".txt"

    def get_features(self, indices_vector, image_index):
        return np.concatenate((self.U[indices_vector, image_index - 1].reshape((-1, 1)),
                                        self.V[indices_vector, image_index - 1].reshape((-1, 1))), axis=1)

    def initial_axillary(self):
        self.PIE.f_3 = np.zeros((self.B.shape[0], 3))
        self.PIE. vis = np.zeros((self.B.shape[0], self.num_imgs))
        self.PIE.r_b = np.zeros((self.B.shape[0], 1))

    def update_local_params(self, im_index, C, R):
        self.PIE.vis[self.PIE.re_indices, im_index] = 1
        self.PIE.q.append(im_index)
        self.PIE.Cset.append(C)
        self.PIE.Rset.append(R)

    def find_match(self, first_image, second_image):
        image_path = self.get_image_path(first_image)
        matchings = self.read_matching(image_path)
        new_u = []; new_v = []; new_b = []; new_rgb = []

        for i, matching in enumerate(matchings):
            u = []; v = []; b = []; rgb = []
            row = matching[1:]
            second_index = np.where(row == second_image)
            self.append_first_features(row, u, v, b, rgb)
            self.append_second_features(row, second_index, u, v, b, rgb)
            add_flag = len(u) != 0
            self.add_matching(new_u, new_v, new_b, new_rgb, u , v, b, rgb, flag=add_flag)

        return np.array(new_u), np.array(new_v), np.array(new_b), np.array(
            new_rgb)

    def estimate_E(self):
        self.initial_axillary()
        self.PIE.set_parameters(self.outliers, self.B, self.U, self.V)
        self.PIE.E = EssentialMatrixFromFundamentalMatrix(self.PIE.F, self.K)


    def extract_pose(self):
        self.PIE.C_set, self.PIE.R_set = ExtractCameraPose(self.PIE.E)

    def correct_x(self):
        for i, row in enumerate(self.PIE.f_3):
            if self.PIE.f_3[i, 2] < 0:
                self.PIE.r_b[i] = 0
                self.PIE.vis[i, :] = 0

    def get_indices(self, first, sec):
        return np.where(np.logical_and(self.B[:, first - 1], self.B[:, sec - 1]) == True)[0]

    def check_cheirality(self):
        self.PIE.C, self.PIE.R, self.PIE.L_X = DisambiguateCameraPose(self.PIE.C_set, self.PIE.R_set, self.PIE.X_set)
        self.PIE.Cset.append(self.PIE.C)
        self.PIE.Rset.append(self.PIE.R)

    def skip(self, image_index, offset=1):
        return np.isin(self.PIE.q, image_index)[0]

    def set_axillary(self):
        self.PIE.r_b[self.PIE.features_indices] = 1
        self.PIE.f_3[self.PIE.features_indices, :] = self.PIE.NL_X
        self.PIE.vis[self.PIE.features_indices, self.PIE.image1_index - 1] = 1
        self.PIE.vis[self.PIE.features_indices, self.PIE.image2_index - 1] = 1


    def get_PnP_args(self, im_index):
        self.PIE.re_indices = np.where(np.logical_and(self.PIE.r_b, self.B[:, im_index].reshape((-1, 1))) == True)[0]
        if (len(self.PIE.re_indices) < 8):
            return None, None, True
        x = np.transpose([self.U[self.PIE.re_indices, im_index], self.V[self.PIE.re_indices, im_index]])
        X = self.PIE.f_3[self.PIE.re_indices, :]
        return x, X, False

    def concat(self, index, ind_vec):
        return np.concatenate((self.U[ind_vec, index].reshape((-1, 1)), self.V[ind_vec, index].reshape((-1, 1))), axis=1)

    def update_vis_X(self, ind_vec, X, j):

        self.PIE.vis[ind_vec, self.PIE.q[j]] = 1
        self.PIE.vis[ind_vec, j] = 1
        self.PIE.f_3[ind_vec, :] = X
        self.PIE.r_b[ind_vec] = 1


    def update_x(self, im_index, j, C, R):
        ind_vec = np.where(np.logical_and(np.logical_and(1 - self.PIE.r_b,
                                                         self.B[:, self.PIE.q[j]].reshape((-1, 1))),
                                          self.B[:, im_index].reshape((-1, 1))) == True)[0]
        if len(ind_vec) < 8:
            return False
        x1 = self.concat(self.PIE.q[j], ind_vec)
        x2 = self.concat(im_index, ind_vec)
        X = LinearTriangulation(self.K, self.PIE.Cset[j], self.PIE.Rset[j], C, R, x1, x2)
        self.update_vis_X(ind_vec, X, j)
        return True

    def set_poits(self, im_index):
        self.PIE.cam_ind = im_index * np.ones((len(np.where(self.PIE.r_b == 1)[0]), 1))
        self.PIE.f_2 = self.concat(im_index, np.where(self.PIE.r_b == 1)[0])




    def plot_tri(self):
        X_L = self.PIE.L_X
        X_NL = self.PIE.NL_X
        name = 'Triangulation.png'
        plt.scatter(X_L[:, 0], X_L[:, 2], c='b', s=4, label='Linear')
        plt.scatter(X_NL[:, 0], X_NL[:, 2], c='g', s=4, label='Nonlinear')
        ax = plt.gca()
        plt.grid('true')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(name)

    def non_linear_trangulation(self):
        self.PIE.NL_X = NonLinearTriangulation(self.K, self.PIE.featues_points1, self.PIE.featues_points2, self.PIE.L_X,
                                   np.eye(3), np.zeros((3, 1)), self.PIE.R, self.PIE.C)
        self.set_axillary()

    def register_and_add_3d_points(self, print_vis = False):
        for im_index in range(self.num_imgs):
            if self.skip(im_index):
                continue
            x, X, flag = self.get_PnP_args(im_index)
            if flag:
                continue
            C, R = PnPRANSAC(X, x, self.K)
            C, R = NonLinearPnP(X, x, self.K, C, R)
            self.update_local_params(im_index, C, R)
            for j in range(0, len(self.PIE.q) - 1):
                flag = self.update_x(im_index, j, C, R)
                if flag:
                    continue
            self.correct_x()
            V = BuildVisibilityMatrix(self.PIE)
            if print_vis:
                print(V)
            self.set_poits(im_index)
            self.PIE.Cset, self.PIE.Rset, self.PIE.f_3 = BundleAdjustment(self.PIE.Cset, self.PIE.Rset,
                                                                          self.PIE.f_3, self.K, self.PIE, V)
        self.update_global_params()

    def outlier_rejection(self):
        print("Outlier Rejection Process ... ")

        self.outliers = np.zeros(self.B.shape)
        for first_index in range(1, self.num_imgs):
            for second_index in range(first_index + 1, self.num_imgs + 1):
                ind_vec = self.get_indices(first_index, second_index)

                if len(ind_vec) < self.corres_threshold:
                    continue
                features_1 = self.get_features(ind_vec, first_index)
                features_2 = self.get_features(ind_vec, second_index)

                inlier_indices, F, inliers_1, inliers_2 = GetInliersRANSAC(features_1, features_2, ind_vec)

                for i, index in enumerate(ind_vec):
                    if index not in inlier_indices:
                        self.update_B_mat(index, first_index)
                        self.update_outlier_mat(index, first_index, second_index)
        # self.save_class()
        print("Outlier Rejection Done Successfully!")

    def linear_trangulation(self):

        for i in range(4):
            self.PIE.X_set.append(LinearTriangulation(self.K, np.zeros((3, 1)), np.identity(3),
                                                      self.PIE.C_set[i].T, self.PIE.R_set[i],
                                                      self.PIE.featues_points1, self.PIE.featues_points2))


