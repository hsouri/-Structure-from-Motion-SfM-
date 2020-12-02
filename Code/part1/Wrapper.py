import cv2
import numpy as np
from Utility import utility









def main():



    utils = utility()
    utils.set_environmnet(image_index1=1, image_index2=2)
    # utils.initialize()
    # utils.outlier_rejection()
    # utils.plot_matchings(inliers=True)
    # utils.plot_matchings(inliers=True, outliers=True)
    # utils.plot_matchings(inliers=False, outliers=True)
    # utils.load_class()
    utils.estimate_E()
    utils.extract_pose()
    utils.linear_trangulation()
    utils.check_cheirality()
    utils.non_linear_trangulation()
    # utils.plot_tri()
    utils.register_and_add_3d_points()
    utils.plot_points()

if __name__ == "__main__":
    main()