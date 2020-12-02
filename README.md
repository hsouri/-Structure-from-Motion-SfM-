# SfM-Structure-from-Motion
Python implementation of classical and unsupervised Structure from Motion(SfM)


In this project the goal is to reconstruct a whole 3D scene
from a set of images taken by a camera at different locations
and poses. The problem here is often referred as Structure from
Motion (SfM). To tackle this problm, we are using two well
know approaches; Classical and deep learning base techniques.
In classical approach We are using a dataset consisted of
6 images of a street scene with a building in it, a text
file describing the 2D image point correspondences between
all possible image pairs and the calibration matrix of the
camera used for capturing the images. In the deep learning
approach we explored an unsupervised deep learning approach
to retrieve depth and Ego-Motion from motion. 
