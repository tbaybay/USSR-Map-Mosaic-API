import cv2
import numpy as np
from scipy import ndimage
import glob
from skimage.feature import canny
from skimage.transform import rotate
from matplotlib import patches
np.random.seed(42)
