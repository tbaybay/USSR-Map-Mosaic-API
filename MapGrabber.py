import cv2
import numpy as np
from numpy.linalg import norm
from skimage.morphology import binary_dilation

BORDER_H = 66
BORDER_W = 56

# Idea: > Correct skew then crop.
#       > Locate outer border by finding edge points nearest to image corners.
#       > Locate inner border using a manually tuned line filter.
# Assumptions (from crudest to reasonable):
#       - Fixed map border widths (i.e. images scanned using same scanner)
#       - Map outer and inner borders are parallel

def crop_border(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    smoothed = cv2.bilateralFilter(gray.copy(), 10, 30, 12)
    edges = cv2.Canny(smoothed.copy(), 110, 230)
    dilated = binary_dilation(edges).astype('uint8')
    _, cnts, hier = cv2.findContours(dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    border = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    perimeter = cv2.arcLength(border, True)
    border_contour = [cv2.approxPolyDP(border, 0.1*perimeter, True)]
    box_pts = np.reshape(border_contour, [4, 2])
    warped_img = correct_perspective_shift(img.copy(), box_pts)
    cropped_img = warped_img[BORDER_H:-BORDER_H, BORDER_W:-BORDER_W]
    cropped_img = cv2.bilateralFilter(cropped_img, 10, 30, 12)
    return histogram_equalization(cropped_img)

def correct_perspective_shift(img, box_pts):
    # Arrange detected box points in clockwise order from top-left
    cbox_pts = box_pts - np.mean(box_pts, axis=0)
    tl = box_pts[np.argmax(-cbox_pts[:, 0] - cbox_pts[:, 1])] # --
    tr = box_pts[np.argmax(-cbox_pts[:, 0] + cbox_pts[:, 1])]# -+
    br = box_pts[np.argmax(cbox_pts[:, 0] + cbox_pts[:, 1])]# ++
    bl = box_pts[np.argmax(cbox_pts[:, 0] - cbox_pts[:, 1])]# +-
    src = np.array([tl, tr, br, bl]).astype('float32')
    maxWidth = int( max(norm(tr - tl), norm(br - bl)) )
    maxHeight = int( max(norm(tl - bl), norm(tr - br)) )
    # Describe destination box shape
    dst = np.array([
            [0, 0], # tl
            [0, maxHeight-1], # tr
            [maxWidth-1, maxHeight-1],
            [maxWidth-1, 0]
        ], dtype='float32')
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp

def histogram_equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
