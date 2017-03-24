import cv2
import numpy as np
from numpy.linalg import norm
from mpl_toolkits.basemap import Basemap
from skimage.morphology import binary_dilation
from scipy.misc import imresize
from scipy.ndimage import imread
from matplotlib import pyplot as plt
import requests
import time

BORDER_H = 66
BORDER_W = 56

# Idea: > Download maps
#       > Correct skew then crop.
#       > Locate inner border using manually tuned parameters.
#       > Stitch together results
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

def download_maps():
    # Need to make folder k38_lom_imgs beforehand
    pfx = 'http://nav.lom.name/maps_scan/K38/100k/100k--k38-';
    img_urls = [pfx+str(0)*(3-len(str(i)))+str(i)+'.gif' for i in range(1, 145)];
    for url in img_urls:
        map_img = requests.get(url).content
        filename = url.split('/')[-1]
        with open('./k38_lom_imgs/'+filename, 'wb') as file:
            file.write(map_img)
    time.sleep(5) # Be polite
    
def stitch(img1, img2, dim=1):
    max_dim = np.max([img1.shape[dim], img2.shape[dim]]);
    if dim == 1:
        img1 = imresize(img1, (img1.shape[0], max_dim));
        img2 = imresize(img2, (img2.shape[0], max_dim));
    if dim == 0:
        img1 = imresize(img1, (max_dim, img1.shape[1]));
        img2 = imresize(img2, (max_dim, img1.shape[1]));
    return np.concatenate([img1, img2], axis=abs(dim-1))

def add_grid_references(map_img, llcrnr, urcrnr, c='#ff0080'):
    m = Basemap(llcrnrlon=llcrnr[1], llcrnrlat=llcrnr[0],
                urcrnrlon=urcrnr[1], urcrnrlat=urcrnr[0],
                lon_0=np.mean([llcrnr[1], urcrnr[1]]), lat_0=np.mean([urcrnr[0], llcrnr[0]]),
                projection = 'tmerc',
                ellps = 'WGS84')
    parallels = np.arange(llcrnr[0], urcrnr[0], 0.025)
    meridians = np.arange(llcrnr[1], urcrnr[1], 0.025)
    m.drawparallels(parallels, color=c, linewidth=1)
    m.drawmeridians(meridians, color=c, linewidth=1)
    delta_x = (m(meridians[1], parallels[0])[0] - m(meridians[0], parallels[0])[0])/2
    delta_y = (m(meridians[0], parallels[1])[1] - m(meridians[0], parallels[0])[1])/2
    m.imshow(map_img);
    for i, lat in enumerate(parallels[1:]):
        for j, lon in enumerate(meridians[:-1]):
            if (i % 2 == 0 and j % 2 == 0):
                x, y = m(lon, lat);
                plt.text(x, y,
                         str(round(lat, 2)),
                         color=c, alpha=1,
                         size=12);
                plt.text(x - 0.2*delta_x, y - delta_y,
                         str(round(lon, 2)),
                         rotation=-90,
                         color=c, alpha=1,
                         size=12)