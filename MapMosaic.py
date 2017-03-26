# -*- coding: utf-8 -*-

import cv2
import numpy as np
from numpy.linalg import norm
from mpl_toolkits.basemap import Basemap
from skimage.morphology import binary_dilation
from scipy.misc import imresize
from scipy.ndimage import imread
from matplotlib import pyplot as plt
import glob
import requests
import time

BORDER_H = 66
BORDER_W = 56

# Idea: > Download maps
#       > Correct skew and perspective then crop
#       > Locate inner border using manually tuned parameters.
#       > Stitch together results into 3x3 blocks
#       > Add grid references and write to .png

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
    t_dim = abs(dim - 1)
    max_dim = np.max([img1.shape[t_dim], img2.shape[t_dim]]);
    if dim == 1:
        img1 = imresize(img1, (max_dim, img1.shape[1]));
        img2 = imresize(img2, (max_dim, img2.shape[1]));
    if dim == 0:
        img1 = imresize(img1, (img1.shape[0], max_dim));
        img2 = imresize(img2, (img2.shape[0], max_dim));
    return np.concatenate([img1, img2], axis=dim)

def add_grid_references(map_img, llcrnr, urcrnr, c='#000000'):
    f = plt.figure(figsize=(10, 10));
    m = Basemap(llcrnrlon=llcrnr[1], llcrnrlat=llcrnr[0],
                urcrnrlon=urcrnr[1], urcrnrlat=urcrnr[0],
                lon_0=np.mean([llcrnr[1], urcrnr[1]]), lat_0=np.mean([urcrnr[0], llcrnr[0]]),
                projection = 'tmerc',
                ellps = 'WGS84')
    parallels = np.arange(llcrnr[0], urcrnr[0], 0.05)
    meridians = np.arange(llcrnr[1], urcrnr[1], 0.05)
    m.drawparallels(parallels, color=c, linewidth=.1, dashes=[5, 2])
    m.drawmeridians(meridians, color=c, linewidth=.1, dashes=[5, 2])
    delta_x = (m(meridians[1], parallels[0])[0] - m(meridians[0], parallels[0])[0])/2
    delta_y = (m(meridians[0], parallels[1])[1] - m(meridians[0], parallels[0])[1])/2
    m.imshow(map_img, origin='upper');
    for i, lat in enumerate(parallels[1:]):
        for j, lon in enumerate(meridians[:-1]):
            if (i % 2 == 0 and j % 2 == 0):
                x, y = m(lon, lat);
                plt.text(x + delta_x, y,
                         str(round(lat, 2))+'°',
                         color=c, alpha=1, ha='center',
                         va='center', size=2);
                plt.text(x, y - delta_y,
                         str(round(lon, 2))+'°',
                         rotation=-90,
                         color=c, alpha=1, ha='center',
                         va='center', size=2)
    return f

def create_map(r_ix, c_ix):
    UR_LONG = np.arange(42.5, 48.1, .5)
    UR_LAT = np.arange(44, 40.3, -1./3.)
    LL_LONG = np.arange(42, 47.6, .5)
    LL_LAT = np.arange(43.666666666, 39.9, -1./3.)
    img_fps = glob.glob('./k38_lom_cropped/*')
    col_ix = np.array([0, 1, 2]) + c_ix;
    row_ix = np.array([1, 2]) + r_ix;
    for i in col_ix:
        img_col = imread(img_fps[r_ix*12 + i]);
        for j in row_ix:
            img_col = stitch(img_col, imread(img_fps[j*12 + i]), dim=0)
        if i == col_ix[0]:
            stitched_img = img_col
        else:
            stitched_img = stitch(stitched_img, img_col, dim=1)
    llcrnr = [LL_LAT[r_ix+2], LL_LONG[c_ix]]
    urcrnr = [UR_LAT[r_ix], UR_LONG[c_ix+2]]
    f = add_grid_references(stitched_img, llcrnr, urcrnr)
    name = str(r_ix)+'_'+str(c_ix)
    f.savefig(name+'.png', dpi=1500, transparent=True, bbox_inches='tight')
    plt.close('all')
