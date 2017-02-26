import cv2
import numpy as np
from scipy import ndimage
import glob
from skimage.transform import rotate
from matplotlib import patches
np.random.seed(42)
BORDER_W = 60 # Number of pixels between outer and inner border

# Idea: > Correct skew then crop.
#       > Locate outer border by finding edge points nearest to image corners.
#       > Locate inner border using a manually tuned line filter.
# Assumptions (from crudest to reasonable):
#       - Map inner border is always ~60px in from outer border (i.e. scan resolutions are identical)
#       - No gunk between map corners and image corners
#       - Map borders form rectangles (i.e. no perspective shift / maps are scanned)
#       - Map outer and inner borders are parallel

# To add:
#   > weight horizontal line filter toward r.h.s. to mitigate map sag

def crop_border(img): # Accepts an RGB image
    img_rows, img_cols, _ = img.shape
    ob_vertices = _find_outer_border(img.copy()) # Outer border vertices
    map_center = _find_center(ob_vertices)
    top_border = ob_vertices[1] - ob_vertices[0] # Vector from tl to tr of outer border
    th = np.arctan(top_border[0]/top_border[1]) # Skew of map relative to horizontal
    rot_img = _rotate_and_crop(img, th, map_center)
    [tl, tr, lr, ll] = _find_outer_border(rot_img.copy())
    cropped_img = rot_img[tl[0]:ll[0], tl[1]:tr[1], :]
    hpos = get_line(cropped_img, filter_='vertical') # Col. extent of horizontal border
    vpos = get_line(cropped_img, filter_='horizontal') # Row extent of vertical border
    return cropped_img[vpos[0]:vpos[1], hpos[0]:hpos[1], :]

def _rotate_and_crop(img, th, rot_center):
    delta_x, delta_y = abs(np.ceil(np.array(img.shape[:2])*np.sin(th)).astype(int)) # Background exposed by rotation
    rot_img = rotate(img, th, center=rot_center)
    cropped_img = rot_img[delta_y:min(-delta_y, -1), delta_x:min(-delta_x, -1), :]
    formatted_img = (cropped_img*255).astype('uint8')
    return formatted_img

def _normal_filter(filter_width): # 1D kernel of normal pdf
    N_SAMPLES = 100000
    return np.histogram(np.clip(np.random.randn(N_SAMPLES), -3, 3)+3, bins=np.round(filter_width), normed=True)[0]

def _find_outer_border(map_img): # Returns outer border corners, tl , tr, lr, ll in [row, col] coords
    gray = cv2.cvtColor(map_img.copy(), cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 10, 17, 17)
    thresh = cv2.Canny(gray, 30, 200)
    nz_px_posn = np.array(np.where(thresh != 0))
    img_height, img_width, _ = map_img.shape
    img_corners = np.array([[0, 0, img_height, img_height], [0, img_width, img_width, 0]])
    border = []
    for ix in range(4): # Get the nearest point to each corner
        corner = img_corners[:, ix, None]
        distances = np.linalg.norm(nz_px_posn-corner, axis=0)
        border.append(nz_px_posn[:, np.argmin(distances)])
    border = np.array([pos.flatten() for pos in border])
    return border

def _find_center(quad_vertices): # Returns [row, col] of center. Quad vertices are defined as [tl, tr, lr, ll].
    tle, tri, lri, lle = quad_vertices
    diagonal = tri - lle
    center = np.round(lle + diagonal/2).astype(int)
    return center

def _interior_filter(img_width, filter_width, filter_ix): # Returns a 1D kernel with normal dist at filter_ix
    interior_filter = np.zeros(img_width)
    ixs = range(int(filter_ix-filter_width/2), int(filter_ix+filter_width/2))
    interior_filter[ixs] = _normal_filter(filter_width)
    interior_filter[(np.array(img_width)-ixs-1).astype(int)] = -_normal_filter(filter_width)
    return interior_filter

def get_line(map_img, mode='position', filter_='vertical'):
    img_height, img_width, _ = map_img.shape
    gray = cv2.cvtColor(map_img, cv2.COLOR_RGB2GRAY)
    if filter_=='vertical':
        scaler = _interior_filter(img_width, int(BORDER_W/2), BORDER_W)
        line_filter = np.ones([img_height])
        line_scores = np.matmul(line_filter, gray)
        line_scores = scaler*line_scores
    elif filter_=='horizontal':
        scaler = _interior_filter(img_height, int(BORDER_W/2), BORDER_W)
        line_filter = np.ones([img_width])
        line_scores = np.matmul(gray, line_filter)
        line_scores = scaler*line_scores
    if mode=='scores':
        return line_scores
    elif mode=='position':
        return np.array([np.argmax(line_scores), np.argmin(line_scores)]).flatten()
