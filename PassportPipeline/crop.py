# Stage 1: OpenCV document extraction — find the passport's rectangular boundary in
# a photo and perspective-warp/crop it to just the document.
#
# Standard "document scanner" CV pipeline (downscale -> edge detect -> contour ->
# 4-point perspective transform), with a fallback chain for low-contrast/difficult
# inputs, and validation of candidate quadrilaterals against the known aspect ratio
# of a passport data page (ICAO 9303: 125mm x 88mm =~ 1.42).
#
# If no confident quadrilateral is found, the ORIGINAL image is returned unchanged
# (never a forced/bad warp) and the failure is reported explicitly, not swallowed.

import cv2
import numpy as np

PASSPORT_ASPECT_RATIO = 125.0 / 88.0  # ICAO 9303 data page, long/short edge
ASPECT_RATIO_TOLERANCE = 0.35          # generous: photos are rarely perfectly fronto-parallel
MIN_AREA_FRACTION = 0.20               # candidate must cover at least 20% of the frame
DOWNSCALE_WIDTH = 1000                 # work at this width for contour detection, scale back up


def _order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _validate_quad(approx, frame_area):
    if len(approx) != 4:
        return False
    area = cv2.contourArea(approx)
    if area < MIN_AREA_FRACTION * frame_area:
        return False
    if not cv2.isContourConvex(approx):
        return False

    corners = _order_corners(approx)
    top_w = np.linalg.norm(corners[1] - corners[0])
    bottom_w = np.linalg.norm(corners[2] - corners[3])
    left_h = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])
    width = max(top_w, bottom_w)
    height = max(left_h, right_h)
    if height == 0:
        return False
    aspect = max(width, height) / min(width, height)
    if abs(aspect - PASSPORT_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
        return False
    return True


def _find_quad_candidates(gray):
    """Try several edge-detection strategies, return the first validated quadrilateral."""
    frame_area = gray.shape[0] * gray.shape[1]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Strategy 1: auto-Canny from the image median
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edge_sets = [cv2.Canny(blurred, lower, upper)]

    # Strategy 2: a couple of fixed alternative Canny parameter sets
    edge_sets.append(cv2.Canny(blurred, 50, 150))
    edge_sets.append(cv2.Canny(blurred, 30, 100))

    for edges in edge_sets:
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if _validate_quad(approx, frame_area):
                return _order_corners(approx)

    # Strategy 3: adaptive threshold instead of Canny (helps low-contrast cases)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if _validate_quad(approx, frame_area):
            return _order_corners(approx)

    # Strategy 4: minAreaRect over the single largest contour from Canny, as a last resort
    if edge_sets:
        contours, _ = cv2.findContours(edge_sets[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= MIN_AREA_FRACTION * frame_area:
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                approx = box.reshape(-1, 1, 2).astype(np.int32)
                if _validate_quad(approx, frame_area):
                    return _order_corners(approx)

    return None


def extract_document(img_rgb):
    """
    Attempt to find and perspective-crop the document in img_rgb (HxWx3 uint8, RGB).
    Returns (cropped_or_original_rgb, info_dict).
    info_dict['cropped'] is True/False; when False the original image is returned
    unchanged and info_dict['reason'] explains why.
    """
    h, w = img_rgb.shape[:2]
    scale = DOWNSCALE_WIDTH / w if w > DOWNSCALE_WIDTH else 1.0
    small = cv2.resize(img_rgb, (int(w * scale), int(h * scale))) if scale != 1.0 else img_rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    quad_small = _find_quad_candidates(gray)
    if quad_small is None:
        return img_rgb, {'cropped': False, 'reason': 'no_confident_quadrilateral_found'}

    quad = quad_small / scale  # map back to full-resolution coordinates

    (tl, tr, br, bl) = quad
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 10 or max_height < 10:
        return img_rgb, {'cropped': False, 'reason': 'degenerate_quadrilateral'}

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_rgb, matrix, (max_width, max_height))

    return warped, {
        'cropped': True,
        'reason': None,
        'corners': quad.tolist(),
        'output_size': [max_width, max_height],
    }
