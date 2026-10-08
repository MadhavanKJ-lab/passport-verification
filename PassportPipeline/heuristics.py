# Stage 2: OpenCV lenient tamper heuristics — classical CV, no ML model.
#
# Narrow, explainable checks for *obvious* localized tampering (e.g. a pasted/
# blurred date field). Explicitly tuned to be LENIENT: must not confuse a
# low-quality photo, poor lighting, or JPEG compression with evidence of tampering.
#
# Design rules (from research, applied throughout):
# - Judge every region against the image's OWN statistics (local baseline), never a
#   fixed global threshold.
# - Require at least two independent cues to agree on the same region before
#   flagging anything.
# - Always allow an explicit "cannot assess" result rather than forcing a verdict on
#   poor-quality input.
# - These checks run on the ORIGINAL, unwarped image — perspective-warp
#   interpolation destroys the noise/compression signals they depend on. Call this
#   on the same image passed into crop.extract_document, not its output.
#
# Known hard limitation, stated explicitly: a photo of a physically forged passport
# (printed fake, photographed normally) leaves no digital manipulation traces at
# all. These checks cannot catch that class of forgery — MRZ checksum/cross-field
# validation (handled separately in this project) is the relevant defense there.

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw

TILE_SIZE = 48
MIN_OVERALL_SHARPNESS = 15.0      # below this, the whole image is too blurry to assess
SHARPNESS_ZSCORE_THRESHOLD = 2.5  # robust z-score (MAD-based) to flag a tile
ELA_QUALITY = 90
ELA_ZSCORE_THRESHOLD = 2.5
NOISE_ZSCORE_THRESHOLD = 2.5
MIN_CUES_TO_FLAG = 2


SPINE_BAND_FRACTION = 0.06  # exclude a band this wide (fraction of image height) around
                             # a detected horizontal book-spine/fold line


def _tile_grid(h, w, tile=TILE_SIZE):
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            yield y, x


SPINE_WIDTH_COVERAGE = 0.75  # the discontinuity must span at least this fraction of
                              # the image width to count as a genuine page fold,
                              # rather than a short localized edit boundary


def _detect_spine_band(img_gray):
    """
    Open-booklet photos (two facing pages) have a strong horizontal discontinuity
    at the spine/fold that spans nearly the full width of the image — a legitimate
    structural feature (shadow + page-edge), not tampering evidence. Exclude a band
    around it from the sharpness/ELA/noise checks, so a real passport spread doesn't
    get flagged just for being a photographed booklet.

    Critically, this must NOT trigger on a short, localized edit boundary (e.g. a
    pasted/blurred field) — that would hide the exact tampering it should help
    detect. The width-coverage requirement below is what tells the two apart: a
    spine crosses the whole page, a tampered field does not.
    """
    h, w = img_gray.shape
    gray_f = img_gray.astype(np.float64)
    row_diff = np.abs(np.diff(gray_f, axis=0))  # per-pixel vertical gradient, HxW
    row_grad_sum = row_diff.sum(axis=1)

    lo, hi = int(h * 0.33), int(h * 0.67)
    if hi <= lo:
        return None

    median_val = np.median(row_grad_sum)
    if median_val < 1e-9:
        return None

    # candidate rows by total gradient energy, strongest first
    candidates = lo + np.argsort(row_grad_sum[lo:hi])[::-1]
    for peak_row in candidates[:10]:
        if row_grad_sum[peak_row] < 4 * median_val:
            break  # remaining candidates are weaker still, no point continuing
        # require the gradient to be spread across most of the row's width, not
        # concentrated in one narrow patch (which would indicate a local edit, not
        # a full-width page fold)
        row_vals = row_diff[peak_row]
        strong_pixel_threshold = np.percentile(row_vals, 90) * 0.3
        coverage = np.mean(row_vals > strong_pixel_threshold)
        if coverage >= SPINE_WIDTH_COVERAGE:
            band = int(h * SPINE_BAND_FRACTION)
            return max(0, peak_row - band), min(h, peak_row + band)

    return None  # no full-width discontinuity found — don't exclude anything


def _in_band(y, tile, band):
    if band is None:
        return False
    lo, hi = band
    return not (y + tile <= lo or y >= hi)


def _robust_zscores(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-9:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def _has_text_content(tile_gray):
    # a tile with near-zero gradient energy is blank background, not a text field
    grad = cv2.Laplacian(tile_gray, cv2.CV_64F)
    return grad.var() > 5.0


BORDER_MARGIN_FRACTION = 0.03  # exclude this fraction of width/height at each edge —
                                # scan/photo frame borders are real edges but not
                                # content fields, and must not be analyzed as such


def _content_tile_coords(img_gray, spine_band=None):
    """
    Single shared pass to find which tiles contain actual document content (text,
    printed patterns) as opposed to blank page margin / scan background / the
    image's own outer frame border. All three checks below call this so they stay
    consistent instead of each re-deriving their own notion of "relevant region" —
    a lone outlier check (e.g. ELA having no content filter) was exactly what let a
    plain background or document-frame-edge tile get flagged.
    """
    h, w = img_gray.shape
    margin_y = int(h * BORDER_MARGIN_FRACTION)
    margin_x = int(w * BORDER_MARGIN_FRACTION)
    coords = []
    for y, x in _tile_grid(h, w):
        if y < margin_y or y + TILE_SIZE > h - margin_y:
            continue
        if x < margin_x or x + TILE_SIZE > w - margin_x:
            continue
        if _in_band(y, TILE_SIZE, spine_band):
            continue
        tile = img_gray[y:y + TILE_SIZE, x:x + TILE_SIZE]
        if _has_text_content(tile):
            coords.append((y, x))
    return coords


def check_sharpness_consistency(img_gray):
    h, w = img_gray.shape
    overall_sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if overall_sharpness < MIN_OVERALL_SHARPNESS:
        return {'status': 'cannot_assess', 'reason': 'overall image too blurry to evaluate', 'flags': []}

    spine_band = _detect_spine_band(img_gray)
    content_coords = _content_tile_coords(img_gray, spine_band)
    tiles = []
    for y, x in content_coords:
        tile = img_gray[y:y + TILE_SIZE, x:x + TILE_SIZE]
        var = cv2.Laplacian(tile, cv2.CV_64F).var()
        tiles.append({'y': y, 'x': x, 'sharpness': var})

    if len(tiles) < 8:
        return {'status': 'cannot_assess', 'reason': 'not enough text-bearing regions found', 'flags': []}

    flags = []
    for t in tiles:
        # compare each tile only against its local neighborhood ring (same approach
        # used for noise/brightness banding below) — a field is judged against
        # nearby text, never against far-apart regions of different density/font
        neighbors = [
            n for n in tiles
            if n is not t and abs(n['y'] - t['y']) <= 3 * TILE_SIZE and abs(n['x'] - t['x']) <= 3 * TILE_SIZE
        ]
        if len(neighbors) < 5:
            continue
        local_vals = np.log(np.array([n['sharpness'] for n in neighbors]) + 1e-6)
        this_val = np.log(t['sharpness'] + 1e-6)
        local_z = _robust_zscores(np.append(local_vals, this_val))[-1]
        if local_z < -SHARPNESS_ZSCORE_THRESHOLD:  # much blurrier than its own neighborhood
            flags.append({'y': t['y'], 'x': t['x'], 'size': TILE_SIZE, 'zscore': float(local_z)})

    return {'status': 'ok', 'overall_sharpness': float(overall_sharpness), 'flags': flags}


def _is_jpeg(image_path):
    try:
        with Image.open(image_path) as im:
            return im.format == 'JPEG'
    except Exception:
        return False


def check_error_level_analysis(image_path):
    if not _is_jpeg(image_path):
        return {'status': 'not_applicable', 'reason': 'input is not a JPEG (no recompression history to analyze)', 'flags': []}

    original = Image.open(image_path).convert('RGB')
    buf = io.BytesIO()
    original.save(buf, 'JPEG', quality=ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert('RGB')

    orig_arr = np.asarray(original, dtype=np.int16)
    recompressed_arr = np.asarray(recompressed, dtype=np.int16)
    diff = np.abs(orig_arr - recompressed_arr).sum(axis=2).astype(np.float64)

    gray = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2GRAY)
    edge_strength = cv2.Laplacian(gray, cv2.CV_64F)
    edge_strength = np.abs(edge_strength)
    # normalize ELA response by local edge strength to suppress the "text/edges always
    # light up" false-positive mode
    normalized = diff / (edge_strength + 10.0)

    spine_band = _detect_spine_band(gray)
    content_coords = _content_tile_coords(gray, spine_band)
    tiles = []
    for y, x in content_coords:
        tile = normalized[y:y + TILE_SIZE, x:x + TILE_SIZE]
        tiles.append({'y': y, 'x': x, 'score': float(tile.mean())})

    if len(tiles) < 8:
        return {'status': 'cannot_assess', 'reason': 'image too small to tile', 'flags': []}

    flags = []
    for t in tiles:
        neighbors = [
            n for n in tiles
            if n is not t and abs(n['y'] - t['y']) <= 3 * TILE_SIZE and abs(n['x'] - t['x']) <= 3 * TILE_SIZE
        ]
        if len(neighbors) < 5:
            continue
        local_vals = np.array([n['score'] for n in neighbors] + [t['score']])
        local_z = _robust_zscores(local_vals)[-1]
        if local_z > ELA_ZSCORE_THRESHOLD:
            flags.append({'y': t['y'], 'x': t['x'], 'size': TILE_SIZE, 'zscore': float(local_z)})

    return {'status': 'ok', 'flags': flags}


def check_noise_consistency(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    median_blur = cv2.medianBlur(img_rgb, 5)
    median_blur_gray = cv2.cvtColor(median_blur, cv2.COLOR_RGB2GRAY).astype(np.float64)
    residual = np.abs(gray - median_blur_gray)

    h, w = gray.shape
    spine_band = _detect_spine_band(gray.astype(np.uint8))
    tiles = []
    for y, x in _tile_grid(h, w):
        if _in_band(y, TILE_SIZE, spine_band):
            continue
        brightness = gray[y:y + TILE_SIZE, x:x + TILE_SIZE].mean()
        if brightness > 245:  # blown-out glare — no noise present by nature, not evidence
            continue
        noise_level = residual[y:y + TILE_SIZE, x:x + TILE_SIZE].std()
        tiles.append({'y': y, 'x': x, 'brightness': brightness, 'noise': noise_level})

    if len(tiles) < 8:
        return {'status': 'cannot_assess', 'reason': 'not enough comparable regions found', 'flags': []}

    # group into brightness bands so noise is only compared within similar-brightness tiles
    brightness_vals = np.array([t['brightness'] for t in tiles])
    bands = np.digitize(brightness_vals, bins=[64, 128, 192])

    flags = []
    for band in np.unique(bands):
        band_tiles = [t for t, b in zip(tiles, bands) if b == band]
        if len(band_tiles) < 6:
            continue
        noise_vals = [t['noise'] for t in band_tiles]
        zscores = _robust_zscores(noise_vals)
        for t, z in zip(band_tiles, zscores):
            if z < -NOISE_ZSCORE_THRESHOLD:  # unnaturally clean relative to same-brightness peers
                flags.append({'y': t['y'], 'x': t['x'], 'size': TILE_SIZE, 'zscore': float(z)})

    return {'status': 'ok', 'flags': flags}


def _tiles_overlap(a, b, tile=TILE_SIZE):
    return abs(a['y'] - b['y']) < tile and abs(a['x'] - b['x']) < tile


def combine_flags(sharpness_result, ela_result, noise_result):
    """Require at least MIN_CUES_TO_FLAG independent checks to agree on the same
    localized region before reporting a tamper flag."""
    cue_sets = []
    for result in (sharpness_result, ela_result, noise_result):
        if result['status'] == 'ok':
            cue_sets.append(result['flags'])

    combined = []
    if len(cue_sets) >= MIN_CUES_TO_FLAG:
        for i, flags_a in enumerate(cue_sets):
            for flag_a in flags_a:
                agreeing = 1
                for j, flags_b in enumerate(cue_sets):
                    if j <= i:
                        continue
                    if any(_tiles_overlap(flag_a, flag_b) for flag_b in flags_b):
                        agreeing += 1
                if agreeing >= MIN_CUES_TO_FLAG:
                    combined.append(flag_a)

    return combined


def run_heuristics(image_path, img_rgb):
    """
    img_rgb: the ORIGINAL (unwarped) image as HxWx3 uint8 RGB array.
    image_path: path to the same image on disk (needed for JPEG-byte-level ELA).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    sharpness_result = check_sharpness_consistency(gray)
    ela_result = check_error_level_analysis(image_path)
    noise_result = check_noise_consistency(img_rgb)

    confirmed_flags = combine_flags(sharpness_result, ela_result, noise_result)

    return {
        'result_type': 'lenient classical-CV tamper heuristics (localized, high-confidence only)',
        'disclaimer': (
            'These checks only flag regions where at least two independent classical '
            'signals (sharpness consistency, error-level analysis, noise consistency) '
            'agree. They are intentionally lenient and will NOT flag whole-image '
            'quality issues (blur, poor lighting, compression). They CANNOT detect a '
            'photo of a physically forged document — only digital post-processing '
            'traces. Absence of a flag is not proof of authenticity.'
        ),
        'sharpness_check': {k: v for k, v in sharpness_result.items() if k != 'flags'},
        'ela_check': {k: v for k, v in ela_result.items() if k != 'flags'},
        'noise_check': {k: v for k, v in noise_result.items() if k != 'flags'},
        'confirmed_flags': confirmed_flags,
        'flagged': len(confirmed_flags) > 0,
    }


def visualize_flags(img_rgb, heuristics_result):
    """
    Draws each confirmed flag as a labeled red box on the original image, for
    inclusion in a per-image results folder. Returns a PIL Image (RGB). If no
    flags were found, returns the original image annotated with a plain "no
    localized tampering flags" caption instead of leaving the caller to guess
    whether an absent file means "clean" or "not generated".
    """
    pil_img = Image.fromarray(img_rgb).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    flags = heuristics_result.get('confirmed_flags', [])

    caption_h = 28
    canvas = Image.new('RGB', (pil_img.width, pil_img.height + caption_h), 'white')
    canvas.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for flag in flags:
        y, x, size = flag['y'], flag['x'], flag['size']
        draw.rectangle([x, y, x + size, y + size], outline='red', width=3)

    caption = (
        f'{len(flags)} confirmed tamper flag(s) - regions where 2+ independent checks agree'
        if flags else
        'No localized tampering flags (does not confirm authenticity - see disclaimer)'
    )
    draw.text((8, pil_img.height + 6), caption, fill='black')

    return canvas
