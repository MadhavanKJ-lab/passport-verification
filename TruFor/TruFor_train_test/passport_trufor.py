# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Wrapper around the official TruFor inference pipeline for document images.
#
# TruFor is a general-purpose image-forensics model (CVPR 2023). It is NOT a
# government passport-authentication system and has no notion of passport
# security features (MRZ, holograms, microprint, etc). Its output here is
# reported strictly as "image manipulation / forgery risk" produced by the
# official pretrained model - never as "passport authenticity".
#
# All processing is local. No network calls are made from this script.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import sys
import json
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from lib.config import config, update_config
from lib.utils import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run official pretrained TruFor on a single document image '
                     'and report image manipulation / forgery risk (NOT passport authenticity).')
    parser.add_argument('--image', required=True, help='path to input image')
    parser.add_argument('--output', required=True, help='output directory')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU index, -1 for CPU')
    parser.add_argument('-exp', '--experiment', type=str, default='trufor_ph3')
    parser.add_argument('--model_file', type=str,
                         default=os.path.join('pretrained_models', 'trufor.pth.tar'))
    return parser.parse_args()


def load_model(args, device):
    ns = argparse.Namespace(
        experiment=args.experiment,
        opts=['TEST.MODEL_FILE', args.model_file],
    )
    update_config(config, ns)

    if device != 'cpu':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

    model_file = config.TEST.MODEL_FILE
    if not model_file:
        raise ValueError('TEST.MODEL_FILE is not set.')
    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f'Pretrained TruFor weights not found at "{model_file}". '
            f'Download them from https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip '
            f'and unzip into "pretrained_models/".')

    print(f'=> loading model from {model_file}')
    checkpoint = torch.load(model_file, map_location=torch.device(device))
    print(f"Epoch: {checkpoint['epoch']}")

    model = get_model(config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    return model


MAX_DIMENSION = 2000  # longer side, px — guards against OOM on very large scans


def run_inference(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    if max(img.size) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(img.size)
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        print(f'Resizing from {img.size} to {new_size} to avoid out-of-memory.')
        img = img.resize(new_size, Image.LANCZOS)
    img_rgb = np.array(img)
    rgb = torch.tensor(img_rgb.transpose(2, 0, 1), dtype=torch.float) / 256.0
    rgb = rgb.unsqueeze(0).to(device)

    with torch.no_grad():
        pred, conf, det, _ = model(rgb, save_np=False)

        if conf is not None:
            conf = torch.squeeze(conf, 0)
            conf = torch.sigmoid(conf)[0]
            conf = conf.cpu().numpy()

        score = None
        if det is not None:
            score = torch.sigmoid(det).item()

        pred = torch.squeeze(pred, 0)
        pred = F.softmax(pred, dim=0)[1]
        pred = pred.cpu().numpy()

    return img_rgb, pred, conf, score


def risk_label(score):
    if score is None:
        return 'unavailable'
    if score < 0.33:
        return 'low'
    if score < 0.66:
        return 'medium'
    return 'high'


LOCALIZED_THRESHOLD = 0.9    # anomaly-map value above which a pixel counts as "anomalous"
LOCALIZED_MIN_COMPONENT_PX = 20  # ignore components smaller than this (single-pixel noise)


def compute_localized_score(anomaly_map, threshold=LOCALIZED_THRESHOLD,
                             min_component_size=LOCALIZED_MIN_COMPONENT_PX):
    """
    TruFor's pooled trufor_score averages confidence-weighted evidence across the
    WHOLE image, so a small, surgical edit (e.g. two added characters) in an
    otherwise large pristine document barely moves it — confirmed empirically on
    a real edited passport image where trufor_score was 0.078 but the anomaly map
    had a 0.997-confidence region exactly at the edit.

    However, raw peak/percentile values of the anomaly map are NOT usable as a
    standalone signal either — clean images were found to have comparably high
    (even higher) peak pixel values, just scattered across many small regions
    (natural edges: text boundaries, security-pattern lines, document borders)
    rather than one cohesive blob. The discriminating signal is CONCENTRATION,
    not magnitude: a genuine localized edit shows up as one tight connected
    component containing most/all of the high-confidence pixels; natural edge
    noise in a clean image is spread across many small, separate components.

    Returns (score in [0,1], debug_info dict). Validated on 3 real images (1
    confirmed-edited, 2 confirmed-clean) during development — a small sample,
    stated plainly, not claimed as a validated general-purpose detector.
    """
    mask = (anomaly_map > threshold).astype(np.uint8)
    total_fg = int(mask.sum())

    debug_info = {
        'threshold': threshold,
        'total_fg_pixels': total_fg,
        'num_components': 0,
        'largest_component_px': 0,
        'concentration': 0.0,
    }

    if total_fg < min_component_size:
        return 0.0, debug_info

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    if len(areas) == 0:
        return 0.0, debug_info

    largest = int(areas.max())
    if largest < min_component_size:
        return 0.0, debug_info

    num_components = len(areas)
    concentration = largest / total_fg

    # high concentration in few components -> confident localized signal;
    # many scattered components dilute the score even if one happens to be largest
    score = concentration if num_components <= 3 else concentration * (3.0 / num_components)
    score = float(min(1.0, score))

    debug_info.update({
        'num_components': num_components,
        'largest_component_px': largest,
        'concentration': float(concentration),
    })
    return score, debug_info


def save_outputs(out_dir, basename, img_rgb, anomaly_map, conf_map, score):
    os.makedirs(out_dir, exist_ok=True)

    # 1. original image
    Image.fromarray(img_rgb).save(os.path.join(out_dir, f'{basename}_original.png'))

    # 2. raw anomaly/manipulation map (float32, values in [0,1])
    np.save(os.path.join(out_dir, f'{basename}_anomaly_map.npy'), anomaly_map)

    # 3. heatmap visualization of the anomaly map
    heatmap_uint8 = (cm.get_cmap('RdBu_r')(anomaly_map)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(heatmap_uint8).save(os.path.join(out_dir, f'{basename}_heatmap.png'))

    # 4. overlay of heatmap on top of the original image
    h, w = anomaly_map.shape
    base = Image.fromarray(img_rgb).resize((w, h)).convert('RGB')
    heat_img = Image.fromarray(heatmap_uint8).convert('RGB')
    overlay = Image.blend(base, heat_img, alpha=0.45)
    overlay.save(os.path.join(out_dir, f'{basename}_overlay.png'))

    # 5. confidence/reliability map, if provided
    if conf_map is not None:
        conf_uint8 = (np.clip(conf_map, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(conf_uint8, mode='L').save(os.path.join(out_dir, f'{basename}_confidence_map.png'))

    # 6. JSON with the raw TruFor output score (no invented probabilities)
    localized_score, localized_debug = compute_localized_score(anomaly_map)
    result = {
        'model': 'TruFor (official pretrained checkpoint, trufor_ph3)',
        'result_type': 'image manipulation / forgery risk',
        'disclaimer': (
            'TruFor is a general image-forensics model, not a government '
            'passport-authentication system. This score reflects statistical '
            'evidence of local image manipulation, not document legitimacy.'
        ),
        'trufor_score': score,
        'risk_label': risk_label(score),
        'trufor_localized_score': localized_score,
        'trufor_localized_debug': localized_debug,
        'anomaly_map_shape': list(anomaly_map.shape),
        'confidence_map_available': conf_map is not None,
    }
    with open(os.path.join(out_dir, f'{basename}_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    args = parse_args()
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    if args.gpu >= 0 and not torch.cuda.is_available():
        print('Warning: CUDA not available, falling back to CPU.')

    model = load_model(args, device)
    img_rgb, anomaly_map, conf_map, score = run_inference(model, args.image, device)

    basename = os.path.splitext(os.path.basename(args.image))[0]
    result = save_outputs(args.output, basename, img_rgb, anomaly_map, conf_map, score)

    print(json.dumps(result, indent=2))
    print(f'=> outputs saved to {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
