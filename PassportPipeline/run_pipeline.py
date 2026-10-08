# Orchestrator: runs all 4 pipeline stages on one image and assembles everything —
# the combined verdict, every stage's raw output, heatmaps, and a plain-language
# justification — into one results/<image-basename>/ folder.
#
#   Stage 1 (crop.py)       - OpenCV document extraction (in-process)
#   Stage 2 (heuristics.py) - lenient classical-CV tamper heuristics (in-process,
#                             runs on the ORIGINAL image, not the Stage-1 crop)
#   Stage 3 (passport_bfree.py)  - B-Free synthetic-image detector (subprocess,
#                                   'bfree' Conda env)
#   Stage 4 (passport_trufor.py) - TruFor general forensics (subprocess,
#                                   'trufor' Conda env)
#
# Stages 3 and 4 run as subprocesses in their own Conda envs because their pinned
# dependency versions are mutually incompatible (torch 2.0.1 vs 1.11.0) — see
# BFree/passport_bfree.py and TruFor/TruFor_train_test/passport_trufor.py headers.
#
# All processing is local. No network calls are made from this script.

import os
import sys
import json
import shutil
import argparse
import tempfile
import subprocess

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from crop import extract_document
from heuristics import run_heuristics, visualize_flags
from combine_scores import build_combined_result
from justification import build_justification

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CONDA_EXE = os.path.join(os.path.expanduser('~'), 'miniconda3', 'Scripts', 'conda.exe')
BFREE_SCRIPT = os.path.join(REPO_ROOT, 'BFree', 'passport_bfree.py')
TRUFOR_SCRIPT = os.path.join(REPO_ROOT, 'TruFor', 'TruFor_train_test', 'passport_trufor.py')
TRUFOR_CWD = os.path.join(REPO_ROOT, 'TruFor', 'TruFor_train_test')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the full 4-stage passport verification pipeline on one '
                     'image and produce a single results/<basename>/ folder with '
                     'the combined verdict, every stage\'s output, and a '
                     'plain-language justification.')
    parser.add_argument('--image', required=True, help='path to input image')
    parser.add_argument('--output', required=True,
                         help='parent output directory (a subfolder named after the '
                              'image is created inside it)')
    parser.add_argument('-g', '--gpu', type=int, default=-1,
                         help='GPU index for the subprocess stages, -1 for CPU (default: CPU, '
                              'since the 4GB test GPU has OOM\'d on several images this session)')
    return parser.parse_args()


def run_subprocess_stage(conda_env, script, cwd, image_path, out_dir, gpu):
    image_path = os.path.abspath(image_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [CONDA_EXE, 'run', '-n', conda_env, 'python', script,
           '--image', image_path, '--output', out_dir, '-g', str(gpu)]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f'Warning: {conda_env} stage failed (exit {result.returncode}):')
        print(f'  stdout: {result.stdout[-2000:]}')
        print(f'  stderr: {result.stderr[-2000:]}')
        return None

    basename = os.path.splitext(os.path.basename(image_path))[0]
    result_json_candidates = [
        os.path.join(out_dir, f'{basename}_result.json'),
        os.path.join(out_dir, f'{basename}_bfree_result.json'),
    ]
    for path in result_json_candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)

    print(f'Warning: {conda_env} stage produced no result JSON', file=sys.stderr)
    return None


def _copy_if_exists(src, dst):
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        return True
    return False


def main():
    args = parse_args()
    basename = os.path.splitext(os.path.basename(args.image))[0]
    image_dir = os.path.join(os.path.abspath(args.output), basename)
    os.makedirs(image_dir, exist_ok=True)

    # Stage 1: crop (in-process)
    img_rgb = np.array(Image.open(args.image).convert('RGB'))
    Image.fromarray(img_rgb).save(os.path.join(image_dir, 'original.png'))
    cropped, crop_info = extract_document(img_rgb)
    if crop_info['cropped']:
        Image.fromarray(cropped).save(os.path.join(image_dir, 'cropped.png'))
    print(f'Stage 1 (crop): {crop_info}')

    # Stage 2: heuristics (in-process, on the ORIGINAL image — see module docstring)
    heuristics_result = run_heuristics(args.image, img_rgb)
    flags_img = visualize_flags(img_rgb, heuristics_result)
    flags_img.save(os.path.join(image_dir, 'heuristics_flags.png'))
    print(f"Stage 2 (heuristics): flagged={heuristics_result['flagged']}, "
          f"num_flags={len(heuristics_result['confirmed_flags'])}")

    # Stage 3: B-Free (subprocess, 'bfree' env) — writes into a temp dir, we keep
    # only the result JSON since B-Free produces no spatial/heatmap output.
    with tempfile.TemporaryDirectory() as bfree_tmp:
        bfree_result = run_subprocess_stage('bfree', BFREE_SCRIPT, REPO_ROOT,
                                             args.image, bfree_tmp, args.gpu)
    print(f"Stage 3 (B-Free): {'ok' if bfree_result else 'FAILED'}")

    # Stage 4: TruFor (subprocess, 'trufor' env) — copy its heatmap/overlay/
    # confidence-map outputs into the unified per-image folder.
    with tempfile.TemporaryDirectory() as trufor_tmp:
        trufor_result = run_subprocess_stage('trufor', TRUFOR_SCRIPT, TRUFOR_CWD,
                                              args.image, trufor_tmp, args.gpu)
        if trufor_result:
            _copy_if_exists(os.path.join(trufor_tmp, f'{basename}_heatmap.png'),
                             os.path.join(image_dir, 'trufor_heatmap.png'))
            _copy_if_exists(os.path.join(trufor_tmp, f'{basename}_overlay.png'),
                             os.path.join(image_dir, 'trufor_overlay.png'))
            _copy_if_exists(os.path.join(trufor_tmp, f'{basename}_confidence_map.png'),
                             os.path.join(image_dir, 'trufor_confidence_map.png'))
    print(f"Stage 4 (TruFor): {'ok' if trufor_result else 'FAILED'}")

    # Combine
    combined = build_combined_result(
        args.image, crop_info, trufor_result, bfree_result, heuristics_result)
    combined['heuristics_detail'] = {
        'sharpness_check': heuristics_result.get('sharpness_check'),
        'ela_check': heuristics_result.get('ela_check'),
        'noise_check': heuristics_result.get('noise_check'),
        'confirmed_flags': heuristics_result.get('confirmed_flags'),
    }

    with open(os.path.join(image_dir, 'combined_result.json'), 'w') as f:
        json.dump(combined, f, indent=2)

    justification_text = build_justification(combined, heuristics_result)
    with open(os.path.join(image_dir, 'justification.md'), 'w', encoding='utf-8') as f:
        f.write(justification_text)

    print(json.dumps(combined, indent=2))
    print(f'=> results saved to {image_dir}')


if __name__ == '__main__':
    main()
