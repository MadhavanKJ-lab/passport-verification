# FastAPI wrapper around the 4-stage passport verification pipeline, for Azure
# Container Apps deployment. Mirrors PassportPipeline/run_pipeline.py's logic
# exactly (same stage functions, same subprocess-per-conda-env pattern) but as an
# HTTP endpoint instead of a CLI script.
#
# PII handling, explicit design decision: nothing from a request is ever written
# to persistent storage. Every uploaded image and every stage's output lives only
# in a per-request tempfile.TemporaryDirectory() that is deleted before the
# response is sent. All visual outputs (original, cropped, heuristics flags,
# TruFor heatmap/overlay/confidence map) are returned as base64 in the JSON
# response body instead of being saved to disk/blob storage.
#
# All processing is local to the container. No network calls are made except the
# ones Azure Container Apps itself handles (ingress).

import os
import io
import sys
import json
import base64
import shutil
import tempfile
import subprocess

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PassportPipeline'))
from crop import extract_document
from heuristics import run_heuristics, visualize_flags
from combine_scores import build_combined_result
from justification import build_justification

APP_ROOT = '/app'
CONDA_EXE = '/opt/conda/bin/conda'
BFREE_SCRIPT = os.path.join(APP_ROOT, 'BFree', 'passport_bfree.py')
TRUFOR_SCRIPT = os.path.join(APP_ROOT, 'TruFor', 'TruFor_train_test', 'passport_trufor.py')
TRUFOR_CWD = os.path.join(APP_ROOT, 'TruFor', 'TruFor_train_test')
BFREE_CWD = APP_ROOT

API_KEY = os.environ.get('API_KEY')

app = FastAPI(title='Passport Verification Pipeline')


def run_subprocess_stage(conda_env, script, cwd, image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [CONDA_EXE, 'run', '-n', conda_env, 'python', script,
           '--image', image_path, '--output', out_dir, '-g', '-1']
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f'{conda_env} stage failed (exit {result.returncode}): {result.stderr[-2000:]}')
        return None

    basename = os.path.splitext(os.path.basename(image_path))[0]
    for suffix in ('_result.json', '_bfree_result.json'):
        candidate = os.path.join(out_dir, f'{basename}{suffix}')
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


def _b64_file(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def _b64_image(img_rgb):
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _check_api_key(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='missing or invalid X-API-Key header')


@app.get('/health')
def health():
    checks = {}
    for env in ('trufor', 'bfree'):
        try:
            r = subprocess.run(
                [CONDA_EXE, 'run', '-n', env, 'python', '-c', 'import torch'],
                capture_output=True, text=True, timeout=30)
            checks[env] = 'ok' if r.returncode == 0 else f'failed: {r.stderr[-300:]}'
        except Exception as e:
            checks[env] = f'error: {e}'
    healthy = all(v == 'ok' for v in checks.values())
    status_code = 200 if healthy else 503
    return JSONResponse(status_code=status_code, content={'healthy': healthy, 'envs': checks})


@app.post('/verify')
async def verify(file: UploadFile = File(...), x_api_key: str = Header(default=None)):
    _check_api_key(x_api_key)

    with tempfile.TemporaryDirectory() as request_dir:
        image_path = os.path.join(request_dir, file.filename or 'upload.jpg')
        with open(image_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        try:
            img_rgb = np.array(Image.open(image_path).convert('RGB'))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'could not read image: {e}')

        # Stage 1: crop
        cropped, crop_info = extract_document(img_rgb)

        # Stage 2: heuristics (on the ORIGINAL image, not the crop)
        heuristics_result = run_heuristics(image_path, img_rgb)
        flags_img = visualize_flags(img_rgb, heuristics_result)

        # Stage 3: B-Free
        bfree_out = os.path.join(request_dir, 'bfree')
        bfree_result = run_subprocess_stage('bfree', BFREE_SCRIPT, BFREE_CWD, image_path, bfree_out)

        # Stage 4: TruFor
        trufor_out = os.path.join(request_dir, 'trufor')
        trufor_result = run_subprocess_stage('trufor', TRUFOR_SCRIPT, TRUFOR_CWD, image_path, trufor_out)

        basename = os.path.splitext(os.path.basename(image_path))[0]

        # Combine
        combined = build_combined_result(
            file.filename, crop_info, trufor_result, bfree_result, heuristics_result)
        combined['heuristics_detail'] = {
            'sharpness_check': heuristics_result.get('sharpness_check'),
            'ela_check': heuristics_result.get('ela_check'),
            'noise_check': heuristics_result.get('noise_check'),
            'confirmed_flags': heuristics_result.get('confirmed_flags'),
        }
        justification_text = build_justification(combined, heuristics_result)

        images_b64 = {
            'original': _b64_image(img_rgb),
            'cropped': _b64_image(cropped) if crop_info.get('cropped') else None,
            'heuristics_flags': _b64_image(np.array(flags_img)),
            'trufor_heatmap': _b64_file(os.path.join(trufor_out, f'{basename}_heatmap.png')),
            'trufor_overlay': _b64_file(os.path.join(trufor_out, f'{basename}_overlay.png')),
            'trufor_confidence_map': _b64_file(os.path.join(trufor_out, f'{basename}_confidence_map.png')),
        }

        # request_dir and everything in it (uploaded image, subprocess outputs)
        # is deleted automatically on exiting this `with` block — nothing
        # persists beyond this request.
        return {
            'combined_result': combined,
            'justification': justification_text,
            'images_base64_png': images_b64,
        }
