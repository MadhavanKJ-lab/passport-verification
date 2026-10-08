"""PaddleOCR service exposing an Azure Document Intelligence-compatible async API.

The n8n Passport_Verify flow was built against DI's prebuilt-layout contract
(POST analyze -> Operation-Location -> poll GET -> analyzeResult.pages[].words/lines
with polygons, confidences and spans). This service keeps that contract so the
downstream MRZ/validation logic needs no changes; only the OCR URL differs.

Nothing is persisted: uploaded documents and results live in memory only and are
dropped after RESULT_TTL seconds.
"""
import io
import os
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from paddleocr import PaddleOCR
from PIL import Image, ImageOps

API_KEY = os.environ.get('API_KEY')
RESULT_TTL = int(os.environ.get('RESULT_TTL', '600'))
MAX_PAGES = int(os.environ.get('MAX_PAGES', '5'))

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, det_limit_side_len=1600)
ocr_lock = threading.Lock()  # PaddleOCR predictors are not thread-safe
results = {}  # operation id -> (expires_at, payload)

app = FastAPI(title='PaddleOCR (Document Intelligence compatible)')


def check_auth(subscription_key, x_api_key):
    if API_KEY and API_KEY not in (subscription_key, x_api_key):
        raise HTTPException(status_code=401, detail='missing or invalid API key')


def load_pages(data):
    """Return a list of RGB PIL images from an image or PDF upload (sniffed, not trusted)."""
    if data[:5] == b'%PDF-':
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(data)
        return [pdf[i].render(scale=2.5).to_pil().convert('RGB') for i in range(min(len(pdf), MAX_PAGES))]
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    return [img.convert('RGB')]


def lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def order_box(box):
    """Order 4 points as TL, TR, BR, BL like Document Intelligence polygons."""
    pts = sorted(box, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[1], bottom[0]]


def analyze_page(img, page_number, offset):
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR for paddle
    with ocr_lock:
        raw = ocr.ocr(arr, cls=True)
    detections = raw[0] if raw and raw[0] else []

    items = []
    for box, (text, conf) in detections:
        if not text.strip():
            continue
        tl, tr, br, bl = order_box(box)
        items.append({'tl': tl, 'tr': tr, 'br': br, 'bl': bl, 'text': text.strip(), 'conf': float(conf)})

    # reading order: rows by vertical centre, then left-to-right
    heights = sorted(abs(i['bl'][1] - i['tl'][1]) for i in items) or [10]
    tol = max(heights[len(heights) // 2] * 0.5, 5)
    items.sort(key=lambda i: (i['tl'][1] + i['bl'][1]) / 2)
    row, centre = -1, -1e9
    for i in items:
        c = (i['tl'][1] + i['bl'][1]) / 2
        if abs(c - centre) > tol:
            row, centre = row + 1, c
        i['row'] = row
    items.sort(key=lambda i: (i['row'], i['tl'][0]))

    words, lines, content = [], [], ''
    for item in items:
        if content:
            content += '\n'
            offset_in_content = len(content)
        else:
            offset_in_content = 0
        line_start = offset + offset_in_content
        content += item['text']
        lines.append({
            'content': item['text'],
            'polygon': [c for p in (item['tl'], item['tr'], item['br'], item['bl']) for c in p],
            'spans': [{'offset': line_start, 'length': len(item['text'])}],
        })
        # split the line box proportionally by character position to approximate word boxes
        n, pos = max(len(item['text']), 1), 0
        for token in item['text'].split(' '):
            if token:
                a, b = pos / n, (pos + len(token)) / n
                wt, wr = lerp(item['tl'], item['tr'], a), lerp(item['tl'], item['tr'], b)
                wb, wl = lerp(item['bl'], item['br'], b), lerp(item['bl'], item['br'], a)
                words.append({
                    'content': token,
                    'polygon': [wt[0], wt[1], wr[0], wr[1], wb[0], wb[1], wl[0], wl[1]],
                    'confidence': item['conf'],
                    'span': {'offset': line_start + pos, 'length': len(token)},
                })
            pos += len(token) + 1

    page = {
        'pageNumber': page_number, 'angle': 0, 'width': img.width, 'height': img.height,
        'unit': 'pixel', 'words': words, 'lines': lines, 'spans': [{'offset': offset, 'length': len(content)}],
    }
    return page, content


def run_ocr(data):
    pages, texts, offset = [], [], 0
    for number, img in enumerate(load_pages(data), start=1):
        page, content = analyze_page(img, number, offset)
        pages.append(page)
        texts.append(content)
        offset += len(content) + 1
    return {
        'status': 'succeeded',
        'analyzeResult': {
            'apiVersion': '2024-11-30', 'modelId': 'paddleocr-ppocrv4-en-compat',
            'stringIndexType': 'textElements', 'content': '\n'.join(texts), 'pages': pages, 'styles': [],
        },
    }


def purge():
    now = time.time()
    for key in [k for k, (exp, _) in results.items() if exp < now]:
        results.pop(key, None)


@app.get('/health')
def health():
    return {'healthy': True, 'engine': 'paddleocr', 'lang': 'en'}


@app.post('/documentintelligence/documentModels/{model_id}:analyze')
async def analyze(model_id: str, request: Request,
                  ocp_apim_subscription_key: str = Header(default=None),
                  x_api_key: str = Header(default=None)):
    check_auth(ocp_apim_subscription_key, x_api_key)
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail='empty body')
    purge()
    try:
        payload = run_ocr(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'could not process document: {e}')
    op_id = uuid.uuid4().hex
    results[op_id] = (time.time() + RESULT_TTL, payload)
    proto = request.headers.get('x-forwarded-proto', 'https')
    host = request.headers.get('x-forwarded-host') or request.headers.get('host')
    location = f'{proto}://{host}/documentintelligence/documentModels/{model_id}/analyzeResults/{op_id}'
    return Response(status_code=202, headers={'Operation-Location': location})


@app.get('/documentintelligence/documentModels/{model_id}/analyzeResults/{op_id}')
def get_result(model_id: str, op_id: str,
               ocp_apim_subscription_key: str = Header(default=None),
               x_api_key: str = Header(default=None)):
    check_auth(ocp_apim_subscription_key, x_api_key)
    entry = results.get(op_id)
    if not entry:
        raise HTTPException(status_code=404, detail='unknown or expired operation')
    return JSONResponse(entry[1])
