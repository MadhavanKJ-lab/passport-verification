# PaddleOCR service (CPU) with a Document Intelligence-compatible API.
# Debian 12 base pinned (see deploy/Dockerfile for why newer bases are avoided
# with older compiled ML wheels).
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir \
      paddlepaddle==2.6.2 paddleocr==2.8.1 "numpy<2" opencv-python-headless==4.6.0.66 \
      pypdfium2 pillow fastapi "uvicorn[standard]" python-multipart

# pyclipper arrived as a corrupt wheel in a first build (zlib error on import);
# reinstall a pinned build and fail fast here if it still can't import.
RUN pip install --no-cache-dir --force-reinstall pyclipper==1.3.0.post5 \
 && python -c "import pyclipper; print('pyclipper ok')"

# Bake the detection/recognition/angle models into the image so containers start
# without downloading anything. Done with curl|tar (not PaddleOCR's own Python
# downloader): once paddle is imported, Python's tarfile/zlib segfaults in
# inflateReset2 (zlib symbol clash with Paddle's native libs), killing the build.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && B=https://paddleocr.bj.bcebos.com && W=/root/.paddleocr/whl \
 && mkdir -p $W/det/en/en_PP-OCRv3_det_infer $W/rec/en/en_PP-OCRv4_rec_infer $W/cls/ch_ppocr_mobile_v2.0_cls_infer \
 && curl -fsSL $B/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar | tar -x --strip-components=1 -C $W/det/en/en_PP-OCRv3_det_infer \
 && curl -fsSL $B/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar | tar -x --strip-components=1 -C $W/rec/en/en_PP-OCRv4_rec_infer \
 && curl -fsSL $B/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar | tar -x --strip-components=1 -C $W/cls/ch_ppocr_mobile_v2.0_cls_infer \
 && ls $W/det/en/en_PP-OCRv3_det_infer $W/rec/en/en_PP-OCRv4_rec_infer $W/cls/ch_ppocr_mobile_v2.0_cls_infer

COPY app.py /app/app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
