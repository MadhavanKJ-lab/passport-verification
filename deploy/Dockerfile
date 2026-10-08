# Single-container deployment of the passport verification pipeline (CPU-only).
# Two Conda environments inside (trufor: py3.7/torch1.11, bfree: py3.10/torch2.0.1)
# because their pinned dependencies are mutually incompatible — mirrors the local
# Windows setup, ported to Linux CPU wheels (no CUDA toolkit needed in-image).
#
# Build context must be the repository root (Passport_verification/), since this
# needs to COPY PassportPipeline/, TruFor/, and BFree/ from alongside deploy/.

# Pinned to an older release (Debian 12 "bookworm" base) rather than :latest —
# :latest now tracks Debian 13 "trixie", whose stricter executable-stack
# enforcement breaks PyTorch 1.11.0's 2022-era compiled libtorch_cpu.so
# ("cannot enable executable stack as shared object requires"). Confirmed via
# direct testing in this container before settling on this fix.
#
# No --platform flag: Azure Container Registry's build service doesn't support
# that FROM syntax (older Dockerfile parser), and its agents are amd64 natively
# anyway. Locally on Docker Desktop, pinning this specific tag (rather than
# :latest) was what actually resolved the earlier architecture mismatch.
FROM continuumio/miniconda3:24.7.1-0

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# System libs cv2 needs at import time (libGL.so.1 etc). Installing these directly
# is more robust than trying to force every package to use opencv-python-headless
# only — some transitive dependency (likely one of the mm* packages) pulls in
# regular opencv-python regardless, so the actual .so needs to exist either way.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Conda's default channels require ToS acceptance non-interactively on newer
# conda releases (same gate hit during local Windows setup) — this base image's
# bundled conda predates the 'tos' subcommand entirely, so these are no-ops here;
# kept with `|| true` so the build stays correct if the base image is ever
# updated to a conda version that does enforce it.
RUN (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true) \
 && (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true)

# ---- trufor env: Python 3.7, torch 1.11.0 CPU, mmcv-full 1.5.3, same pins used
# locally (see pip freeze captured during development), minus the Windows-only
# wincertstore package and with CPU wheel sources instead of +cu113.
RUN conda create -n trufor python=3.7 -y
RUN conda run -n trufor python -m pip install --no-cache-dir \
      torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html
# opencv-python-headless ONLY (not opencv-python) — the non-headless build needs
# libGL.so.1, which this minimal server image doesn't have and doesn't need
# (no GUI/display ever used here). Headless provides the same cv2 API.
RUN conda run -n trufor python -m pip install --no-cache-dir \
      numpy==1.21.5 scipy==1.7.3 opencv-python-headless==4.5.5.64 \
      pillow==9.0.1 matplotlib==3.5.2 pandas==1.3.5 scikit-image==0.19.2 \
      scikit-learn==1.0.2 albumentations==1.2.1 timm==0.5.4 tqdm==4.64.0 \
      yacs==0.1.8 addict==2.4.0 einops==0.6.1 imageio==2.21.1 \
      fastapi uvicorn[standard] python-multipart
RUN conda run -n trufor python -m pip install --no-cache-dir \
      mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html
RUN conda run -n trufor python -m pip install --no-cache-dir \
      mmsegmentation==0.25.0 mmcls==0.23.1

# ---- bfree env: Python 3.10, torch 2.0.1 CPU, DINOv2-based B-Free stack.
RUN conda create -n bfree python=3.10 -y
RUN conda run -n bfree python -m pip install --no-cache-dir \
      torch==2.0.1+cpu torchvision==0.15.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu
RUN conda run -n bfree python -m pip install --no-cache-dir "numpy<2"
RUN conda run -n bfree python -m pip install --no-cache-dir \
      tqdm scikit-learn pyyaml "timm>=1.0.12" "transformers>=4.43.4"

# ---- application code and model weights (weights baked into the image for v1
# simplicity — see plan's "what this does NOT do": no Blob Storage step yet).
COPY PassportPipeline/ /app/PassportPipeline/
COPY TruFor/TruFor_train_test/ /app/TruFor/TruFor_train_test/
COPY BFree/ /app/BFree/
COPY deploy/api.py /app/api.py

EXPOSE 8000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trufor", \
            "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
