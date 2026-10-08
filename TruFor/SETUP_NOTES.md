# TruFor Local Setup — Notes & FAQ

Internal reference for this install. Covers how the model works, repo layout,
Docker vs. Conda, what broke and how it was fixed, and answers to questions
raised during setup.

---

## 1. How TruFor works

TruFor (CVPR 2023, GRIP-UNINA) is a **general-purpose image forgery
detector**, not a document- or passport-specific tool. It has no concept of
MRZ codes, holograms, microprint, or any passport security feature.

### Architecture (high level)

```
Input RGB image
      │
      ├────────────────────────────┐
      │                             │
  RGB branch                   Noiseprint++ branch (DnCNN, 17 conv layers)
  (raw pixels)                 learns a "camera processing fingerprint"
      │                        trained self-supervised on PRISTINE images only
      │                             │
      └──────────┬──────────────────┘
                 CMX fusion encoder (SegFormer-B2 backbone, cross-modal)
                 — combines high-level semantic (RGB) + low-level noise traces
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                   │
   Localization head   Confidence head     Detection head
   (per-pixel map,     (per-pixel          (whole-image score,
    "how forged")       reliability)        pooled from loc+conf maps)
```

**Core idea:** Noiseprint++ learns what a camera's internal processing
pipeline (demosaicing, JPEG compression, denoising, etc.) "normally" looks
like, having only ever seen authentic images during its self-supervised
training. When part of an image was edited (spliced, cloned, inpainted,
AI-generated), that local region's noise fingerprint deviates from the
camera's expected pattern. The model flags that deviation as an anomaly —
it is **not** pattern-matching against known forgery templates, it is
detecting "this region doesn't belong statistically."

### What you get per image (the `.npz` / our wrapper's outputs)
- **`map`** — pixel-level localization map (probability each pixel is forged)
- **`conf`** — reliability map (where the model's own prediction should be trusted)
- **`score`** — single whole-image number in [0,1], pooled from the above two

### Why this matters for the "not a passport verifier" caveat
Because the whole mechanism is "does this patch look statistically different
from the rest of the image," TruFor:
- Works well on natural photographs with localized edits (splicing in a face,
  removing an object, copy-moving a region).
- Has **no notion of correctness** for passport layout, security fonts, MRZ
  checksum, hologram presence, etc. A passport that's info­rmation-wrong but
  pixel-consistent will score **low** (looks "unforged") even if it's a
  complete fake built from scratch in a single clean export.

---

## 2. Repo structure

```
TruFor/
├── README.md                     top-level pointers (Docker vs train/test)
├── test_docker/                  self-contained Docker inference image
│   ├── Dockerfile                 builds from pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
│   ├── docker_build.sh            builds image + bakes in downloaded weights
│   ├── docker_run.sh              runs container against images/ folder
│   └── src/                       trimmed inference-only copy of the code
└── TruFor_train_test/            full source — training AND inference
    ├── trufor_conda.yaml          reference Conda env (Linux-authored)
    ├── test.py                    official inference script (what we run)
    ├── train.py                   training entrypoint (NOT used — out of scope)
    ├── visualize.py                quick matplotlib viewer for a result .npz
    ├── metrics.py                  F1/IoU scoring utilities (for labeled eval sets)
    ├── project_config.py           dataset path config (training only)
    ├── passport_trufor.py          *** our wrapper, added this session ***
    ├── lib/
    │   ├── config/                 YAML configs (trufor_ph2.yaml, trufor_ph3.yaml = inference config)
    │   ├── models/cmx/              CMX fusion encoder, SegFormer backbone, decoders
    │   ├── models/DnCNN.py          Noiseprint++ architecture
    │   └── utils.py                 get_model(), training loop helpers
    ├── dataset/                     dataset loaders (training only, except dataset_test.py)
    └── pretrained_models/
        ├── segformers/mit_b2.pth    SegFormer-B2 ImageNet backbone init (included in repo)
        ├── noiseprint++/noiseprint++.th  Noiseprint++ pretrained init (included in repo)
        └── trufor.pth.tar           *** downloaded this session, 281MB, the actual usable weights ***
```

**Important nuance:** `segformers/mit_b2.pth` and `noiseprint++/noiseprint++.th`
are *initialization* weights — only relevant if you were training from
scratch. For inference, `test.py` loads `trufor.pth.tar` which overwrites the
entire model's `state_dict` (backbone + Noiseprint++ + all heads, already
fused and trained end-to-end). The two "init" files are functionally unused
once `trufor.pth.tar` is loaded.

---

## 3. Why Docker exists here, and can you just use Docker Desktop instead

`test_docker/` is the repo authors' answer to "I don't want to fight Conda/CUDA
versions, give me a container that just works." It:
- Builds `FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime` (official PyTorch
  image, Linux-based, CUDA 11.3 pre-baked)
- `pip install`s only the minimal inference deps (no mmcv/mmseg — the Docker
  `src/` copy apparently vendors or avoids that dependency, lighter than full repo)
- Auto-downloads and unzips `trufor.pth.tar` **during the build**
- Entrypoint runs inference directly: `docker run ... -in images/ -out output/`

**Yes — you can absolutely run this on Docker Desktop instead of the Conda
route we did.** Trade-offs:

| | Conda (what we set up) | Docker Desktop |
|---|---|---|
| GPU access | Native, direct CUDA | Needs Docker Desktop's WSL2 GPU passthrough enabled (Windows → works since Docker Desktop 4.x with NVIDIA driver support, but is an extra layer) |
| Setup friction | Version-pinning by hand (what we just did) | Pull/build once, environment is frozen in the image |
| Debuggability | Full access to edit `passport_trufor.py`, inspect intermediate tensors, iterate fast | Need to rebuild image or bind-mount source to iterate |
| Portability | Tied to this machine's Conda install | Image runs identically on any Docker host |
| Disk cost | ~6-8GB in a Conda env | Similar, inside a Docker image layer |

**Why we didn't just use Docker for this task:** you asked for pretrained
inference you can iterate on (the `passport_trufor.py` wrapper, custom output
formats, JSON scoring) — that kind of iteration is much faster directly in a
Conda env than rebuilding a Docker image on every change. If this were going
into a **deployed service** (an API endpoint, a background worker), Docker
would be the better end state — pin the image once, ship it, no "works on my
machine" risk. Natural next step if you want: write a `Dockerfile` that
`COPY`s in `passport_trufor.py` on top of the existing `test_docker` base, so
you get the container's portability with our wrapper's output format.

---

## 4. What issues we actually hit and fixed

1. **Conda ToS gate** — new conda (26.x) refuses to solve against
   `repo.anaconda.com` default channels until Terms of Service are accepted
   non-interactively. Fixed with `conda tos accept --override-channels
   --channel <url>` for `pkgs/main`, `pkgs/r`, `pkgs/msys2` (you approved this).

2. **`trufor_conda.yaml` is Linux-only** — build strings like
   `h06a4308_3561`, `_libgcc_mutex`, `libgcc-ng` don't exist for `win-64`.
   We didn't try to force that YAML; instead built the equivalent env
   package-by-package with pinned versions (Python 3.7.16, torch
   1.11.0+cu113 from `download.pytorch.org`, mmcv-full 1.5.3 from
   `download.openmmlab.com`'s win_amd64/cp37 wheel index — confirmed it
   exists before committing to the approach).

3. **Python 3.7 env had a broken `ssl` module** when invoking
   `python.exe` directly — `ImportError: DLL load failed`. Root cause: the
   DLLs (`libssl-1_1-x64.dll`, `libcrypto-1_1-x64.dll`) live in
   `envs\trufor\Library\bin`, which is only on `PATH` when the env is
   properly activated. Direct `python.exe` calls skip that. Fix:
   always invoke through `conda run -n trufor python ...` (or fully activate
   the env), never call the env's `python.exe` in isolation.

4. **`opencv-python-headless` got silently upgraded to 5.0.0.93** —
   `albumentations` pulled it in unpinned, overriding the intended
   `4.5.5.64` from `opencv-python`. Re-pinned explicitly to avoid a
   `cv2` version/ABI mismatch between the two opencv packages.

5. **MD5 "mismatch" on the weights** — initially hashed the *extracted*
   `trufor.pth.tar` and got a different MD5 than the README's. The README's
   MD5 (`7bee48f3476c75616c3c5721ab256ff8`) is for the **zip file**, not the
   extracted tar. Re-checked against the zip — matched exactly. No actual
   corruption; just hashed the wrong artifact first.

6. **`test.py`'s Windows path bug** — when `-out` is a directory, the script
   does POSIX-style path math (`root = input.split('*')[0]`, then strips
   leading `/`) to compute the output sub-path. On Windows, `input` has a
   drive letter and backslashes, so this logic produces garbage like
   `C:\real.png.npz` — a write at the filesystem root, which fails with
   `PermissionError`. **Workaround (not a repo patch):** pass a full output
   *filename* to `-out` instead of a directory, which skips that broken
   branch entirely. Our `passport_trufor.py` wrapper sidesteps this
   altogether by writing files itself instead of relying on `test.py`'s path logic.

7. **GPU OOM on a 962×657 image** — `RuntimeError: CUDA out of memory.
   Tried to allocate 1.36 GiB` on a 4GB RTX 3050 Ti (shared with Windows
   desktop compositor, so usable VRAM is less than the nameplate 4GB).
   TruFor's CMX+SegFormer+Noiseprint++ stack is memory-heavy relative to
   the card. Fixes, in order of preference:
   - Run on CPU for that image: `-g -1` flag (slower, always works)
   - Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` to reduce
     fragmentation before retrying on GPU
   - Downscale the input image before inference if GPU speed matters

---

## 5. "No PyTorch/TensorFlow allowed" — what's the alternative for inference

If you're on a locked-down company machine where you can't install PyTorch
(policy restriction, no admin rights, air-gapped, etc.), options in order of
how much they preserve TruFor's actual behavior:

1. **ONNX export + ONNX Runtime** (best option if feasible)
   Export the loaded PyTorch model once (`torch.onnx.export`) on a machine
   where PyTorch *is* allowed, then ship only the `.onnx` file + `onnxruntime`
   (a much smaller, non-training-framework dependency, often allowed where
   PyTorch isn't) to the restricted machine. Caveat: TruFor's SegFormer
   backbone and the custom `weighted_statistics_pooling` op may need manual
   verification that they export cleanly — some interpolation/attention ops
   in older SegFormer implementations have had ONNX tracing issues. Would
   need a one-time validation pass (compare ONNX output vs PyTorch output on
   a test image) before trusting it.

2. **TorchScript (`torch.jit.trace` / `.script`) + `libtorch` (C++ runtime)**
   Similar idea — convert once elsewhere, run inference via the C++ LibTorch
   runtime, which doesn't require the Python PyTorch package. More setup
   effort than ONNX Runtime, same core advantage (no Python ML framework on
   the target machine).

3. **Remote inference / API** — run TruFor on an allowed machine or server,
   expose a minimal internal HTTP endpoint, have the locked-down machine
   just POST images and get JSON back. Not "local inference" anymore in the
   literal sense, but keeps the actual image data wherever you control it
   (e.g., your own server, not a third party) — compatible with the
   "no external APIs" spirit of this task if the server is yours.

4. **OpenVINO** (if Intel hardware / no GPU) — Intel's inference runtime can
   consume ONNX models too, sometimes with better CPU performance than
   plain PyTorch CPU inference.

**What won't work:** there's no pip package that runs this specific model
without some ML runtime (PyTorch, ONNX Runtime, or LibTorch) — the model
architecture (custom CMX fusion, DnCNN, SegFormer) isn't something you can
hand-reimplement in pure NumPy without a lot of engineering effort and risk
of subtly wrong results.

---

## 6. Would fine-tuning make this better for passports specifically?

**Probably yes for detecting the *kinds* of edits present in your training
data, but it comes with real costs and risks:**

**What fine-tuning could improve:**
- If you build a labeled dataset of real vs. tampered passport/ID images
  (ideally with pixel-level forgery masks), fine-tuning phase 2/3
  (`train.py -exp trufor_ph2` then `trufor_ph3`) would adapt the model to:
  - Document-specific textures (security paper patterns, printed microtext,
    guilloché patterns) that differ from natural photos
  - Typical edit types seen in real document fraud (photo swap, DOB
    alteration, name alteration) rather than generic splicing/COCO-style edits
  - Scan/photograph artifacts (different from DSLR/phone camera noise the
    base model was trained on)

**Why it's non-trivial:**
- Needs a **decent-sized labeled dataset** — likely hundreds to low-thousands
  of tampered + pristine document images with ground-truth masks, which is
  hard to source (real fraud examples are sensitive/rare; synthetic tampering
  has to be representative of real fraud techniques, not just random splices)
- Risk of **overfitting to your synthetic tampering style** — if your training
  tampered set is "obvious Photoshop splices," fine-tuned model gets *worse*
  at generic graceful edits, not better, unless your fakes are diverse and
  high-quality
- The paper's phase 2/3 split (`trufor_ph2` → localization network,
  `trufor_ph3` → detection + confidence) exists because training end-to-end
  from scratch is unstable; realistically you'd fine-tune phase 3 on top of
  the existing `trufor.pth.tar` weights (transfer learning), not retrain
  phase 2, to avoid needing the huge original COCO/CASIA/IMD/FantasticReality
  datasets
- You explicitly said "I do NOT want to train the model" for this task — this
  section is informational, not a recommendation to do it now

**Bottom line:** fine-tuning is the right lever *if* document-fraud detection
becomes a serious product requirement and you can source/label real fraud
examples. For a prototype or risk-flagging tool, the pretrained general model
plus a human review step is the more defensible starting point.

---

## 7. Can it detect synthetic / AI-generated documents (e.g. fully GAN/diffusion-made passports)?

**Weak to unreliable, by design.** TruFor's core mechanism is "find the
boundary between a pristine region and a locally-edited region" — it needs a
*contrast* between untouched camera-noise-consistent pixels and
manipulated ones.

- A **fully synthetic image** (the entire passport generated by a diffusion
  model or GAN from scratch, no real photo as a base) has **no pristine
  region to contrast against** — the whole image is uniformly "synthetic,"
  so there's no local anomaly to localize. The `map` output may come back
  uniformly low, uniformly high, or noisy/meaningless — none of which
  reliably signals "this is fully fake."
- Some fully-synthetic detectors work differently: they look for
  **global** statistical fingerprints of a specific GAN/diffusion
  architecture (checkerboard artifacts, frequency-domain signatures unique
  to the generator), which is a different task than TruFor's localized
  splice detection. TruFor was not trained or evaluated for this
  (the paper's datasets — tampCOCO, CASIA, IMD, FantasticReality, CocoGlide —
  are all about edits to real photos, not fully-synthetic generation).
- **Partial synthesis** (e.g., a real passport template with the photo
  region replaced by a diffusion-inpainted face) is much closer to TruFor's
  actual strength — that's a local edit on a real-photo base, which is
  exactly the splice/inpainting scenario the model is trained to flag.

**Practical takeaway:** don't rely on TruFor as a synthetic-document
detector. If fully-AI-generated documents are a real threat model for you,
that needs a separate, purpose-built classifier (trained specifically on
GAN/diffusion-generated vs. real images) run alongside TruFor, not instead
of it.

---

## 8. Quick reference — commands

```powershell
# Activate / invoke the env (PATH issue means direct python.exe calls can fail — use conda run)
C:\Users\Madhavan\miniconda3\Scripts\conda.exe run -n trufor python <script> ...

# Official inference script (output must be a FILENAME on Windows, not a directory — see bug #6 above)
cd TruFor\TruFor_train_test
C:\Users\Madhavan\miniconda3\Scripts\conda.exe run -n trufor python test.py `
  -in "path\to\image.jpg" -out "path\to\output.npz" `
  -exp trufor_ph3 TEST.MODEL_FILE "pretrained_models/trufor.pth.tar"

# Our wrapper (recommended — handles paths correctly, writes images + JSON)
C:\Users\Madhavan\miniconda3\Scripts\conda.exe run -n trufor python passport_trufor.py `
  --image "path\to\image.jpg" --output results

# Force CPU (use if you hit CUDA OOM on the 4GB RTX 3050 Ti)
... passport_trufor.py --image "..." --output results -g -1
```

**Environment summary:** Conda env `trufor`, Python 3.7.16, PyTorch
1.11.0+cu113, torchvision 0.12.0+cu113, mmcv-full 1.5.3, mmsegmentation
0.25.0, GPU = NVIDIA RTX 3050 Ti Laptop GPU (4GB VRAM), weights at
`TruFor_train_test/pretrained_models/trufor.pth.tar` (MD5 of source zip
verified: `7bee48f3476c75616c3c5721ab256ff8`).
