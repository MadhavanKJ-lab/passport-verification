# Passport Verification

Decision support for passport images: OCR, AI validation against MRZ rules, and a forensic check for image manipulation or synthetic content. Output is a risk signal, not an authentication verdict.

## How it fits together

```
 browser ──► n8n workflow (Passport_Verify)
               ├─ OCR ............ deploy/paddle_ocr   (PaddleOCR, Document Intelligence-compatible API)
               ├─ AI validation .. Azure OpenAI (gpt-5.1) + deterministic MRZ checks
               └─ Forensics ...... deploy/             (FastAPI: crop → heuristics → B-Free → TruFor → combiner)
```

| Path | What it is |
|---|---|
| `PassportPipeline/` | Forensic stages: `crop.py`, `heuristics.py`, `combine_scores.py`, `justification.py`, `run_pipeline.py` (local CLI) |
| `TruFor/` | Upstream [TruFor](https://github.com/grip-unina/TruFor) with `passport_trufor.py` wrapper and local patches |
| `BFree/` | Upstream [B-Free](https://github.com/grip-unina/B-Free) code with `passport_bfree.py` wrapper |
| `deploy/` | Forensic API container (`Dockerfile`, `api.py`) and `deploy.sh` for Azure Container Apps |
| `deploy/paddle_ocr/` | OCR container exposing a Document Intelligence-style async API |
| `frontend/` | Web UI (`public/`), Vercel config, and a local dev server |

## Combined verdict

TruFor 0.5, B-Free 0.3, classical heuristics 0.2, plus a small bonus when models disagree. Scores ≥ 0.55 are **high**, ≥ 0.30 **elevated**. If a model stage fails, the verdict is "cannot assess" and never "low". The weights are hand-set: there is no labelled passport data to fit a classifier.

## Not in this repository

Model weights and test data are excluded on purpose.

- **Weights:** download TruFor (`trufor.pth.tar`) and B-Free (`model_epoch_best.pth`) from their upstream repos and place them where `deploy/Dockerfile` expects (`TruFor/TruFor_train_test/pretrained_models/`, `BFree/weights/BFREE_dino2reg4/`).
- **Images and results:** real passport images and per-image outputs are personal data and are git-ignored.
- **Secrets:** API keys live in Azure Container App secrets and n8n credentials.

## Run the UI locally

```bash
cd frontend
N8N_WEBHOOK_URL=<your n8n webhook URL> python server.py   # http://localhost:8080
```

The local server proxies the upload to the webhook. Set `N8N_API_KEY` if the webhook requires the `API` header.

## Deploy the UI to Vercel

Set the Vercel project's **Root Directory** to `frontend` and add the environment variable `N8N_WEBHOOK_URL`. The page reads it from `/api/config` and calls the webhook from the browser, so the n8n webhook must allow that origin (CORS).

## Deploy the backend to Azure

See `deploy/deploy.sh`. Build from the repository root with the weights in place.

## Licences

TruFor and B-Free are research code from the GRIP group (University of Naples Federico II) under their own licences, which restrict commercial use. Check them before using this outside research.
