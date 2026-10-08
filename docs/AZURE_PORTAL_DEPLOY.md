# Deploy to Azure using only the portal

This guide deploys the two backend containers from this public GitHub repository using the Azure portal in a browser. No CLI and no local Docker are needed.

| Container | Build context | Port | Size |
|---|---|---|---|
| Forensics API | repo root, `deploy/Dockerfile` | 8000 | **4 vCPU / 8 GiB** |
| OCR service | `deploy/paddle_ocr`, `Dockerfile` | 8000 | 2 vCPU / 4 GiB |

The forensic image downloads the TruFor and B-Free weights from their authors while it builds, so nothing large has to be uploaded. Expect the first forensic build to take **20-30 minutes**.

Button labels in the portal change from time to time. If one differs slightly, look for the nearest equivalent.

## 1. Resource group and registry

1. Portal → **Resource groups** → **Create**. Pick your subscription, name it (for example `rg-passport-verify`) and choose a region (for example **Central India**). Create.
2. Portal → **Container registries** → **Create**.
   - Resource group: the one above. Registry name: globally unique, lowercase letters and digits only.
   - Location: same region. Pricing plan: **Basic**.
   - **Review + create** → **Create**.
3. Open the registry → **Settings → Access keys** → turn **Admin user** on. Container Apps uses this to pull images.

## 2. Build the images in the cloud from GitHub

In the registry: **Services → Tasks** → **Quick run** (sometimes labelled *Run*).

**Forensics image**

| Field | Value |
|---|---|
| Source location | `https://github.com/MadhavanKJ-lab/passport-verification.git#main` |
| Dockerfile | `deploy/Dockerfile` |
| Image name | `passport-verify:v1` |
| Platform | Linux / amd64 |

**OCR image** (a second run)

| Field | Value |
|---|---|
| Source location | `https://github.com/MadhavanKJ-lab/passport-verification.git#main:deploy/paddle_ocr` |
| Dockerfile | `Dockerfile` |
| Image name | `passport-ocr:v1` |
| Platform | Linux / amd64 |

Watch **Services → Tasks → Runs** until both show **Succeeded**, then confirm the tags under **Services → Repositories**. A failed run has a log: open it and read the last lines.

If you fork the repository, use your own GitHub URL. Private forks need an access token in the source URL, so the public repository is simpler.

## 3. Container Apps environment

Portal → **Container Apps Environments** → **Create**. Same resource group and region, environment type **Consumption only**. Leave logging at the default. Create.

## 4. Create the two container apps

Portal → **Container Apps** → **Create**. Do this once per app.

**Basics:** resource group, app name (`passport-verify-api` / `passport-ocr-api`), region, and the environment from step 3.

**Container tab**
- Uncheck **Use quickstart image**.
- Image source **Azure Container Registry**, then pick your registry, image and tag.
- **CPU and memory:** 4 CPU / 8 Gi for forensics, 2 CPU / 4 Gi for OCR. The forensic app is killed (exit 137) at 2 CPU / 4 Gi.
- **Environment variables:** add `API_KEY` with *Source* **Manual entry** and a long random value. Save that value: callers send it as the `X-API-Key` header.

**Ingress tab**
- Enable ingress, **Accepting traffic from anywhere**, protocol HTTP, **target port 8000**.

**Create**, then wait for the deployment to finish.

**Afterwards, for each app:**
1. **Settings → Secrets** → add a secret `api-key` with the same value, then **Application → Containers → Edit and deploy** and change `API_KEY` to *Reference a secret → api-key*. This keeps the key out of the revision's plain settings.
2. **Application → Scale**: OCR must be **min 1, max 1**, because it keeps results in memory between two calls. For forensics choose min 0 (cheapest, but the first request after idle can fail with a 507) or min 1 (always warm, higher cost), and max 2.

## 5. Test

Copy each app's **Application Url** from its Overview page.

- Forensics: open `https://<app-url>/health`. Expect `{"healthy": true, ...}`. The first call after a cold start can take a minute.
- OCR: open `https://<app-url>/health`. Expect `{"healthy": true, "engine": "paddleocr", ...}`.

To test uploads without a terminal, use Postman or any REST client:
- Forensics: `POST https://<app-url>/verify`, header `X-API-Key: <key>`, body type **form-data** with a **File** field named `file`.
- Opening `/verify` in a browser returns "method not allowed" because it only accepts POST.

## 6. Connect n8n and the UI

- In n8n, point the forensic HTTP node at `https://<forensics-url>/verify` and the OCR nodes at `https://<ocr-url>/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30`, each with a Header Auth credential (`X-API-Key`).
- In Vercel → your project → **Settings → Environment Variables**, set `N8N_WEBHOOK_URL` and `N8N_WEBHOOK_KEY`, then redeploy.

## Common problems

| Symptom | Cause and fix |
|---|---|
| Quick run fails while downloading weights | The authors' download host was unreachable. Re-run it later |
| Container shows `Killed` / exit 137 | Too little memory. Use 4 CPU / 8 Gi |
| `507 exceeded request buffer limit` | Large upload hit a scaled-to-zero app. Set min replicas to 1 |
| 401 from the API | The `X-API-Key` header does not match the `API_KEY` value |
| OCR result `404 unknown or expired operation` | More than one OCR replica. Set max replicas to 1 |
| Cannot create a 4 CPU app | Your subscription's Container Apps core quota is too low. Request an increase under **Usage + quotas** |

## Cost

The OCR app (always on) and a warm forensic app are billed for the time they run. Delete the resource group when you are finished to stop all charges.
