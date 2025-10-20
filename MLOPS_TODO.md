# End-to-end MLOps TODO (Backend + Web + Mobile)

This document turns the backend, training, and frontends into a reproducible, observable, and continuously validated system. All commands are for Windows PowerShell.

## Goals

- Reliable API with health, readiness, metrics, and version endpoints.
- Observable stack with Prometheus (+ optional Grafana).
- Reproducible training with DVC + MLflow, exporting a model artifact used by the API.
- CI smoke tests for API (and basic frontends build checks).
- Clear environment-driven configuration across backend/web/mobile.

---

## 1) Prerequisites

- Python 3.10+ and pip
- Node 18+ (for frontends), npm or pnpm
- Docker Desktop (for compose stack)

Project layout of this workspace:
- Backend: `product-recognition-backend/`
- Web: `product-recognition-frontend-web/`
- Mobile (Expo): `product-recognition-frontend-mobile/`

---

## 2) Configure environment (.env)

Create `product-recognition-backend/.env` (and commit a `.env.example`). Suggested keys:

```
MODEL_URI=./yolov8n.pt
HOST=0.0.0.0
PORT=8000
APP_VERSION=0.1.0
CORS_ORIGINS=*
MONGO_URI=mongodb://mongo:27017
SKIP_MODEL_LOAD=0
```

Notes:
- Set `SKIP_MODEL_LOAD=1` to quickly boot the API for health/metrics without a model (predict will return 503).
- For local only, you can use `MONGO_URI=mongodb://localhost:27017` if Mongo runs on your host.

---

## 3) Backend: install and run locally

From `product-recognition-backend/`:

```powershell
# create venv (if not present) and install
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# start without model (fast boot)
$env:SKIP_MODEL_LOAD=1; python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Acceptance (without model):
- GET http://localhost:8000/healthz -> 200 with status ok
- GET http://localhost:8000/version -> 200 with version
- GET http://localhost:8000/metrics -> Prometheus text
- GET http://localhost:8000/readyz -> 503 with ready=false (not ready)
- POST /predict -> 503 Service Unavailable (expected since model is not loaded)

Run with model loaded:

```powershell
$env:SKIP_MODEL_LOAD=0; $env:MODEL_URI="$PWD\yolov8n.pt"; python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Acceptance (with model):
- GET /readyz -> ready=true
- POST /predict with a test image -> 200 JSON with detections

Optional smoke test script in this repo:

```powershell
python .\tests\smoke_predict.py
```

---

## 4) Monitoring stack (Docker Compose)

We provide `docker-compose.yml` to run API + Mongo + Prometheus (+ Grafana). From `product-recognition-backend/`:

```powershell
docker compose up -d
```

Ports (default):
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (if enabled in compose)
- Mongo: localhost:27017

Acceptance:
- Prometheus “Status > Targets” shows the API target up and scraping /metrics
- Hitting /predict increases request counters/histograms (check Prometheus expression browser)

If Grafana is enabled in compose, import a simple dashboard that graphs:
- request_count by route/status
- request_latency_seconds histogram (p50/p95)
- errors (5xx) over time

---

## 5) Training pipeline (DVC + MLflow)

Training code and pipeline configs are in `product-recognition-backend/training/`, `params.yaml`, and `dvc.yaml`.

Recommended local run with tracking to local MLflow store:

```powershell
cd .\product-recognition-backend
$env:MLFLOW_TRACKING_URI="$PWD\mlruns"
# Optionally view UI in another terminal:
# mlflow ui --backend-store-uri $env:MLFLOW_TRACKING_URI --port 5001

# Reproduce full pipeline
python -m pip install -r requirements.txt
dvc repro
```

Acceptance:
- DVC completes stages: prepare -> train -> eval -> export
- MLflow run created with params/metrics/artifacts
- Exported model file exists (e.g., `exported_yolov8n.pt` or path configured in params)

Promoting the model to the API:

```powershell
Copy-Item .\exported_yolov8n.pt .\yolov8n.pt -Force
$env:SKIP_MODEL_LOAD=0; $env:MODEL_URI="$PWD\yolov8n.pt"; python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Acceptance:
- /readyz -> ready=true
- /predict -> 200 with reasonable detections on sample image

---

## 6) CI recommendations (GitHub Actions)

Backend minimal CI (lint + boot + health/metrics):

```yaml
name: backend-ci
on: [push, pull_request]
jobs:
	api:
		runs-on: ubuntu-latest
		defaults:
			run:
				working-directory: product-recognition-backend
		steps:
			- uses: actions/checkout@v4
			- uses: actions/setup-python@v5
				with: { python-version: '3.11' }
			- run: pip install -r requirements.txt
			- name: Boot API (no model)
				run: |
					export SKIP_MODEL_LOAD=1
					python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
					sleep 3
			- run: curl -f http://localhost:8000/healthz
			- run: curl -f http://localhost:8000/version
			- run: curl -f http://localhost:8000/metrics | head -n 20
```

Optional nightly job (with small model artifact) to exercise `/predict`.

Web minimal CI (install + build):

```yaml
name: web-ci
on: [push, pull_request]
jobs:
	build:
		runs-on: ubuntu-latest
		defaults:
			run:
				working-directory: product-recognition-frontend-web
		steps:
			- uses: actions/checkout@v4
			- uses: actions/setup-node@v4
				with: { node-version: '18' }
			- run: npm ci
			- run: npm run build
```

Mobile minimal CI (typecheck/lint or Expo web build smoke):

```yaml
name: mobile-ci
on: [push, pull_request]
jobs:
	build-web:
		runs-on: ubuntu-latest
		defaults:
			run:
				working-directory: product-recognition-frontend-mobile
		steps:
			- uses: actions/checkout@v4
			- uses: actions/setup-node@v4
				with: { node-version: '18' }
			- run: npm ci
			- run: npm run lint --if-present
			- run: npm run web -- --non-interactive --no-open & sleep 5; pkill -f "webpack|vite" || true
```

---

## 7) Frontends: environment and smoke checks

Web (`product-recognition-frontend-web/`):
- Configuration: set `VITE_API_BASE_URL` (and `VITE_WS_URL` if used) in `.env.local`.
- Dev: `npm run dev`
- Acceptance: Can call http://localhost:8000/predict and render boxes in the UI.

Mobile (Expo, `product-recognition-frontend-mobile/`):
- Configuration: set `EXPO_PUBLIC_API_BASE_URL` (and `EXPO_PUBLIC_WS_URL` if used) in `app.json` or `.env` for Expo.
- Dev: `npx expo start` then run on device or emulator.
- Acceptance: Captured/uploaded image returns predictions; green bounding boxes overlay correctly.

---

## 8) Release and deployment (compose/K8s)

Container image (example):

```powershell
cd .\product-recognition-backend
docker build -t your-registry/product-recognition-api:$(git rev-parse --short HEAD) .
# docker push your-registry/product-recognition-api:<tag>
```

Deploy with Docker Compose (production-like):
- Set `MODEL_URI` to the promoted artifact
- `docker compose up -d` on the target host
- Acceptance: Health/ready checks pass, Prometheus scraping, baseline latency within SLO

---

## 9) Troubleshooting

- /predict returns 503: Model not loaded. Set `SKIP_MODEL_LOAD=0` and ensure `MODEL_URI` points to a valid file.
- API not reachable from phone: Use your host LAN IP in the frontends (not 127.0.0.1). Confirm firewall allows port 8000.
- Prometheus empty: Hit endpoints to generate traffic; confirm `monitoring/prometheus.yml` targets match service name/port.
- DVC errors about remotes: You can use local cache first; add remotes later (S3, Azure, GCS).

---

## 10) Acceptance checklist (summary)

- Backend dev run: health/version/metrics OK; readiness reflects model presence; predict 200 with model
- Compose stack: Prometheus up; dashboard shows request counts and latency
- Training pipeline: `dvc repro` succeeds; MLflow run logged; exported model ready
- Web app: builds and can call API; boxes render
- Mobile app: runs on device, calls API over LAN; boxes align with images/camera

Done means the above five bullets are satisfied for this repo and the two frontends in this workspace.
