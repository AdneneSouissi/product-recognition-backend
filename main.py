import os
import time
import io
import json
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, WebSocket, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from prometheus_client import Histogram, Counter, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
from pymongo import MongoClient
from PIL import Image
from ultralytics import YOLO

# MongoDB connection (default localhost)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["product_db"]
collection = db["products"]


APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"

app = FastAPI(title="Product Recognition API", version=APP_VERSION)

cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "0") == "1"
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"]
)


model: Optional[YOLO] = None
if not SKIP_MODEL_LOAD:
    model_uri = os.getenv("MODEL_URI", "yolov8n.pt")
    model = YOLO(model_uri)




############################
# Metrics & Middleware
############################

registry = CollectorRegistry()
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["method", "endpoint", "status"],
    registry=registry,
)
REQUEST_COUNT = Counter(
    "requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry,
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.perf_counter() - start
        method = request.method
        endpoint = request.url.path
        status = getattr(response, "status_code", 500)
        REQUEST_LATENCY.labels(method, endpoint, status).observe(elapsed)
        REQUEST_COUNT.labels(method, endpoint, status).inc()


############################
# Health, Ready, Version, Metrics
############################

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    ready = model is not None
    return JSONResponse({"ready": ready, "model_loaded": ready}, status_code=200 if ready else 503)


@app.get("/version")
def version():
    return {"version": APP_VERSION}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/add_to_database")
async def add_to_database(
    file: UploadFile = File(...),
    predictions: str = Form(...)
):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    filename = file.filename
    preds = json.loads(predictions)
    saved_products = []
    for pred in preds:
        bbox = pred["bbox"]
        class_name = pred["class"]
        confidence = pred["confidence"]
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image.crop((x1, y1, x2, y2))
        buffered = io.BytesIO()
        cropped.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        doc = {
            "class": class_name,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "original_filename": filename,
            "cropped_image_base64": img_str
        }
    result = collection.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    saved_products.append(doc)
    return {"saved_products": saved_products}

# WebSocket
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            if model is None:
                await websocket.send_json({"error": "Model not loaded"})
                break
            results = model.predict(image)
            predictions = []
            for r in results:
                for box in r.boxes:
                    predictions.append({
                        "class": model.names[int(box.cls)],
                        "confidence": round(float(box.conf), 3),
                        "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                    })
            await websocket.send_json({"predictions": predictions})
        except Exception as e:
            await websocket.send_json({"error": str(e)})
            break

@app.get("/")
def root():
    return {"message": "YOLOv8 Local Deployment is running!"}

@app.post("/predict")
async def predict(file: UploadFile):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Run YOLO prediction
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    results = model.predict(image)

    # Extract predictions
    predictions = []
    for r in results:
        for box in r.boxes:
            predictions.append({
                "class": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
            })

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=True)
