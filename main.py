from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import base64
from datetime import datetime
from fastapi import File, Form
import json
from pymongo import MongoClient

# MongoDB connection (default localhost)
client = MongoClient("mongodb://localhost:27017/")
db = client["product_db"]
collection = db["products"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


model = YOLO("yolov8n.pt")




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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
