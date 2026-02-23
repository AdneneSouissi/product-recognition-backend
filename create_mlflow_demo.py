import mlflow
import time
from pathlib import Path

# Set experiment
mlflow.set_experiment("product-recognition")

# Create 3 sample runs with different parameters
configs = [
    {"epochs": 100, "batch": 16, "lr": 0.01, "imgsz": 640, "model": "yolov8n.pt"},
    {"epochs": 50, "batch": 32, "lr": 0.005, "imgsz": 416, "model": "yolov8n.pt"},
    {"epochs": 150, "batch": 8, "lr": 0.02, "imgsz": 640, "model": "yolov8s.pt"},
]

metrics_sets = [
    {"mAP50": 0.523, "mAP50-95": 0.373, "precision": 0.638, "recall": 0.522, "box_loss": 0.045},
    {"mAP50": 0.498, "mAP50-95": 0.351, "precision": 0.612, "recall": 0.501, "box_loss": 0.052},
    {"mAP50": 0.547, "mAP50-95": 0.389, "precision": 0.651, "recall": 0.538, "box_loss": 0.041},
]

print("Creating MLflow experiment runs...")

for i, (config, metrics) in enumerate(zip(configs, metrics_sets)):
    run_name = f"yolov8-run-{i+1}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for key, value in config.items():
            mlflow.log_param(key, value)
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log additional training metrics over "epochs"
        for epoch in range(1, min(config["epochs"], 10) + 1):
            mlflow.log_metric("train/box_loss", 0.05 - epoch * 0.002, step=epoch)
            mlflow.log_metric("train/cls_loss", 0.03 - epoch * 0.001, step=epoch)
            mlflow.log_metric("val/mAP50", 0.3 + epoch * 0.02, step=epoch)
        
        print(f"✓ Created run: {run_name}")

print(f"\n✅ Successfully created {len(configs)} MLflow runs!")
print("Open http://localhost:5000 to view experiments")
