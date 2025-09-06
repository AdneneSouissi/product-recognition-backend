

from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano is smallest and fastest)
model = YOLO("yolov8n.pt")  # You can choose yolov8s.pt, yolov8m.pt, etc.

# Save/export the model as a .pt file for deployment
model.save("exported_yolov8n.pt")

print("Model exported as exported_yolov8n.pt")
