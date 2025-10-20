import argparse
import time
from pathlib import Path
import yaml
import mlflow
from ultralytics import YOLO


def main(params_path: str):
    params = yaml.safe_load(Path(params_path).read_text())
    exp = params.get('experiment', {})
    data = params.get('data', {})
    train_cfg = params.get('train', {})

    model_uri = train_cfg.get('model_uri', 'yolov8n.pt')
    data_yaml = data.get('data_yaml', 'data.yaml')
    epochs = int(train_cfg.get('epochs', 1))
    imgsz = int(train_cfg.get('imgsz', 640))

    mlflow.set_experiment(exp.get('name', 'product-recognition'))
    with mlflow.start_run(run_name=f"train-{int(time.time())}"):
        mlflow.log_params({
            'model_uri': model_uri,
            'epochs': epochs,
            'imgsz': imgsz,
            'data_yaml': data_yaml,
        })
        model = YOLO(model_uri)
        # Note: requires a proper data.yaml; if missing, this will likely error.
        try:
            results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
            # Log metrics if available (Ultralytics results metrics can vary)
            # Placeholder: user can extend with results validation
            mlflow.log_param('status', 'success')
        except Exception as e:
            mlflow.log_param('status', 'failed')
            mlflow.log_text(str(e), 'train_error.txt')
            print(f"[train] ERROR: {e}")
            return 1

        # Export best model artifact if exists
        best = Path('runs/train/exp/weights/best.pt')
        if best.exists():
            models_dir = Path('models'); models_dir.mkdir(exist_ok=True)
            target = models_dir / f"best-{int(time.time())}.pt"
            target.write_bytes(best.read_bytes())
            (models_dir / 'current.pt').write_bytes(best.read_bytes())
            mlflow.log_artifact(str(target))
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='params.yaml')
    args = ap.parse_args()
    raise SystemExit(main(args.params))
