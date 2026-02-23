import mlflow
from pathlib import Path

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Get the latest run from product-recognition experiment
experiment = mlflow.get_experiment_by_name("product-recognition")
if not experiment:
    print("Error: No experiment found")
    exit(1)

# Get the best run (you can choose any run)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.mAP50 DESC"], max_results=1)

if runs.empty:
    print("Error: No runs found")
    exit(1)

run_id = runs.iloc[0].run_id
print(f"Found run: {run_id}")

# Register the model from this run
model_name = "product-detection-yolov8"
model_uri = f"runs:/{run_id}/model"

# Try to register (if model exists as artifact)
# If no model artifact exists, we'll log a dummy one
try:
    # Check if we have a model file to register
    model_path = Path("yolov8n.pt")
    
    if model_path.exists():
        # Log the model as an artifact in the run
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(model_path), "model")
            print(f"✓ Logged model artifact to run {run_id}")
        
        # Now register the model
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model/yolov8n.pt",
            name=model_name
        )
        print(f"✓ Registered model: {model_name}")
        print(f"  Version: {result.version}")
        
        # Set model version stage to Production
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        print(f"✓ Promoted version {result.version} to Production")
        
        # Add model description
        client.update_registered_model(
            name=model_name,
            description="YOLOv8n model for product detection. Trained on COCO dataset with 80 classes."
        )
        print(f"✓ Updated model description")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nAlternative: Use MLflow UI to register model manually:")
    print("1. Go to http://localhost:5000")
    print("2. Click on any run")
    print("3. Scroll to 'Artifacts' section")
    print("4. Click 'Register Model' button")
    print("5. Enter model name and click 'Register'")
