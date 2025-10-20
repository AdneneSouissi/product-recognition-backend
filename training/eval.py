import argparse
from pathlib import Path
import yaml
import mlflow


def main(params_path: str):
    params = yaml.safe_load(Path(params_path).read_text())
    exp = params.get('experiment', {})
    eval_cfg = params.get('eval', {})

    mlflow.set_experiment(exp.get('name', 'product-recognition'))
    with mlflow.start_run(run_name='eval', nested=True):
        # Placeholder: compute metrics on a val set and log to MLflow
        # Log dummy metric to confirm MLflow wiring
        mlflow.log_metric('dummy_eval_metric', 1.0)
        print('[eval] Dummy metric logged')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='params.yaml')
    args = ap.parse_args()
    raise SystemExit(main(args.params))
