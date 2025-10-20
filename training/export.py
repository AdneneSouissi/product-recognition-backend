import argparse
from pathlib import Path
import shutil
import yaml


def main(params_path: str) -> int:
    params = yaml.safe_load(Path(params_path).read_text())
    export_cfg = params.get('export', {})
    artifact_path = export_cfg.get('artifact_path', 'exported_yolov8n.pt')

    src = Path('models/current.pt')
    dst = Path(artifact_path)

    if not src.exists():
        print(f"[export] WARNING: source model not found: {src}. Did training finish and produce models/current.pt?")
        return 0

    # Ensure parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy the model as the exported artifact (extend later to real format conversions if needed)
    shutil.copy2(src, dst)
    print(f"[export] Exported artifact -> {dst}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='params.yaml')
    args = ap.parse_args()
    raise SystemExit(main(args.params))
