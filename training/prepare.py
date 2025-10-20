import argparse
import yaml
from pathlib import Path


def main(params_path: str):
    params = yaml.safe_load(Path(params_path).read_text())
    data_dir = Path(params.get('data', {}).get('raw_dir', 'data/raw'))
    processed_dir = Path(params.get('data', {}).get('processed_dir', 'data/processed'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"[prepare] WARNING: raw data dir not found: {data_dir}. Skipping.")
        return 0

    # Placeholder: copy/check data, build train/val split files, etc.
    print(f"[prepare] OK. raw={data_dir} processed={processed_dir}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='params.yaml')
    args = ap.parse_args()
    raise SystemExit(main(args.params))
