import os
import json
import datetime

def make_run_dir(base_dir: str = "prediction/cache") -> dict:
    base = os.path.abspath(base_dir)
    os.makedirs(base, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(base, ts)
    preprocess_dir = os.path.join(run_dir, "preprocess")
    model_dir = os.path.join(run_dir, "model")
    raw_dir = os.path.join(run_dir, "raw")
    for d in (preprocess_dir, model_dir, raw_dir):
        os.makedirs(d, exist_ok=True)
    return {"root": run_dir, "preprocess": preprocess_dir, "model": model_dir, "raw": raw_dir}

def get_latest_run(base_dir: str = "prediction/cache") -> str | None:
    base = os.path.abspath(base_dir)
    if not os.path.exists(base):
        return None
    entries = [e for e in os.listdir(base) if os.path.isdir(os.path.join(base, e))]
    if not entries:
        return None
    entries.sort()
    return os.path.join(base, entries[-1])

def get_latest_model_dir(base_dir: str = "prediction/cache") -> str | None:
    run = get_latest_run(base_dir)
    if not run:
        return None
    model_dir = os.path.join(run, "model")
    return model_dir if os.path.exists(model_dir) else None

def write_run_metadata(run_root: str, meta: dict):
    path = os.path.join(run_root, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
