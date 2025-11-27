import os, json, torch
from typing import Dict, List, Tuple

def save_checkpoint(out_dir: str, epoch: int, model) -> None:
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "model": model.state_dict()}, path)

def load_chain(out_dir: str):
    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {out_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cps = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("epoch_") and fn.endswith(".pt"):
            cps.append(os.path.join(out_dir, fn))
    checkpoints = [torch.load(p, map_location='cpu') for p in cps]
    return checkpoints, meta