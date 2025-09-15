# src/data/build_labels_csv.py
from __future__ import annotations
from pathlib import Path
import re, sys
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "raw" / "mias"
OUT_CSV = REPO / "data" / "labels.csv"

def _to_float(v):
    try:
        return float(str(v).strip())
    except Exception:
        return None

def _canon_class(s: str) -> str:
    if s is None: return "unknown"
    s = str(s).strip().lower()
    if s in {"n", "norm", "normal"}: return "normal"
    if s in {"b", "benign"}: return "benign"
    if s in {"m", "malignant"}: return "malignant"
    return s

def _to_binary_target(row) -> int | None:
    c = str(row.get("class", "")).lower()
    sev = str(row.get("severity", "")).lower()
    if c == "normal": return 0
    if sev in {"benign", "malignant"}: return 1
    # Any non-normal MIAS class (calc, circ, spic, misc, arch, asym) counts as abnormal
    if c in {"calc","circ","spic","misc","arch","asym","abnormal"}: return 1
    return None

def _find_metadata_csv(root: Path) -> Path | None:
    for name in ["mias-metadata.csv","mias_metadata.csv","mias.csv","metadata.csv","labels.csv"]:
        p = root / name
        if p.exists(): return p
    for p in root.rglob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=3); cols = {c.lower() for c in head.columns}
        except Exception:
            continue
        if {"class","label","severity","abnormality","refnum","id","filename"} & cols:
            return p
    return None

def _parse_info_txt(root: Path) -> pd.DataFrame | None:
    # Find Info.txt (may live in root or a subfolder)
    info = None
    for p in [root / "Info.txt", root / "info.txt"]:
        if p.exists():
            info = p
            break
    if info is None:
        for p in root.rglob("Info.txt"):
            info = p
            break
    if info is None:
        return None

    rows = []
    for raw in info.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip comments or header-like lines
        lower = line.lower()
        if lower.startswith(("#", "%", "//")):
            continue
        if "refnum" in lower and "class" in lower:
            # header line e.g. "REFNUM BG CLASS SEVERITY X Y RADIUS"
            continue

        # tokenise on any whitespace
        toks = line.split()
        if len(toks) < 3:
            continue

        refnum = toks[0].lower()
        bg = toks[1]
        mclass = toks[2].lower()

        severity = None
        x = y = radius = None

        # Normal cases usually have only 3 tokens (no severity/x/y/r)
        if mclass not in {"norm", "normal"}:
            # Guard against another stray header row that slipped through
            if len(toks) >= 4 and toks[3].upper() in {"SEVERITY", "X", "Y", "RADIUS"}:
                # definitely a header-ish line -> skip
                continue
            if len(toks) >= 4:
                sev_token = toks[3].lower()
                severity = {"b": "benign", "benign": "benign",
                            "m": "malignant", "malignant": "malignant"}.get(sev_token, sev_token)
            if len(toks) >= 7:
                x = _to_float(toks[4])
                y = _to_float(toks[5])
                radius = _to_float(toks[6])

        rows.append({
            "refnum": refnum,
            "bg": bg,
            "class": mclass,
            "severity": severity,
            "x": x, "y": y, "radius": radius,
        })

    return pd.DataFrame(rows)

def _index_images(root: Path):
    exts = {".pgm",".png",".jpg",".jpeg",".tif",".tiff",".bmp"}
    paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    by_stem = {}
    for p in paths:
        by_stem.setdefault(p.stem.lower(), []).append(p)
    return by_stem

def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] Expected raw MIAS at {RAW_DIR}.", file=sys.stderr); sys.exit(1)

    # 1) Prefer CSV if present
    meta_csv = _find_metadata_csv(RAW_DIR)
    if meta_csv:
        df = pd.read_csv(meta_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        # Try to align to MIAS canonical columns
        if "refnum" not in df.columns:
            if "id" in df.columns: df["refnum"] = df["id"]
            elif "filename" in df.columns: df["refnum"] = df["filename"].astype(str).str.replace(r"\..*$","",regex=True)
            elif "name" in df.columns: df["refnum"] = df["name"]
        if "class" not in df.columns and "abnormality" in df.columns:
            df["class"] = df["abnormality"]
        if "severity" not in df.columns and "label" in df.columns:
            df["severity"] = df["label"]
    else:
        # 2) Fall back to Info.txt
        df = _parse_info_txt(RAW_DIR)
        if df is None or df.empty:
            print(f"[ERROR] No metadata CSV and no Info.txt found under {RAW_DIR}.", file=sys.stderr); sys.exit(2)

    # Build output schema
    out = pd.DataFrame()
    out["id"] = df.get("refnum", df.get("id")).astype(str).str.strip().str.lower()
    out["bg"] = df.get("bg")
    # Normalize class/severity to friendly terms
    cls_raw = df.get("class")
    sev_raw = df.get("severity")
    # Preferred class: if severity exists and is benign/malignant use that, else use MIAS class
    out["class"] = [
        _canon_class((sv if str(sv).lower() in {"b","benign","m","malignant"} else cr))
        for sv, cr in zip(sev_raw if sev_raw is not None else [None]*len(df), cls_raw if cls_raw is not None else [None]*len(df))
    ]
    # Keep separate severity for reference
    out["severity"] = [ {"b":"benign","m":"malignant"}.get(str(s).lower(), str(s).lower() if pd.notna(s) else None) for s in (sev_raw if sev_raw is not None else []) ] + ([None] * (len(df) - len(sev_raw))) if sev_raw is not None else None
    # Coordinates if present
    for c in ["x","y","radius"]:
        if c in df.columns: out[c] = df[c]

    # Find image paths by stem
    by_stem = _index_images(RAW_DIR)
    def match_path(stem: str):
        cands = by_stem.get(stem.lower(), [])
        if cands: return str(cands[0].resolve())
        return None
    out["path_raw"] = out["id"].apply(match_path)

    # Binary target: normal=0, others (benign/malignant/other MIAS classes)=1
    out["target"] = out.apply(_to_binary_target, axis=1)

    # Drop rows with no image path
    before = len(out)
    out = out.dropna(subset=["path_raw"]).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows with missing image files.", file=sys.stderr)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {OUT_CSV.relative_to(REPO)} with {len(out)} rows.")
    print("\n[Class counts]\n", out["class"].value_counts(dropna=False))
    if out["target"].notna().any():
        print("\n[Binary target counts]\n", out["target"].value_counts())

if __name__ == "__main__":
    main()
