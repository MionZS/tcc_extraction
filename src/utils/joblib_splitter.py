"""Joblib Splitter and Chunking Utility.

Splits large serialized model objects into compressed, versioned binary chunks
(<= 25 MB by default) so that they comply with GitHub's 100 MB hard limit and
50 MB warning threshold, while allowing seamless transparent loading.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import joblib


def dump_split_joblib(
    obj: Any,
    target_path: str | Path,
    chunk_size_mb: int = 25,
    compress: int = 3,
) -> list[Path]:
    """Serialize an object with joblib and split into chunks if exceeding chunk_size_mb.

    Args:
        obj: Python object to serialize (e.g. scikit-learn / PyOD model or pipeline).
        target_path: Destination path (e.g. `path/to/model.joblib`).
        chunk_size_mb: Maximum size in MB per chunk (default 25 MB).
        compress: Joblib zlib compression level (default 3).

    Returns:
        List of paths created (either the single file or the list of part files).
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Serialize into an in-memory byte buffer
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=compress)
    buf.seek(0)
    data = buf.getvalue()
    total_bytes = len(data)
    chunk_bytes = chunk_size_mb * 1024 * 1024

    # Calculate SHA256 of the complete serialized payload
    full_sha256 = hashlib.sha256(data).hexdigest()

    if total_bytes <= chunk_bytes:
        # Fits in a single file
        target.write_bytes(data)
        # Clean up any leftover parts if previously split
        for old_part in target.parent.glob(f"{target.name}.part*"):
            old_part.unlink(missing_ok=True)
        old_meta = target.with_suffix(target.suffix + ".meta.json")
        old_meta.unlink(missing_ok=True)
        return [target]

    # Split into chunks
    parts: list[Path] = []
    part_hashes: list[str] = []
    num_parts = (total_bytes + chunk_bytes - 1) // chunk_bytes

    for idx in range(num_parts):
        start = idx * chunk_bytes
        end = min(start + chunk_bytes, total_bytes)
        chunk = data[start:end]

        part_path = target.parent / f"{target.name}.part{idx:02d}"
        part_path.write_bytes(chunk)
        parts.append(part_path)
        part_hashes.append(hashlib.sha256(chunk).hexdigest())

    # Write metadata manifest
    meta = {
        "target_name": target.name,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "chunk_size_mb": chunk_size_mb,
        "num_parts": num_parts,
        "full_sha256": full_sha256,
        "parts": [
            {
                "part_filename": p.name,
                "bytes": p.stat().st_size,
                "sha256": h,
            }
            for p, h in zip(parts, part_hashes)
        ],
    }
    meta_path = target.parent / f"{target.name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # If the uncompressed/monolithic target exists, remove it to prevent git tracking
    if target.exists():
        target.unlink()

    return parts


def load_split_joblib(target_path: str | Path) -> Any:
    """Load a joblib object from a single file or reassembled split parts.

    Args:
        target_path: Path to the joblib file (e.g. `path/to/model.joblib`).

    Returns:
        The deserialized Python object.
    """
    target = Path(target_path)

    # 1. Direct single file
    if target.exists() and target.is_file() and target.stat().st_size > 0:
        return joblib.load(target)

    # 2. Check for parts
    part_files = sorted(target.parent.glob(f"{target.name}.part*"))
    if not part_files:
        raise FileNotFoundError(
            f"Cannot find model file '{target}' or split parts '{target.name}.part*'"
        )

    # Check metadata if available
    meta_path = target.parent / f"{target.name}.meta.json"
    expected_hash = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_hash = meta.get("full_sha256")
        except Exception:
            pass

    # Assemble parts into buffer
    buf = io.BytesIO()
    for pf in part_files:
        buf.write(pf.read_bytes())
    buf.seek(0)

    if expected_hash is not None:
        actual_hash = hashlib.sha256(buf.getvalue()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for reassembled {target.name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    return joblib.load(buf)


def main():
    parser = argparse.ArgumentParser(description="Split or join large joblib model files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_p = subparsers.add_parser("split", help="Split a joblib file into chunks")
    split_p.add_argument("file", help="Path to joblib file")
    split_p.add_argument("--chunk-size", type=int, default=25, help="Max chunk size in MB (default 25)")
    split_p.add_argument("--compress", type=int, default=3, help="Joblib compression level (default 3)")

    join_p = subparsers.add_parser("join", help="Verify and load split parts")
    join_p.add_argument("file", help="Base path to joblib file")

    args = parser.parse_args()

    if args.command == "split":
        p = Path(args.file)
        if not p.exists():
            print(f"Error: file '{p}' not found")
            return
        print(f"Loading '{p}' ({p.stat().st_size / (1024*1024):.2f} MB)...")
        obj = joblib.load(p)
        parts = dump_split_joblib(obj, p, chunk_size_mb=args.chunk_size, compress=args.compress)
        print(f"Successfully serialized and split into {len(parts)} part(s):")
        for part in parts:
            print(f"  - {part.name} ({part.stat().st_size / (1024*1024):.2f} MB)")
    elif args.command == "join":
        obj = load_split_joblib(args.file)
        print(f"Successfully loaded model object: {type(obj)}")


if __name__ == "__main__":
    main()
