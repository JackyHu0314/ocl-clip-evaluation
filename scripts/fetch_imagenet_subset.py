#!/usr/bin/env python3
"""Fetch required original ImageNet training files using validated HTTP byte ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.request
import urllib.parse

from PIL import Image

from prepare_ocl_data import (
    atomic_json, check_space, image_info, select_objects, write_jsonl,
)


ARCHIVE_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
ARCHIVE_BYTES = 147897477120
INDEX_REPO = "AlienKevin/imagenet-1k"
INDEX_COMMIT = "7864e8a21416c48304bba492d2be7cfec52ffa6d"
INDEX_URL = (
    f"https://huggingface.co/datasets/{INDEX_REPO}/resolve/{INDEX_COMMIT}/"
    "imagenet_train_index_overall.json?download=true"
)


def index_key(relative_path):
    path = Path(relative_path)
    return f"{path.parent.name}.tar/{path.name}"


def load_required_index(root, required, prefix):
    cache_path = root / f"resources/imagenet_index_{prefix}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if set(cached["entries"]) == required:
            return cached
    print("ImageNet: reading the filename/byte-offset index", flush=True)
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import requests

    # The viewer's Parquet export stores the same filename/offset/size index in
    # about 28 MB instead of the original 110 MB JSON. No image mirror is used.
    viewer_url = "https://datasets-server.huggingface.co/parquet?" + urllib.parse.urlencode(
        {"dataset": INDEX_REPO})
    with urllib.request.urlopen(viewer_url, timeout=45) as response:
        exports = json.load(response)
    if exports.get("partial") or exports.get("pending") or exports.get("failed"):
        raise RuntimeError("A complete filename index export is not available")
    entries, index_files = {}, []
    for shard in exports["parquet_files"]:
        if shard["config"] != "default" or shard["split"] != "train":
            continue
        last_error = None
        for attempt in range(3):
            try:
                response = requests.get(shard["url"], timeout=90)
                response.raise_for_status()
                payload = response.content
                if len(payload) != shard["size"]:
                    raise OSError("Incomplete filename-index response")
                break
            except Exception as error:
                last_error = error
                if attempt == 2:
                    raise RuntimeError(f"Cannot read the filename index: {last_error}")
                time.sleep(attempt + 1)
        table = pq.read_table(io.BytesIO(payload), columns=["name", "offset", "size"])
        rows = table.filter(pc.is_in(table["name"], value_set=pa.array(sorted(required)))).to_pylist()
        entries.update({entry["name"]: entry for entry in rows})
        index_files.append({"url": shard["url"], "sha256": hashlib.sha256(payload).hexdigest()})
    if set(entries) != required:
        raise ValueError(f"Index is missing required originals: {sorted(required - set(entries))}")
    cached = {
        "repository": INDEX_REPO, "commit": INDEX_COMMIT, "index_url": INDEX_URL,
        "index_exports": index_files,
        "archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES,
        "entries": entries,
    }
    atomic_json(cache_path, cached)
    print(f"ImageNet: located all {len(entries)} required filenames", flush=True)
    return cached


def fetch_original(root, relative, entry, reserve_gb):
    target = root / relative
    if target.is_file():
        return {"image": relative, "status": "cached", **image_info(target)}
    offset, size = int(entry["offset"]), int(entry["size"])
    if not (512 <= offset < ARCHIVE_BYTES and 0 < size < 64 * 1024**2):
        raise ValueError(f"Unexpected index range for {relative}")
    start, end = offset - 512, offset + size - 1
    expected_range = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
    last_error = None
    for attempt in range(3):
        try:
            check_space(root, reserve_gb)
            request = urllib.request.Request(ARCHIVE_URL, headers={
                "Range": f"bytes={start}-{end}", "User-Agent": "OCL-stage0-data/1.0",
            })
            with urllib.request.urlopen(request, timeout=45) as response:
                actual_range = response.headers.get("Content-Range")
                if response.status != 206 or actual_range != expected_range:
                    raise OSError(f"Unexpected HTTP range: {response.status} {actual_range}")
                data = response.read(size + 513)
            if len(data) != size + 512:
                raise OSError(f"Incomplete range: {len(data)}/{size + 512}")
            header = tarfile.TarInfo.frombuf(data[:512], "utf-8", "strict")
            if Path(header.name).name != Path(relative).name or header.size != size:
                raise ValueError(f"Tar header does not match the annotated filename: {header.name}")
            if not header.isfile():
                raise ValueError("Index points to a non-file tar member")
            image_bytes = data[512:]
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                dimensions, image_format = list(image.size), image.format
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            check_space(root, reserve_gb)
            temporary.write_bytes(image_bytes)
            temporary.replace(target)
            return {
                "image": relative, "status": "downloaded", "bytes": size,
                "size": dimensions, "format": image_format,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "archive_url": ARCHIVE_URL, "range": expected_range,
                "tar_header_filename": header.name,
            }
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    return {"image": relative, "status": "missing", "error": str(last_error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gb", type=float, default=3)
    options = parser.parse_args()
    root = options.data_dir.resolve()
    objects, _, _, _ = select_objects(root / "resources", options.split, options.top_k)
    wanted = sorted({row["image"] for row in objects
                     if row["source"] == "ImageNet" and row["image"].startswith("ImageNet/train/")})
    prefix = f"{options.split}_top{options.top_k}"
    catalog = load_required_index(root, {index_key(name) for name in wanted}, prefix)
    results = []
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(fetch_original, root, name, catalog["entries"][index_key(name)],
                               options.reserve_gb) for name in wanted]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0 or len(results) == len(wanted):
                succeeded = sum(row["status"] != "missing" for row in results)
                print(f"ImageNet: checked {len(results)}/{len(wanted)}, available {succeeded}", flush=True)
    write_jsonl(root / f"manifests/imagenet_{prefix}.jsonl", results)
    failed = [row for row in results if row["status"] == "missing"]
    print(json.dumps({"source_split": "ImageNet/train", "required": len(wanted),
                      "available": len(wanted) - len(failed),
                      "bytes": sum(row.get("bytes", 0) for row in results),
                      "failed": failed}, indent=2), flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
