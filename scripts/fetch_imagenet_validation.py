#!/usr/bin/env python3
"""Stream the official ImageNet validation tar, saving only OCL's required files.

The archive is never stored on disk. Its tar headers also provide a reusable
byte-offset index, so a repeated run can fetch any missing selected image directly.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import urllib.request

from PIL import Image

from prepare_ocl_data import atomic_json, check_space, image_info, select_objects, write_jsonl


ARCHIVE_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
ARCHIVE_BYTES = 6744924160


def scan_header_range(start, end):
    """Scan aligned tar blocks in a bounded byte range without saving the tar."""
    request = urllib.request.Request(ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"})
    entries = {}
    with urllib.request.urlopen(request, timeout=90) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise OSError("Server did not honor a validation archive scan range")
        consumed = 0
        while block := response.read(4 * 1024**2):
            position = block.find(b"ILSVRC2012_val_")
            while position != -1:
                absolute = start + consumed + position
                if absolute % 512 == 0 and position + 512 <= len(block):
                    try:
                        member = tarfile.TarInfo.frombuf(block[position:position + 512], "utf-8", "strict")
                        name = Path(member.name).name
                        if member.isfile() and re.fullmatch(r"ILSVRC2012_val_\d{8}\.JPEG", name):
                            entries[name] = {"name": member.name, "offset": absolute + 512,
                                             "size": member.size}
                    except (tarfile.HeaderError, ValueError, UnicodeError):
                        pass
                position = block.find(b"ILSVRC2012_val_", position + 1)
            consumed += len(block)
        if consumed != end - start + 1:
            raise OSError("Incomplete archive scan range")
    return entries


def fetch_indexed_image(root, relative, entry, reserve_gb):
    start, end = entry["offset"] - 512, entry["offset"] + entry["size"] - 1
    request = urllib.request.Request(ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_BYTES}":
            raise OSError("Server did not honor the validation image byte range")
        payload = response.read(entry["size"] + 513)
    header = tarfile.TarInfo.frombuf(payload[:512], "utf-8", "strict")
    if header.name != entry["name"] or header.size != entry["size"]:
        raise ValueError("Validation filename does not match its tar header")
    return validate_and_save(root, relative, payload[512:], entry, reserve_gb)


def validate_and_save(root, relative, payload, entry, reserve_gb):
    check_space(root, reserve_gb)
    if len(payload) != entry["size"]:
        raise OSError("Validation image is incomplete")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        dimensions, image_format = list(image.size), image.format
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return {
        "image": relative, "status": "downloaded", "size": dimensions,
        "format": image_format, "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "archive_url": ARCHIVE_URL, "offset": entry["offset"],
        "tar_header_filename": entry["name"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reserve-gb", type=float, default=3)
    parser.add_argument("--workers", type=int, default=16)
    options = parser.parse_args()
    root = options.data_dir.resolve()
    objects, _, _, _ = select_objects(root / "resources", options.split, options.top_k)
    paths = sorted({row["image"] for row in objects
                    if row["source"] == "ImageNet" and row["image"].startswith("ImageNet/val/")})
    wanted = {Path(name).name: name for name in paths}
    results = []
    missing = {}
    for name, relative in wanted.items():
        target = root / relative
        if target.is_file():
            results.append({"image": relative, "status": "cached", **image_info(target)})
        else:
            missing[name] = relative
    index_path = root / "resources/imagenet_validation_offsets.json"
    prefix = f"{options.split}_top{options.top_k}"
    if missing and index_path.exists():
        # A previous successful scan makes full streaming unnecessary on reruns.
        index = json.loads(index_path.read_text())
        if set(missing).issubset(index["entries"]):
            for name, relative in missing.items():
                results.append(fetch_indexed_image(root, relative, index["entries"][name], options.reserve_gb))
            missing = {}
    if missing:
        print(f"ImageNet validation: scanning bounded archive ranges for {len(missing)} required images", flush=True)
        entries = json.loads(index_path.read_text())["entries"] if index_path.exists() else {}
        chunk_size = 64 * 1024**2
        chunks = [(start, min(start + chunk_size, ARCHIVE_BYTES) - 1)
                  for start in range(0, ARCHIVE_BYTES, chunk_size)]
        with ThreadPoolExecutor(max_workers=options.workers) as scanners, ThreadPoolExecutor(max_workers=8) as downloads:
            futures = [scanners.submit(scan_header_range, start, end) for start, end in chunks]
            image_futures = []
            completed = 0
            for future in as_completed(futures):
                found = future.result()
                entries.update(found)
                completed += 1
                for name in list(missing):
                    if name in entries:
                        image_futures.append(downloads.submit(fetch_indexed_image, root, missing.pop(name),
                                                               entries[name], options.reserve_gb))
                if completed % 5 == 0 or not missing:
                    atomic_json(index_path, {"archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES,
                                             "entries": entries})
                    print(f"ImageNet validation: scanned {completed}/{len(chunks)} ranges, indexed {len(paths) - len(missing)}/{len(paths)} needed images", flush=True)
                if not missing:
                    for pending in futures:
                        pending.cancel()
                    break
            for future in as_completed(image_futures):
                results.append(future.result())
            atomic_json(index_path, {"archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES,
                                     "entries": entries})
    write_jsonl(root / f"manifests/imagenet_validation_{prefix}.jsonl", results)
    if missing:
        raise RuntimeError(f"Required validation images were not found: {sorted(missing)}")
    print(json.dumps({"required": len(paths), "available": len(results),
                      "bytes": sum(row["bytes"] for row in results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
