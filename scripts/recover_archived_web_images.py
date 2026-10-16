#!/usr/bin/env python3
"""Recover unavailable original image URLs from Internet Archive captures."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image
import requests

from prepare_ocl_data import check_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve-gb", type=float, default=3)
    options = parser.parse_args()
    root = options.data_dir.resolve()
    prefix = f"{options.split}_top{options.top_k}"
    rows = [json.loads(line) for line in (root / f"manifests/{prefix}.jsonl").read_text().splitlines()]
    boxes = defaultdict(list)
    for row in rows:
        if row["source"] == "from_web" and row["bbox_xyxy"]:
            boxes[row["image"]].append(row["bbox_xyxy"])
    urls = {name: url for url, name in json.loads((root / "resources/url_to_filename.json").read_text())}
    wanted = sorted({row["image"] for row in rows
                     if row["source"] == "from_web" and not (root / row["image"]).is_file()})

    def recover(name):
        original = urls.get(name)
        attempts = []
        if not original:
            return {"image": name, "status": "missing", "error": "No official URL"}
        with requests.Session() as session:
            parameters = {"url": original, "output": "json", "filter": ["statuscode:200", "mimetype:image/.*"],
                          "fl": "timestamp,original", "collapse": "digest", "limit": 5, "to": "2023"}
            captures = None
            for attempt in range(3):
                try:
                    response = session.get("https://web.archive.org/cdx/search/cdx", params=parameters, timeout=(10, 35))
                    response.raise_for_status()
                    captures = response.json()
                    break
                except Exception as error:
                    attempts.append({"stage": "capture_lookup", "error": str(error)})
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
            if captures is None:
                return {"image": name, "status": "missing", "original_url": original, "attempts": attempts}
            for timestamp, captured_url in sorted(captures[1:], reverse=True):
                archived_url = f"https://web.archive.org/web/{timestamp}id_/{captured_url}"
                try:
                    with session.get(archived_url, stream=True, timeout=(10, 35)) as response:
                        response.raise_for_status()
                        payload = bytearray()
                        for chunk in response.iter_content(1024**2):
                            payload.extend(chunk)
                            if len(payload) > 64 * 1024**2:
                                raise ValueError("Archived image exceeds 64 MiB limit")
                    payload = bytes(payload)
                    with Image.open(io.BytesIO(payload)) as image:
                        image.load()
                        width, height = image.size
                    for x1, y1, x2, y2 in boxes[name]:
                        if min(width, x2) <= max(0, x1) or min(height, y2) <= max(0, y1):
                            raise ValueError("Archived image cannot contain the annotated object")
                    check_space(root, options.reserve_gb)
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".part")
                    temporary.write_bytes(payload)
                    temporary.replace(target)
                    return {"image": name, "status": "recovered", "original_url": original,
                            "archived_url": archived_url, "timestamp": timestamp,
                            "size": [width, height], "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(), "attempts": attempts}
                except Exception as error:
                    attempts.append({"stage": "archived_image", "url": archived_url, "error": str(error)})
        return {"image": name, "status": "missing", "original_url": original,
                "captures_found": max(0, len(captures) - 1), "attempts": attempts}

    print(f"Archive recovery: checking {len(wanted)} original image URLs", flush=True)
    total, recovered = 0, 0
    log_path = root / f"manifests/{prefix}.archive_recovery.jsonl"
    with log_path.open("a") as log, ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(recover, name) for name in wanted]
        for future in as_completed(futures):
            result = future.result()
            total += 1
            recovered += result["status"] == "recovered"
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            if total % 5 == 0 or total == len(wanted):
                print(f"Archive recovery: checked {total}/{len(wanted)}, recovered {recovered}", flush=True)
    print(json.dumps({"checked": total, "recovered": recovered}, indent=2), flush=True)


if __name__ == "__main__":
    main()
