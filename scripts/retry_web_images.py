#!/usr/bin/env python3
"""Retry original web-image paths over HTTPS and ordinary browser headers."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PIL import Image
import requests

from prepare_ocl_data import check_space, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reserve-gb", type=float, default=3)
    options = parser.parse_args()
    root = options.data_dir.resolve()
    prefix = f"{options.split}_top{options.top_k}"
    objects = [json.loads(line) for line in (root / f"manifests/{prefix}.jsonl").read_text().splitlines()]
    boxes = defaultdict(list)
    for row in objects:
        if row["source"] == "from_web" and row["bbox_xyxy"]:
            boxes[row["image"]].append(row["bbox_xyxy"])
    urls = {name: url for url, name in json.loads((root / "resources/url_to_filename.json").read_text())}
    wanted = {row["image"] for row in objects
              if row["source"] == "from_web" and not (root / row["image"]).is_file()}
    invalid_path = root / f"manifests/{prefix}.invalid_boxes.jsonl"
    if invalid_path.exists():
        wanted.update(json.loads(line)["image"] for line in invalid_path.read_text().splitlines())

    def recover(name):
        original = urls.get(name)
        if not original:
            return {"image": name, "status": "missing", "error": "No official URL"}
        parsed = urlsplit(original)
        secure = urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
        variants = list(dict.fromkeys([secure, original]))
        attempts = []
        for use_proxy in (True, False):
            with requests.Session() as session:
                session.trust_env = use_proxy
                for url in variants:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                        "Referer": f"{urlsplit(url).scheme}://{parsed.netloc}/",
                    }
                    try:
                        check_space(root, options.reserve_gb)
                        with session.get(url, headers=headers, stream=True, timeout=(6, 15)) as response:
                            response.raise_for_status()
                            data = bytearray()
                            for chunk in response.iter_content(1024**2):
                                data.extend(chunk)
                                if len(data) > 64 * 1024**2:
                                    raise ValueError("Image exceeds 64 MiB limit")
                        payload = bytes(data)
                        with Image.open(io.BytesIO(payload)) as image:
                            image.load()
                            width, height = image.size
                        for x1, y1, x2, y2 in boxes[name]:
                            if min(width, x2) <= max(0, x1) or min(height, y2) <= max(0, y1):
                                raise ValueError(f"Downloaded image {width}x{height} cannot contain the annotated object")
                        target = root / name
                        if target.exists():
                            quarantine = root / "quarantine" / target.name
                            quarantine.parent.mkdir(exist_ok=True)
                            target.replace(quarantine)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(target.suffix + ".part")
                        temporary.write_bytes(payload)
                        temporary.replace(target)
                        return {"image": name, "status": "recovered", "original_url": original,
                                "download_url": url, "proxy": use_proxy, "size": [width, height],
                                "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                                "previous_attempts": attempts}
                    except Exception as error:
                        attempts.append({"url": url, "proxy": use_proxy, "error": str(error)})
        return {"image": name, "status": "missing", "original_url": original, "attempts": attempts}

    results = []
    print(f"Web recovery: checking {len(wanted)} original image paths", flush=True)
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        for future in as_completed([pool.submit(recover, name) for name in sorted(wanted)]):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(wanted):
                recovered = sum(row["status"] == "recovered" for row in results)
                print(f"Web recovery: checked {len(results)}/{len(wanted)}, recovered {recovered}", flush=True)
    write_jsonl(root / f"manifests/{prefix}.web_recovery.jsonl", results)
    print(json.dumps({"checked": len(results), "recovered": sum(row["status"] == "recovered" for row in results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
