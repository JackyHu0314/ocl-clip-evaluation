#!/usr/bin/env python3
"""Prepare the official OCL test subset without downloading full source datasets."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tarfile
import time
import urllib.request


OCL_COMMIT = "4ad86d929513961ff018fc53db5c7b40910ca88a"
RESOURCE_URL = (
    "https://raw.githubusercontent.com/silicx/ObjectConceptLearning/"
    f"{OCL_COMMIT}/data/resources/"
)
WEB_ARCHIVE_ID = "1TMAAvgRh7EmNvQu40lRIVD594LcDH2zy"
RESOURCE_NAMES = [
    "OCL_annot.tgz", "OCL_class_object.json", "OCL_class_attribute.json",
    "OCL_class_affordance.json", "url_to_filename.json",
]


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected pickle class: {module}.{name}")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def check_space(root, reserve_gb):
    free = shutil.disk_usage(root).free
    if free < reserve_gb * 1024**3:
        raise OSError(f"Space guard: only {free / 1024**3:.2f} GiB free")


def transfer(url, destination, reserve_gb=3, image=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(3):
        try:
            check_space(destination.parent, reserve_gb)
            request = urllib.request.Request(url, headers={"User-Agent": "OCL-stage0-data/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                length = response.headers.get("Content-Length")
                if image and length and int(length) > 64 * 1024**2:
                    raise ValueError("Image exceeds 64 MiB limit")
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024**2):
                        received += len(chunk)
                        if image and received > 64 * 1024**2:
                            raise ValueError("Image exceeds 64 MiB limit")
                        check_space(destination.parent, reserve_gb)
                        output.write(chunk)
                if length and received != int(length):
                    raise OSError(f"Incomplete response: {received}/{length} bytes")
            if image:
                image_info(temporary)
            temporary.replace(destination)
            return
        except Exception as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def image_info(path):
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        dimensions = list(image.size)
        image_format = image.format
    return {
        "size": dimensions, "format": image_format, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_metadata(root, reserve_gb):
    resources = root / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in RESOURCE_NAMES:
        target = resources / name
        if not target.is_file():
            print(f"metadata: {name}", flush=True)
            transfer(RESOURCE_URL + name, target, reserve_gb)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    with tarfile.open(resources / "OCL_annot.tgz", "r:gz") as archive:
        for split in ("train", "val", "test"):
            name = f"OCL_annot_{split}.pkl"
            target = resources / name
            if not target.is_file():
                member = archive.getmember(name)
                target.write_bytes(archive.extractfile(member).read())
    provenance = {
        "repository": "https://github.com/silicx/ObjectConceptLearning",
        "commit": OCL_COMMIT, "resource_sha256": hashes,
        "web_archive_id": WEB_ARCHIVE_ID,
    }
    atomic_json(resources / "provenance.json", provenance)
    return resources, provenance


def select_objects(resources, split, top_k):
    classes = json.loads((resources / "OCL_class_object.json").read_text())
    attributes = json.loads((resources / "OCL_class_attribute.json").read_text())
    affordances = json.loads((resources / "OCL_class_affordance.json").read_text())
    class_ids = {name: index for index, name in enumerate(classes)}
    with (resources / f"OCL_annot_{split}.pkl").open("rb") as stream:
        images = PlainUnpickler(stream).load()
    counts = Counter(obj["obj"] for image in images for obj in image["objects"])
    # Count annotations before checking image availability; never change the task
    # population to fit the sources that happen to download successfully.
    selected = sorted(counts, key=lambda name: (-counts[name], class_ids[name]))[:top_k]
    selected_ids = {name: index for index, name in enumerate(selected)}
    objects = []
    for image_index, image in enumerate(images):
        relative_path = Path(image["name"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe annotated path: {relative_path}")
        for object_index, obj in enumerate(image["objects"]):
            category = obj["obj"]
            if category not in selected_ids:
                continue
            if not all(0 <= index < len(attributes) for index in obj["attr"]):
                raise ValueError("Invalid attribute index")
            if not all(0 <= index < len(affordances) for index in obj["aff"]):
                raise ValueError("Invalid affordance index")
            box = obj.get("box")
            if box is not None and len(box) != 4:
                raise ValueError("Invalid object box")
            objects.append({
                "sample_id": f"{split}:{image_index}:{object_index}",
                "split": split, "image_index": image_index, "object_index": object_index,
                "image": image["name"], "source": image["source"],
                "category": category, "category_id": class_ids[category],
                "category_index": selected_ids[category], "bbox_xyxy": box,
                "attribute_indices": obj["attr"], "affordance_indices": obj["aff"],
            })
    return objects, selected, counts, {
        "annotation_images": len(images), "annotation_objects": sum(counts.values()),
        "annotation_categories": len(counts), "attribute_count": len(attributes),
        "affordance_count": len(affordances),
    }


def extract_web_archive(archive_path, root, wanted, reserve_gb):
    extracted = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if "from_web" not in parts:
                continue
            relative = Path(*parts[parts.index("from_web"):]).as_posix()
            if relative not in wanted or (root / relative).exists():
                continue
            check_space(root, reserve_gb)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
            extracted += 1
    print(f"web archive: extracted {extracted} required images", flush=True)


def download_images(root, images, options):
    url_map = {name: url for url, name in json.loads(
        (root / "resources/url_to_filename.json").read_text())}
    web_wanted = {name for name, source in images.items() if source == "from_web"}
    exclusions_path = root / "resources/image_qa_exclusions.json"
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    if options.fetch_web_archive and web_wanted and not options.verify_only:
        import gdown

        archive_path = root / "downloads/from_web.tgz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            check_space(root, options.reserve_gb)
            print("web archive: downloading the official supplemental images", flush=True)
            result = gdown.download(id=WEB_ARCHIVE_ID, output=str(archive_path),
                                    quiet=True, resume=True)
            if result is None:
                raise RuntimeError("Official web archive download did not finish")
        extract_web_archive(archive_path, root, web_wanted, options.reserve_gb)

    def download_one(item):
        name, source = item
        path = root / name
        if path.is_file():
            try:
                details = image_info(path)
                exclusion = exclusions.get(name)
                if exclusion and details["sha256"] == exclusion.get("sha256"):
                    return name, {"status": "invalid", "error": exclusion["reason"], **details}
                return name, {"status": "available", **details}
            except Exception as error:
                return name, {"status": "invalid", "error": str(error)}
        if name in exclusions:
            return name, {"status": "invalid", "error": exclusions[name]["reason"]}
        if options.verify_only:
            return name, {"status": "missing", "url": url_map.get(name),
                          "error": "Original image unavailable; see download and recovery logs"}
        if source == "ImageNet":
            if options.imagenet_root:
                source_path = options.imagenet_root / Path(name).relative_to("ImageNet")
                if source_path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.symlink_to(source_path.resolve())
                    return name, {"status": "available", **image_info(path)}
            return name, {"status": "missing", "error": "Original ImageNet file required"}
        if source == "COCO":
            # Same official bucket as images.cocodataset.org, with a valid HTTPS
            # hostname. The dotted bucket's custom HTTPS hostname mismatches TLS.
            url = "https://s3.amazonaws.com/images.cocodataset.org/" + name.removeprefix("COCO/")
        elif source == "from_web" and name in url_map:
            url = url_map[name]
        else:
            return name, {"status": "missing", "error": f"No official direct URL for {source}"}
        try:
            transfer(url, path, options.reserve_gb, image=True)
            return name, {"status": "available", "url": url, **image_info(path)}
        except Exception as error:
            return name, {"status": "missing", "url": url, "error": str(error)}

    results = {}
    log_path = root / "manifests/download_log.jsonl"
    with log_path.open("a") as log, ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(download_one, item) for item in images.items()]
        for future in as_completed(futures):
            name, status = future.result()
            results[name] = status
            log.write(json.dumps({"image": name, **status}, ensure_ascii=False) + "\n")
            if len(results) % 100 == 0 or len(results) == len(images):
                log.flush()
                available = sum(value["status"] == "available" for value in results.values())
                print(f"images: checked {len(results)}/{len(images)}, available {available}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-only", action="store_true",
                        help="Decode local images and rebuild manifests without retrying downloads")
    parser.add_argument("--fetch-web-archive", action="store_true")
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gb", type=float, default=3)
    options = parser.parse_args()
    if options.top_k < 1 or options.workers < 1 or options.reserve_gb < 0:
        parser.error("Invalid positive-count or disk-reserve argument")
    root = options.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(exist_ok=True)
    resources, provenance = load_metadata(root, options.reserve_gb)
    objects, selected, counts, population = select_objects(resources, options.split, options.top_k)
    images = {row["image"]: row["source"] for row in objects}
    prefix = f"{options.split}_top{options.top_k}"
    write_jsonl(root / f"manifests/{prefix}.jsonl", objects)
    print(json.dumps({"categories": {name: counts[name] for name in selected},
                      "objects": len(objects), "images": len(images),
                      "sources": dict(Counter(images.values()))}, indent=2), flush=True)
    if not (options.download or options.verify_only):
        return
    statuses = download_images(root, images, options)
    available_objects, invalid_boxes = [], []
    for row in objects:
        status = statuses[row["image"]]
        if status["status"] != "available":
            continue
        width, height = status["size"]
        box = row["bbox_xyxy"] or [0, 0, width, height]
        clipped = [max(0, min(width, box[0])), max(0, min(height, box[1])),
                   max(0, min(width, box[2])), max(0, min(height, box[3]))]
        prepared = {**row, "image_size": [width, height], "bbox_xyxy_clipped": clipped,
                    "bbox_coordinate_review_required": row["source"] == "ImageNet"}
        if row["source"] == "ImageNet":
            # Preserve every available ImageNet annotation. Its source frame is
            # unresolved, so do not silently interpret the raw box as native
            # pixels or discard it when that interpretation is degenerate.
            prepared["bbox_xyxy_clipped"] = None
            available_objects.append(prepared)
        elif clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            invalid_boxes.append(prepared)
        else:
            available_objects.append(prepared)
    write_jsonl(root / f"manifests/{prefix}.available.jsonl", available_objects)
    missing = [{"image": name, "source": images[name], **status}
               for name, status in statuses.items() if status["status"] != "available"]
    write_jsonl(root / f"manifests/{prefix}.missing_images.jsonl", missing)
    write_jsonl(root / f"manifests/{prefix}.invalid_boxes.jsonl", invalid_boxes)
    write_jsonl(root / f"manifests/{prefix}.coordinate_review.jsonl",
                [row for row in available_objects if row["bbox_coordinate_review_required"]])
    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
        "split": options.split, "top_k": options.top_k,
        "selection_rule": "Object annotation counts in the requested official split, before downloads",
        **population, "selected_categories": selected,
        "expected_category_counts": {name: counts[name] for name in selected},
        "expected_images": len(images), "expected_objects": len(objects),
        "available_images": sum(status["status"] == "available" for status in statuses.values()),
        "available_objects": len(available_objects), "missing_images": len(missing),
        "invalid_images": sum(status["status"] == "invalid" for status in statuses.values()),
        "invalid_boxes": len(invalid_boxes),
        "available_category_counts": dict(Counter(row["category"] for row in available_objects)),
        "missing_sources": dict(Counter(row["source"] for row in missing)),
        "image_bytes": sum(status.get("bytes", 0) for status in statuses.values()),
        "bbox_coordinate_review_objects": sum(row["source"] == "ImageNet" for row in available_objects),
        "bbox_coordinate_note": "ImageNet annotation coordinates cluster around a 256-sized frame; native-image scaling needs confirmation. Raw boxes are unchanged.",
        "complete": not missing and not invalid_boxes,
    }
    atomic_json(root / f"manifests/{prefix}.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["complete"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
