"""Compare OCL boxes with official ILSVRC localization annotations by filename.

Downloads are intentionally separate. The three archives and their MD5 hashes
are listed at https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.
Overlap validates geometry; OCL may annotate different instances/boundaries.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import xml.etree.ElementTree as ET


ARCHIVES = {
    "ILSVRC2012_bbox_train_v2.tar.gz": "9271167e2176350e65cfe4e546f14b17",
    "ILSVRC2012_bbox_train_dogs.tar.gz": "61ebd3cc0e4793899a841b6b27f3d6d8",
    "ILSVRC2012_bbox_val_v3.tgz": "f4cd18b5ea29fe6bbea62ec9c20d80f0",
}


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + \
        max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union else 0


def clip_box(box, width, height):
    return [max(0, min(width, box[0])), max(0, min(height, box[1])),
            max(0, min(width, box[2])), max(0, min(height, box[3]))]


def valid_box(box):
    return box[2] > box[0] and box[3] > box[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    wanted = {Path(row["image"]).stem for row in rows}
    references, provenance = {}, {}
    for filename, expected in ARCHIVES.items():
        path = args.annotations_dir / filename
        raw = path.read_bytes()
        actual = hashlib.md5(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Official annotation MD5 mismatch: {filename}")
        provenance[filename] = {"md5": actual, "sha256": hashlib.sha256(raw).hexdigest(),
                                "url": "https://image-net.org/data/ILSVRC/2012/" + filename}
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                name = Path(member.name).stem
                if not member.isfile() or name not in wanted:
                    continue
                xml = ET.fromstring(archive.extractfile(member).read())
                boxes = [[int(obj.findtext("bndbox/" + key))
                          for key in ["xmin", "ymin", "xmax", "ymax"]]
                         for obj in xml.findall("object")]
                references[name] = {
                    "size": [int(xml.findtext("size/width")), int(xml.findtext("size/height"))],
                    "boxes": boxes,
                }
        print(f"Read {filename}; matched {len(references)} images", flush=True)
    results = []
    for row in rows:
        reference = references.get(Path(row["image"]).stem)
        if reference is None or not reference["boxes"]:
            continue
        width, height = row["image_size"]
        raw = row["bbox_xyxy"]
        native = clip_box(raw, width, height)
        scaled = clip_box([raw[0] * width / 256, raw[1] * height / 256,
                           raw[2] * width / 256, raw[3] * height / 256], width, height)
        results.append({
            "sample_id": row["sample_id"], "image": row["image"],
            "image_size": row["image_size"], "xml_size": reference["size"],
            "raw_box": raw, "native_box": native, "scaled_box": scaled,
            "official_boxes": reference["boxes"],
            "native_valid": valid_box(native), "scaled_valid": valid_box(scaled),
            "native_iou": max(iou(native, box) for box in reference["boxes"]),
            "scaled_iou": max(iou(scaled, box) for box in reference["boxes"]),
        })
    if not results:
        raise ValueError("No filenames matched the official localization annotations")
    summary = {
        "objects": len(rows), "matched_objects": len(results), "matched_images": len(references),
        "dimension_matches": sum(row["image_size"] == row["xml_size"] for row in results),
        "native_valid_boxes": sum(row["native_valid"] for row in results),
        "scaled_valid_boxes": sum(row["scaled_valid"] for row in results),
        "native_median_iou": statistics.median(row["native_iou"] for row in results),
        "scaled_median_iou": statistics.median(row["scaled_iou"] for row in results),
        "scaled_iou_ge_0_9": sum(row["scaled_iou"] >= .9 for row in results),
        "scaled_better_than_native": sum(row["scaled_iou"] > row["native_iou"] for row in results),
        "unmatched_sample_ids": [row["sample_id"] for row in rows
                                 if Path(row["image"]).stem not in references],
    }
    result = {"summary": summary, "provenance": provenance, "objects": results,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "interpretation": "Geometry cross-check only; OCL and ILSVRC may annotate different instances or boundaries. This is not an official OCL coordinate specification."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
