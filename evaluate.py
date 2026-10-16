"""Frozen CLIP evaluation on OCL. Run python evaluate.py --help."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--top-n", type=int, default=10, help="Number of categories to retain")
    parser.add_argument("--model", default="ViT-B/32", help="Official CLIP name or local checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--missing-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--imagenet-box-mode", choices=["error", "native", "scale256"], default="error")
    parser.add_argument("--category-template", default="a photo of a {label}.")
    parser.add_argument("--attribute-template", default="a photo of a {label} object.")
    parser.add_argument("--affordance-template", default="an object associated with the action: {label}.")
    parser.add_argument("--affordance-text", choices=["word", "definition", "word-definition"],
                        default="word-definition")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/clip_test"))
    parser.add_argument("--save-scores", action="store_true", help="Save raw cosine scores and ground truth")
    parser.add_argument("--inspect-only", action="store_true", help="Write data coverage without loading CLIP")
    parser.add_argument("--max-samples", type=int, help="First N available objects; smoke testing only")
    parser.add_argument("--sources", nargs="+", choices=["COCO", "ImageNet", "from_web", "APY", "SUN"],
                        help="Optional source subset after fixing top-N categories")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.text_batch_size, args.top_n) < 1 or args.workers < 0:
        parser.error("Batch sizes/top-n must be positive; workers must be nonnegative")
    if any(k < 1 or k > args.top_n for k in args.top_k):
        parser.error("top-k values must be between 1 and top-n")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max-samples must be positive")
    return args


def load_model(name, device, cache=None):
    import clip
    model, preprocess = clip.load(name, device=device, jit=False,
                                  download_root=str(cache) if cache else None)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess


def encode_text(model, text_tokens, device, batch_size=64):
    import torch
    import torch.nn.functional as F
    result = {}
    with torch.inference_mode():
        for kind, tokens in text_tokens.items():
            features = [model.encode_text(batch.to(device)).float()
                        for batch in tokens.split(batch_size)]
            result[kind] = F.normalize(torch.cat(features), dim=-1)
    return result


def run_inference(model, loader, text_features, device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    scores = {kind: [] for kind in text_features}
    truth = {"category": [], "attribute": [], "affordance": []}
    sample_ids, image_paths = [], []
    boxes = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"CLIP rank {os.environ.get('RANK', '0')}"):
            features = F.normalize(model.encode_image(batch["image"].to(device)).float(), dim=-1)
            for kind, candidates in text_features.items():
                # Raw cosine scores preserve cross-image rankings for each attribute.
                # A softmax over labels would make different labels compete.
                scores[kind].append((features @ candidates.T).cpu().numpy())
            truth["category"].append(batch["category"].numpy())
            truth["attribute"].append(batch["attributes"].numpy())
            truth["affordance"].append(batch["affordances"].numpy())
            sample_ids.extend(batch["sample_id"])
            image_paths.extend(batch["image_path"])
            boxes.append(batch["box"].numpy())
    if not sample_ids:
        raise ValueError("Cannot evaluate an empty DataLoader")
    return ({kind: np.concatenate(values) for kind, values in scores.items()},
            {kind: np.concatenate(values) for kind, values in truth.items()},
            sample_ids, image_paths, np.concatenate(boxes))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def merge_predictions(parts, expected_samples):
    """Merge disjoint rank shards in dataset order; reject padded/missing rows."""
    import numpy as np
    indices = np.concatenate([part["indices"] for part in parts])
    if not np.array_equal(np.sort(indices), np.arange(expected_samples)):
        raise ValueError("Distributed predictions contain duplicate or missing sample indices")
    order = np.argsort(indices)
    scores = {kind: np.concatenate([part["scores"][kind] for part in parts])[order]
              for kind in parts[0]["scores"]}
    truth = {kind: np.concatenate([part["truth"][kind] for part in parts])[order]
             for kind in parts[0]["truth"]}
    ids = [value for part in parts for value in part["ids"]]
    paths = [value for part in parts for value in part["paths"]]
    boxes = np.concatenate([part["boxes"] for part in parts])[order]
    return scores, truth, [ids[i] for i in order], [paths[i] for i in order], boxes


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from dataset import OCLDataset
    from metrics import multilabel_map, topk_accuracy

    dataset = OCLDataset(args.data_dir, split=args.split, top_n=args.top_n,
                         missing_policy=args.missing_policy,
                         imagenet_box_mode=args.imagenet_box_mode)
    if args.sources:
        dataset.samples = [sample for sample in dataset.samples if sample["source"] in args.sources]
        dataset.coverage["source_filter"] = args.sources
        dataset.coverage["available_objects_after_source_filter"] = len(dataset.samples)
        if not dataset.samples:
            raise ValueError("No available samples after the source filter")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "data_coverage.json", dataset.coverage)
        with (args.output_dir / "skipped_samples.jsonl").open("w") as output:
            for sample in dataset.skipped:
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    if args.inspect_only:
        if rank == 0:
            print(json.dumps(dataset.coverage, ensure_ascii=False, indent=2))
        return
    import clip
    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
    dataset.ensure_boxes_resolved()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix the GPU environment or explicitly use --device cpu")
    if world_size > len(dataset):
        raise ValueError("Use no more processes than evaluated objects")
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if world_size > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world_size > 1:
        # Frozen independent forwards need no DDP gradient synchronization.
        torch.distributed.init_process_group(backend="gloo")
    indices = np.arange(rank, len(dataset), world_size)
    model, dataset.preprocess = load_model(args.model, device, args.model_cache)
    model_device = next(model.parameters()).device
    if model_device != device:
        raise RuntimeError(f"Model landed on {model_device}, expected {device}")
    device_record = {"rank": rank, "local_rank": local_rank, "device": str(device),
                     "model_device": str(model_device), "pid": os.getpid(),
                     "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable"))
                         if device.type == "cuda" else None}
    print(f"Rank binding: {device_record}", flush=True)
    rank_devices = [device_record]
    dataset.tokenize_labels(clip.tokenize, args.category_template, args.attribute_template,
                            args.affordance_template, args.affordance_text)
    # Stride shards avoid DistributedSampler padding and duplicate objects.
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    text_features = encode_text(model, dataset.text_tokens, device, args.text_batch_size)
    if world_size > 1:
        torch.distributed.barrier()
    start = time.perf_counter()
    scores, truth, sample_ids, image_paths, boxes = run_inference(model, loader, text_features, device)
    if world_size > 1:
        part = {"indices": indices, "scores": scores, "truth": truth, "ids": sample_ids,
                "paths": image_paths, "boxes": boxes, "device_record": device_record}
        gathered = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(part, gathered, dst=0)
        torch.distributed.destroy_process_group()
        if rank != 0:
            return
        scores, truth, sample_ids, image_paths, boxes = merge_predictions(gathered, len(dataset))
        rank_devices = [part["device_record"] for part in gathered]
        if device.type == "cuda" and len({part["model_device"] for part in rank_devices}) != world_size:
            raise RuntimeError("Distributed workers did not bind to distinct CUDA devices")
    inference_seconds = time.perf_counter() - start
    category = topk_accuracy(scores["category"], truth["category"], tuple(args.top_k))
    attribute = multilabel_map(scores["attribute"], truth["attribute"])
    affordance = multilabel_map(scores["affordance"], truth["affordance"])
    per_category = []
    for index, name in enumerate(dataset.categories):
        mask = truth["category"] == index
        values = topk_accuracy(scores["category"][mask], truth["category"][mask], tuple(args.top_k)) \
            if mask.any() else {f"top{k}": None for k in args.top_k}
        per_category.append({"category": name, "objects": int(mask.sum()), **values})
    versions = {}
    for name in ["torch", "torchvision", "numpy", "scikit-learn", "Pillow", "clip"]:
        versions[name] = importlib.metadata.version(name)
    coordinate_audit = dataset.root / "manifests/imagenet_bbox_geometry_audit.json"
    result = {
        "run": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv] if argv is None else ["evaluate.py", *argv],
            "config": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
            "device": str(device), "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "world_size": world_size, "batch_size_per_process": args.batch_size,
            "effective_batch_size": world_size * args.batch_size,
            "rank_devices": rank_devices,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(world_size)]
                if device.type == "cuda" and world_size > 1 else [],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": platform.python_version(), "versions": versions,
            "frozen_parameters": all(not p.requires_grad for p in model.parameters()),
            "model_dtype": str(getattr(model, "dtype", next(model.parameters()).dtype)),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ["dataset.py", "metrics.py", "evaluate.py"]},
            "preprocess": repr(dataset.preprocess), "inference_seconds": inference_seconds,
            "objects_per_second": len(sample_ids) / inference_seconds,
            "smoke_test": args.max_samples is not None,
            "full_selected_population": len(sample_ids) == dataset.coverage["expected_objects"],
            "imagenet_scaling_assumption": args.imagenet_box_mode == "scale256",
            "metric_scale": "fractions in [0,1]; multiply by 100 for percent",
            "ap_definition": "sklearn non-interpolated AP; exclude labels with no positives, matching OCL",
            "score_definition": "L2-normalized image features @ L2-normalized text features.T; no softmax",
        },
        "coverage": dataset.coverage, "evaluated_objects": len(sample_ids),
        "resource_sha256": dataset.resource_hashes, "prompts": dataset.prompts,
        "category": category, "attribute": attribute, "affordance": affordance,
        "per_category": per_category,
        "imagenet_box_audit_summary": json.loads(coordinate_audit.read_text())["summary"]
            if coordinate_audit.exists() else None,
    }
    write_json(args.output_dir / "metrics.json", result)
    with (args.output_dir / "per_label_ap.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["task", "label_index", "label", "definition", "positives", "AP"])
        for kind, labels, metric in [("attribute", dataset.attributes, attribute),
                                      ("affordance", dataset.affordances, affordance)]:
            for i, label in enumerate(labels):
                word = label if kind == "attribute" else label["word"][0]
                definition = "" if kind == "attribute" else label["define"].strip()
                writer.writerow([kind, i, word, definition, metric["positive_counts"][i],
                                 metric["per_label_ap"][i]])
    with (args.output_dir / "evaluated_samples.jsonl").open("w") as output:
        for sample_id, image_path, box in zip(sample_ids, image_paths, boxes):
            output.write(json.dumps({"sample_id": sample_id, "image": image_path,
                                     "crop_box": box.tolist()}) + "\n")
    if args.save_scores:
        np.savez_compressed(args.output_dir / "scores.npz", sample_ids=np.asarray(sample_ids),
                            **{f"{kind}_scores": value for kind, value in scores.items()},
                            **{f"{kind}_targets": value for kind, value in truth.items()})
    print(json.dumps({"evaluated_objects": len(sample_ids), "category": category,
                      "attribute_mAP": attribute["mAP"], "affordance_mAP": affordance["mAP"],
                      "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
