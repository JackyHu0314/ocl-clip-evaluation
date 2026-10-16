"""Read official OCL annotations as one object per PyTorch sample."""

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import tarfile

from PIL import Image
import torch
from torch.utils.data import Dataset


class PlainUnpickler(pickle.Unpickler):
    """OCL annotations contain built-in containers, not executable classes."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected pickle class: {module}.{name}")


class OCLDataset(Dataset):
    def __init__(self, data_dir, preprocess=None, split="test", top_n=10,
                 missing_policy="error", imagenet_box_mode="error"):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if missing_policy not in {"error", "skip"}:
            raise ValueError("missing_policy must be error or skip")
        if imagenet_box_mode not in {"error", "native", "scale256"}:
            raise ValueError("imagenet_box_mode must be error, native or scale256")
        if top_n < 1:
            raise ValueError("top_n must be positive")
        self.root = Path(data_dir).expanduser().resolve()
        self.preprocess = preprocess
        self.split = split
        self.imagenet_box_mode = imagenet_box_mode
        self.text_tokens = {}
        self.prompts = {}
        resources = self.root / "resources"
        self.resource_hashes = {}

        def read_labels(kind):
            path = resources / f"OCL_class_{kind}.json"
            raw = path.read_bytes()
            self.resource_hashes[path.name] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)

        self.all_categories = read_labels("object")
        self.attributes = read_labels("attribute")
        self.affordances = read_labels("affordance")
        annotation_name = f"OCL_annot_{split}.pkl"
        annotation_path = resources / annotation_name
        if annotation_path.is_file():
            raw = annotation_path.read_bytes()
        else:
            with tarfile.open(resources / "OCL_annot.tgz", "r:gz") as archive:
                member = next(m for m in archive.getmembers()
                              if Path(m.name).name == annotation_name)
                raw = archive.extractfile(member).read()
        self.resource_hashes[annotation_name] = hashlib.sha256(raw).hexdigest()
        images = PlainUnpickler(io.BytesIO(raw)).load()
        counts = Counter(obj["obj"] for image in images for obj in image["objects"])
        category_ids = {name: i for i, name in enumerate(self.all_categories)}
        if not set(counts).issubset(category_ids):
            raise ValueError("Annotation contains a category absent from the label table")
        if top_n > len(counts):
            raise ValueError(f"Requested {top_n} categories; this split has {len(counts)}")
        # Select BEFORE checking images, so unavailable files cannot change the task.
        self.categories = sorted(counts, key=lambda x: (-counts[x], category_ids[x]))[:top_n]
        selected_ids = {name: i for i, name in enumerate(self.categories)}
        self.category_counts = {name: counts[name] for name in self.categories}
        exclusion_path = resources / "image_qa_exclusions.json"
        exclusions = json.loads(exclusion_path.read_text()) if exclusion_path.exists() else {}
        self.samples = []
        self.skipped = []
        availability = {}
        selected_images = set()
        for image_index, image in enumerate(images):
            for object_index, obj in enumerate(image["objects"]):
                if obj["obj"] not in selected_ids:
                    continue
                relative = Path(image["name"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe image path: {relative}")
                for key, labels in [("attr", self.attributes), ("aff", self.affordances)]:
                    if any(type(i) is not int or not 0 <= i < len(labels) for i in obj[key]):
                        raise ValueError(f"Invalid {key} indices in {relative}")
                sample = {
                    "sample_id": f"{split}:{image_index}:{object_index}",
                    "image": image["name"], "source": image["source"],
                    "category": selected_ids[obj["obj"]], "category_name": obj["obj"],
                    "box": obj.get("box"), "attr": obj["attr"], "aff": obj["aff"],
                }
                selected_images.add(image["name"])
                if image["name"] not in availability:
                    path = self.root / relative
                    issue = None if path.is_file() else "missing image"
                    exclusion = exclusions.get(image["name"])
                    if issue is None and exclusion:
                        expected_hash = exclusion.get("sha256")
                        if not expected_hash or hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash:
                            issue = exclusion.get("reason", "image excluded by data QA")
                    availability[image["name"]] = issue
                issue = availability[image["name"]]
                if issue:
                    self.skipped.append({**sample, "reason": issue})
                else:
                    self.samples.append(sample)
        self.coverage = {
            "split": split,
            "selection_rule": "Object counts in the full requested split; ties use official category ID",
            "selected_categories": self.categories,
            "expected_category_counts": self.category_counts,
            "expected_images": len(selected_images),
            "expected_objects": sum(self.category_counts.values()),
            "available_images": sum(issue is None for issue in availability.values()),
            "available_objects": len(self.samples),
            "skipped_images": sum(issue is not None for issue in availability.values()),
            "skipped_objects": len(self.skipped),
            "imagenet_objects_with_boxes": sum(s["source"] == "ImageNet" and s["box"] is not None
                                              for s in self.samples),
            "missing_policy": missing_policy,
            "imagenet_box_mode": imagenet_box_mode,
            "availability_check": "file presence and QA exclusion hash; decode/crop failures raise during iteration",
        }
        if self.skipped and missing_policy == "error":
            first = self.skipped[0]
            raise FileNotFoundError(
                f"{len(self.skipped)} selected objects have unavailable/QA-excluded images. "
                f"First: {first['image']} ({first['reason']}). "
                "Restore images, or explicitly use --missing-policy skip for a partial evaluation."
            )
        if not self.samples:
            raise ValueError("No available objects in the selected data")

    def ensure_boxes_resolved(self):
        if self.imagenet_box_mode == "error" and any(
            sample["source"] == "ImageNet" and sample["box"] is not None for sample in self.samples
        ):
            raise ValueError(
                "ImageNet box coordinates still need confirmation. Choose "
                "--imagenet-box-mode native or scale256 explicitly after reviewing the convention. "
                "scale256 is an experimental assumption, not a verified official convention."
            )

    def tokenize_labels(self, tokenizer,
                        category_template="a photo of a {label}.",
                        attribute_template="a photo of a {label} object.",
                        affordance_template="an object associated with the action: {label}.",
                        affordance_text="word-definition"):
        if affordance_text not in {"word", "definition", "word-definition"}:
            raise ValueError("Unknown affordance text mode")
        affordance_labels = []
        for label in self.affordances:
            word, definition = label["word"][0], label["define"].strip()
            affordance_labels.append({"word": word, "definition": definition,
                                      "word-definition": f"{word}: {definition}"}[affordance_text])
        for kind, labels, template in [
            ("category", self.categories, category_template),
            ("attribute", self.attributes, attribute_template),
            ("affordance", affordance_labels, affordance_template),
        ]:
            if "{label}" not in template:
                raise ValueError(f"{kind} template must include {{label}}")
            self.prompts[kind] = [template.format(label=label) for label in labels]
            # Tokenize constant candidate labels once, not once per image.
            self.text_tokens[kind] = tokenizer(self.prompts[kind])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.preprocess is None:
            raise ValueError("Set dataset.preprocess to the transform returned by clip.load")
        with Image.open(self.root / sample["image"]) as original:
            image = original.convert("RGB")
        width, height = image.size
        box = sample["box"]
        if box is None:
            box = [0, 0, width, height]
        else:
            if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
                raise ValueError(f"Invalid box for {sample['sample_id']}: {box}")
            if sample["source"] == "ImageNet":
                if self.imagenet_box_mode == "error":
                    self.ensure_boxes_resolved()
                if self.imagenet_box_mode == "scale256":
                    box = [box[0] * width / 256, box[1] * height / 256,
                           box[2] * width / 256, box[3] * height / 256]
        box = [max(0, min(width, box[0])), max(0, min(height, box[1])),
               max(0, min(width, box[2])), max(0, min(height, box[3]))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Degenerate box for {sample['sample_id']}: {box}")
        crop = image.crop(box)
        if min(crop.size) < 1:
            raise ValueError(f"Empty crop for {sample['sample_id']}: {box}")
        attributes = torch.zeros(len(self.attributes), dtype=torch.float32)
        affordances = torch.zeros(len(self.affordances), dtype=torch.float32)
        attributes[sample["attr"]] = 1
        affordances[sample["aff"]] = 1
        return {
            "image": self.preprocess(crop), "category": sample["category"],
            "attributes": attributes, "affordances": affordances,
            "sample_id": sample["sample_id"], "image_path": sample["image"],
            "box": torch.tensor(box, dtype=torch.float32),
        }
