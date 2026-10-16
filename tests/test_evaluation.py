"""Small, deterministic checks for annotation alignment and evaluation math."""

import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset import OCLDataset
from evaluate import encode_text, load_model, merge_predictions, run_inference
from metrics import multilabel_map, topk_accuracy


class MetricsTests(unittest.TestCase):
    def test_topk_counts_objects(self):
        scores = np.array([[1, 0]] * 5)
        result = topk_accuracy(scores, np.array([0, 0, 0, 0, 1]), (1, 2))
        self.assertAlmostEqual(result["top1"], 4 / 5)
        self.assertEqual(result["top2"], 1)

    def test_ap_matches_hand_calculation_and_excludes_absent_labels(self):
        result = multilabel_map([[.9, 1], [.8, 0], [.7, .5], [.2, .2]],
                                [[1, 0], [0, 0], [1, 0], [0, 0]])
        self.assertAlmostEqual(result["mAP"], (1 + 2 / 3) / 2)
        self.assertEqual(result["per_label_ap"][1], None)
        self.assertEqual(result["excluded_no_positive_labels"], [1])

    def test_ap_handles_ties_as_one_threshold(self):
        result = multilabel_map([[.5], [.5]], [[1], [0]])
        self.assertEqual(result["mAP"], .5)

    def test_all_absent_labels_are_undefined_and_json_safe(self):
        result = multilabel_map([[.5, .8]], [[0, 0]])
        self.assertIsNone(result["mAP"])
        json.dumps(result, allow_nan=False)

    def test_invalid_metric_inputs_fail(self):
        with self.assertRaises(ValueError):
            topk_accuracy([[1, 0]], [0], (3,))
        with self.assertRaises(ValueError):
            multilabel_map([[float("nan")]], [[1]])
        with self.assertRaises(ValueError):
            multilabel_map([[.5]], [[-1]])


def make_fixture(root):
    resources = root / "resources"
    resources.mkdir()
    for kind, data in {
        "object": ["cat", "dog", "bird"], "attribute": ["red", "furry"],
        "affordance": [{"word": ["eat"], "define": "take in solid food"},
                       {"word": ["eat"], "define": "eat a meal"}],
    }.items():
        (resources / f"OCL_class_{kind}.json").write_text(json.dumps(data))
    records = [
        {"name": "a.png", "source": "COCO", "objects": [
            {"obj": "cat", "box": [1, 2, 5, 8], "attr": [1], "aff": [0, 1]},
            {"obj": "dog", "box": [-1, -2, 4, 5], "attr": [0], "aff": []},
        ]},
        {"name": "b.png", "source": "COCO", "objects": [
            {"obj": "cat", "attr": [0, 1], "aff": [1]},
            {"obj": "cat", "attr": [], "aff": []},
        ]},
    ]
    (resources / "OCL_annot_test.pkl").write_bytes(pickle.dumps(records))
    for name in ["a.png", "b.png"]:
        Image.new("RGB", (8, 10), color=(255, 0, 0)).save(root / name)
    return records


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = make_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_multiple_objects_crop_and_labels_remain_aligned(self):
        dataset = OCLDataset(self.root, preprocess=lambda image: image.size, top_n=2)
        self.assertEqual(dataset.categories, ["cat", "dog"])
        self.assertEqual(len(dataset), 4)
        cat, dog = dataset[0], dataset[1]
        self.assertEqual(cat["image"], (4, 6))
        self.assertEqual(dog["image"], (4, 5))
        self.assertEqual(cat["category"], 0)
        self.assertEqual(dog["category"], 1)
        self.assertEqual(cat["attributes"].tolist(), [0, 1])
        self.assertEqual(cat["affordances"].tolist(), [1, 1])
        self.assertEqual(dog["affordances"].tolist(), [0, 0])
        self.assertEqual(dataset[2]["image"], (8, 10))

    def test_missing_images_do_not_change_top_category_selection(self):
        (self.root / "b.png").unlink()
        with self.assertRaises(FileNotFoundError):
            OCLDataset(self.root, top_n=1)
        dataset = OCLDataset(self.root, top_n=1, missing_policy="skip")
        self.assertEqual(dataset.categories, ["cat"])
        self.assertEqual(dataset.coverage["expected_objects"], 3)
        self.assertEqual(dataset.coverage["available_objects"], 1)
        self.assertEqual(dataset.coverage["skipped_objects"], 2)

    def test_qa_placeholder_is_excluded_but_replaced_image_is_not(self):
        exclusion = {"a.png": {"reason": "placeholder", "sha256": hashlib.sha256(
            (self.root / "a.png").read_bytes()).hexdigest()}}
        (self.root / "resources/image_qa_exclusions.json").write_text(json.dumps(exclusion))
        dataset = OCLDataset(self.root, top_n=2, missing_policy="skip")
        self.assertEqual(dataset.coverage["skipped_objects"], 2)
        Image.new("RGB", (8, 10), color="blue").save(self.root / "a.png")
        dataset = OCLDataset(self.root, top_n=2)
        self.assertEqual(len(dataset), 4)

    def test_imagenet_coordinate_assumption_requires_explicit_choice(self):
        self.records[0]["source"] = "ImageNet"
        (self.root / "resources/OCL_annot_test.pkl").write_bytes(pickle.dumps(self.records))
        with self.assertRaisesRegex(ValueError, "coordinates still need confirmation"):
            OCLDataset(self.root, top_n=2).ensure_boxes_resolved()
        dataset = OCLDataset(self.root, preprocess=lambda image: image.size,
                             top_n=2, imagenet_box_mode="scale256")
        Image.new("RGB", (512, 256)).save(self.root / "a.png")
        dataset.samples[0]["box"] = [128, 64, 256, 128]
        self.assertEqual(dataset[0]["image"], (256, 64))
        self.assertEqual(dataset[0]["box"].tolist(), [256, 64, 512, 128])

    def test_reads_compressed_official_annotation_and_preserves_verb_senses(self):
        path = self.root / "resources/OCL_annot_test.pkl"
        raw = path.read_bytes()
        with tarfile.open(self.root / "resources/OCL_annot.tgz", "w:gz") as archive:
            info = tarfile.TarInfo(path.name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        path.unlink()
        dataset = OCLDataset(self.root, top_n=2)
        dataset.tokenize_labels(lambda texts: torch.zeros(len(texts), 77, dtype=torch.long))
        self.assertEqual(dataset.text_tokens["affordance"].shape, (2, 77))
        self.assertNotEqual(dataset.prompts["affordance"][0], dataset.prompts["affordance"][1])


class InferenceTests(unittest.TestCase):
    def test_distributed_merge_restores_order_and_rejects_padded_samples(self):
        def part(indices):
            values = np.array(indices)[:, None]
            return {"indices": np.array(indices), "scores": {"category": values},
                    "truth": {"category": np.array(indices)},
                    "ids": [str(i) for i in indices], "paths": [f"{i}.png" for i in indices],
                    "boxes": np.tile(values, (1, 4))}
        scores, truth, ids, paths, boxes = merge_predictions([part([0, 2, 4]), part([1, 3])], 5)
        self.assertEqual(ids, ["0", "1", "2", "3", "4"])
        np.testing.assert_array_equal(truth["category"], np.arange(5))
        np.testing.assert_array_equal(scores["category"][:, 0], np.arange(5))
        self.assertEqual(paths[-1], "4.png")
        with self.assertRaises(ValueError):
            merge_predictions([part([0, 2, 4]), part([1, 3, 0])], 5)

    def test_model_is_frozen_and_in_eval_mode(self):
        model = torch.nn.Linear(2, 2)
        fake_clip = types.SimpleNamespace(load=lambda *args, **kwargs: (model, "preprocess"))
        with patch.dict("sys.modules", {"clip": fake_clip}):
            loaded, transform = load_model("fake", "cpu")
        self.assertFalse(loaded.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        self.assertEqual(transform, "preprocess")

    def test_forward_uses_no_grad_and_keeps_short_final_batch(self):
        class Model(torch.nn.Module):
            def encode_image(inner_self, images):
                self.assertFalse(torch.is_grad_enabled())
                return images

            def encode_text(inner_self, tokens):
                self.assertFalse(torch.is_grad_enabled())
                return tokens.float()

        model = Model()
        text_features = encode_text(model, {kind: torch.eye(2) for kind in
                                            ["category", "attribute", "affordance"]}, "cpu", 1)
        rows = [{"image": torch.tensor([2., 0.]), "category": 0 if i < 4 else 1,
                 "attributes": torch.tensor([1., 0.]), "affordances": torch.tensor([0., 1.]),
                 "sample_id": str(i), "image_path": "example.png", "box": torch.zeros(4)}
                for i in range(5)]
        scores, truth, ids, paths, boxes = run_inference(
            model, DataLoader(rows, batch_size=4), text_features, "cpu")
        self.assertEqual(ids, ["0", "1", "2", "3", "4"])
        self.assertEqual(scores["category"].shape, (5, 2))
        np.testing.assert_allclose(scores["category"], [[1, 0]] * 5)
        self.assertAlmostEqual(topk_accuracy(scores["category"], truth["category"], (1,))["top1"], .8)
        self.assertEqual(boxes.shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
