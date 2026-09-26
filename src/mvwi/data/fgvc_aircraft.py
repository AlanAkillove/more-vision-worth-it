"""FGVC-Aircraft dataset access: official 2013b splits + variant classification."""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

SPLIT_FILES = {
    "train": "images_variant_train.txt",
    "val": "images_variant_val.txt",
    "test": "images_variant_test.txt",
}
SPLITS = ["train", "val", "test"]


def load_variants(data_dir):
    """Return ordered variant names (class index = position). Names may contain spaces."""
    path = Path(data_dir) / "variants.txt"
    names = [ln.rstrip("\n").strip() for ln in open(path, encoding="utf-8")]
    names = [n for n in names if n != ""]
    return names


def name_to_idx(variant_names):
    return {n: i for i, n in enumerate(variant_names)}


def load_split(data_dir, split):
    """Parse official split file -> list[(sample_id, variant_name)]. id maps to images/<id>.jpg.

    Annotation line format is ``<id> <variant name (may contain spaces)>``; split on
    the first space only.
    """
    path = Path(data_dir) / SPLIT_FILES[split]
    recs = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if ln == "":
            continue
        img_id, name = ln.split(" ", 1)
        recs.append((img_id, name.strip()))
    return recs


def image_filename(sample_id):
    return f"{sample_id}.jpg"


def image_path(images_dir, sample_id):
    return str(Path(images_dir) / image_filename(sample_id))


def orig_size(images_dir, sample_id):
    with Image.open(image_path(images_dir, sample_id)) as im:
        return im.width, im.height


class AircraftDataset:
    """Map-style dataset yielding a normalized tensor at a single target resolution.

    Uses the shared canonical preprocessing pipeline so every resolution of a given
    image observes the identical field of view (plan §3).
    """

    def __init__(self, records, images_dir, label_idx, target_size, preproc):
        from mvwi.preprocess.canonical import preprocess_to_tensor

        self.records = records  # list[(sample_id, variant_name)]
        self.images_dir = images_dir
        self.label_idx = label_idx
        self.target_size = target_size
        self._to_tensor = preprocess_to_tensor
        self.preproc = preproc

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import torch

        sid, name = self.records[i]
        path = image_path(self.images_dir, sid)
        with Image.open(path) as im:
            im = im.convert("RGB")
            ow, oh = im.width, im.height
            tensor = self._to_tensor(im, self.preproc, self.target_size)
        label = int(self.label_idx[name])
        return sid, label, ow, oh, tensor


def collate_fn(batch):
    import torch

    sids = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ws = torch.tensor([b[2] for b in batch], dtype=torch.int32)
    hs = torch.tensor([b[3] for b in batch], dtype=torch.int32)
    imgs = torch.stack([b[4] for b in batch], dim=0)
    return sids, labels, ws, hs, imgs
