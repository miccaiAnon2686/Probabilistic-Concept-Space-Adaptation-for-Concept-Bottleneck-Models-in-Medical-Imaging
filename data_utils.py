# data_utils.py
#
# Add drop-in support for a dataset named "sparwious" (also accepts the alias "spawrious")
# WITHOUT reorganizing files on disk.
#
# It mirrors the Waterbirds integration:
#   - get_data("sparwious_train"/"sparwious_val"/"sparwious_test") -> Dataset
#   - get_targets_only("sparwious_*") -> list[int]
#   - get_filenames_only("sparwious_*") -> list[str]
#   - get_groups_only("sparwious_*") -> list[int] where group_id = y * 2 + place
#
# Assumptions for Sparwious:
#   * Root:   <sparwious_root> (configured via config/paths.yaml or env)
#   * Images: <sparwious_root>/images/...
#   * CSV:    <sparwious_root>/metadata.csv
#   * CSV columns: img_id, img_filename, y, split, place
#                  split_map = {0: train, 1: val, 2: test}
#
# Also keeps the Waterbirds support you already use (per-split dirs with filtered metadata.csv).

import os
import csv
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms, models
import yaml

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "config")
DEFAULT_PATHS_FILE = os.path.join(CONFIG_ROOT, "paths.yaml")


# ------------------------------
# Paths & label files
# ------------------------------

def _expand_path_value(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_path_value(v) for v in value]
    return value


def _default_dataset_roots() -> dict:
    return {
        # Generic dataset fallbacks
        "imagenet_train": os.path.join(DATA_ROOT, "imagenet", "train"),
        "imagenet_val": os.path.join(DATA_ROOT, "imagenet", "val"),
        "cub_train": os.path.join(DATA_ROOT, "CUB", "train"),
        "cub_val": os.path.join(DATA_ROOT, "CUB", "test"),

        # Waterbirds / Spawrious (optional in this release)
        "waterbirds_train": os.path.join(DATA_ROOT, "waterbirds", "splits", "train"),
        "waterbirds_val": os.path.join(DATA_ROOT, "waterbirds", "splits", "val"),
        "waterbirds_test": os.path.join(DATA_ROOT, "waterbirds", "splits", "test"),
        "sparwious_root": os.path.join(DATA_ROOT, "spawrious", "all"),

        # WBC / dermatology datasets used in the paper
        "pbc_root": os.path.join(DATA_ROOT, "wbc", "images", "pbc"),
        "raabinwbc_testA_root": os.path.join(DATA_ROOT, "wbc", "images", "raabin_testA"),
        "raabinwbc_testA_csv": os.path.join(DATA_ROOT, "wbc", "RaabinWBCTestA.csv"),
        "scirep_images_root": os.path.join(DATA_ROOT, "wbc", "images", "scirep"),
        "scirep_csv": os.path.join(DATA_ROOT, "wbc", "scirep_test.csv"),

        "fitz17k_csv": os.path.join(DATA_ROOT, "skincon", "fitzpatrick17k.csv"),
        "fitz17k_images": os.path.join(DATA_ROOT, "skincon", "fitzpatrick17k_images"),
        "fitzskin_root": os.path.join(DATA_ROOT, "skincon"),
        "fitzskin_train_csv": os.path.join(DATA_ROOT, "skincon", "fitz_v3_train.csv"),
        "fitzskin_test_csv": os.path.join(DATA_ROOT, "skincon", "fitz_v3_test.csv"),
        "ddi_images_root": os.path.join(DATA_ROOT, "skincon"),
        "ddi_test_csv": os.path.join(DATA_ROOT, "skincon", "ddi_v1_test.csv"),
        "ddi_label_2_class": os.path.join(DATA_ROOT, "skincon", "label_2classes.yml"),

        # Additional benchmarks (optional)
        "rimone_hosp_train": os.path.join(DATA_ROOT, "rimone_hosp", "training_set"),
        "rimone_hosp_val": os.path.join(DATA_ROOT, "rimone_hosp", "test_set"),
        "drishti_gs": os.path.join(DATA_ROOT, "drishti_gs", "test", "Images"),
        "drishti_gs_train": os.path.join(DATA_ROOT, "drishti_gs", "train", "Images"),
        "drishti_gs_val": os.path.join(DATA_ROOT, "drishti_gs", "test", "Images"),
        "drishti_gs_test": os.path.join(DATA_ROOT, "drishti_gs", "test", "Images"),
        "ham_metadata": os.path.join(DATA_ROOT, "ham", "HAM10000_metadata.csv"),
        "ham_image_roots": [
            os.path.join(DATA_ROOT, "ham", "ham10000_images_part_1"),
            os.path.join(DATA_ROOT, "ham", "ham10000_images_part_2"),
            os.path.join(DATA_ROOT, "ham", "HAM10000_images_part_1"),
            os.path.join(DATA_ROOT, "ham", "HAM10000_images_part_2"),
        ],
        "isic19_metadata": os.path.join(DATA_ROOT, "isic19", "ISIC_2019_Training_Metadata.csv"),
        "isic19_images_root": os.path.join(DATA_ROOT, "isic19", "ISIC_2019_Training_Input"),
        "isic19_groundtruth": os.path.join(DATA_ROOT, "isic19", "ISIC_2019_Training_GroundTruth.csv"),
    }


def _load_dataset_roots_with_overrides(default_roots: dict) -> dict:
    roots = dict(default_roots)
    paths_file = os.environ.get("LF_CBM_PATHS_FILE", DEFAULT_PATHS_FILE)

    if os.path.exists(paths_file):
        with open(paths_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Path config must be a mapping: {paths_file}")
        unknown = sorted(set(cfg.keys()) - set(roots.keys()))
        if unknown:
            raise ValueError(f"Unknown keys in {paths_file}: {unknown}")
        roots.update(cfg)

    # Per-key env override, e.g. LF_CBM_PATH_PBC_ROOT=/path/to/pbc/images
    for k in list(roots.keys()):
        env_key = f"LF_CBM_PATH_{k.upper()}"
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            roots[k] = env_val

    return {k: _expand_path_value(v) for k, v in roots.items()}


DATASET_ROOTS = _load_dataset_roots_with_overrides(_default_dataset_roots())

LABEL_FILES = {
    "places365": os.path.join(DATA_ROOT, "categories_places365_clean.txt"),
    "imagenet":  os.path.join(DATA_ROOT, "imagenet_classes.txt"),
    "cifar10":   os.path.join(DATA_ROOT, "cifar10_classes.txt"),
    "cifar100":  os.path.join(DATA_ROOT, "cifar100_classes.txt"),
    "cub":       os.path.join(DATA_ROOT, "cub_classes.txt"),
    "waterbirds": os.path.join(DATA_ROOT, "waterbirds_classes.txt"),
    # Provided by you:
    "sparwious": os.path.join(DATA_ROOT, "sparwious_classes.txt"),
    # Alias key so either spelling works when code does LABEL_FILES[args.dataset]
    "spawrious": os.path.join(DATA_ROOT, "sparwious_classes.txt"),
}

# Add a label-file entry so args.dataset='cifar10c' works
LABEL_FILES.update({
    "cifar10c": os.path.join(DATA_ROOT, "cifar10_classes.txt"),   # 10 lines with CIFAR-10 class names
})

LABEL_FILES.update({
    # Keep the same 5-class order everywhere:
    "pbc":       os.path.join(DATA_ROOT, "wbc_classes.txt"),        # Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil
    "raabinwbc": os.path.join(DATA_ROOT, "wbc_classes.txt"),        # Same order and spelling/casing
})

LABEL_YAML = os.path.join(DATA_ROOT, "wbc", "label.yml")

# Optional global exclusion list (e.g., to drop HAM overlaps from ISIC).
EXCLUDE_IDS: set = set()


def set_exclude_ids(ids):
    """Register a set of image IDs (without extensions) to skip when loading derm datasets."""
    global EXCLUDE_IDS
    EXCLUDE_IDS = {
        (str(i) or "").strip().split(".")[0].lower()
        for i in (ids or [])
        if (str(i) or "").strip()
    }


# Default config for CIFAR-10-C; will be overwritten by set_cifar10c_options(...)
CIFAR10C_CONFIG = {
    "root": os.path.join(DATA_ROOT, "cifar10c"),
    "corruptions": ["gaussian_noise", "shot_noise"],
    "severities": [1, 2, 3, 4, 5],
}
LABEL_FILES.update({
    "rimone_hosp": os.path.join(DATA_ROOT, "rimone_hosp_classes.txt"),
})

LABEL_FILES.update({
    "drishti_gs": os.path.join(DATA_ROOT, "rimone_hosp_classes.txt"),
})

# HAM10000 / ISIC2019 (shared 7-class set; ISIC SCC dropped)
LABEL_FILES.update({
    "ham": os.path.join(DATA_ROOT, "ham_isic_classes.txt"),
    "isic19": os.path.join(DATA_ROOT, "ham_isic_classes.txt"),
    "isic19_4": os.path.join(DATA_ROOT, "ham_isic_classes_4.txt"),
    "ham_4": os.path.join(DATA_ROOT, "ham_isic_classes_4.txt"),
    "isic19_3": os.path.join(DATA_ROOT, "ham_isic_classes_3.txt"),
    "ham_3": os.path.join(DATA_ROOT, "ham_isic_classes_3.txt"),
})


def set_cifar10c_options(root=None, corruptions=None, severities=None):
    if root is not None: CIFAR10C_CONFIG["root"] = root
    if corruptions is not None: CIFAR10C_CONFIG["corruptions"] = list(corruptions)
    if severities is not None: CIFAR10C_CONFIG["severities"] = list(severities)


SPLIT_MAP = {0: "train", 1: "val", 2: "test"}  # for reference

def _load_pbc_attr_vocab(attr_yaml_path: str, attr_bin_yaml_path: str):
    with open(attr_yaml_path, "r") as f:
        attr_map = yaml.safe_load(f)  # dict: attribute -> {value: idx}

    with open(attr_bin_yaml_path, "r") as f:
        bin_list = yaml.safe_load(f)  # list like ['cell_size=big', ...]
    # Normalize: replace spaces with '_' so keys match your CSV normalization below
    concept_names = []
    for item in bin_list:
        key = item.strip().lstrip("-").strip()   # 'cell_size=big'
        key = key.replace(" ", "_")              # 'light blue' -> 'light_blue'
        concept_names.append(key)
    return attr_map, concept_names  # dict, list[str]


# ------------------------------
# Common helpers
# ------------------------------

def get_concepts_only(dataset_name: str):
    ds = _as_dataset(dataset_name)
    if isinstance(ds, PBCConceptsDataset):
        return [c.tolist() for c in ds.concepts]
    # if isinstance(ds, Fitzpatrick17kDataset):
    #     return [c.tolist() for c in ds.concepts]
    if isinstance(ds, FitzSkinV3Dataset):
        return [c.tolist() for c in ds.concepts]
    if isinstance(ds, DDiV1Dataset):
        return [c.tolist() for c in ds.concepts]
    if isinstance(ds, RaabinWBCSplitDataset):
        return [c.tolist() for c in ds.concepts]
    raise AttributeError(f"{dataset_name} does not expose concept annotations")

def get_concept_names(dataset_name: str):
    ds = _as_dataset(dataset_name)
    if isinstance(ds, PBCConceptsDataset):
        return list(ds.concept_vocab)
    # if isinstance(ds, Fitzpatrick17kDataset):
    #     return [f"fitzpatrick={i}" for i in range(1,7)]
    if isinstance(ds, FitzSkinV3Dataset):
        return list(ds.concept_vocab)  # CSV header order
    if isinstance(ds, DDiV1Dataset):
        return list(ds.concept_vocab)
    if isinstance(ds, RaabinWBCSplitDataset):
        return list(ds.concept_vocab)
    return None




def _canon_dataset_name(name: str) -> str:
    """Normalize dataset name spelling (sparwious <-> spawrious)."""
    if name.startswith("spawrious"):
        return name.replace("spawrious", "sparwious", 1)
    return name

def _lstrip_slash(p: str) -> str:
    return p[1:] if p.startswith("/") else p

def _sniffed_reader(fp):
    """CSV/TSV tolerant DictReader."""
    sample = fp.read(4096)
    fp.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return csv.DictReader(fp, dialect=dialect)


def _deterministic_split(key: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """
    Deterministic split chooser using SHA1 hash of a key (e.g., image_id).
    Returns 'train', 'val', or 'test'.
    """
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < train_ratio:
        return "train"
    if v < train_ratio + val_ratio:
        return "val"
    return "test"


def _load_isic19_groundtruth(gt_csv: str, columns: List[str]) -> dict:
    """
    Load ISIC 2019 ground-truth one-hot columns -> class index.
    columns should align with desired class order (e.g., AK,BCC,BKL,DF,MEL,NV,VASC).
    """
    lookup = {}
    with open(gt_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ["image"] + columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"ISIC19 ground truth missing columns: {missing}")
        for row in reader:
            img = (row.get("image") or "").strip()
            if not img:
                continue
            scores = [float(row.get(c, 0.0)) for c in columns]
            if not scores:
                lookup[img] = -1
                continue
            best = int(np.argmax(scores))
            # If all zeros (e.g., SCC or UNK), mark as unlabeled (-1) but keep for alignment.
            if scores[best] <= 0.0:
                lookup[img] = -1
                continue
            lookup[img] = best
    return lookup


def _resolve_image_path(img_id: str, roots: List[Path], exts=(".jpg", ".jpeg", ".png", ".JPG", ".PNG")):
    """
    Returns (path, rel_name) if found under any root with any extension; else (None, None).
    """
    for r in roots:
        for ext in exts:
            cand = r / f"{img_id}{ext}"
            if cand.exists():
                return cand, f"{img_id}{ext}"
    return None, None


# ------------------------------
# Preprocess (used elsewhere)
# ------------------------------

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=target_mean, std=target_std),
    ])


# ------------------------------
# WATERBIRDS dataset (unchanged)
# ------------------------------

class WaterbirdsDataset(Dataset):
    """
    Expects split directories with per-split metadata.csv and images under <split>/<y>/<img_filename>.
    """
    classes = ["landbird", "waterbird"]
    class_to_idx = {"landbird": 0, "waterbird": 1}

    def __init__(self, root: str, transform=None, metadata_name: str = "metadata.csv"):
        self.root = Path(root)
        self.transform = transform
        self.meta = self.root / metadata_name
        if not self.meta.exists():
            raise FileNotFoundError(f"metadata.csv not found at {self.meta}")

        self.filenames: List[str] = []
        self.targets:   List[int] = []
        self.places:    List[int] = []
        self.groups:    List[int] = []  # (1 - y) * 2 + (1 - place)
        self.paths:     List[Path] = []

        with self.meta.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            need = {"img_filename", "y", "place"}
            miss = need - set(reader.fieldnames or [])
            if miss:
                raise ValueError(f"metadata.csv missing columns: {sorted(miss)}")

            for row in reader:
                try:
                    y = int(row["y"]); place = int(row["place"])
                except Exception:
                    continue
                rel = _lstrip_slash(row["img_filename"])
                abs_path = self.root / str(y) / rel

                self.filenames.append(row["img_filename"])
                self.targets.append(y)
                self.places.append(place)
                self.groups.append((1 - y) * 2 + (1 - place))
                self.paths.append(abs_path)

        assert len(self.paths) == len(self.targets) == len(self.filenames) == len(self.groups)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        with Image.open(img_path).convert("RGB") as im:
            if self.transform is not None:
                im = self.transform(im)
        return im, self.targets[idx]


# ------------------------------
# SPARWIOUS dataset (no reorg)
# ------------------------------

class SparwiousDataset(Dataset):
    """
    Loads directly from a single root directory using metadata.csv with split column.
    - root points to: <sparwious_root>
    - metadata.csv there contains: img_filename, y, split, place
    - Images live at: root / <img_filename>  (e.g., 'images/008365.png')
    - Group ID (per your earlier code): group_id = y * 2 + place
    """
    def __init__(self, root: str, split_code: int, transform=None, metadata_name: str = "metadata.csv"):
        assert split_code in (0, 1, 2), "split_code must be 0(train),1(val),2(test)"
        self.root = Path(root)
        self.transform = transform
        self.meta = self.root / metadata_name
        if not self.meta.exists():
            raise FileNotFoundError(f"metadata.csv not found at {self.meta}")

        self.filenames: List[str] = []
        self.targets:   List[int] = []
        self.places:    List[int] = []
        self.groups:    List[int] = []  # y * 2 + place
        self.paths:     List[Path] = []

        with self.meta.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            need = {"img_filename", "y", "split", "place"}
            miss = need - set(reader.fieldnames or [])
            if miss:
                raise ValueError(f"metadata.csv missing columns: {sorted(miss)}")

            for row in reader:
                try:
                    y = int(row["y"]); place = int(row["place"]); sp = int(row["split"])
                except Exception:
                    continue
                if sp != split_code:
                    continue
                rel = _lstrip_slash(row["img_filename"])  # e.g., 'images/000123.png'
                abs_path = self.root / rel

                self.filenames.append(row["img_filename"])
                self.targets.append(y)
                self.places.append(place)
                self.groups.append(y * 2 + place)
                self.paths.append(abs_path)

        assert len(self.paths) == len(self.targets) == len(self.filenames) == len(self.groups)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        with Image.open(img_path).convert("RGB") as im:
            if self.transform is not None:
                im = self.transform(im)
        return im, self.targets[idx]

class DermCSVImageDataset(Dataset):
    """
    Generic CSV-backed dermatology dataset.

    Args:
      images_roots: list of base dirs to search for <image_id>.<ext>
      metadata_csv: CSV with at least an id column and optional label/split columns
      split: 'train' | 'val' | 'test' (hash-based if split_col missing)
      id_col: column name for image id (without extension)
      label_col: column name for label (string). If None/missing, labels default to -1.
      label_map: dict label -> int index. If provided and require_label=True, rows without a mapped label are dropped.
      split_col: optional column with split name; if absent, a deterministic hash split is used.
      allow_labels: optional set of labels to keep (others dropped if present).
      require_label: if True, skip rows with missing/unmapped labels.
    """
    def __init__(
        self,
        images_roots: List[str],
        metadata_csv: str,
        split: str,
        id_col: str = "image_id",
        label_col: Optional[str] = "dx",
        label_map: Optional[dict] = None,
        split_col: Optional[str] = None,
        allow_labels: Optional[set] = None,
        require_label: bool = False,
        transform=None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        label_lookup: Optional[dict] = None,
    ):
        assert split in ("train", "val", "test")
        self.images_roots = [Path(r) for r in images_roots]
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.id_col = id_col
        self.label_col = label_col
        self.label_map = label_map or {}
        self.split_col = split_col
        self.allow_labels = allow_labels
        self.require_label = require_label
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.label_lookup = label_lookup or {}

        self.filenames: List[str] = []
        self.targets:   List[int] = []
        self.paths:     List[Path] = []

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"metadata.csv not found at {self.metadata_csv}")

        with self.metadata_csv.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            if self.id_col not in (reader.fieldnames or []):
                raise ValueError(f"{self.metadata_csv} missing id_col '{self.id_col}'")

            for row in reader:
                img_id = (row.get(self.id_col) or "").strip()
                if not img_id:
                    continue
                norm_id = img_id.split(".")[0].lower()
                if norm_id in EXCLUDE_IDS:
                    continue

                # split selection
                if self.split_col and row.get(self.split_col, "").strip():
                    split_val = row[self.split_col].strip().lower()
                else:
                    split_val = _deterministic_split(img_id, self.train_ratio, self.val_ratio)
                if split_val != self.split:
                    continue

                # labels: prefer explicit lookup (e.g., ISIC ground-truth), otherwise map from label_col
                label_idx = -1
                if self.label_lookup:
                    label_idx = self.label_lookup.get(img_id, -1)
                    if self.require_label and label_idx < 0:
                        continue
                else:
                    label_name = (row.get(self.label_col) or "").strip() if self.label_col else ""
                    if label_name:
                        if self.allow_labels and label_name not in self.allow_labels:
                            continue
                        if label_name in self.label_map:
                            label_idx = self.label_map[label_name]
                        elif self.require_label:
                            continue
                    elif self.require_label:
                        continue

                path, rel_name = _resolve_image_path(img_id, self.images_roots)
                if path is None:
                    continue

                self.filenames.append(rel_name or img_id)
                self.targets.append(label_idx)
                self.paths.append(path)

        assert len(self.paths) == len(self.targets) == len(self.filenames)
        self.classes = list(self.label_map.keys())

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        with Image.open(img_path).convert("RGB") as im:
            if self.transform is not None:
                im = self.transform(im)
        return im, self.targets[idx]

class FitzSkinV3Dataset(Dataset):
    """
    Fitzpatrick17k SkinCon v3

    • CSV schema (v3):
        img_name, label, <binary concept columns...>, path
      - Concepts are EVERY column strictly between 'label' and 'path' (CSV order preserved).
      - 'label' is expected to be 'benign' or 'malignant'.

    • Splits:
      - If split_name in {'train','val'} and csv_path ends with '_train.csv':
          Create a deterministic ~val_ratio holdout from the train CSV using SHA1(img_name).
      - If split_name == 'test':
          csv_path must point to the test CSV (ends with '_test.csv').

    • Image resolution:
      - Primary: <root>/<path> (e.g., root/fitz_images/<img>).
      - Fallbacks try common locations under ALT_ROOTS (including your real downloads).

    Args:
        root:       Base directory (e.g., "data/skincon").
        csv_path:   Path to v3 CSV file.
        split_name: "train", "val", or "test".
        transform:  Torchvision transform to apply (default: ToTensor()).
        val_ratio:  Fraction drawn as validation from train CSV (default 0.1).
        alt_roots:  Optional list of extra root dirs to try for images.
        strict:     If True, raise if dataset ends up empty.
    """
    def __init__(self, root: str, csv_path: str, split_name: str,
                 transform=None, val_ratio: float = 0.1,
                 alt_roots=None, strict: bool = True):
        assert split_name in ("train", "val", "test")
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.val_ratio = float(val_ratio)
        self.strict = bool(strict)

        # Optional alternate locations for Fitzpatrick17k images.
        if alt_roots is None:
            fitz17k_images = DATASET_ROOTS.get("fitz17k_images")
            fitz17k_root = Path(fitz17k_images).parent if fitz17k_images else None
            alt_roots = [p for p in [fitz17k_root, Path(fitz17k_images) if fitz17k_images else None] if p]
        self.alt_roots = [Path(p) for p in alt_roots]

        # --- Load CSV and basic checks
        import pandas as pd, hashlib
        df = pd.read_csv(self.csv_path)

        required = ["img_name", "label", "path"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"FitzSkinV3 CSV missing columns: {missing}")

        # Concept columns = strictly between 'label' and 'path' in header order
        cols = list(df.columns)
        i_label = cols.index("label")
        i_path  = cols.index("path")
        concept_cols = cols[i_label + 1 : i_path]
        self.concept_vocab = concept_cols  # preserve CSV order

        # --- Split logic
        is_train_csv = self.csv_path.name.endswith("_train.csv")
        is_test_csv  = self.csv_path.name.endswith("_test.csv")

        if split_name in ("train", "val"):
            if not is_train_csv:
                raise ValueError(f"Requested {split_name} from non-train CSV: {self.csv_path}")
            # deterministic val mask via SHA1(img_name)
            def _is_val(img_name: str) -> bool:
                h = hashlib.sha1(str(img_name).encode("utf-8")).hexdigest()
                r = int(h[:8], 16) / 0xFFFFFFFF
                return r < self.val_ratio
            mask_val = df["img_name"].map(_is_val)
            df = df[mask_val] if split_name == "val" else df[~mask_val]

        elif split_name == "test":
            if not is_test_csv:
                raise ValueError("split_name='test' requires the test CSV (name must end with _test.csv)")

        # --- Class mapping
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # --- Row -> item conversion
        self.paths, self.targets, self.concepts, self.filenames = [], [], [], []

        # Helper to generate a prioritized list of candidate file paths
        def _candidates_for(row):
            rel = str(row["path"]).lstrip("/").strip()
            img_name = str(row["img_name"]).strip()

            cand = []
            # Primary: as-given under root
            cand.append(self.root / rel)
            # Common subdir under root
            cand.append(self.root / "fitz_images" / img_name)
            # Try alternate roots
            for R in self.alt_roots:
                cand.append(R / rel)
                cand.append(R / rel.replace("fitz_images/", "fitzpatrick17k_images/"))
                cand.append(R / "fitz_images" / img_name)
                cand.append(R / "fitzpatrick17k_images" / img_name)
            return cand

        # Parse rows
        for _, row in df.iterrows():
            y_name = str(row["label"]).strip().lower()
            if y_name not in self.class_to_idx:
                continue
            y = self.class_to_idx[y_name]

            # Resolve image path
            p = next((q for q in _candidates_for(row) if q.exists()), None)
            if p is None:
                # Skip missing files silently; set strict=True to raise if all get skipped
                continue

            # Build concept vector
            vec = torch.zeros(len(self.concept_vocab), dtype=torch.float32)
            for i, cname in enumerate(self.concept_vocab):
                val = str(row[cname]).strip().lower()
                if val in ("yes", "y", "1", "true", "t"):
                    vec[i] = 1.0
                elif val in ("no", "n", "0", "false", "f", ""):
                    vec[i] = 0.0
                else:
                    # Unknown token -> treat as 0; customize if needed
                    vec[i] = 0.0

            self.paths.append(p)
            self.targets.append(y)
            self.concepts.append(vec)
            self.filenames.append(str(row["img_name"]).strip())

        if self.strict and len(self.paths) == 0:
            raise ValueError(
                "FitzSkinV3Dataset constructed an EMPTY split. "
                "Check that image files exist under root/alt_roots for the CSV 'path' layout."
            )

        assert len(self.paths) == len(self.targets) == len(self.concepts)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]).convert("RGB") as im:
            x = self.transform(im) if self.transform else transforms.ToTensor()(im)
        return x, int(self.targets[idx]), self.concepts[idx]



    
# class PBCConceptsDataset(Dataset):
#     """
#     Reads from CSV like pbc_attr_v1_train.csv:
#       img_name,label,cell_size,...,granularity,path
#     Builds:
#       - x: image
#       - y: class id 0..4
#       - c: 31-dim one-hot concept vector (attribute=value bins; order from attribute_binarized.yml)
#     """
#     classes = ["Basophil","Eosinophil","Lymphocyte","Monocyte","Neutrophil"]
#     # map various casings/spellings to canonical class ids
#     _cls_to_id = {name.lower(): i for i, name in enumerate(classes)}

#     def __init__(self, root: str, csv_path: str,
#                  attr_yaml: str, attr_bin_yaml: str,
#                  transform=None):
#         from collections import defaultdict
#         self.root = Path(root)
#         self.transform = transform
#         self.csv_path = Path(csv_path)

#         self.attr_map, self.concept_vocab = _load_pbc_attr_vocab(attr_yaml, attr_bin_yaml)
#         # reverse index for concept name -> index
#         self._concept_index = {name: i for i, name in enumerate(self.concept_vocab)}

#         # columns in CSV we care about (attribute names)
#         self.attr_names = list(self.attr_map.keys())  # 11 attributes in attribute.yml order

#         self.paths, self.filenames, self.targets, self.concepts = [], [], [], []

#         with self.csv_path.open("r", newline="") as f:
#             reader = _sniffed_reader(f)
#             need = {"img_name","label","path"} | set(self.attr_names)
#             miss = need - set(reader.fieldnames or [])
#             if miss:
#                 raise ValueError(f"PBC CSV missing: {sorted(miss)}")

#             for row in reader:
#                 # class id
#                 y_name = row["label"].strip().lower()
#                 if y_name not in self._cls_to_id:
#                     # some rows (e.g., erythroblast, platelet, ig) are not in the 5-class list → skip them
#                     # (matches your folder tree that also has those extra classes)
#                     continue
#                 y = self._cls_to_id[y_name]

#                 # image path (resolve under root)
#                 rel = _lstrip_slash(row["path"])
#                 rel = rel.removeprefix("PBC_dataset_normal_DIB/")
#                 p = self.root / rel
#                 s = 4
#                 if not p.exists():
#                     # fallback: try img_name under class folder
#                     rel2 = f"{row['label']}/{row['img_name']}"
#                     p2 = self.root / rel2
#                     p = p2 if p2.exists() else p

#                 # build 31-d concept vector
#                 vec = torch.zeros(len(self.concept_vocab), dtype=torch.float32)
#                 for a in self.attr_names:
#                     v = row[a].strip().replace(" ", "_")  # normalize to match binarized.yml
#                     name = f"{a}={v}"
#                     idx = self._concept_index.get(name, None)
#                     if idx is not None:
#                         vec[idx] = 1.0
#                     else:
#                         # if a value not present in binarized.yml appears, we silently ignore
#                         # or you can raise/warn here
#                         pass

#                 self.filenames.append(row["img_name"])
#                 self.paths.append(p)
#                 self.targets.append(y)
#                 self.concepts.append(vec)

#         assert len(self.paths) == len(self.targets) == len(self.concepts)

#     def __len__(self): return len(self.targets)

#     def __getitem__(self, idx: int):
#         with Image.open(self.paths[idx]).convert("RGB") as im:
#             x = self.transform(im) if self.transform else transforms.ToTensor()(im)
#         return x, int(self.targets[idx]), self.concepts[idx]

from pathlib import Path
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import csv, yaml

def _sniffed_reader(fobj):
    sniffer = csv.Sniffer()
    sample = fobj.read(4096)
    fobj.seek(0)
    dialect = sniffer.sniff(sample)
    return csv.DictReader(fobj, dialect=dialect)

def _lstrip_slash(s: str) -> str:
    return s[1:] if s.startswith("/") else s

# def _load_pbc_attr_vocab(attr_yaml: str, attr_bin_yaml: str):
#     with open(attr_yaml, "r") as f:
#         attr_map = yaml.safe_load(f)  # attribute -> list of values (human names)
#     with open(attr_bin_yaml, "r") as f:
#         attr_bin = yaml.safe_load(f)  # attribute -> list of bin/value tokens used to build one-hots

#     # concept order is attribute_binarized.yml order: for each attribute, iterate its bins
#     concept_vocab: List[str] = []
#     for a in attr_map.keys():
#         bins = attr_bin[a]
#         for b in bins:
#             # normalize “value” to token used in CSV → attr=value
#             concept_vocab.append(f"{a}={b}")
#     return attr_map, concept_vocab

def _load_pbc_attr_vocab(attr_yaml: str, attr_bin_yaml: str):
    """
    Loads:
      - attr_yaml: dict mapping attribute -> values (human names), used for column presence & order
      - attr_bin_yaml: EITHER
          (A) dict mapping attribute -> list of bin tokens   OR
          (B) list of strings like 'attribute=value'
    Returns:
      attr_map (dict), concept_vocab (list[str])  # tokens 'attribute=value' in the chosen order
    """
    with open(attr_yaml, "r") as f:
        attr_map = yaml.safe_load(f)  # dict: attribute -> list[values]

    with open(attr_bin_yaml, "r") as f:
        bins_yaml = yaml.safe_load(f)

    concept_vocab = []

    if isinstance(bins_yaml, dict):
        # Respect attribute order from attr_map keys
        for a in attr_map.keys():
            bins = bins_yaml.get(a, [])
            for b in bins:
                concept_vocab.append(f"{a}={str(b).replace(' ', '_')}")
    elif isinstance(bins_yaml, list):
        # One entry per concept; normalize spacing to match CSV normalization
        for item in bins_yaml:
            key = str(item).strip().lstrip("-").strip().replace(" ", "_")
            concept_vocab.append(key)
    else:
        raise TypeError(f"Unsupported YAML type for attribute_binarized.yml: {type(bins_yaml)}")

    return attr_map, concept_vocab


class PBCConceptsDataset(Dataset):
    """
    Reads from CSV like pbc_attr_v1_train.csv:
      img_name,label,cell_size,...,granularity,path

    Uses label order strictly from label.yml (key 'label').
    Builds:
      - x: image
      - y: id in 0..K-1 from label.yml
      - c: 31-d one-hot concept vector (attribute=value bins; order from attribute_binarized.yml)
    """
    def __init__(
        self,
        root: str,
        csv_path: str,
        attr_yaml: str,
        attr_bin_yaml: str,
        label_yaml: str = DATA_ROOT + "/wbc/label.yml",
        transform=None,
    ):
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.transform = transform

        # --- labels from YAML (single source of truth) ---
        with open(label_yaml, "r") as f:
            ymap_all = yaml.safe_load(f)
        if "label" not in ymap_all or not isinstance(ymap_all["label"], dict):
            raise ValueError(f"{label_yaml} must contain a 'label' mapping")
        ymap = ymap_all["label"]  # e.g., {'Basophil':0,...}

        # guarantee class order is 0..K-1 and keep names in that order
        names = list(ymap.keys())
        ids = [ymap[n] for n in names]
        if ids != list(range(len(ids))):
            names = [k for k, _ in sorted(ymap.items(), key=lambda kv: kv[1])]

        self.classes: List[str] = names
        self.class_to_idx = {k.lower(): ymap[k] for k in ymap}

        # --- concept vocab / binarization ---
        self.attr_map, self.concept_vocab = _load_pbc_attr_vocab(attr_yaml, attr_bin_yaml)
        self._concept_index = {name: i for i, name in enumerate(self.concept_vocab)}
        self.attr_names = list(self.attr_map.keys())  # 11 attributes in attribute.yml order

        self.paths, self.filenames, self.targets, self.concepts = [], [], [], []

        with self.csv_path.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            need = {"img_name","label","path"} | set(self.attr_names)
            miss = need - set(reader.fieldnames or [])
            if miss:
                raise ValueError(f"PBC CSV missing: {sorted(miss)}")

            for row in reader:
                # map label → id by YAML order
                raw_name = str(row["label"]).strip()
                key = raw_name.lower()
                if key not in self.class_to_idx:
                    # skip extra classes not in 5-class YAML
                    continue
                y = int(self.class_to_idx[key])

                # resolve image path
                rel = _lstrip_slash(str(row["path"]))
                rel = rel.replace("PBC_dataset_normal_DIB/", "")
                p = self.root / rel
                if not p.exists():
                    alt = self.root / f"{raw_name}/{row['img_name']}"
                    p = alt if alt.exists() else p
                if not p.exists():
                    # last fallback: try just img_name anywhere under root (optional, can remove)
                    alt2 = next(self.root.rglob(row["img_name"]), None)
                    if alt2 is not None:
                        p = alt2
                if not p.exists():
                    continue  # or raise

                # build 31-d concept vector
                vec = torch.zeros(len(self.concept_vocab), dtype=torch.float32)
                for a in self.attr_names:
                    v = str(row[a]).strip().replace(" ", "_")
                    idx = self._concept_index.get(f"{a}={v}", None)
                    if idx is not None:
                        vec[idx] = 1.0

                self.filenames.append(row["img_name"])
                self.paths.append(p)
                self.targets.append(y)
                self.concepts.append(vec)

        assert len(self.paths) == len(self.targets) == len(self.concepts)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]).convert("RGB") as im:
            x = self.transform(im) if self.transform else transforms.ToTensor()(im)
        return x, int(self.targets[idx]), self.concepts[idx]


from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import yaml


class DDiV1Dataset(Dataset):
    """
    DDI v1 CSV with per-sample concepts.

    CSV schema:
      img_name, label, <binary concept columns...>, path

    Splits:
      - train/val: deterministic val holdout from the full CSV via SHA1(img_name)
      - test:      uses the full CSV
    """
    def __init__(self, csv_path: str, images_root: str, label_yaml: str,
                 split_name: str = "train", transform=None,
                 val_ratio: float = 0.1, strict: bool = True):
        assert split_name in ("train", "val", "test")
        self.root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.val_ratio = float(val_ratio)
        self.strict = bool(strict)

        import pandas as pd
        df = pd.read_csv(self.csv_path)

        required = ["img_name", "label", "path"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DDI CSV missing columns: {missing}")

        cols = list(df.columns)
        i_label = cols.index("label")
        i_path = cols.index("path")
        concept_cols = cols[i_label + 1 : i_path]
        self.concept_vocab = concept_cols  # preserve CSV order

        if split_name in ("train", "val"):
            def _is_val(img_name: str) -> bool:
                h = hashlib.sha1(str(img_name).encode("utf-8")).hexdigest()
                r = int(h[:8], 16) / 0xFFFFFFFF
                return r < self.val_ratio
            mask_val = df["img_name"].map(_is_val)
            df = df[mask_val] if split_name == "val" else df[~mask_val]

        with open(label_yaml, "r") as f:
            ymap_all = yaml.safe_load(f)
        if "label" not in ymap_all or not isinstance(ymap_all["label"], dict):
            raise ValueError(f"{label_yaml} must contain a 'label' mapping (e.g., benign: 0, malignant: 1)")
        ymap = ymap_all["label"]

        names = list(ymap.keys())
        ids = [ymap[n] for n in names]
        if ids != list(range(len(ids))):
            names = [k for k, _ in sorted(ymap.items(), key=lambda kv: kv[1])]
        self.classes = names
        self.class_to_idx = {k: int(v) for k, v in ymap.items()}

        self.paths, self.targets, self.concepts, self.filenames = [], [], [], []

        for _, row in df.iterrows():
            y_name = str(row["label"]).strip().lower()
            if y_name not in self.class_to_idx:
                continue
            y = int(self.class_to_idx[y_name])

            rel = str(row["path"]).lstrip("/").strip()
            img_name = str(row["img_name"]).strip()
            candidates = [
                self.root / rel,
                self.root / "ddi_images" / img_name,
                self.root / img_name,
            ]
            p = next((q for q in candidates if q.exists()), None)
            if p is None:
                continue

            vec = torch.zeros(len(self.concept_vocab), dtype=torch.float32)
            for i, cname in enumerate(self.concept_vocab):
                val = str(row[cname]).strip().lower()
                if val in ("yes", "y", "1", "true", "t"):
                    vec[i] = 1.0
                elif val in ("no", "n", "0", "false", "f", ""):
                    vec[i] = 0.0
                else:
                    vec[i] = 0.0

            self.paths.append(p)
            self.targets.append(y)
            self.concepts.append(vec)
            self.filenames.append(img_name)

        if self.strict and len(self.paths) == 0:
            raise ValueError("DDiV1Dataset constructed an EMPTY split. Check CSV paths and images_root.")

        assert len(self.paths) == len(self.targets) == len(self.concepts)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]).convert("RGB") as im:
            x = self.transform(im) if self.transform else transforms.ToTensor()(im)
        return x, int(self.targets[idx]), self.concepts[idx]


class DDiCSVTestDataset(Dataset):
    """
    CSV-backed SciRep test set, consistent with the second script.
    Expects:
      - DATASET_ROOTS['scirep_csv'] (CSV with columns: path,label)
      - DATASET_ROOTS['scirep_images_root'] (images root)
      - LABEL_YAML mapping label names -> ids under key 'label'
    """
    def __init__(self, csv_path: str, images_root: str, label_yaml: str, transform=None, strict: bool = True):
        import pandas as pd
        self.transform = transform
        self.images_root = Path(images_root)

        df = pd.read_csv(csv_path)

        with open(label_yaml, "r") as f:
            ymap_all = yaml.safe_load(f)
        if "label" not in ymap_all or not isinstance(ymap_all["label"], dict):
            raise ValueError(f"{label_yaml} must contain a 'label' mapping (e.g., Basophil: 0, ...)")
        ymap = ymap_all["label"]

        # Preserve YAML order if already 0..K-1; otherwise sort by id
        names = list(ymap.keys())
        ids = [ymap[n] for n in names]
        if ids != list(range(len(ids))):
            names = [k for k, _ in sorted(ymap.items(), key=lambda kv: kv[1])]

        self.classes = names
        self.class_to_idx = {k: int(v) for k, v in ymap.items()}

        self.paths, self.targets, self.filenames = [], [], []
        for _, row in df.iterrows():
            # --- build clean relative path as a STRING ---
            rel = str(row["path"]).lstrip("/").strip()
            rel = rel.replace("*", "_")  # string replace (safe)

            yname = str(row["label"]).strip()
            if yname not in self.class_to_idx:
                if strict:
                    raise ValueError(f"Unknown label in CSV: {yname}")
                else:
                    continue

            # join AFTER cleaning the string
            p = self.images_root / rel
            if not p.exists():
                if strict:
                    raise FileNotFoundError(f"Missing image: {p}")
                else:
                    continue

            self.paths.append(p)
            self.targets.append(self.class_to_idx[yname])
            self.filenames.append(Path(rel).name)

        if strict and len(self.paths) == 0:
            raise ValueError("Empty SciRep CSV dataset (no valid rows or files).")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]).convert("RGB") as im:
            x = self.transform(im) if self.transform else transforms.ToTensor()(im)
        return x, int(self.targets[idx])

class ScirepCSVTestDataset(Dataset):
    """
    CSV-backed SciRep test set, consistent with the second script.
    Expects:
      - DATASET_ROOTS['scirep_csv'] (CSV with columns: path,label)
      - DATASET_ROOTS['scirep_images_root'] (images root)
      - LABEL_YAML mapping label names -> ids under key 'label'
    """
    def __init__(self, csv_path: str, images_root: str, label_yaml: str, transform=None, strict: bool = True):
        import pandas as pd
        self.transform = transform
        self.images_root = Path(images_root)

        df = pd.read_csv(csv_path)

        with open(label_yaml, "r") as f:
            ymap_all = yaml.safe_load(f)
        if "label" not in ymap_all or not isinstance(ymap_all["label"], dict):
            raise ValueError(f"{label_yaml} must contain a 'label' mapping (e.g., Basophil: 0, ...)")
        ymap = ymap_all["label"]

        # Preserve YAML order if already 0..K-1; otherwise sort by id
        names = list(ymap.keys())
        ids = [ymap[n] for n in names]
        if ids != list(range(len(ids))):
            names = [k for k, _ in sorted(ymap.items(), key=lambda kv: kv[1])]

        self.classes = names
        self.class_to_idx = {k: int(v) for k, v in ymap.items()}

        self.paths, self.targets, self.filenames = [], [], []
        for _, row in df.iterrows():
            # --- build clean relative path as a STRING ---
            rel = str(row["path"]).lstrip("/").strip()
            rel = rel.replace("*", "_")  # string replace (safe)

            yname = str(row["label"]).strip()
            if yname not in self.class_to_idx:
                if strict:
                    raise ValueError(f"Unknown label in CSV: {yname}")
                else:
                    continue

            # join AFTER cleaning the string
            p = self.images_root / rel
            if not p.exists():
                if strict:
                    raise FileNotFoundError(f"Missing image: {p}")
                else:
                    continue

            self.paths.append(p)
            self.targets.append(self.class_to_idx[yname])
            self.filenames.append(Path(rel).name)

        if strict and len(self.paths) == 0:
            raise ValueError("Empty SciRep CSV dataset (no valid rows or files).")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]).convert("RGB") as im:
            x = self.transform(im) if self.transform else transforms.ToTensor()(im)
        return x, int(self.targets[idx])



# class ScirepWBCTestADataset(Dataset):
#     """
#     SciRep WBC Test (folder-structured, no CSV).

#     Tree (under DATASET_ROOTS['scirep_root']):
#         Basophil/
#         Eosinophil/
#         Lymphocyte/
#         Monocyte/
#         Neutrophil/

#     Exposes:
#       - __getitem__ -> (image_tensor, y:int)
#       - .targets    -> List[int]
#       - .filenames  -> List[str]  (basename only)
#       - .paths      -> List[Path] (absolute paths)
#     """
#     classes = ["Basophil","Eosinophil","Lymphocyte","Monocyte","Neutrophil"]
#     _cls_to_id = {name.lower(): i for i, name in enumerate(classes)}
#     _valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#     def __init__(self, root: str, transform=None, strict: bool = True, recursive: bool = True):
#         self.root = Path(root)
#         self.transform = transform
#         self.strict = bool(strict)
#         self.recursive = bool(recursive)

#         self.paths: List[Path] = []
#         self.filenames: List[str] = []
#         self.targets: List[int] = []

#         # For stability and reproducibility, iterate classes in the canonical order above
#         for cname in self.classes:
#             cdir = self.root / cname
#             if not cdir.exists():
#                 # Allow missing classes (some test sets may be partial), unless strict=True and we end up empty
#                 continue

#             it = cdir.rglob("*") if self.recursive else cdir.iterdir()
#             # Collect files with valid image suffixes; sort for deterministic order
#             files = sorted([p for p in it if p.is_file() and p.suffix.lower() in self._valid_exts])

#             y = self._cls_to_id[cname.lower()]
#             for p in files:
#                 self.paths.append(p)
#                 self.filenames.append(p.name)
#                 self.targets.append(y)

#         if self.strict and len(self.paths) == 0:
#             raise ValueError(f"ScirepWBCTestADataset found no images under {self.root}")

#         assert len(self.paths) == len(self.targets) == len(self.filenames)

#     def __len__(self): 
#         return len(self.targets)

#     def __getitem__(self, idx: int):
#         with Image.open(self.paths[idx]).convert("RGB") as im:
#             x = self.transform(im) if self.transform else transforms.ToTensor()(im)
#         return x, int(self.targets[idx])


class RaabinWBCTestADataset(Dataset):
    def __init__(self, root: str, csv_path: str, transform=None):
        self.root = Path(root); self.transform = transform
        self.csv_path = Path(csv_path)

        classes = ["Basophil","Eosinophil","Lymphocyte","Monocyte","Neutrophil"]
        self._cls_to_id = {name.lower(): i for i, name in enumerate(classes)}

        self.paths, self.targets = [], []
        with self.csv_path.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            need = {"path","label"}
            miss = need - set(reader.fieldnames or [])
            if miss: raise ValueError(f"Raabin CSV missing: {sorted(miss)}")
            for row in reader:
                y = self._cls_to_id[row["label"].strip().lower()]
                rel = _lstrip_slash(row["path"])
                rel = rel.removeprefix("TestA/")
                # s = 4
                # print(row["path"])
                # print(rel)
                # print(self.root)
                # exit()
                self.paths.append(self.root / rel)
                self.targets.append(y)

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        with Image.open(self.paths[idx]).convert("RGB") as im:
            x = self.transform(im) if self.transform else transforms.ToTensor()(im)
        return x, int(self.targets[idx])


class RaabinWBCSplitDataset(Dataset):
    """
    Raabin WBC TestA with deterministic train/val splits and proxy concepts.
    Concept supervision is a one-hot vector of the class label.
    """
    def __init__(self, root: str, csv_path: str, split_name: str,
                 transform=None, val_ratio: float = 0.1, strict: bool = True):
        assert split_name in ("train", "val", "test")
        self.root = Path(root)
        self.transform = transform
        self.csv_path = Path(csv_path)
        self.val_ratio = float(val_ratio)
        self.strict = bool(strict)

        self.classes = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
        self.class_to_idx = {name.lower(): i for i, name in enumerate(self.classes)}
        self.concept_vocab = [f"class={name}" for name in self.classes]

        self.paths, self.targets, self.concepts, self.filenames = [], [], [], []

        with self.csv_path.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            need = {"path", "label"}
            miss = need - set(reader.fieldnames or [])
            if miss:
                raise ValueError(f"Raabin CSV missing: {sorted(miss)}")

            for row in reader:
                rel = _lstrip_slash(row["path"])
                rel = rel.removeprefix("TestA/")
                key = str(rel)

                if split_name in ("train", "val"):
                    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
                    r = int(h[:8], 16) / 0xFFFFFFFF
                    is_val = r < self.val_ratio
                    if split_name == "val" and not is_val:
                        continue
                    if split_name == "train" and is_val:
                        continue

                y_name = row["label"].strip().lower()
                if y_name not in self.class_to_idx:
                    if self.strict:
                        raise ValueError(f"Unknown Raabin label in CSV: {row['label']}")
                    else:
                        continue
                y = self.class_to_idx[y_name]

                p = self.root / rel
                if not p.exists():
                    if self.strict:
                        continue
                    else:
                        continue

                vec = torch.zeros(len(self.concept_vocab), dtype=torch.float32)
                vec[y] = 1.0

                self.paths.append(p)
                self.targets.append(y)
                self.concepts.append(vec)
                self.filenames.append(Path(rel).name)

        if self.strict and len(self.paths) == 0:
            raise ValueError("RaabinWBCSplitDataset constructed an EMPTY split.")

        assert len(self.paths) == len(self.targets) == len(self.concepts)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]).convert("RGB") as im:
            x = self.transform(im) if self.transform else transforms.ToTensor()(im)
        return x, int(self.targets[idx]), self.concepts[idx]

import numpy as np

class Cifar10CSubset(Dataset):
    """
    Loads a subset of CIFAR-10-C given a list of corruptions and severities.
    - root: directory that contains e.g. gaussian_noise.npy, shot_noise.npy, labels.npy
    - Each corruption file has shape (50000, 32, 32, 3) with 10k images per severity (1..5) in order.
    - We concatenate [ (corr, sev) for corr in corruptions for sev in severities ] in that order.
    Exposes:
      - .targets (List[int])
      - .filenames (synthetic, for compatibility)
      - .segments: List[dict{name, severity, start, end}]  inclusive-exclusive indices for each (corr,sev)
    """
    def __init__(self, root: str, corruptions, severities, transform=None):
        self.root = Path(root)
        self.corruptions = list(corruptions)
        self.severities = list(severities)
        self.transform = transform

        labels = np.load(self.root / "labels.npy")  # (50000,)
        self._segments = []   # book-keeping for metrics
        self._items = []      # list of (arr_ref, idx_in_arr)
        self.targets = []     # aligned with _items
        self.filenames = []   # optional: fake names for compatibility

        # Load each corruption mmap'ed to keep RAM small
        per_sev = 10000
        for corr in self.corruptions:
            arr = np.load(self.root / f"{corr}.npy", mmap_mode="r")  # (50000, 32,32,3), uint8
            for sev in self.severities:
                s = (sev - 1) * per_sev
                e = sev * per_sev
                start = len(self._items)
                for i in range(s, e):
                    self._items.append((arr, i))
                    self.targets.append(int(labels[i]))
                    # fabricate a filename-like id (purely informational)
                    self.filenames.append(f"{corr}/severity{sev}/{i - s:05d}.png")
                end = len(self._items)
                self._segments.append({"name": corr, "severity": sev, "start": start, "end": end})

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        arr, j = self._items[idx]
        img = Image.fromarray(arr[j])  # HWC uint8 -> PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]

    # Expose for metrics
    @property
    def segments(self):
        return list(self._segments)

def get_cifar10c_segments(dataset_name: str):
    if dataset_name not in ("cifar10c_val", "cifar10c_test"):
        return None
    ds = _as_dataset(dataset_name)
    return getattr(ds, "segments", None)


# ------------------------------
# Factory & convenience accessors
# ------------------------------

def get_data(dataset_name: str, preprocess=None):
    """
    Returns dataset instance for:
      - waterbirds_{train,val,test} -> WaterbirdsDataset
      - sparwious_{train,val,test}  -> SparwiousDataset
      - (legacy alias) spawrious_*  -> SparwiousDataset
      - others in DATASET_ROOTS     -> ImageFolder fallback
    """
    dataset_name = _canon_dataset_name(dataset_name)

    # Short-hands: allow base name to default to train split
    if dataset_name == "ham_4":
        return get_data("ham_4_train", preprocess)
    if dataset_name == "ham":
        return get_data("ham_train", preprocess)
    if dataset_name == "ham_3":
        return get_data("ham_3_train", preprocess)
    if dataset_name == "isic19_4":
        return get_data("isic19_4_train", preprocess)
    if dataset_name == "isic19":
        return get_data("isic19_train", preprocess)
    if dataset_name == "isic19_3":
        return get_data("isic19_3_train", preprocess)

    # HAM10000 (source)
    if dataset_name in ("ham_train", "ham_val", "ham_test"):
        split = dataset_name.rsplit("_", 1)[1]
        with open(LABEL_FILES["ham"], "r") as f:
            labels = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        label_map = {c: i for i, c in enumerate(labels)}
        return DermCSVImageDataset(
            images_roots=DATASET_ROOTS["ham_image_roots"],
            metadata_csv=DATASET_ROOTS["ham_metadata"],
            split=split,
            id_col="image_id",
            label_col="dx",
            label_map=label_map,
            allow_labels=set(label_map.keys()),
            require_label=True,
            transform=preprocess,
            train_ratio=0.8,
            val_ratio=0.1,
        )

    # HAM10000 (source, 4-class subset: AKIEC, BKL, DF, VASC)
    if dataset_name in ("ham_4_train", "ham_4_val", "ham_4_test"):
        split = dataset_name.rsplit("_", 1)[1]
        with open(LABEL_FILES["ham_4"], "r") as f:
            labels = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        label_map = {c: i for i, c in enumerate(labels)}
        return DermCSVImageDataset(
            images_roots=DATASET_ROOTS["ham_image_roots"],
            metadata_csv=DATASET_ROOTS["ham_metadata"],
            split=split,
            id_col="image_id",
            label_col="dx",
            label_map=label_map,
            allow_labels=set(label_map.keys()),
            require_label=True,
            transform=preprocess,
            train_ratio=0.8,
            val_ratio=0.1,
        )

    # HAM10000 (source, 3-class subset: BCC, BKL, MEL)
    if dataset_name in ("ham_3_train", "ham_3_val", "ham_3_test"):
        split = dataset_name.rsplit("_", 1)[1]
        with open(LABEL_FILES["ham_3"], "r") as f:
            labels = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        label_map = {c: i for i, c in enumerate(labels)}
        return DermCSVImageDataset(
            images_roots=DATASET_ROOTS["ham_image_roots"],
            metadata_csv=DATASET_ROOTS["ham_metadata"],
            split=split,
            id_col="image_id",
            label_col="dx",
            label_map=label_map,
            allow_labels=set(label_map.keys()),
            require_label=True,
            transform=preprocess,
            train_ratio=0.8,
            val_ratio=0.1,
        )

    # ISIC 2019 (target, labels optional; keep all rows)
    if dataset_name in ("isic19_train", "isic19_val", "isic19_test"):
        split = dataset_name.rsplit("_", 1)[1]
        with open(LABEL_FILES["isic19"], "r") as f:
            labels = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        label_map = {c: i for i, c in enumerate(labels)}
        gt_lookup = _load_isic19_groundtruth(
            DATASET_ROOTS["isic19_groundtruth"],
            ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"],
        )
        return DermCSVImageDataset(
            images_roots=[DATASET_ROOTS["isic19_images_root"]],
            metadata_csv=DATASET_ROOTS["isic19_metadata"],
            split=split,
            id_col="image",
            label_col=None,  # labels supplied via ground-truth lookup instead
            label_map=label_map,
            allow_labels=set(label_map.keys()),
            require_label=False,  # keep rows even if ground-truth is missing/UNK (-1)
            label_lookup=gt_lookup,
            transform=preprocess,
            train_ratio=0.8,
            val_ratio=0.1,
        )

    # ISIC 2019 (target, 4-class subset: AKIEC, BKL, DF, VASC)
    if dataset_name in ("isic19_4_train", "isic19_4_val", "isic19_4_test"):
        split = dataset_name.rsplit("_", 1)[1]
        with open(LABEL_FILES["isic19_4"], "r") as f:
            labels = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        label_map = {c: i for i, c in enumerate(labels)}
        gt_lookup = _load_isic19_groundtruth(
            DATASET_ROOTS["isic19_groundtruth"],
            ["AK", "BKL", "DF", "VASC"],
        )
        return DermCSVImageDataset(
            images_roots=[DATASET_ROOTS["isic19_images_root"]],
            metadata_csv=DATASET_ROOTS["isic19_metadata"],
            split=split,
            id_col="image",
            label_col=None,  # labels supplied via ground-truth lookup instead
            label_map=label_map,
            allow_labels=set(label_map.keys()),
            require_label=False,  # keep rows even if ground-truth is missing/UNK (-1)
            label_lookup=gt_lookup,
            transform=preprocess,
            train_ratio=0.8,
            val_ratio=0.1,
        )

    # ISIC 2019 (target, 3-class subset: BCC, BKL, MEL)
    if dataset_name in ("isic19_3_train", "isic19_3_val", "isic19_3_test"):
        split = dataset_name.rsplit("_", 1)[1]
        with open(LABEL_FILES["isic19_3"], "r") as f:
            labels = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        label_map = {c: i for i, c in enumerate(labels)}
        gt_lookup = _load_isic19_groundtruth(
            DATASET_ROOTS["isic19_groundtruth"],
            ["BCC", "BKL", "MEL"],
        )
        return DermCSVImageDataset(
            images_roots=[DATASET_ROOTS["isic19_images_root"]],
            metadata_csv=DATASET_ROOTS["isic19_metadata"],
            split=split,
            id_col="image",
            label_col=None,  # labels supplied via ground-truth lookup instead
            label_map=label_map,
            allow_labels=set(label_map.keys()),
            require_label=False,  # keep rows even if ground-truth is missing/UNK (-1)
            label_lookup=gt_lookup,
            transform=preprocess,
            train_ratio=0.8,
            val_ratio=0.1,
        )

    #    # Fitzpatrick17k skincon v3 (2-class benign/malignant + CSV-provided concepts)
    if dataset_name in ("fitzskin_train", "fitzskin_val", "fitzskin_test"):
        split = dataset_name.rsplit("_", 1)[1]
        if split in ("train", "val"):
            csv_path = DATASET_ROOTS["fitzskin_train_csv"]
        else:
            csv_path = DATASET_ROOTS["fitzskin_test_csv"]
        return FitzSkinV3Dataset(
            root=DATASET_ROOTS["fitzskin_root"],
            csv_path=csv_path,
            split_name=split,
            transform=preprocess,
        )



    if dataset_name in ("pbc_train","pbc_val","pbc_test"):
        split2csv = {
            "pbc_train": DATA_ROOT + "/wbc/pbc_attr_v1_train.csv",
            "pbc_val":   DATA_ROOT + "/wbc/pbc_attr_v1_val.csv",
            "pbc_test":  DATA_ROOT + "/wbc/pbc_attr_v1_test.csv",
        }
        return PBCConceptsDataset(
            root=DATASET_ROOTS["pbc_root"],
            csv_path=split2csv[dataset_name],
            attr_yaml=DATA_ROOT + "/wbc/attribute.yml",
            attr_bin_yaml=DATA_ROOT + "/wbc/attribute_binarized.yml",
            transform=preprocess
        )

    if dataset_name in ("raabinwbc_train", "raabinwbc_val", "raabinwbc_test", "raabinwbc_testA"):
        split = "test" if dataset_name in ("raabinwbc_test", "raabinwbc_testA") else dataset_name.rsplit("_", 1)[1]
        return RaabinWBCSplitDataset(
            root=DATASET_ROOTS["raabinwbc_testA_root"],
            csv_path=DATASET_ROOTS["raabinwbc_testA_csv"],
            split_name=split,
            transform=preprocess,
        )
    if dataset_name in ("scirep",):
        # return ScirepWBCTestADataset(
        #     root=DATASET_ROOTS["scirep_root"],
        #     transform=preprocess
        # )
        return ScirepCSVTestDataset(
            csv_path=DATASET_ROOTS["scirep_csv"],
            images_root=DATASET_ROOTS["scirep_images_root"],
            label_yaml=LABEL_YAML,
            transform=preprocess,
            strict=True
        )
    if dataset_name in ("ddi_train", "ddi_val", "ddi_test"):
        split = dataset_name.rsplit("_", 1)[1]
        return DDiV1Dataset(
            csv_path=DATASET_ROOTS["ddi_test_csv"],
            images_root=DATASET_ROOTS["ddi_images_root"],
            label_yaml=DATASET_ROOTS["ddi_label_2_class"],
            split_name=split,
            transform=preprocess,
            strict=True,
        )
    if dataset_name in ("ddi",):
        # return ScirepWBCTestADataset(
        #     root=DATASET_ROOTS["scirep_root"],
        #     transform=preprocess
        # )
        return DDiCSVTestDataset(
            csv_path=DATASET_ROOTS["ddi_test_csv"],
            images_root=DATASET_ROOTS["ddi_images_root"],
            label_yaml=DATASET_ROOTS["ddi_label_2_class"],
            transform=preprocess,
            strict=True
        )


    # Waterbirds custom
    if dataset_name in ("waterbirds_train", "waterbirds_val", "waterbirds_test"):
        root = DATASET_ROOTS[dataset_name]
        return WaterbirdsDataset(root=root, transform=preprocess)

    # Sparwious custom (single root + split filter)
    if dataset_name in ("sparwious_train", "sparwious_val", "sparwious_test"):
        split_code = {"sparwious_train": 0, "sparwious_val": 1, "sparwious_test": 2}[dataset_name]
        root = DATASET_ROOTS["sparwious_root"]
        return SparwiousDataset(root=root, split_code=split_code, transform=preprocess)
    
        # CIFAR-10-C support
    if dataset_name == "cifar10c_train":
        # train on clean CIFAR-10
        return datasets.CIFAR10(
            root=os.path.expanduser("~/.cache"),
            download=True, train=True, transform=preprocess
        )

    if dataset_name in ("cifar10c_val", "cifar10c_test"):
        # evaluate on the chosen CIFAR-10-C subset
        cfg = CIFAR10C_CONFIG
        return Cifar10CSubset(
            root=cfg["root"],
            corruptions=cfg["corruptions"],
            severities=cfg["severities"],
            transform=preprocess
        )


    # Torchvision built-ins (if you still rely on them here)
    if dataset_name == "cifar100_train":
        return datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=preprocess)
    if dataset_name == "cifar100_val":
        return datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
    if dataset_name == "cifar10_train":
        return datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=preprocess)
    if dataset_name == "cifar10_val":
        return datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
    if dataset_name == "places365_train":
        try:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True, transform=preprocess)
        except RuntimeError:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False, transform=preprocess)
    if dataset_name == "places365_val":
        try:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True, transform=preprocess)
        except RuntimeError:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False, transform=preprocess)

    # Fallback ImageFolder for other custom names in DATASET_ROOTS
    if dataset_name in DATASET_ROOTS:
        return datasets.ImageFolder(DATASET_ROOTS[dataset_name], transform=preprocess)

    # Alias path: if someone calls spawrious_* explicitly
    alias = _canon_dataset_name(dataset_name)
    if alias != dataset_name:
        return get_data(alias, preprocess)

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def _as_dataset(dataset_name: str):
    """Internal helper returning a dataset with a safe default preprocess."""
    preprocess = get_resnet_imagenet_preprocess()
    return get_data(dataset_name, preprocess=preprocess)


def get_targets_only(dataset_name: str) -> List[int]:
    ds = _as_dataset(dataset_name)
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "samples"):
        return [y for _, y in ds.samples]
    if hasattr(ds, "imgs"):
        return [y for _, y in ds.imgs]
    raise AttributeError(f"{dataset_name} dataset does not expose targets")


def get_filenames_only(dataset_name: str) -> List[str]:
    ds = _as_dataset(dataset_name)
    if hasattr(ds, "filenames"):
        return list(ds.filenames)
    if hasattr(ds, "samples"):
        root = Path(DATASET_ROOTS[dataset_name])
        return [str(Path(p).relative_to(root)) for p, _ in ds.samples]
    if hasattr(ds, "imgs"):
        root = Path(DATASET_ROOTS[dataset_name])
        return [str(Path(p).relative_to(root)) for p, _ in ds.imgs]
    raise AttributeError(f"{dataset_name} dataset does not expose filenames")


def get_groups_only(dataset_name: str) -> Optional[List[int]]:
    """
    Returns group IDs aligned with dataset order.
      * waterbirds_* : (1 - y) * 2 + (1 - place)
      * sparwious_*  : y * 2 + place
    """
    dataset_name = _canon_dataset_name(dataset_name)
    if dataset_name.startswith("waterbirds_") or dataset_name.startswith("sparwious_"):
        ds = _as_dataset(dataset_name)
        return list(getattr(ds, "groups", []))
    if dataset_name.startswith("spawrious_"):
        return get_groups_only(_canon_dataset_name(dataset_name))
    return None


# ------------------------------
# Target model factory
# ------------------------------

def get_target_model(name: str, device: str="cpu"):
    """
    Returns (model, preprocess) for common names.
    Supports torchvision and timm if available.
    name examples:
      - 'resnet50', 'resnet101'
      - 'ViT-B/16', 'ViT-B/32'
    NOTE: CLIP models should be requested as 'clip_<CLIP_NAME>' and are handled in utils.save_activations.
    """
    from torchvision import models as tvm
    import torch
    model = None
    preprocess = get_resnet_imagenet_preprocess()

    # Try torchvision aliases
    tv_alias = {
        "resnet18":"resnet18",
        "resnet50":"resnet50",
        "resnet101":"resnet101",
        "vit_b_16":"vit_b_16",
        "vit_b_32":"vit_b_32",
        "ViT-B/16":"vit_b_16",
        "ViT-B/32":"vit_b_32",
    }
    key = tv_alias.get(name, name.replace("-","_").replace("/","_").lower())

    if hasattr(tvm, key):
        ctor = getattr(tvm, key)
        try:
            model = ctor(weights=None).eval()
        except TypeError:
            model = ctor(pretrained=False).eval()
        return model.to(device), preprocess

    # Fallback: timm
    try:
        import timm
        tim_alias = {
            "ViT-B/16": "vit_base_patch16_224",
            "ViT-B/32": "vit_base_patch32_224",
        }
        tname = tim_alias.get(name, name)
        model = timm.create_model(tname, pretrained=False).eval().to(device)
        # Basic ImageNet preprocess; adjust if you need timm's exact cfg
        return model, preprocess
    except Exception:
        pass

    raise ValueError(f"Unknown/non-loadable backbone name: {name}")

# --- Add this in data_utils.py ---

def _strip_split_suffix(name: str) -> str:
    # Turn 'raabinwbc_testA' -> 'raabinwbc', 'waterbirds_train' -> 'waterbirds', etc.
    for suf in ["_train", "_val", "_test", "_testA", "_testB"]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name

def get_class_names(dataset_key: str):
    """
    Returns a list of human-readable class names for zero-shot prompts.
    Backed by LABEL_FILES; falls back to a few known datasets.
    Examples:
      'pbc_train'   -> use LABEL_FILES['pbc']
      'raabinwbc_*' -> use LABEL_FILES['raabinwbc']
      'waterbirds_*'-> use LABEL_FILES['waterbirds']
      'sparwious_*' -> use LABEL_FILES['sparwious']
    """
    base = _strip_split_suffix(_canon_dataset_name(dataset_key))
    # map some aliases
    if base.startswith("raabinwbc"):
        key = "raabinwbc"
    elif base.startswith("waterbirds"):
        key = "waterbirds"
    elif base.startswith("sparwious"):
        key = "sparwious"
    elif base.startswith("ham"):
        key = "ham"
    elif base.startswith("isic19"):
        key = "isic19"
    elif base.startswith("ddi"):
        try:
            import yaml
            with open(DATASET_ROOTS["ddi_label_2_class"], "r") as f:
                ymap_all = yaml.safe_load(f)
            ymap = ymap_all.get("label", {})
            if isinstance(ymap, dict) and len(ymap) > 0:
                names = list(ymap.keys())
                ids = [ymap[n] for n in names]
                if ids != list(range(len(ids))):
                    names = [k for k, _ in sorted(ymap.items(), key=lambda kv: kv[1])]
                return names
        except Exception:
            pass
    # Prefer YAML order for WBC datasets (keeps ids consistent with CSV/YAML)
    if base in ("scirep", "raabinwbc", "pbc"):
        try:
            import yaml
            with open(LABEL_YAML, "r") as f:
                ymap_all = yaml.safe_load(f)
            ymap = ymap_all.get("label", {})
            if isinstance(ymap, dict) and len(ymap) > 0:
                names = list(ymap.keys())
                ids = [ymap[n] for n in names]
                if ids != list(range(len(ids))):
                    # sort by id if YAML keys not in id order
                    names = [k for k, _ in sorted(ymap.items(), key=lambda kv: kv[1])]
                return names
        except Exception:
            # fall through to the rest of the function
            pass

    elif base.startswith("cifar10c"):
        key = "cifar10c"
    elif base.startswith("cifar10"):
        key = "cifar10"
    elif base.startswith("cifar100"):
        key = "cifar100"
    elif base.startswith("fitz17k9"):
        # derive from CSV, nine_partition_label
        import pandas as pd
        csv_path = DATASET_ROOTS.get("fitz17k_csv", "")
        if os.path.exists(csv_path):
            labs = sorted({str(x).strip() for x in pd.read_csv(csv_path)["nine_partition_label"] if isinstance(x, str)})
            return labs
        return None
    elif base.startswith("fitz17k3"):
        import pandas as pd
        csv_path = DATASET_ROOTS.get("fitz17k_csv", "")
        if os.path.exists(csv_path):
            labs = sorted({str(x).strip() for x in pd.read_csv(csv_path)["three_partition_label"] if isinstance(x, str)})
            return labs
        return None
    elif base.startswith("fitzskin"):
        return ["benign", "malignant"]

    else:
        key = base

    path = LABEL_FILES.get(key, None)
    if path is None or not os.path.exists(path):
        # Last-resort fallback: if the dataset object exposes .classes
        try:
            ds = _as_dataset(dataset_key)
            return list(getattr(ds, "classes"))
        except Exception:
            return None

    with open(path, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names
