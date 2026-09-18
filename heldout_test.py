# Train on a fixed 80% split and test on the same matched 20% with_IPI split.
#
# Four-run experiment:
#   1) EfficientNet trained on 80% IPI_removed, tested on 20% with_IPI
#   2) PaSST trained on 80% IPI_removed, tested on 20% with_IPI
#   3) EfficientNet trained on 80% with_IPI, tested on the same 20% with_IPI
#   4) PaSST trained on 80% with_IPI, tested on the same 20% with_IPI
#
# Examples:
#  python train_80_20_four.py --run-four
#   python train_80_20_four.py --model effnet --train-source IPI_removed
#   python train_80_20_four.py --model passt --train-source with_IPI
#   python train_80_20_four.py --model both --train-source both
#   python train_80_20_four.py --model effnet --train-source IPI_removed --grid
#
# This script splits by shared filename key. The test keys are always selected
# once from the shared filenames and are always evaluated using with_IPI files.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import json
import glob
import re
import itertools
import argparse
from collections import Counter
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchaudio
import soundfile as sf

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import timm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from hear21passt.base import get_basic_model, get_model_passt


CFG = {
    # Data
    "data_dir_IPI_removed": " ", # add path to standardized IPI dataset
    "data_dir_with_IPI": " ", # add path the original IPI dataset
    "pattern": "**/*.wav",
    "label_regex": r"^([^_]+_[^_]+)",
    "match_key": "basename",  # "basename" or "relative"

    # Split
    "seed": 42,
    "train_fraction": 0.80,
    "test_fraction": 0.20,
    "train_source": "IPI_removed",  # "IPI_removed" when you want to train the standardized IPI dataset and choose "with_IPI" to train with original IPI dataset

    # Training
    "num_workers": 0,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "max_epochs": 50,
    "use_weighted_sampler": True,
    "drop_last": True,
    "saved_model_name": "final_model.ckpt",

    # Grid search  - did not use
    "grid_search": {
        "lr": [5e-5],
        "weight_decay": [1e-5],
    },

    # EfficientNet-specific
    "effnet_target_sr": 250000,
    "effnet_clip_seconds": 2.0,
    "effnet_batch_size": 4,
    "effnet_accumulate_grad_batches": 8,
    "effnet_name": "efficientnet_b0",
    "effnet_pretrained": True,
    "effnet_n_fft": 2048,
    "effnet_win_length": 2048,
    "effnet_hop_length": 256,
    "effnet_spec_power": 1.0,
    "effnet_image_height": 128,
    "effnet_image_width": 1024,
    "effnet_normalize_spectrogram": True,
    "effnet_results_dir_cross_dataset": " ", # add path for storage results of the heldout test
    "effnet_log_dir_cross_dataset": " ", # add path for logs of the heldout test

    # PaSST-specific
    "passt_target_sr": 32000,
    "passt_clip_seconds": 20.0,
    "passt_time_expand_factor": 10.0,
    "passt_batch_size": 1,
    "passt_accumulate_grad_batches": 32,
    "passt_arch": "passt_20sec",
    "passt_input_tdim": 2000,
    "passt_use_class_weighted_loss": False,
    "passt_results_dir_cross_dataset": " ", # add path for storage results of the heldout test
    "passt_log_dir_cross_dataset": " ", # add path for logs of the heldout test
}


# ===================================================================================================================
# Configuration
# ===================================================================================================================

def select_model_config(cfg: dict, model_name: str) -> dict:
    cfg = cfg.copy()

    if model_name == "effnet":
        cfg["target_sr"] = cfg["effnet_target_sr"]
        cfg["clip_seconds"] = cfg["effnet_clip_seconds"]
        cfg["batch_size"] = cfg["effnet_batch_size"]
        cfg["accumulate_grad_batches"] = cfg["effnet_accumulate_grad_batches"]
        cfg["results_dir"] = cfg["effnet_results_dir_cross_dataset"]
        cfg["log_dir"] = cfg["effnet_log_dir_cross_dataset"]
        cfg["efficientnet_name"] = cfg["effnet_name"]
        cfg["pretrained"] = cfg["effnet_pretrained"]
        cfg["n_fft"] = cfg["effnet_n_fft"]
        cfg["win_length"] = cfg["effnet_win_length"]
        cfg["hop_length"] = cfg["effnet_hop_length"]
        cfg["spec_power"] = cfg["effnet_spec_power"]
        cfg["image_height"] = cfg["effnet_image_height"]
        cfg["image_width"] = cfg["effnet_image_width"]
        cfg["normalize_spectrogram"] = cfg["effnet_normalize_spectrogram"]
        cfg["use_class_weighted_loss"] = False

    elif model_name == "passt":
        cfg["target_sr"] = cfg["passt_target_sr"]
        cfg["clip_seconds"] = cfg["passt_clip_seconds"]
        cfg["time_expand_factor"] = cfg["passt_time_expand_factor"]
        cfg["batch_size"] = cfg["passt_batch_size"]
        cfg["accumulate_grad_batches"] = cfg["passt_accumulate_grad_batches"]
        cfg["results_dir"] = cfg["passt_results_dir_cross_dataset"]
        cfg["log_dir"] = cfg["passt_log_dir_cross_dataset"]
        cfg["passt_arch"] = cfg["passt_arch"]
        cfg["input_tdim"] = cfg["passt_input_tdim"]
        cfg["use_class_weighted_loss"] = cfg["passt_use_class_weighted_loss"]

    else:
        raise ValueError("Choose model_name: 'effnet' or 'passt'")

    cfg["model_name"] = model_name
    cfg["experiment"] = "fixed_80_20_test_with_IPI"
    return cfg


def save_config(cfg: dict, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "config.json")
    serializable_cfg = {}
    for k, v in cfg.items():
        try:
            json.dumps(v)
            serializable_cfg[k] = v
        except TypeError:
            serializable_cfg[k] = str(v)
    with open(path, "w") as f:
        json.dump(serializable_cfg, f, indent=2)
    print(f"Saved config to: {path}")


# ===================================================================================================================
# Shared utility functions
# ===================================================================================================================

def list_wavs(data_dir: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(data_dir, pattern), recursive=True)
    paths = [p for p in paths if p.lower().endswith(".wav")]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No wavs found under {data_dir} with pattern {pattern}")
    return paths


def filter_readable_wavs(paths: List[str]) -> List[str]:
    good, bad = [], []
    for p in paths:
        try:
            with sf.SoundFile(p) as f:
                _ = f.samplerate
                _ = f.frames
            good.append(p)
        except Exception as e:
            bad.append((p, str(e)))

    print(f"Readable wavs: {len(good)} | Bad wavs: {len(bad)}")
    if bad:
        print("Example bad file:", bad[0][0])
        print("Reason:", bad[0][1])
    return good


def parse_label_from_filename(path: str, label_regex: str) -> str:
    base = os.path.basename(path)
    m = re.match(label_regex, base)
    return m.group(1) if m else base.split("_")[0]


def build_label_map(paths: List[str], label_regex: str) -> Tuple[Dict[str, int], List[int]]:
    labels = [parse_label_from_filename(p, label_regex) for p in paths]
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    y = [label2id[label] for label in labels]
    return label2id, y


def save_label_map(label2id: Dict[str, int], results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    label2id_path = os.path.join(results_dir, "label2id.json")
    id2label_path = os.path.join(results_dir, "id2label.json")
    id2label = {str(v): k for k, v in label2id.items()}
    with open(label2id_path, "w") as f:
        json.dump(label2id, f, indent=2)
    with open(id2label_path, "w") as f:
        json.dump(id2label, f, indent=2)
    print(f"Saved label2id to: {label2id_path}")
    print(f"Saved id2label to: {id2label_path}")


def save_class_distribution(y: List[int], id2label: Dict[str, str], results_dir: str, filename: str):
    rows = []
    y_t = torch.tensor(y)
    for class_id in sorted(set(y)):
        rows.append({
            "class_id": class_id,
            "label": id2label[str(class_id)],
            "count": int((y_t == class_id).sum().item()),
        })
    path = os.path.join(results_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved class distribution to: {path}")


def make_key(path: str, root: str, match_key: str) -> str:
    if match_key == "basename":
        return os.path.basename(path)
    if match_key == "relative":
        return os.path.relpath(path, root)
    raise ValueError("match_key must be 'basename' or 'relative'")


def make_unique_path_map(paths: List[str], root: str, match_key: str, dataset_name: str) -> Dict[str, str]:
    keys = [make_key(p, root, match_key) for p in paths]
    counts = Counter(keys)
    duplicates = sorted([k for k, c in counts.items() if c > 1])
    if duplicates:
        raise ValueError(
            f"Duplicate match keys found in {dataset_name}. Example: {duplicates[:5]}. "
            f"Use --match-key relative if basenames are duplicated across subfolders."
        )
    return dict(zip(keys, paths))


def build_paired_80_20_split(cfg: dict):
    removed_paths = filter_readable_wavs(list_wavs(cfg["data_dir_IPI_removed"], cfg["pattern"]))
    with_ipi_paths = filter_readable_wavs(list_wavs(cfg["data_dir_with_IPI"], cfg["pattern"]))

    removed_map = make_unique_path_map(
        removed_paths,
        cfg["data_dir_IPI_removed"],
        cfg.get("match_key", "basename"),
        "IPI_removed",
    )
    with_ipi_map = make_unique_path_map(
        with_ipi_paths,
        cfg["data_dir_with_IPI"],
        cfg.get("match_key", "basename"),
        "with_IPI",
    )

    shared_keys = sorted(set(removed_map) & set(with_ipi_map))
    removed_only = sorted(set(removed_map) - set(with_ipi_map))
    with_ipi_only = sorted(set(with_ipi_map) - set(removed_map))

    print(f"Matched files: {len(shared_keys)}")
    print(f"Only in IPI_removed: {len(removed_only)}")
    print(f"Only in with_IPI: {len(with_ipi_only)}")

    if not shared_keys:
        raise ValueError("No matching filenames found between datasets.")

    shared_removed_paths = [removed_map[k] for k in shared_keys]
    label2id, y = build_label_map(shared_removed_paths, cfg["label_regex"])

    class_counts = Counter(y)
    min_count = min(class_counts.values())
    stratify_y = y if min_count >= 2 else None
    if stratify_y is None:
        print("WARNING: At least one class has <2 samples. Using non-stratified split.")

    train_keys, test_keys, train_y, test_y = train_test_split(
        shared_keys,
        y,
        test_size=cfg.get("test_fraction", 0.20),
        random_state=cfg["seed"],
        stratify=stratify_y,
    )

    train_source = cfg.get("train_source", "IPI_removed")
    if train_source == "IPI_removed":
        train_paths = [removed_map[k] for k in train_keys]
    elif train_source == "with_IPI":
        train_paths = [with_ipi_map[k] for k in train_keys]
    else:
        raise ValueError("train_source must be 'IPI_removed' or 'with_IPI'")

    # The test set is always the same held-out 20% of keys, using the with_IPI files.
    test_paths = [with_ipi_map[k] for k in test_keys]

    overlap = set(train_keys) & set(test_keys)
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} filenames are in both train and test.")

    split_rows = []
    for key, y_i, p in zip(train_keys, train_y, train_paths):
        split_rows.append({
            "key": key,
            "split": "train",
            "source_dataset": train_source,
            "path": p,
            "label_id": y_i,
        })
    for key, y_i, p in zip(test_keys, test_y, test_paths):
        split_rows.append({
            "key": key,
            "split": "test",
            "source_dataset": "with_IPI",
            "path": p,
            "label_id": y_i,
        })

    split_df = pd.DataFrame(split_rows)
    return train_paths, train_y, test_paths, test_y, label2id, split_df


def make_weighted_sampler(y: List[int], n_classes: int) -> WeightedRandomSampler:
    y_t = torch.tensor(y, dtype=torch.long)
    counts = torch.bincount(y_t, minlength=n_classes).float()
    class_weights = 1.0 / torch.clamp(counts, min=1.0)
    sample_weights = class_weights[y_t]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def compute_class_weights(y: List[int], n_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(y), minlength=n_classes).float()
    weights = 1.0 / torch.clamp(counts, min=1.0)
    return weights / weights.mean()


def build_train_loader(train_ds: Dataset, cfg: dict, n_classes: int, train_y: List[int]):
    sampler = None
    shuffle = True
    if cfg.get("use_weighted_sampler", True):
        sampler = make_weighted_sampler(train_y, n_classes)
        shuffle = False

    return DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 4),
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", torch.cuda.is_available()),
        sampler=sampler,
        shuffle=shuffle,
        drop_last=cfg.get("drop_last", True),
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )


def build_eval_loader(eval_ds: Dataset, cfg: dict):
    return DataLoader(
        eval_ds,
        batch_size=cfg.get("batch_size", 4),
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", torch.cuda.is_available()),
        shuffle=False,
        drop_last=False,
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )


# ===================================================================================================================
# Dataset classes
# ===================================================================================================================

class BaseAudioDataset(Dataset):
    def __init__(self, paths: List[str], y: List[int], cfg: dict):
        self.paths = paths
        self.y = y
        self.cfg = cfg
        self.target_sr = int(cfg["target_sr"])
        self.clip_seconds = float(cfg["clip_seconds"])
        self.target_len = int(self.target_sr * self.clip_seconds)
        self.resampler_cache = {}

    def __len__(self):
        return len(self.paths)

    def get_resampler(self, orig_sr: int, new_sr: int):
        key = (int(orig_sr), int(new_sr))
        if key not in self.resampler_cache:
            self.resampler_cache[key] = torchaudio.transforms.Resample(orig_freq=int(orig_sr), new_freq=int(new_sr))
        return self.resampler_cache[key]

    def load_mono_wav(self, path: str):
        wav, sr = torchaudio.load(path)
        sr = int(sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, sr

    def pad_or_crop(self, wav: torch.Tensor):
        wav = wav.squeeze(0)
        if wav.numel() >= self.target_len:
            wav = wav[:self.target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.numel()))
        return wav


class PaSSTDataset(BaseAudioDataset):
    def __init__(self, paths: List[str], y: List[int], cfg: dict):
        super().__init__(paths, y, cfg)
        self.time_expand_factor = float(cfg.get("time_expand_factor", 1.0))
        if self.time_expand_factor <= 0:
            raise ValueError("time_expand_factor must be > 0")

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.y[idx]
        wav, sr = self.load_mono_wav(path)

        effective_sr = int(round(sr / self.time_expand_factor))
        effective_sr = max(1, effective_sr)

        if effective_sr != self.target_sr:
            wav = self.get_resampler(effective_sr, self.target_sr)(wav)

        wav = self.pad_or_crop(wav)
        return wav, torch.tensor(label, dtype=torch.long), path


class EfficientNetDataset(BaseAudioDataset):
    def __init__(self, paths: List[str], y: List[int], cfg: dict):
        super().__init__(paths, y, cfg)
        self.image_height = int(cfg["image_height"])
        self.image_width = int(cfg["image_width"])
        self.normalize_spectrogram = bool(cfg.get("normalize_spectrogram", True))
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=int(cfg["n_fft"]),
            win_length=int(cfg["win_length"]),
            hop_length=int(cfg["hop_length"]),
            power=float(cfg["spec_power"]),
            center=True,
            pad_mode="reflect",
        )

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.y[idx]
        wav, sr = self.load_mono_wav(path)

        if sr != self.target_sr:
            wav = self.get_resampler(sr, self.target_sr)(wav)

        wav = self.pad_or_crop(wav)
        spec = self.spec(wav)

        if self.normalize_spectrogram:
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        spec = spec.unsqueeze(0)
        spec = TF.resize(
            spec,
            size=[self.image_height, self.image_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        spec = spec.repeat(3, 1, 1)
        return spec, torch.tensor(label, dtype=torch.long), path


def make_dataset(paths: List[str], y: List[int], cfg: dict) -> Dataset:
    if cfg["model_name"] == "effnet":
        return EfficientNetDataset(paths, y, cfg)
    if cfg["model_name"] == "passt":
        return PaSSTDataset(paths, y, cfg)
    raise ValueError("Unknown model_name in cfg.")


# ===================================================================================================================
# Lightning model classes
# ===================================================================================================================

class BaseFinetuner(pl.LightningModule):
    def __init__(self, n_classes: int, cfg: dict, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "class_weights"])
        self.cfg = cfg
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.model = self.build_model()
        self.criterion = self.build_loss()

        self.train_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.train_f1 = MulticlassF1Score(num_classes=n_classes, average="macro")

        self.test_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.test_f1 = MulticlassF1Score(num_classes=n_classes, average="macro")
        self.test_precision = MulticlassPrecision(num_classes=n_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=n_classes, average="macro")
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)
        self.test_outputs = []

    def build_model(self):
        raise NotImplementedError

    def build_loss(self):
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def on_fit_start(self):
        if hasattr(self.criterion, "weight") and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        f1 = self.train_f1(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1_macro", f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        paths = batch[2] if len(batch) > 2 else None
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.test_cm.update(preds, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_f1_macro", self.test_f1(logits, y), prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_precision_macro", self.test_precision(logits, y), prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_recall_macro", self.test_recall(logits, y), prog_bar=False, on_step=False, on_epoch=True)

        for i in range(y.size(0)):
            self.test_outputs.append({
                "path": paths[i] if paths is not None else "",
                "target": int(y[i].detach().cpu()),
                "pred": int(preds[i].detach().cpu()),
                "correct": int(preds[i].detach().cpu()) == int(y[i].detach().cpu()),
                "prob": probs[i].detach().cpu().tolist(),
            })

    def on_test_epoch_end(self):
        os.makedirs(self.cfg["results_dir"], exist_ok=True)
        predictions_path = os.path.join(self.cfg["results_dir"], "test_predictions.csv")
        confusion_matrix_path = os.path.join(self.cfg["results_dir"], "test_confusion_matrix.csv")

        with open(predictions_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "target", "pred", "correct"] + [f"prob_class_{i}" for i in range(self.n_classes)])
            for row in self.test_outputs:
                writer.writerow([row["path"], row["target"], row["pred"], row["correct"], *row["prob"]])

        cm = self.test_cm.compute().detach().cpu().numpy()
        with open(confusion_matrix_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true/pred"] + [str(i) for i in range(self.n_classes)])
            for i in range(self.n_classes):
                writer.writerow([str(i)] + cm[i].tolist())

        self.test_cm.reset()
        print(f"Saved test predictions to: {predictions_path}")
        print(f"Saved test confusion matrix to: {confusion_matrix_path}")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.get("lr", 2e-5),
            weight_decay=self.cfg.get("weight_decay", 1e-4),
        )


class EfficientNetFinetuner(BaseFinetuner):
    def build_model(self):
        return timm.create_model(
            self.cfg.get("efficientnet_name", "efficientnet_b0"),
            pretrained=bool(self.cfg.get("pretrained", True)),
            num_classes=self.n_classes,
            in_chans=3,
        )


class PaSSTFinetuner(BaseFinetuner):
    def build_model(self):
        model = get_basic_model(mode="logits")
        model.net = get_model_passt(
            arch=self.cfg.get("passt_arch", "passt_20sec"),
            n_classes=self.n_classes,
            input_tdim=self.cfg.get("input_tdim", 2000),
        )
        return model


def make_model(n_classes: int, cfg: dict, train_y: List[int]):
    class_weights = None
    if cfg.get("use_class_weighted_loss", False):
        class_weights = compute_class_weights(train_y, n_classes)

    if cfg["model_name"] == "effnet":
        return EfficientNetFinetuner(n_classes=n_classes, cfg=cfg, class_weights=class_weights)
    if cfg["model_name"] == "passt":
        return PaSSTFinetuner(n_classes=n_classes, cfg=cfg, class_weights=class_weights)
    raise ValueError("Unknown model_name in cfg.")


# ===================================================================================================================
# Reporting and visualization
# ===================================================================================================================

def save_classification_outputs(results_dir: str, n_classes: int, id2label: Dict[str, str]):
    pred_path = os.path.join(results_dir, "test_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"No predictions found at {pred_path}; skipping report plots.")
        return {}

    preds_df = pd.read_csv(pred_path)
    y_true = preds_df["target"].to_numpy()
    y_pred = preds_df["pred"].to_numpy()
    labels = list(range(n_classes))
    target_names = [id2label[str(i)] for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_path = os.path.join(results_dir, "test_classification_report.csv")
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_norm = np.nan_to_num(cm_norm)

    cm_norm_path = os.path.join(results_dir, "test_confusion_matrix_normalized.csv")
    with open(cm_norm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + [str(i) for i in labels])
        for i in labels:
            writer.writerow([str(i)] + cm_norm[i].tolist())

    # Plot normalized confusion matrix.
    fig_size = max(8, min(24, 0.45 * n_classes))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm_norm, interpolation="nearest", aspect="auto")
    plt.title("Normalized confusion matrix: test 20% with_IPI")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, target_names, rotation=90, fontsize=7)
    plt.yticks(tick_marks, target_names, fontsize=7)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, "test_confusion_matrix_normalized.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # Plot per-class F1 scores.
    class_rows = report_df.loc[target_names]
    f1_values = class_rows["f1-score"].to_numpy()
    order = np.argsort(f1_values)
    plt.figure(figsize=(10, max(6, 0.28 * n_classes)))
    plt.barh(np.array(target_names)[order], f1_values[order])
    plt.xlabel("F1-score")
    plt.title("Per-class F1 on test 20% with_IPI")
    plt.xlim(0, 1)
    plt.tight_layout()
    f1_plot_path = os.path.join(results_dir, "test_per_class_f1.png")
    plt.savefig(f1_plot_path, dpi=200)
    plt.close()

    # Plot support per class.
    support_values = class_rows["support"].to_numpy()
    support_order = np.argsort(support_values)
    plt.figure(figsize=(10, max(6, 0.28 * n_classes)))
    plt.barh(np.array(target_names)[support_order], support_values[support_order])
    plt.xlabel("Support")
    plt.title("Test support per class: 20% with_IPI")
    plt.tight_layout()
    support_plot_path = os.path.join(results_dir, "test_class_support.png")
    plt.savefig(support_plot_path, dpi=200)
    plt.close()

    summary = {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "report_path": report_path,
        "cm_norm_path": cm_norm_path,
        "cm_plot_path": cm_plot_path,
        "f1_plot_path": f1_plot_path,
        "support_plot_path": support_plot_path,
    }

    summary_path = os.path.join(results_dir, "test_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = summary_path

    print("\n===== TEST METRICS: 20% with_IPI =====")
    print(json.dumps(summary, indent=2))
    return summary


def plot_split_class_distribution(results_dir: str, id2label: Dict[str, str]):
    split_path = os.path.join(results_dir, "split_assignment.csv")
    if not os.path.exists(split_path):
        return

    df = pd.read_csv(split_path)
    df["label"] = df["label_id"].astype(str).map(id2label)
    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    counts_path = os.path.join(results_dir, "split_class_counts.csv")
    counts.to_csv(counts_path)

    counts = counts.sort_index()
    ax = counts.plot(kind="bar", figsize=(max(10, 0.35 * len(counts)), 6))
    ax.set_title("Class counts by split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "split_class_counts.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved split class counts to: {counts_path}")
    print(f"Saved split class count plot to: {plot_path}")


# ===================================================================================================================
# Experiment runner
# ===================================================================================================================

def run_cross_dataset_experiment(cfg: dict):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg["seed"], workers=True)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    save_config(cfg, cfg["results_dir"])

    if cfg.get("use_weighted_sampler", False) and cfg.get("use_class_weighted_loss", False):
        raise ValueError("Choose either weighted sampler OR class-weighted loss, not both.")

    train_paths, train_y, test_paths, test_y, label2id, split_df = build_paired_80_20_split(cfg)
    n_classes = len(label2id)
    id2label = {str(v): k for k, v in label2id.items()}

    split_path = os.path.join(cfg["results_dir"], "split_assignment.csv")
    split_df.to_csv(split_path, index=False)
    print(f"Saved split assignment to: {split_path}")

    save_label_map(label2id, cfg["results_dir"])
    save_class_distribution(train_y, id2label, cfg["results_dir"], "train_class_distribution.csv")
    save_class_distribution(test_y, id2label, cfg["results_dir"], "test_class_distribution.csv")
    plot_split_class_distribution(cfg["results_dir"], id2label)

    print("\nClasses:")
    for label, idx in label2id.items():
        print(f"{idx}: {label}")

    print("\n===== SPLIT =====")
    print(f"Train samples from {cfg.get('train_source', 'IPI_removed')}: {len(train_paths)}")
    print(f"Test samples from with_IPI: {len(test_paths)}")

    train_ds = make_dataset(train_paths, train_y, cfg)
    test_ds = make_dataset(test_paths, test_y, cfg)
    train_loader = build_train_loader(train_ds, cfg, n_classes, train_y)
    test_loader = build_eval_loader(test_ds, cfg)
    model = make_model(n_classes, cfg, train_y)

    logger = CSVLogger(save_dir=cfg.get("log_dir", "logs_cross_dataset"), name="train_80_removed_test_20_with_ipi")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg["results_dir"],
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-train-loss-{epoch:02d}-{train_loss:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 20),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 8),
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader)

    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"Best train-loss checkpoint: {best_ckpt_path}")

    final_model_path = os.path.join(cfg["results_dir"], cfg.get("saved_model_name", "final_model.ckpt"))
    trainer.save_checkpoint(final_model_path)
    print(f"Saved final model to: {final_model_path}")

    test_results = trainer.test(model, dataloaders=test_loader, ckpt_path=best_ckpt_path if best_ckpt_path else None)
    metric_summary = save_classification_outputs(cfg["results_dir"], n_classes, id2label)

    result = {
        "model_name": cfg["model_name"],
        "train_source": cfg.get("train_source", "IPI_removed"),
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "results_dir": cfg["results_dir"],
        "best_ckpt_path": best_ckpt_path,
        "final_model_path": final_model_path,
    }
    if test_results:
        result.update({k: float(v) for k, v in test_results[0].items() if isinstance(v, (int, float))})
    result.update({k: v for k, v in metric_summary.items() if isinstance(v, (int, float))})
    return result


def run_grid_search(cfg: dict):
    base_results_dir = cfg["results_dir"]
    base_log_dir = cfg["log_dir"]

    grid = cfg.get("grid_search", {})
    lr_values = grid.get("lr", [cfg["lr"]])
    weight_decay_values = grid.get("weight_decay", [cfg["weight_decay"]])

    grid_results = []
    for lr, weight_decay in itertools.product(lr_values, weight_decay_values):
        run_name = f"lr_{lr:g}_wd_{weight_decay:g}".replace("-", "m").replace(".", "p")
        run_cfg = cfg.copy()
        run_cfg["lr"] = lr
        run_cfg["weight_decay"] = weight_decay
        run_cfg["results_dir"] = os.path.join(base_results_dir, run_name)
        run_cfg["log_dir"] = os.path.join(base_log_dir, run_name)

        print("\n" + "=" * 80)
        print(f"GRID RUN: model={run_cfg['model_name']}, lr={lr}, weight_decay={weight_decay}")
        print("=" * 80)

        result = run_cross_dataset_experiment(run_cfg)
        grid_results.append(result)

        os.makedirs(base_results_dir, exist_ok=True)
        pd.DataFrame(grid_results).to_csv(os.path.join(base_results_dir, "grid_search_results_partial.csv"), index=False)

    grid_df = pd.DataFrame(grid_results).sort_values("macro_f1", ascending=False)
    grid_results_path = os.path.join(base_results_dir, "grid_search_results.csv")
    grid_df.to_csv(grid_results_path, index=False)

    print("\n===== GRID SEARCH RESULTS =====")
    print(grid_df)
    print(f"Saved grid search results to: {grid_results_path}")
    print("\n===== BEST CONFIG =====")
    print(grid_df.iloc[0])


# ===================================================================================================================
# Main
# ===================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train on a fixed 80% split and always test on the same matched "
            "20% with_IPI split."
        )
    )
    parser.add_argument("--model", choices=["effnet", "passt", "both"], default="effnet")
    parser.add_argument(
        "--train-source",
        choices=["IPI_removed", "with_IPI", "both"],
        default="IPI_removed",
        help="Dataset used for the 80% training split. The 20% test split is always with_IPI.",
    )
    parser.add_argument(
        "--run-four",
        action="store_true",
        help="Run all four combinations: 2 models x 2 train sources, with one shared test split.",
    )
    parser.add_argument("--grid", action="store_true", help="Run grid search using CFG['grid_search'].")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=None, help="Default 0.80. Test fraction becomes 1 - train_fraction.")
    parser.add_argument("--match-key", choices=["basename", "relative"], default=None, help="How to match files across datasets.")
    parser.add_argument("--results-dir", type=str, default=None, help="Override base results dir. If multiple runs are requested, subfolders are added.")
    parser.add_argument("--log-dir", type=str, default=None, help="Override base log dir. If multiple runs are requested, subfolders are added.")
    return parser.parse_args()


def default_dir_for_train_source(path: str, train_source: str) -> str:
    """Change default output folders so IPI_removed and with_IPI runs do not overwrite each other."""
    if train_source == "IPI_removed":
        return path
    return path.replace("80_IPI_removed", "80_with_IPI")


def apply_cli_overrides(cfg: dict, args, train_source: str, multi_run: bool = False) -> dict:
    cfg = cfg.copy()
    cfg["train_source"] = train_source
    cfg["experiment"] = f"train_80_{train_source}_test_20_with_IPI"

    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.train_fraction is not None:
        if not 0 < args.train_fraction < 1:
            raise ValueError("--train-fraction must be between 0 and 1")
        cfg["train_fraction"] = args.train_fraction
        cfg["test_fraction"] = 1.0 - args.train_fraction
    if args.match_key is not None:
        cfg["match_key"] = args.match_key

    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
        if multi_run:
            cfg["results_dir"] = os.path.join(
                cfg["results_dir"],
                f"{cfg['model_name']}_train_{train_source}_test_with_IPI",
            )
    else:
        cfg["results_dir"] = default_dir_for_train_source(cfg["results_dir"], train_source)

    if args.log_dir is not None:
        cfg["log_dir"] = args.log_dir
        if multi_run:
            cfg["log_dir"] = os.path.join(
                cfg["log_dir"],
                f"{cfg['model_name']}_train_{train_source}_test_with_IPI",
            )
    else:
        cfg["log_dir"] = default_dir_for_train_source(cfg["log_dir"], train_source)

    return cfg


def main():
    args = parse_args()

    if args.run_four:
        models = ["effnet", "passt"]
        train_sources = ["IPI_removed", "with_IPI"]
    else:
        models = ["effnet", "passt"] if args.model == "both" else [args.model]
        train_sources = ["IPI_removed", "with_IPI"] if args.train_source == "both" else [args.train_source]

    multi_run = len(models) > 1 or len(train_sources) > 1

    all_results = []
    for train_source in train_sources:
        for model_name in models:
            cfg = select_model_config(CFG, model_name)
            cfg = apply_cli_overrides(cfg, args, train_source=train_source, multi_run=multi_run)

            print("\n" + "=" * 80)
            print("Selected model:", cfg["model_name"])
            print("Train source:", cfg["train_source"])
            print("Train dataset folder:", cfg["data_dir_IPI_removed"] if cfg["train_source"] == "IPI_removed" else cfg["data_dir_with_IPI"])
            print("Test dataset folder:", cfg["data_dir_with_IPI"])
            print("Results directory:", cfg["results_dir"])
            print("Log directory:", cfg["log_dir"])
            print("Match key:", cfg.get("match_key", "basename"))
            print("=" * 80)

            if args.grid:
                run_grid_search(cfg)
            else:
                result = run_cross_dataset_experiment(cfg)
                all_results.append(result)

    if all_results:
        combined_base = args.results_dir if args.results_dir is not None else os.path.dirname(all_results[0]["results_dir"])
        os.makedirs(combined_base, exist_ok=True)
        combined_path = os.path.join(combined_base, "fixed_test_four_run_results.csv")
        pd.DataFrame(all_results).to_csv(combined_path, index=False)
        print(f"Saved combined model results to: {combined_path}")


if __name__ == "__main__":
    main()
