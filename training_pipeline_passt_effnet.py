# Combined EfficientNet + PaSST cross-validation/grid-search training script

# Usage examples for running in the terminal:

#python training_pipeline.py --model effnet --dataset IPI_removed
#python training_pipeline.py --model passt --dataset with_IPI
#python training_pipeline.py --model passt --dataset IPI_removed --no-grid

# -- model can be "effnet" or "passt" depending on which model you want to train 
# -- dataset can be "with_IPI" or "IPI_removed" depending on which dataset version you want to use


###############
# Libraries
###############

# Environment
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Standard library
import csv
import json
import glob
import re
import itertools
import argparse
from typing import List, Tuple, Dict, Optional

# Data handling
import pandas as pd

# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Audio
import torchaudio
import soundfile as sf

# Training framework
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Metrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix,
)

# Cross-validation / reports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# EfficientNet only
import timm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# PaSST only
from hear21passt.base import get_basic_model, get_model_passt


################
# Configuration
################

CFG = {
    # Data
    "data_dir_IPI_removed": r"/data/welmoed/datasets/European_data/merged_folder_IPI_removed",  # dataset path with IPI removed
    "data_dir_with_IPI": r"/data/welmoed/datasets/European_data/merged_folder_with_IPI",        # dataset path with IPI
    "pattern": "**/*.wav",

    # Shared
    "label_regex": r"^([^_]+_[^_]+)",
    "seed": 42,
    "n_splits": 5,  # number of CV folds
    "num_workers": 0,
    "lr": 2e-5, # learning rate for both models (can be overridden in grid search)
    "weight_decay": 1e-4, # weight decay for both models (can be overridden in grid search)
    "max_epochs": 50, # maximum number of epochs for both models
    "use_weighted_sampler": True,
    "early_stopping_patience": 5,
    "early_stopping_monitor": "val_f1_macro",
    "early_stopping_mode": "max",
    "saved_model_name": "best_model.ckpt",
    "folds_csv": "cv_folds.csv",

    # Shared grid search
    "grid_search": {
        "lr": [1e-5, 2e-5, 5e-5],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },

    # EfficientNet-specific
    "effnet_target_sr": 250000, # sampling rate for efficientnet
    "effnet_clip_seconds": 2.0, # clip length in seconds for efficientnet 
    "effnet_batch_size": 4,
    "effnet_accumulate_grad_batches": 8,
    "effnet_name": "efficientnet_b0", # model specification for timm.create_model()
    "effnet_pretrained": True,
    "effnet_n_fft": 2048,
    "effnet_win_length": 2048,
    "effnet_hop_length": 256,
    "effnet_spec_power": 0.2,
    "effnet_image_height": 128,
    "effnet_image_width": 1024,
    "effnet_normalize_spectrogram": True,
    "effnet_results_dir_with_IPI": "/data/welmoed/models2/Naturalis_code/results_combined/results_efficientnet_with_IPI",
    "effnet_log_dir_with_IPI": "/data/welmoed/models2/Naturalis_code/results_combined/logs_efficientnet_with_IPI",
    "effnet_results_dir_IPI_removed": "/data/welmoed/models2/Naturalis_code/results_combined/results_efficientnet_IPI_removed",
    "effnet_log_dir_IPI_removed": "/data/welmoed/models2/Naturalis_code/results_combined/logs_efficientnet_IPI_removed",

    # PaSST-specific
    "passt_target_sr": 32000, # sampling rate for PaSST (after time expansion)
    "passt_clip_seconds": 20.0, # clip length in seconds for PaSST
    "passt_time_expand_factor": 10.0, # time expansion factor for PaSST
    "passt_batch_size": 1, 
    "passt_accumulate_grad_batches": 32, 
    "passt_arch": "passt_20sec", # model architecture for PaSST
    "passt_input_tdim": 2000,
    "passt_use_class_weighted_loss": False,
    "passt_results_dir_with_IPI": "/data/welmoed/models2/Naturalis_code/results_combined/results_PaSST_with_IPI",
    "passt_log_dir_with_IPI": "/data/welmoed/models2/Naturalis_code/results_combined/logs_PaSST_with_IPI",
    "passt_results_dir_IPI_removed": "/data/welmoed/models2/Naturalis_code/results_combined/results_PaSST_IPI_removed",
    "passt_log_dir_IPI_removed": "/data/welmoed/models2/Naturalis_code/results_combined/logs_PaSST_IPI_removed",
}

###################################
# Configuration selecting functions
###################################


def select_dataset_config(cfg: dict, dataset_version: str) -> dict:
    cfg = cfg.copy()

    if dataset_version == "with_IPI":
        cfg["data_dir"] = cfg["data_dir_with_IPI"]
    elif dataset_version == "IPI_removed":
        cfg["data_dir"] = cfg["data_dir_IPI_removed"]
    else:
        raise ValueError("Choose dataset_version: 'with_IPI' or 'IPI_removed'")

    cfg["dataset_version"] = dataset_version
    return cfg


def select_model_config(cfg: dict, model_name: str) -> dict:
    cfg = cfg.copy()
    dataset_version = cfg.get("dataset_version")
    if dataset_version not in {"with_IPI", "IPI_removed"}:
        raise ValueError("Call select_dataset_config() before select_model_config().")

    if model_name == "effnet":
        cfg["target_sr"] = cfg["effnet_target_sr"]
        cfg["clip_seconds"] = cfg["effnet_clip_seconds"]
        cfg["batch_size"] = cfg["effnet_batch_size"]
        cfg["accumulate_grad_batches"] = cfg["effnet_accumulate_grad_batches"]
        cfg["results_dir"] = cfg[f"effnet_results_dir_{dataset_version}"]
        cfg["log_dir"] = cfg[f"effnet_log_dir_{dataset_version}"]
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
        cfg["results_dir"] = cfg[f"passt_results_dir_{dataset_version}"]
        cfg["log_dir"] = cfg[f"passt_log_dir_{dataset_version}"]
        cfg["passt_arch"] = cfg["passt_arch"]
        cfg["input_tdim"] = cfg["passt_input_tdim"]
        cfg["use_class_weighted_loss"] = cfg["passt_use_class_weighted_loss"]

    else:
        raise ValueError("Choose model_name: 'effnet' or 'passt'")

    cfg["model_name"] = model_name
    return cfg

###########################
# Shared utility functions
###########################

# Finds all wav files under data_dir, returns sorted list of paths.
def list_wavs(data_dir: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(data_dir, pattern), recursive=True)
    paths = [p for p in paths if p.lower().endswith(".wav")]
    paths.sort()

    if not paths:
        raise FileNotFoundError(f"No wavs found under {data_dir} with pattern {pattern}")

    return paths

# Checks which wav files are readable and returns list of good paths.
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

    print(f"Readable wavs: {len(good)} | Bad wavs: {len(bad)}") # print some info about bad files

    if bad:
        print("Example bad file:", bad[0][0])
        print("Reason:", bad[0][1])

    return good

# Parses label from filename using regex. If regex doesn't match, falls back to splitting on underscore and taking first part as label
# file names are e.g. "Genus_species_recordingID.wav"
def parse_label_from_filename(path: str, label_regex: str) -> str:
    base = os.path.basename(path)
    m = re.match(label_regex, base)
    return m.group(1) if m else base.split("_")[0]

# Builds label2id mapping and list of label ids corresponding to each path
def build_label_map(paths: List[str], label_regex: str) -> Tuple[Dict[str, int], List[int]]:
    labels = [parse_label_from_filename(p, label_regex) for p in paths]
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    y = [label2id[label] for label in labels]
    return label2id, y

# Builds stratified K-fold splits and returns list of (fold_idx, train_paths, train_y, val_paths, val_y) for each fold
def build_folds(paths: List[str], y: List[int], n_splits: int, seed: int):
    counts = torch.bincount(torch.tensor(y)).tolist()
    min_count = min(counts)

    if min_count < n_splits:
        raise ValueError(
            f"n_splits={n_splits} is too high. "
            f"The smallest class has only {min_count} samples. "
            f"Set n_splits <= {min_count}."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths, y)):
        train_paths = [paths[i] for i in train_idx]
        train_y = [y[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        val_y = [y[i] for i in val_idx]
        folds.append((fold_idx, train_paths, train_y, val_paths, val_y))

    return folds

# Saves fold assignment to CSV with columns: fold, split (train/val), path, label_id
def save_folds_csv(folds, results_dir: str, filename: str):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)

    rows = []
    for fold_idx, train_paths, train_y, val_paths, val_y in folds:
        for p, label in zip(train_paths, train_y):
            rows.append({"fold": fold_idx, "split": "train", "path": p, "label_id": label})
        for p, label in zip(val_paths, val_y):
            rows.append({"fold": fold_idx, "split": "val", "path": p, "label_id": label})

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved CV fold assignment to: {path}")

# Creates a WeightedRandomSampler for the training set based on class frequencies 
def make_weighted_sampler(y: List[int], n_classes: int) -> WeightedRandomSampler:
    y_t = torch.tensor(y, dtype=torch.long)
    counts = torch.bincount(y_t, minlength=n_classes).float()
    class_weights = 1.0 / torch.clamp(counts, min=1.0)
    sample_weights = class_weights[y_t]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Computes class weights for loss function based on class frequencies
def compute_class_weights(y: List[int], n_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(y), minlength=n_classes).float()
    weights = 1.0 / torch.clamp(counts, min=1.0)
    return weights / weights.mean()

# Builds PyTorch DataLoaders for training and validation datasets, optionally using a weighted sampler for the training set
def build_loaders(train_ds: Dataset, val_ds: Dataset, cfg: dict, n_classes: int, train_y: List[int]):
    batch_size = cfg.get("batch_size", 4)
    num_workers = cfg.get("num_workers", 0)
    pin_memory = cfg.get("pin_memory", torch.cuda.is_available())
    drop_last = cfg.get("drop_last", True)

    sampler = None
    shuffle = True
    if cfg.get("use_weighted_sampler", True): # if True, use weighted sampler to handle class imbalance in training set
        sampler = make_weighted_sampler(train_y, n_classes)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader

# Saves label2id and id2label mappings to JSON files in results_dir
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

# Saves class distribution (class_id, label, count) to CSV in results_dir
def save_class_distribution(y: List[int], id2label: Dict[str, str], results_dir: str):
    rows = []
    y_t = torch.tensor(y)
    for class_id in sorted(set(y)):
        rows.append({
            "class_id": class_id,
            "label": id2label[str(class_id)],
            "count": int((y_t == class_id).sum().item()),
        })
    path = os.path.join(results_dir, "class_distribution.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved class distribution to: {path}")

# Saves config dictionary to JSON in results_dir, converting non-serializable values to strings
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

# Aggregates predictions from all folds, saves combined predictions and classification report to results_dir
def aggregate_cv_results(results_dir: str, n_splits: int, n_classes: int):
    pred_dfs = []
    for fold_idx in range(n_splits):
        pred_path = os.path.join(results_dir, f"fold_{fold_idx}", "val_predictions.csv")
        if os.path.exists(pred_path):
            pred_dfs.append(pd.read_csv(pred_path))
        else:
            print(f"Missing predictions for fold {fold_idx}: {pred_path}")

    if not pred_dfs:
        print("No fold predictions found. Skipping aggregation.")
        return

    all_preds = pd.concat(pred_dfs, ignore_index=True)
    all_preds_path = os.path.join(results_dir, "cv_all_predictions.csv")
    all_preds.to_csv(all_preds_path, index=False)

    y_true = all_preds["target"].to_numpy()
    y_pred = all_preds["pred"].to_numpy()
    labels = list(range(n_classes))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(results_dir, "cv_classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred, labels=labels) # 
    with open(os.path.join(results_dir, "cv_confusion_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + [str(i) for i in labels])
        for i in labels:
            writer.writerow([str(i)] + cm[i].tolist())

    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    with open(os.path.join(results_dir, "cv_confusion_matrix_normalized.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + [str(i) for i in labels])
        for i in labels:
            writer.writerow([str(i)] + cm_norm[i].tolist())

    print(f"Saved all CV predictions to: {all_preds_path}")

# Aggregates fold-level metrics (e.g. val_f1_macro) from each fold, saves fold metrics and summary (mean/std) to results_dir
def summarize_fold_metrics(fold_metrics: List[Dict], results_dir: str):
    df = pd.DataFrame(fold_metrics)
    path = os.path.join(results_dir, "cv_fold_metrics.csv")
    df.to_csv(path, index=False)

    mean_df = df.mean(numeric_only=True).to_frame("mean")
    std_df = df.std(numeric_only=True).to_frame("std")
    summary = pd.concat([mean_df, std_df], axis=1)

    summary_path = os.path.join(results_dir, "cv_metrics_summary.csv")
    summary.to_csv(summary_path)

    print("\n===== CROSS-VALIDATION METRICS =====")
    print(summary)
    print(f"Saved fold metrics to: {path}")
    print(f"Saved metric summary to: {summary_path}")

###########################
# Dataset classes
###########################

# Base class for audio datasets
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

    def get_resampler(self, orig_sr: int, new_sr: int): # caches resamplers to avoid creating a new one for every sample
        key = (int(orig_sr), int(new_sr))
        if key not in self.resampler_cache:
            self.resampler_cache[key] = torchaudio.transforms.Resample(orig_freq=int(orig_sr), new_freq=int(new_sr))
        return self.resampler_cache[key]

    def load_mono_wav(self, path: str): # loads wav file, converts to mono if necessary, and returns (wav_tensor, sample_rate)
        wav, sr = torchaudio.load(path)
        sr = int(sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, sr

    def pad_or_crop(self, wav: torch.Tensor): # crops or pads wav tensor to target_len, returns 1D tensor of length target_len
        wav = wav.squeeze(0)
        if wav.numel() >= self.target_len:
            wav = wav[:self.target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.numel()))
        return wav

# Dataset class for PaSST model 
class PaSSTDataset(BaseAudioDataset):
    def __init__(self, paths: List[str], y: List[int], cfg: dict): # adds time expansion factor to config and checks that it's > 0
        super().__init__(paths, y, cfg)
        self.time_expand_factor = float(cfg.get("time_expand_factor", 1.0))
        if self.time_expand_factor <= 0:
            raise ValueError("time_expand_factor must be > 0")

    def __getitem__(self, idx: int): # loads wav, applies time expansion by resampling to a lower effective sampling rate, then pads/crops to target length and returns (wav_tensor, label, path)
        path = self.paths[idx]
        label = self.y[idx]
        wav, sr = self.load_mono_wav(path)

        effective_sr = int(round(sr / self.time_expand_factor))
        effective_sr = max(1, effective_sr)

        if effective_sr != self.target_sr:
            wav = self.get_resampler(effective_sr, self.target_sr)(wav)

        wav = self.pad_or_crop(wav)
        return wav, torch.tensor(label, dtype=torch.long), path

# Dataset class for EfficientNet model 
class EfficientNetDataset(BaseAudioDataset):
    def __init__(self, paths: List[str], y: List[int], cfg: dict): # adds spectrogram parameters to config and initializes torchaudio.transforms.Spectrogram with those parameters
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

    def __getitem__(self, idx: int): # loads wav, resamples if necessary, pads/crops to target length, converts to spectrogram, normalizes if specified, resizes to image dimensions, repeats to 3 channels, and returns (spec_tensor, label, path)
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


def make_dataset(paths: List[str], y: List[int], cfg: dict) -> Dataset: # factory function to create dataset based on model_name in config
    if cfg["model_name"] == "effnet":
        return EfficientNetDataset(paths, y, cfg)
    if cfg["model_name"] == "passt":
        return PaSSTDataset(paths, y, cfg)
    raise ValueError("Unknown model_name in cfg.")


# ===================================================================================================================
# Lightning classes
# ===================================================================================================================

# Base finetuner class that defines training and validation steps, metrics, and optimizer configuration. 
# #Specific models will inherit from this and implement build_model() to define the architecture.
class BaseFinetuner(pl.LightningModule):
    def __init__(self, n_classes: int, cfg: dict, fold_idx: int): # initializes model, loss function, and metrics based on config and number of classes. fold_idx is used for logging and saving results specific to each fold.
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.n_classes = n_classes
        self.model = self.build_model()
        self.criterion = self.build_loss()

        self.train_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.train_f1 = MulticlassF1Score(num_classes=n_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=n_classes, average="macro")
        self.val_precision = MulticlassPrecision(num_classes=n_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=n_classes, average="macro")
        self.val_cm = MulticlassConfusionMatrix(num_classes=n_classes)
        self.val_outputs = []

    def build_model(self): 
        raise NotImplementedError

    def build_loss(self):
        return nn.CrossEntropyLoss()

    def on_fit_start(self):
        if hasattr(self.criterion, "weight") and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self.device)

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str): #
        x, y = batch[0], batch[1]
        paths = batch[2] if len(batch) > 2 else None

        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            acc = self.train_acc(logits, y)
            f1 = self.train_f1(logits, y)
        else:
            acc = self.val_acc(logits, y)
            f1 = self.val_f1(logits, y)
            precision = self.val_precision(logits, y)
            recall = self.val_recall(logits, y)
            self.val_cm.update(preds, y)
            self.log("val_precision_macro", precision, prog_bar=False, on_step=False, on_epoch=True)
            self.log("val_recall_macro", recall, prog_bar=False, on_step=False, on_epoch=True)

            probs = torch.softmax(logits, dim=1)
            for i in range(y.size(0)):
                self.val_outputs.append({
                    "fold": self.fold_idx,
                    "path": paths[i] if paths is not None else "",
                    "target": int(y[i].detach().cpu()),
                    "pred": int(preds[i].detach().cpu()),
                    "prob": probs[i].detach().cpu().tolist(),
                })

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_f1_macro", f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_cm.reset()
            self.val_outputs = []
            return

        fold_dir = os.path.join(self.cfg["results_dir"], f"fold_{self.fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        predictions_path = os.path.join(fold_dir, "val_predictions.csv")
        confusion_matrix_path = os.path.join(fold_dir, "val_confusion_matrix.csv")

        with open(predictions_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "path", "target", "pred"] + [f"prob_class_{i}" for i in range(self.n_classes)])
            for row in self.val_outputs:
                writer.writerow([row["fold"], row["path"], row["target"], row["pred"], *row["prob"]])

        cm = self.val_cm.compute().detach().cpu()
        with open(confusion_matrix_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true/pred"] + [str(i) for i in range(self.n_classes)])
            for i in range(self.n_classes):
                writer.writerow([str(i)] + cm[i].tolist())

        self.val_cm.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.get("lr", 2e-5),
            weight_decay=self.cfg.get("weight_decay", 1e-4),
        )

# EfficientNet finetuner that inherits from BaseFinetuner and implements build_model() to create an EfficientNet model using timm.create_model() with parameters from config
class EfficientNetFinetuner(BaseFinetuner):
    def build_model(self):
        return timm.create_model(
            self.cfg.get("efficientnet_name", "efficientnet_b0"),
            pretrained=bool(self.cfg.get("pretrained", True)),
            num_classes=self.n_classes,
            in_chans=3,
        )

# PaSST finetuner that inherits from BaseFinetuner and implements build_model() to create a PaSST model using get_model_passt() with parameters from config. Also overrides build_loss() to use class-weighted loss if specified in config.
class PaSSTFinetuner(BaseFinetuner):
    def __init__(self, n_classes: int, cfg: dict, fold_idx: int, class_weights: Optional[torch.Tensor] = None):
        self.class_weights = class_weights
        super().__init__(n_classes, cfg, fold_idx)

    def build_model(self):
        model = get_basic_model(mode="logits")
        model.net = get_model_passt(
            arch=self.cfg.get("passt_arch", "passt_20sec"),
            n_classes=self.n_classes,
            input_tdim=self.cfg.get("input_tdim", 2000),
        )
        return model

    def build_loss(self):
        return nn.CrossEntropyLoss(weight=self.class_weights)

# Factory function to create model based on model_name in config, passing n_classes, cfg, fold_idx, and train_y (for computing class weights if needed)
def make_model(n_classes: int, cfg: dict, fold_idx: int, train_y: List[int]):
    if cfg["model_name"] == "effnet":
        return EfficientNetFinetuner(n_classes=n_classes, cfg=cfg, fold_idx=fold_idx)

    if cfg["model_name"] == "passt":
        class_weights = None
        if cfg.get("use_class_weighted_loss", False):
            class_weights = compute_class_weights(train_y, n_classes)
        return PaSSTFinetuner(n_classes=n_classes, cfg=cfg, fold_idx=fold_idx, class_weights=class_weights)

    raise ValueError("Unknown model_name in cfg.")

###########################
# Experiment runner
###########################

# Main function to run cross-validation experiment based on config. Handles data loading, fold creation, model training, and result aggregation.
def run_cv_experiment(cfg: dict):
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg["seed"], workers=True)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    save_config(cfg, cfg["results_dir"])

    if cfg.get("use_weighted_sampler", False) and cfg.get("use_class_weighted_loss", False):
        raise ValueError("Choose either weighted sampler OR class-weighted loss, not both.")

    paths = list_wavs(cfg["data_dir"], cfg["pattern"])
    paths = filter_readable_wavs(paths)

    label2id, y = build_label_map(paths, cfg["label_regex"])
    n_classes = len(label2id)
    id2label = {str(v): k for k, v in label2id.items()}

    save_label_map(label2id, cfg["results_dir"])
    save_class_distribution(y, id2label, cfg["results_dir"])

    print("Classes:")
    for label, idx in label2id.items():
        print(f"{idx}: {label}")

    folds = build_folds(paths=paths, y=y, n_splits=cfg.get("n_splits", 5), seed=cfg["seed"])
    save_folds_csv(folds=folds, results_dir=cfg["results_dir"], filename=cfg.get("folds_csv", "cv_folds.csv"))

    fold_metrics = []

    for fold_idx, train_paths, train_y, val_paths, val_y in folds:
        print(f"\n===== FOLD {fold_idx + 1}/{cfg['n_splits']} =====")
        print(f"Train samples: {len(train_paths)} | Validation samples: {len(val_paths)}")

        train_ds = make_dataset(train_paths, train_y, cfg)
        val_ds = make_dataset(val_paths, val_y, cfg)
        train_loader, val_loader = build_loaders(train_ds, val_ds, cfg, n_classes, train_y)
        model = make_model(n_classes, cfg, fold_idx, train_y)

        logger = CSVLogger(save_dir=cfg.get("log_dir", "logs_cv"), name=f"fold_{fold_idx}")
        fold_results_dir = os.path.join(cfg["results_dir"], f"fold_{fold_idx}")
        os.makedirs(fold_results_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_results_dir,
            monitor=cfg.get("early_stopping_monitor", "val_f1_macro"),
            mode=cfg.get("early_stopping_mode", "max"),
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}-{val_f1_macro:.4f}",
        )

        early_stopping_callback = EarlyStopping(
            monitor=cfg.get("early_stopping_monitor", "val_f1_macro"),
            mode=cfg.get("early_stopping_mode", "max"),
            patience=cfg.get("early_stopping_patience", 5),
            verbose=True,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.get("max_epochs", 20),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            accumulate_grad_batches=cfg.get("accumulate_grad_batches", 8),
            log_every_n_steps=10,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        trainer.fit(model, train_loader, val_loader)

        best_ckpt_path = checkpoint_callback.best_model_path
        print(f"Best checkpoint for fold {fold_idx}: {best_ckpt_path}")

        final_model_path = os.path.join(fold_results_dir, cfg.get("saved_model_name", "best_model.ckpt"))
        trainer.save_checkpoint(final_model_path)
        print(f"Saved final fold model to: {final_model_path}")

        val_results = trainer.validate(model, dataloaders=val_loader, ckpt_path="best")
        fold_result = val_results[0]
        fold_result["fold"] = fold_idx
        fold_result["lr"] = cfg["lr"]
        fold_result["weight_decay"] = cfg["weight_decay"]
        fold_result["best_ckpt_path"] = best_ckpt_path
        fold_metrics.append(fold_result)

    summarize_fold_metrics(fold_metrics, cfg["results_dir"])
    aggregate_cv_results(cfg["results_dir"], cfg.get("n_splits", 5), n_classes)

    summary_df = pd.read_csv(os.path.join(cfg["results_dir"], "cv_metrics_summary.csv"), index_col=0)
    return {
        "model_name": cfg["model_name"],
        "dataset_version": cfg["dataset_version"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "results_dir": cfg["results_dir"],
        "val_f1_macro_mean": float(summary_df.loc["val_f1_macro", "mean"]),
        "val_f1_macro_std": float(summary_df.loc["val_f1_macro", "std"]),
    }

# Runs a grid search over specified learning rates and weight decays, running a full CV experiment for each combination and saving results to CSV.
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
        print(f"GRID SEARCH RUN: model={run_cfg['model_name']}, dataset={run_cfg['dataset_version']}, lr={lr}, weight_decay={weight_decay}")
        print("=" * 80)

        result = run_cv_experiment(run_cfg)
        grid_results.append(result)

        os.makedirs(base_results_dir, exist_ok=True)
        pd.DataFrame(grid_results).to_csv(os.path.join(base_results_dir, "grid_search_results_partial.csv"), index=False)

    grid_df = pd.DataFrame(grid_results).sort_values("val_f1_macro_mean", ascending=False)
    grid_results_path = os.path.join(base_results_dir, "grid_search_results.csv")
    grid_df.to_csv(grid_results_path, index=False)

    print("\n===== GRID SEARCH RESULTS =====")
    print(grid_df)
    print(f"Saved grid search results to: {grid_results_path}")
    print("\n===== BEST CONFIG =====")
    print(grid_df.iloc[0])


###########################
# Main
###########################

# Parses command-line arguments to select model, dataset, and whether to run grid search or single CV experiment.
def parse_args():
    parser = argparse.ArgumentParser(description="Run EfficientNet or PaSST CV/grid-search experiment.")
    parser.add_argument("--model", choices=["effnet", "passt"], default="effnet")
    parser.add_argument("--dataset", choices=["with_IPI", "IPI_removed"], default="with_IPI")
    parser.add_argument("--no-grid", action="store_true", help="Run one CV experiment using CFG['lr'] and CFG['weight_decay'].")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = select_dataset_config(CFG, args.dataset)
    cfg = select_model_config(cfg, args.model)

    print("Selected model:", cfg["model_name"])
    print("Selected dataset:", cfg["dataset_version"])
    print("Data directory:", cfg["data_dir"])
    print("Results directory:", cfg["results_dir"])
    print("Log directory:", cfg["log_dir"])

    if args.no_grid:
        run_cv_experiment(cfg)
    else:
        run_grid_search(cfg)


if __name__ == "__main__":
    main()
