"""Utility helpers for the KTH ARCCnet practical notebook.

The notebook keeps the analysis and plotting steps explicit for teaching.
This module contains reusable mechanics: path discovery, dataset loading,
quality flag decoding, FITS I/O, and image preprocessing.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import shutil
import tarfile
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import zoom as nd_zoom
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm


DATASET_NAME = "arccnet-ar-classification-toy-v20251016"
MISSING_PATH_TOKENS = {"", "none", "nan", "<na>"}

HMI_GOOD_FLAGS = {"", "0x00000000", "0x00000400"}
MDI_GOOD_FLAGS = {"", "0x00000000", "0x00000200"}

QUALITY_FLAG_MEANINGS = {
    "MDI": {
        0x00000001: "Missing Data",
        0x00000002: "Saturated Pixel",
        0x00000004: "Truncated (Top)",
        0x00000008: "Truncated (Bottom)",
        0x00000200: "Shutterless Mode",
        0x00010000: "Cosmic Ray",
        0x00020000: "Calibration Mode",
        0x00040000: "Image Bad",
    },
    "HMI": {
        0x00000020: "Missing >50% Data",
        0x00000080: "Limb Darkening Correction Bad",
        0x00000400: "Shutterless Mode",
        0x00001000: "Partial/Missing Frame",
        0x00010000: "Cosmic Ray",
    },
}


def find_dataset_dir(dataset_dir=None, dataset_name=DATASET_NAME):
    """Find the toy dataset in local, repo-root, or common Colab locations."""
    candidates = [
        Path(dataset_dir).expanduser(),
    ] if dataset_dir is not None else []

    candidates.extend(
        [
            Path(dataset_name),
            Path("KTH_ML4Space2026") / dataset_name,
            Path("KTH_PhD_Course") / dataset_name,
            Path.cwd() / dataset_name,
            Path.cwd() / "KTH_ML4Space2026" / dataset_name,
            Path.cwd() / "KTH_PhD_Course" / dataset_name,
            Path("/content") / dataset_name,
            Path("/content") / "KTH_ML4Space2026" / dataset_name,
            Path("/content") / "KTH_PhD_Course" / dataset_name,
            Path("/content/drive/MyDrive") / dataset_name,
            Path("/content/drive/MyDrive") / "KTH_ML4Space2026" / dataset_name,
            Path("/content/drive/MyDrive") / "KTH_PhD_Course" / dataset_name,
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        parquet_file = candidate / "region_classification.parq"
        fits_dir = candidate / "fits"
        if parquet_file.exists() and fits_dir.exists():
            return candidate

    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Could not find the toy dataset. Set DATASET_DIR to the folder that "
        "contains region_classification.parq and fits/.\n\nSearched:\n" + searched
    )


def find_dataset_dir_recursive(root_dir):
    """Find a dataset folder below root_dir by looking for parquet + fits/."""
    root_dir = Path(root_dir).expanduser()
    if not root_dir.exists():
        raise FileNotFoundError(f"{root_dir} does not exist.")

    for candidate in [root_dir, *root_dir.rglob("*")]:
        if not candidate.is_dir():
            continue
        if (candidate / "region_classification.parq").exists() and (candidate / "fits").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find region_classification.parq and fits/ below "
        f"{root_dir}."
    )


def download_google_drive_dataset(url, dest_dir=".", dataset_name=DATASET_NAME):
    """Download a Google Drive dataset archive/folder and return its local path.

    The URL may point to a zip/tar archive or a public Drive folder. Archives can
    either contain ``region_classification.parq`` and ``fits/`` directly, or wrap
    them inside a top-level dataset directory.
    """
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        return find_dataset_dir(dest_dir / dataset_name, dataset_name=dataset_name)
    except FileNotFoundError:
        pass

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Google Drive download support requires gdown. Run "
            "`pip install gdown` or rerun the notebook setup cell."
        ) from exc

    download_dir = dest_dir / "_dataset_download"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True)

    if "folders/" in str(url):
        gdown.download_folder(url, output=str(download_dir), quiet=False, use_cookies=False)
        search_root = download_dir
    else:
        archive_path = download_dir / "dataset_download"
        gdown.download(url, str(archive_path), quiet=False, fuzzy=True)
        search_root = dest_dir
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                tf.extractall(dest_dir)
        else:
            raise ValueError(
                "Downloaded file is not a zip/tar archive. If this is a Google "
                "Drive folder, use the folder share URL."
            )

    try:
        return find_dataset_dir(search_root, dataset_name=dataset_name)
    except FileNotFoundError:
        return find_dataset_dir_recursive(search_root)


def is_missing(val):
    """Return True for empty, None, or NaN-like path values."""
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in MISSING_PATH_TOKENS
    try:
        return bool(pd.isna(val))
    except Exception:
        return False


def nonempty_path_mask(series):
    """Vectorized mask for non-empty path values."""
    text = series.astype("string").str.strip().str.lower()
    return series.notna() & (~text.isin(MISSING_PATH_TOKENS))


def load_dataset(parquet_path):
    """Load metadata and derive dates, year, label, and instrument columns."""
    df = pd.read_parquet(parquet_path)

    jd = df["target_time.jd1"] + df["target_time.jd2"]
    df["dates"] = pd.to_datetime(Time(jd.values, format="jd").iso)
    df["year"] = df["dates"].dt.year

    mag = df["magnetic_class"].astype(str).str.strip()
    reg = df["region_type"].astype(str).str.strip()
    df["label"] = np.where((mag != "") & (mag != "nan"), mag, reg)

    hmi_ok = nonempty_path_mask(df["path_image_cutout_hmi"])
    mdi_ok = nonempty_path_mask(df["path_image_cutout_mdi"])
    df["instrument"] = np.where(hmi_ok, "HMI", np.where(mdi_ok, "MDI", "unknown"))

    return df


def normalize_flag(val):
    if val is None:
        return "0x00000000"
    try:
        if pd.isna(val):
            return "0x00000000"
    except Exception:
        pass
    text = str(val).strip().lower()
    if text in MISSING_PATH_TOKENS:
        return "0x00000000"
    hex_part = text[2:] if text.startswith("0x") else text
    if all(c in "0123456789abcdef" for c in hex_part) and hex_part:
        return f"0x{hex_part.zfill(8)}"
    return text


def decode_flag(val, instrument):
    """Return a human-readable description of a quality flag."""
    flag_dict = QUALITY_FLAG_MEANINGS.get(instrument.upper(), {})
    try:
        text = str(val).strip().lower()
        if text in MISSING_PATH_TOKENS:
            return "Good Quality"
        hex_str = text[2:] if text.startswith("0x") else text
        flag_int = int(hex_str, 16)
        if flag_int == 0:
            return "Good Quality"
        meanings = [m for bit, m in flag_dict.items() if flag_int & bit]
        return " | ".join(meanings) or "Unknown Flag"
    except (ValueError, TypeError):
        return "Invalid Format"


def summarise_quality(df, col, instrument):
    """Return a flag-frequency table for one quality column."""
    if col not in df.columns:
        return None
    normalized = df[col].apply(normalize_flag)
    counts = normalized.value_counts().reset_index()
    counts.columns = ["Flag_normalized", "Count"]
    total = counts["Count"].sum()
    counts["Pct"] = (counts["Count"] / total * 100).round(2).map("{:.2f}%".format)
    counts["Description"] = counts["Flag_normalized"].apply(
        lambda f: decode_flag(f, instrument)
    )
    return counts.sort_values("Count", ascending=False).reset_index(drop=True)


def clean_dataset(df):
    """Keep rows with at least one valid, good-quality cutout path."""
    hmi_ok = nonempty_path_mask(df["path_image_cutout_hmi"])
    mdi_ok = nonempty_path_mask(df["path_image_cutout_mdi"])
    hmi_quality = df["QUALITY_hmi"].apply(normalize_flag).isin(HMI_GOOD_FLAGS)
    mdi_quality = df["QUALITY_mdi"].apply(normalize_flag).isin(MDI_GOOD_FLAGS)

    keep = (hmi_ok & hmi_quality) | (mdi_ok & mdi_quality)
    return df[keep].copy().reset_index(drop=True)


def fits_paths_for_row(row, fits_dir):
    """Resolve magnetogram and continuum FITS paths for a dataframe row."""
    fits_dir = Path(fits_dir)
    if not is_missing(row.get("path_image_cutout_hmi")):
        mag_name = os.path.basename(str(row["path_image_cutout_hmi"]))
    else:
        mag_name = os.path.basename(str(row["path_image_cutout_mdi"]))

    cont_name = mag_name.replace("_mag_", "_cont_")
    return fits_dir / mag_name, fits_dir / cont_name


def load_fits_pair(row, fits_dir):
    """Load magnetogram and continuum arrays for a single row."""
    mag_path, cont_path = fits_paths_for_row(row, fits_dir)
    with fits.open(mag_path) as h:
        mag = h[1].data.astype(float)
    with fits.open(cont_path) as h:
        cont = h[1].data.astype(float)
    return mag, cont


def pixel_stats(arr):
    """Compute finite-pixel summary statistics for a FITS image array."""
    v = arr[np.isfinite(arr)]
    nan_frac = 1 - len(v) / arr.size
    return {
        "mean": v.mean(),
        "std": v.std(),
        "min": v.min(),
        "max": v.max(),
        "p1": np.percentile(v, 1),
        "p99": np.percentile(v, 99),
        "nan_fraction": nan_frac,
    }


def parse_dim(val):
    """Parse dimension strings like '(128, 256)' into (height, width)."""
    if is_missing(val):
        return None, None
    try:
        dims = ast.literal_eval(str(val))
        if isinstance(dims, (list, tuple)) and len(dims) == 2:
            return int(dims[0]), int(dims[1])
    except Exception:
        pass
    return None, None


def preprocess_image(arr, target_size=64, divisor=800.0):
    """Sanitize, physically normalize, pad, and resize a 2-D magnetogram array.

    Mirrors the ARCCnet production pipeline (hardtanh normalization):
    1. Replace NaN / ±Inf with 0  (zero field — physically correct).
    2. Divide by ``divisor`` (default 800 G) and clamp to [-1, 1].
    3. Zero-pad to a square canvas; 0 is the neutral midpoint in [-1, 1].
    4. Resize to ``target_size × target_size``.
    """
    arr = np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr / divisor, -1.0, 1.0).astype(np.float32)

    h, w = arr.shape
    s = max(h, w)
    canvas = np.zeros((s, s), dtype=np.float32)
    ph, pw = (s - h) // 2, (s - w) // 2
    canvas[ph : ph + h, pw : pw + w] = arr

    if s != target_size:
        canvas = nd_zoom(canvas, target_size / s, order=1)
    return canvas[:target_size, :target_size].astype(np.float32)


def preprocess_continuum(arr, target_size=64):
    """Sanitize, inverted-normalize, pad, and resize a 2-D continuum array.

    Mirrors the ARCCnet production ``normalize_continuum`` pipeline:
    1. Replace NaN / ±Inf with 0.
    2. Inverted per-image min-max to [0, 1]: background → near 0,
       brightest points → near 1.
    3. Zero-pad to a square canvas.
    4. Resize to ``target_size × target_size``.
    """
    arr = arr.astype(float)
    finite = arr[np.isfinite(arr)]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(finite) == 0 or finite.max() == finite.min():
        arr = np.zeros(arr.shape, dtype=np.float32)
    else:
        min_val, max_val = finite.min(), finite.max()
        arr = 1.0 - (arr - min_val) / (max_val - min_val)
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

    h, w = arr.shape
    s = max(h, w)
    canvas = np.zeros((s, s), dtype=np.float32)
    ph, pw = (s - h) // 2, (s - w) // 2
    canvas[ph : ph + h, pw : pw + w] = arr

    if s != target_size:
        canvas = nd_zoom(canvas, target_size / s, order=1)
    return canvas[:target_size, :target_size].astype(np.float32)


class SolarFITSDataset(Dataset):
    """Loads magnetogram cutouts on the fly for PyTorch models."""

    def __init__(self, df, indices, label_encoder, fits_dir, target_size=64, flatten=False):
        self.df = df.reset_index(drop=True)
        self.indices = indices
        self.le = label_encoder
        self.fits_dir = Path(fits_dir)
        self.target_size = target_size
        self.flatten = flatten

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.indices[idx]]
        mag, _ = load_fits_pair(row, self.fits_dir)
        img = preprocess_image(mag, target_size=self.target_size)
        if self.flatten:
            img = img.ravel()
        else:
            img = img[None, :, :]
        label = self.le.transform([row["label_coarse"]])[0]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


class SolarFITSDualDataset(Dataset):
    """Loads paired magnetogram/continuum cutouts for PyTorch models."""

    def __init__(
        self,
        df,
        indices,
        label_encoder,
        fits_dir,
        target_size=64,
        as_volume=False,
    ):
        self.df = df.reset_index(drop=True)
        self.indices = indices
        self.le = label_encoder
        self.fits_dir = Path(fits_dir)
        self.target_size = target_size
        self.as_volume = as_volume

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.indices[idx]]
        mag, cont = load_fits_pair(row, self.fits_dir)
        mag = preprocess_image(mag, target_size=self.target_size)
        cont = preprocess_continuum(cont, target_size=self.target_size)
        img = np.stack([mag, cont], axis=0).astype(np.float32)
        if self.as_volume:
            img = img[None, :, :, :]
        label = self.le.transform([row["label_coarse"]])[0]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


def train_classifier(
    model,
    train_dl,
    val_dl,
    criterion,
    optimizer,
    scheduler=None,
    epochs=10,
    device="cpu",
    name="Model",
    print_every=5,
    show_progress=False,
    early_stopping_patience=None,
    early_stopping_metric="val_loss",
    early_stopping_min_delta=0.0,
    restore_best=True,
    tb_writer=None,
):
    """Train a classifier and return train/validation loss and accuracy curves.

    By default the function restores the weights from the epoch with the lowest
    validation loss. This keeps the notebook results tied to the best validation
    checkpoint instead of the final, often overfit, epoch.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "best_epoch": None,
        "best_val_loss": None,
        "best_metric": early_stopping_metric,
        "best_score": None,
    }
    metric_modes = {"val_loss": "min", "val_acc": "max", "val_macro_f1": "max"}
    if early_stopping_metric not in metric_modes:
        valid = ", ".join(metric_modes)
        raise ValueError(f"early_stopping_metric must be one of: {valid}")

    best_score = float("inf") if metric_modes[early_stopping_metric] == "min" else -float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_ok = 0.0, 0
        for xb, yb in tqdm(
            train_dl,
            desc=f"{name} {epoch}/{epochs} train",
            unit="batch",
            leave=False,
            disable=not show_progress,
        ):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            train_ok += (out.argmax(1) == yb).sum().item()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss, val_ok = 0.0, 0
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in tqdm(
                val_dl,
                desc=f"{name} {epoch}/{epochs} val",
                unit="batch",
                leave=False,
                disable=not show_progress,
            ):
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * len(yb)
                pred = out.argmax(1)
                val_ok += (pred == yb).sum().item()
                val_pred.extend(pred.cpu().numpy())
                val_true.extend(yb.cpu().numpy())

        history["train_loss"].append(train_loss / len(train_dl.dataset))
        history["val_loss"].append(val_loss / len(val_dl.dataset))
        history["train_acc"].append(train_ok / len(train_dl.dataset))
        history["val_acc"].append(val_ok / len(val_dl.dataset))
        val_macro_f1 = precision_recall_fscore_support(
            val_true,
            val_pred,
            average="macro",
            zero_division=0,
        )[2]
        history["val_macro_f1"].append(val_macro_f1)

        if tb_writer is not None:
            tb_writer.add_scalar("loss/train", history["train_loss"][-1], epoch)
            tb_writer.add_scalar("loss/val", history["val_loss"][-1], epoch)
            tb_writer.add_scalar("accuracy/train", history["train_acc"][-1], epoch)
            tb_writer.add_scalar("accuracy/val", history["val_acc"][-1], epoch)
            tb_writer.add_scalar("macro_f1/val", history["val_macro_f1"][-1], epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        score_now = history[early_stopping_metric][-1]
        if metric_modes[early_stopping_metric] == "min":
            improved = score_now < best_score - early_stopping_min_delta
        else:
            improved = score_now > best_score + early_stopping_min_delta

        if improved:
            best_score = score_now
            history["best_epoch"] = epoch
            history["best_score"] = best_score
            history["best_val_loss"] = history["val_loss"][-1]
            epochs_without_improvement = 0
            if restore_best:
                best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train  loss={history['train_loss'][-1]:.4f}  "
                f"acc={history['train_acc'][-1]:.3f}  |  "
                f"val    loss={history['val_loss'][-1]:.4f}  "
                f"acc={history['val_acc'][-1]:.3f}  "
                f"macro-F1={history['val_macro_f1'][-1]:.3f}"
            )

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; best {early_stopping_metric} was "
                f"{best_score:.4f} at epoch {history['best_epoch']}."
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restored {name} weights from epoch {history['best_epoch']} "
            f"(best {early_stopping_metric}={history['best_score']:.4f})."
        )

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook helpers — reusable plotting, data-loading, and evaluation utilities
# ═══════════════════════════════════════════════════════════════════════════════


def solar_grid(ax, n_meridians=12, n_parallels=12, n_pts=300):
    """Draw heliographic grid on an existing Axes."""
    phis = np.linspace(0, 2 * np.pi, n_meridians, endpoint=False)
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_parallels)
    theta = np.linspace(-np.pi / 2, np.pi / 2, n_pts)
    for phi in phis:
        ax.plot(
            np.cos(theta) * np.sin(phi), np.sin(theta), "k-", lw=0.15, alpha=0.6
        )
    for lat in lats:
        ax.plot(
            np.cos(lat) * np.sin(theta),
            np.full(n_pts, np.sin(lat)),
            "k-",
            lw=0.15,
            alpha=0.6,
        )


def plot_class_histogram(
    series,
    title="Class Distribution",
    color_map=None,
    figsize=(13, 5),
    show_pct=True,
    horizontal=False,
):
    """Bar chart of label frequencies with optional percentage annotations."""
    counts = series.value_counts()
    total = counts.sum()
    colors = (
        [color_map.get(lbl, "#888888") for lbl in counts.index]
        if color_map
        else None
    )

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=figsize)
        if horizontal:
            bars = ax.barh(
                counts.index,
                counts.values,
                color=colors,
                edgecolor="k",
                linewidth=0.5,
            )
            ax.set_xlabel("Count", fontsize=15)
            ax.tick_params(axis="both", labelsize=12)
            for bar, val in zip(bars, counts.values):
                pct = val / total * 100
                label = f"{val:,} ({pct:.1f}%)" if show_pct else f"{val:,}"
                ax.text(
                    bar.get_width() + total * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    va="center",
                    fontsize=12,
                )
        else:
            bars = ax.bar(
                counts.index,
                counts.values,
                color=colors,
                edgecolor="k",
                linewidth=0.5,
            )
            ax.set_ylabel("Count", fontsize=15)
            ax.tick_params(axis="both", labelsize=12)
            for bar, val in zip(bars, counts.values):
                pct = val / total * 100
                label = f"{val:,} ({pct:.1f}%)" if show_pct else f"{val:,}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    clip_on=True,
                )
        ax.set_title(title, fontsize=18, pad=14)
        plt.tight_layout()
        plt.show()


def make_loaders(
    train_ds, val_ds, test_ds, batch_size=32, num_workers=0
):
    """Create train/val/test DataLoaders."""
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(val_ds, batch_size=64, num_workers=num_workers),
        DataLoader(test_ds, batch_size=64, num_workers=num_workers),
    )


def split_tensor_datasets(array, train_idx, val_idx, test_idx, y):
    """Create train/val/test TensorDatasets from a preloaded numpy array."""
    return (
        TensorDataset(
            torch.from_numpy(array[train_idx]).float(),
            torch.from_numpy(y[train_idx]).long(),
        ),
        TensorDataset(
            torch.from_numpy(array[val_idx]).float(),
            torch.from_numpy(y[val_idx]).long(),
        ),
        TensorDataset(
            torch.from_numpy(array[test_idx]).float(),
            torch.from_numpy(y[test_idx]).long(),
        ),
    )


def select_torch_device():
    """Return the best available PyTorch device and a readable backend label."""
    candidates = []
    if torch.cuda.is_available():
        # PyTorch exposes ROCm accelerators through the CUDA device API.
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        candidates.append((torch.device("cuda"), backend))

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        candidates.append((torch.device("mps"), "mps"))

    candidates.append((torch.device("cpu"), "cpu"))

    for device, backend in candidates:
        try:
            _ = torch.empty(1, device=device) + 1
            return device, backend
        except Exception:
            continue

    return torch.device("cpu"), "cpu"


def _clean_run_name(value):
    """Return a filesystem-friendly TensorBoard run name."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "run"


def _jsonable(value):
    """Convert common notebook values to JSON-friendly scalars."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_tensorboard_writer(log_dir="runs", run_name=None, hparams=None):
    """Create a TensorBoard SummaryWriter with a Colab-friendly default layout.

    TensorBoard is imported lazily so the rest of the notebook works even when
    the optional dependency is missing.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the optional 'tensorboard' package. "
            "In Colab, run `%load_ext tensorboard`; if import still fails, run "
            "`!pip install tensorboard` and restart the runtime."
        ) from exc

    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    run_name = _clean_run_name(run_name or f"run_{timestamp}")
    if not run_name.endswith(timestamp):
        run_name = f"{run_name}_{timestamp}"

    run_dir = Path(log_dir) / run_name
    writer = SummaryWriter(log_dir=str(run_dir))

    if hparams:
        clean_hparams = {str(k): _jsonable(v) for k, v in hparams.items()}
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "hparams.json").write_text(
            json.dumps(clean_hparams, indent=2, sort_keys=True)
        )
        writer.add_text(
            "hparams",
            "```json\n" + json.dumps(clean_hparams, indent=2, sort_keys=True) + "\n```",
            0,
        )

    return writer, run_dir


def log_tensorboard_test_metrics(writer, y_true, y_pred, label_encoder, acc, run_dir=None):
    """Log final test metrics to TensorBoard and optionally to metrics.json."""
    labels = np.arange(len(label_encoder.classes_))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "test/accuracy": float(acc),
        "test/macro_precision": float(macro_precision),
        "test/macro_recall": float(macro_recall),
        "test/macro_f1": float(macro_f1),
    }
    for cls, p, r, f in zip(label_encoder.classes_, precision, recall, f1):
        cls_name = _clean_run_name(cls)
        metrics[f"test/{cls_name}_precision"] = float(p)
        metrics[f"test/{cls_name}_recall"] = float(r)
        metrics[f"test/{cls_name}_f1"] = float(f)

    for key, value in metrics.items():
        writer.add_scalar(key, value, 0)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True)
        )

    writer.flush()
    return metrics


def show_tensorboard(log_dir="runs"):
    """Start TensorBoard in notebooks; works in Colab and degrades gracefully."""
    try:
        ip = get_ipython()  # noqa: F821
    except NameError:
        print(f"Run this in a notebook: %load_ext tensorboard; %tensorboard --logdir {log_dir}")
        return

    ip.run_line_magic("load_ext", "tensorboard")
    ip.run_line_magic("tensorboard", f"--logdir {log_dir}")


def plot_training_history(history, title, baselines=None):
    """Plot training/validation loss and accuracy curves."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(ep, history["train_loss"], color="steelblue", label="Train")
    ax1.plot(
        ep, history["val_loss"], color="tomato", label="Val", linestyle="--"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-entropy loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, history["train_acc"], color="steelblue", label="Train")
    ax2.plot(
        ep, history["val_acc"], color="tomato", label="Val", linestyle="--"
    )
    for label, score in (baselines or {}).items():
        ax2.axhline(
            score, linewidth=1.2, linestyle=":", label=f"{label} ({score:.3f})"
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def fit_evaluate_classifier(
    model,
    train_dl,
    val_dl,
    test_dl,
    result_key,
    display_name,
    model_results,
    label_encoder,
    lr=1e-3,
    epochs=30,
    class_weight=None,
    device="cpu",
    print_every=5,
    show_progress=False,
    early_stopping_patience=None,
    early_stopping_metric="val_loss",
    early_stopping_min_delta=0.0,
    restore_best=True,
    history_title=None,
    baselines=None,
    cmap="Blues",
    tensorboard_log_dir=None,
    run_name=None,
    hparams=None,
):
    """Train, plot history, evaluate on test data, and record one classifier."""
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tb_writer = None
    tb_run_dir = None
    if tensorboard_log_dir is not None:
        tb_hparams = {
            "model": display_name,
            "result_key": result_key,
            "lr": lr,
            "epochs": epochs,
            "device": device,
            "class_weighted": class_weight is not None,
            "early_stopping_metric": early_stopping_metric,
            "early_stopping_min_delta": early_stopping_min_delta,
            **(hparams or {}),
        }
        try:
            tb_writer, tb_run_dir = make_tensorboard_writer(
                log_dir=tensorboard_log_dir,
                run_name=run_name or result_key,
                hparams=tb_hparams,
            )
            print(f"TensorBoard run: {tb_run_dir}")
        except ImportError as exc:
            warnings.warn(
                f"{exc} Continuing without TensorBoard logging.",
                RuntimeWarning,
                stacklevel=2,
            )

    try:
        history = train_classifier(
            model,
            train_dl,
            val_dl,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            name=display_name,
            print_every=print_every,
            show_progress=show_progress,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            early_stopping_min_delta=early_stopping_min_delta,
            restore_best=restore_best,
            tb_writer=tb_writer,
        )
        plot_training_history(history, history_title or f"{display_name} training curves", baselines=baselines)

        y_true, y_pred = predict_torch_model(model, test_dl, device=device)
        acc = record_result(result_key, display_name, y_true, y_pred, model_results, label_encoder, cmap=cmap)
        if tb_writer is not None:
            log_tensorboard_test_metrics(
                tb_writer,
                y_true,
                y_pred,
                label_encoder,
                acc,
                run_dir=tb_run_dir,
            )
        return history, acc, y_true, y_pred
    finally:
        if tb_writer is not None:
            tb_writer.close()


def predict_torch_model(model, data_loader, device="cpu"):
    """Run inference and return (true, pred) numpy arrays."""
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for Xb, yb in data_loader:
            batch_preds = model(Xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(batch_preds)
            true.extend(yb.numpy())
    return np.array(true), np.array(preds)


def metrics_table(model_results, label_encoder):
    """Return one row per recorded model with accuracy, macro-F1, and per-class metrics."""
    rows = []
    labels = np.arange(len(label_encoder.classes_))
    for result in model_results.values():
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        row = {
            "model": result["name"],
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1.mean(),
        }
        for i, cls in enumerate(label_encoder.classes_):
            row[f"{cls}_recall"] = recall[i]
            row[f"{cls}_f1"] = f1[i]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def plot_confusion_matrix_grid(model_results, label_encoder, n_cols=3):
    """Plot row-normalized confusion matrices annotated with counts and percentages."""
    n_models = len(model_results)
    if n_models == 0:
        print("No model results to plot.")
        return

    labels = np.arange(len(label_encoder.classes_))
    n_cols = min(n_cols, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    axes = np.atleast_1d(axes).ravel()

    for ax, result in zip(axes, model_results.values()):
        counts = confusion_matrix(result["y_true"], result["y_pred"], labels=labels)
        row_totals = counts.sum(axis=1, keepdims=True)
        pct = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=float),
            where=row_totals != 0,
        )

        ax.imshow(pct, cmap=result.get("cmap", "Blues"), vmin=0, vmax=1)
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(label_encoder.classes_)
        ax.set_yticklabels(label_encoder.classes_)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(result["name"], fontsize=13, fontweight="bold")

        threshold = pct.max() / 2 if pct.size else 0.0
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                color = "white" if pct[i, j] > threshold else "black"
                ax.text(
                    j,
                    i,
                    f"{counts[i, j]}\n({pct[i, j] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    for ax in axes[n_models:]:
        ax.set_visible(False)

    plt.suptitle("Confusion matrices on the test set", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()


def record_result(
    key, display_name, y_true, y_pred, model_results, label_encoder, cmap="Blues"
):
    """Record model results and show confusion-matrix counts plus row percentages."""
    acc = accuracy_score(y_true, y_pred)
    model_results[key] = {
        "name": display_name,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "accuracy": acc,
    }
    print(f"{display_name} test accuracy : {acc:.3f}")
    print()
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    labels = np.arange(len(label_encoder.classes_))
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = counts.sum(axis=1, keepdims=True)
    pct = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row percentage")

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(label_encoder.classes_)
    ax.set_yticklabels(label_encoder.classes_)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    threshold = pct.max() / 2 if pct.size else 0.0
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            color = "white" if pct[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{counts[i, j]}\n({pct[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    ax.set_title(
        f"Confusion matrix — {display_name}  (test set)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()
    return acc
