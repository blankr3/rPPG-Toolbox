"""
Dataset management for rPPG-Toolbox.

Provides auto-download for publicly available datasets (e.g. UBFC-rPPG)
and clear manual instructions for datasets requiring agreements (e.g. MMPD).

Path resolution order:
  1. Explicit path argument
  2. RPPG_DATA_DIR environment variable
  3. ~/.rppg_toolbox/data (default)
"""

import glob
import os
import shutil
import zipfile

DATASETS = {
    "UBFC-rPPG": {
        "auto_download": True,
        "gdrive_id": "1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk",
        "expected_pattern": "subject*/vid.avi",
        "description": "UBFC-rPPG dataset (pulse oximeter ground truth)",
        "instructions": None,
    },
    "MMPD": {
        "auto_download": False,
        "expected_pattern": "subject*/p*_*.mat",
        "description": "Multi-domain Mobile Phone Dataset with Fitzpatrick skin tone labels",
        "instructions": (
            "MMPD requires a signed data release agreement.\n"
            "1. Download the agreement:\n"
            "   https://github.com/McJackTang/MMPD_rPPG_dataset/blob/main/MMPD_Release_Agreement.pdf\n"
            "2. Sign and email to tjk24@mails.tsinghua.edu.cn\n"
            "   (cc: yuntaowang@tsinghua.edu.cn)\n"
            "3. Place received data in: {data_dir}/MMPD/"
        ),
    },
}

DEFAULT_DATA_DIR = os.path.expanduser("~/.rppg_toolbox/data")


def get_data_dir(override=None):
    """Return the base data directory, creating it if needed.

    Args:
        override: Explicit path (highest priority).

    Returns:
        Resolved absolute path to the data directory.
    """
    data_dir = override or os.environ.get("RPPG_DATA_DIR", DEFAULT_DATA_DIR)
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def dataset_path(dataset_name, data_dir=None):
    """Return the expected path for a dataset."""
    return os.path.join(get_data_dir(data_dir), dataset_name)


def check_dataset(dataset_name, data_dir=None):
    """Check whether a dataset exists and has expected structure.

    Returns:
        (bool, str): (is_valid, message)
    """
    if dataset_name not in DATASETS:
        return False, f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}"

    info = DATASETS[dataset_name]
    ds_path = dataset_path(dataset_name, data_dir)

    if not os.path.isdir(ds_path):
        return False, f"Dataset directory not found: {ds_path}"

    pattern = os.path.join(ds_path, info["expected_pattern"])
    matches = glob.glob(pattern)
    if not matches:
        return False, (
            f"Dataset directory exists ({ds_path}) but expected files not found.\n"
            f"Expected pattern: {info['expected_pattern']}"
        )

    return True, f"Dataset OK: {ds_path} ({len(matches)} files matching '{info['expected_pattern']}')"


def ensure_dataset(dataset_name, data_dir=None):
    """Ensure a dataset is available, downloading if possible.

    Args:
        dataset_name: Name from the DATASETS registry.
        data_dir: Override for the base data directory.

    Returns:
        Resolved path to the dataset directory.

    Raises:
        FileNotFoundError: If dataset is missing and cannot be auto-downloaded.
        ValueError: If dataset_name is not in the registry.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}"
        )

    info = DATASETS[dataset_name]
    resolved_dir = get_data_dir(data_dir)
    is_valid, msg = check_dataset(dataset_name, data_dir)

    if is_valid:
        print(f"[data_manager] {msg}")
        return dataset_path(dataset_name, data_dir)

    # Dataset not found — try auto-download or print instructions
    if info["auto_download"]:
        print(f"[data_manager] {msg}")
        print(f"[data_manager] Auto-downloading {dataset_name}...")
        _download_dataset(dataset_name, resolved_dir)

        # Validate after download
        is_valid, msg = check_dataset(dataset_name, data_dir)
        if is_valid:
            print(f"[data_manager] {msg}")
            return dataset_path(dataset_name, data_dir)
        else:
            raise FileNotFoundError(
                f"Download completed but validation failed: {msg}"
            )
    else:
        instructions = info["instructions"].format(data_dir=resolved_dir)
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' is not available at {resolved_dir}/{dataset_name}/.\n\n"
            f"{instructions}"
        )


def _download_dataset(dataset_name, dest_base):
    """Dispatch to dataset-specific download function."""
    if dataset_name == "UBFC-rPPG":
        _download_ubfc_rppg(dest_base)
    else:
        raise NotImplementedError(f"No auto-download for {dataset_name}")


def _download_ubfc_rppg(dest_base):
    """Download UBFC-rPPG from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for auto-download. Install it with:\n"
            "  pip install gdown"
        )

    gdrive_id = DATASETS["UBFC-rPPG"]["gdrive_id"]
    dest_dir = os.path.join(dest_base, "UBFC-rPPG")
    os.makedirs(dest_dir, exist_ok=True)

    # gdown.download_folder downloads a public Google Drive folder
    print(f"[data_manager] Downloading UBFC-rPPG to {dest_dir} ...")
    print("[data_manager] This may take a while (~3 GB).")

    zip_path = os.path.join(dest_base, "ubfc_rppg_download.zip")

    # Try folder download first (Google Drive public folder)
    try:
        gdown.download_folder(
            id=gdrive_id,
            output=dest_dir,
            quiet=False,
        )
    except Exception as e:
        # Fallback: try as a single zip file
        print(f"[data_manager] Folder download failed ({e}), trying as zip...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, zip_path, quiet=False)

        print(f"[data_manager] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_base)
        os.remove(zip_path)

        # If extraction created a nested directory, flatten it
        _flatten_nested_dir(dest_dir)

    print(f"[data_manager] UBFC-rPPG download complete: {dest_dir}")


def _flatten_nested_dir(target_dir):
    """If target_dir contains a single subdirectory with all the data, move contents up."""
    entries = os.listdir(target_dir)
    if len(entries) == 1:
        nested = os.path.join(target_dir, entries[0])
        if os.path.isdir(nested):
            for item in os.listdir(nested):
                shutil.move(os.path.join(nested, item), target_dir)
            os.rmdir(nested)


def list_datasets(data_dir=None):
    """Print status of all registered datasets."""
    resolved_dir = get_data_dir(data_dir)
    print(f"Data directory: {resolved_dir}\n")
    print(f"{'Dataset':<15} {'Auto-DL':<10} {'Status':<50}")
    print("-" * 75)

    for name, info in DATASETS.items():
        is_valid, msg = check_dataset(name, data_dir)
        status = "OK" if is_valid else "Not found"
        auto = "Yes" if info["auto_download"] else "No"
        print(f"{name:<15} {auto:<10} {status:<50}")

    print()
