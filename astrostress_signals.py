"""Read-only WESAD ECG/EDA waveform retrieval for recorded 120 s feature windows.

WESAD pickle files must come from a trusted dataset download. Python pickle can
execute code during deserialization. No user-supplied file names are accepted.
Raw signal data is never sent beyond this localhost-only server.
"""

from functools import lru_cache
from pathlib import Path
import os
import pickle
import re
import zipfile

import numpy as np

ECG_HZ = 700
EDA_HZ = 4
SUBJECT_ID = re.compile(r"^S(?:[2-9]|1[0-7])$")
# Keep 10 s of 700 Hz ECG (QRS morphology) and full 120 s of 4 Hz EDA.
ECG_PREVIEW_SEC = 10


def source_description():
    """A local trusted WESAD directory or ZIP, without loading any subject."""
    directory = Path(os.environ.get("WESAD_DATA_DIR", "WESAD")).expanduser()
    zip_path = Path(os.environ.get("WESAD_ZIP_PATH", "WESAD.zip")).expanduser()
    if directory.is_dir():
        return {"available": True, "kind": "directory", "path": str(directory)}
    if zip_path.is_file():
        return {"available": True, "kind": "zip", "path": str(zip_path)}
    return {"available": False, "kind": None,
            "note": "Set WESAD_DATA_DIR to the extracted WESAD folder or WESAD_ZIP_PATH to the downloaded WESAD.zip. Only use a trusted official WESAD download."}


def _verified_subject_name(subject):
    if not SUBJECT_ID.fullmatch(str(subject)) or str(subject) == "S12":
        raise ValueError("Invalid WESAD subject")
    return str(subject)


def _read_wesad_subject(subject, source):
    """Load trusted WESAD source, matching known subject path exactly."""
    expected = f"{subject}/{subject}.pkl"
    if source["kind"] == "directory":
        path = Path(source["path"]) / expected
        if not path.is_file():
            raise FileNotFoundError(f"Missing WESAD recording: {expected}")
        with path.open("rb") as fh:
            return pickle.load(fh, encoding="latin1")
    if source["kind"] == "zip":
        with zipfile.ZipFile(source["path"]) as zf:
            matches = [name for name in zf.namelist()
                       if name.replace("\\", "/").lstrip("./").endswith("/" + expected)
                       or name.replace("\\", "/").lstrip("./") == expected]
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one {expected} inside WESAD ZIP")
            with zf.open(matches[0]) as fh:
                return pickle.load(fh, encoding="latin1")
    raise FileNotFoundError("Trusted WESAD source not configured")


@lru_cache(maxsize=1)
def _cached_subject(subject, kind, path):
    data = _read_wesad_subject(subject, {"kind": kind, "path": path})
    try:
        ecg = np.asarray(data["signal"]["chest"]["ECG"]).reshape(-1)
        eda = np.asarray(data["signal"]["wrist"]["EDA"]).reshape(-1)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Corrupt WESAD signal arrays for {subject}") from exc
    if not np.issubdtype(ecg.dtype, np.number) or not np.issubdtype(eda.dtype, np.number):
        raise ValueError("WESAD signals must be numeric")
    return ecg, eda


def waveform_payload(features, subject, window_index):
    """Return real ECG 10-second and EDA full-window traces, with source metadata.

    Windows are *indexed from features.csv*, not raw labels, so windows skipped
    by the extraction pipeline never shift the selected time range.
    """
    subject = _verified_subject_name(subject)
    if not isinstance(window_index, int) or window_index < 0:
        raise ValueError("Window index must be nonnegative")
    source = source_description()
    if not source["available"]:
        raise FileNotFoundError(source["note"])
    selected = features.loc[features["subject"].astype(str) == subject]
    if window_index >= len(selected):
        raise ValueError("Window is not in the verified features dataset")
    row = selected.iloc[window_index]
    if "window_start_s" not in row or "window_end_s" not in row:
        raise ValueError("Current features CSV lacks required window timing")
    start = float(row["window_start_s"])
    end = float(row["window_end_s"])
    if not np.isfinite(start) or not np.isfinite(end) or end <= start or (end-start)>121:
        raise ValueError("Invalid recorded WESAD window timing")
    ecg, eda = _cached_subject(subject, source["kind"], source["path"])
    ecg_start = round(start * ECG_HZ)
    ecg_end = min(round(end * ECG_HZ), ecg_start + ECG_PREVIEW_SEC * ECG_HZ)
    eda_start = round(start * EDA_HZ)
    eda_end = round(end * EDA_HZ)
    if ecg_start < 0 or ecg_end > len(ecg) or eda_start < 0 or eda_end > len(eda):
        raise ValueError("Selected window extends beyond WESAD recorded signals")
    ecg_segment = np.asarray(ecg[ecg_start:ecg_end], dtype=float)
    eda_segment = np.asarray(eda[eda_start:eda_end], dtype=float)
    if not np.isfinite(ecg_segment).all() or not np.isfinite(eda_segment).all():
        raise ValueError("Selected raw recording contains nonfinite samples")
    # Native values; chest ECG is in the original RespiBAN signal units, not
    # reliably calibrated millivolts. Do not relabel amplitude as mV.
    return {
        "source": "original_wesad_recording",
        "subject": subject, "window_index": window_index,
        "window_start_s": start, "window_end_s": end,
        "protocol_label": int(row["label"] if "label" in row else row["label_binary"]),
        "ecg": {
            "sampling_rate_hz": ECG_HZ, "preview_seconds": ECG_PREVIEW_SEC,
            "unit": "original ECG sensor units",
            "samples": np.round(ecg_segment, 6).tolist(),
        },
        "eda": {
            "sampling_rate_hz": EDA_HZ, "preview_seconds": end-start,
            "unit": "μS",
            "samples": np.round(eda_segment, 6).tolist(),
        },
        "note": "Real WESAD recordings from an Earth-based study. ECG shows the first 10 seconds for readability; EDA shows the full selected 120-second window. Not astronaut measurements.",
    }
