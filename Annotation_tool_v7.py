import json
import os
import posixpath
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import paramiko
import soundfile as sf


"""
Working of pipeline:
- Connect to a Linux server over SSH/SFTP
- Find the next unannotated WAV file on the server
- Download just that file to a local temp folder
- Trim it to a 2-second local segment
- Open the local 2-second segment in Audacity
- Read labels back from Audacity using mod-script-pipe
- Build a preview WAV by normalizing ONSET-TO-ONSET intervals:
    - keep the original audio between consecutive onsets when it is shorter than target
    - pad the remainder with background so the next onset lands exactly at target IPI
    - if the original interval is longer than target, remove only excess background / non-protected
      audio from that interval so the next onset lands exactly at target IPI
    - keep protected intervals (for example "call" labels) intact as much as possible
- On 's', read labels from the current segment, close Audacity, and open the preview WAV in Audacity
- On '2', discard the current 2-second segment and load the next 2-second segment from the same file
- On 'd', save all outputs together:
    - labels JSON to the server
    - the current 2-second segment WAV to the server
    - processed WAV to the server
  then close Audacity without saving the project
- On 'k', skip just this file and close Audacity without saving the project
- On 'f', skip the current folder and move to the next folder
- Mark completed/skipped items in a local state file
- Delete local temp files again

Requirements:
  pip install paramiko numpy soundfile

Audacity requirements on the LOCAL computer:
- Audacity installed locally
- mod-script-pipe enabled:
    Edit > Preferences > Modules > mod-script-pipe > Enabled
- Restart Audacity after enabling the module

Common error + solution: 
AudacityError: Could not open Audacity write pipe after 30.0s: [Errno 2] The system cannot find the file specified: '\\.\pipe\ToSrvPipe'
You need to close all audacity windows and make sure Audacity is running locally with mod-script-pipe enabled before starting the script.


"""

# Configuration
@dataclass
class Config:
    ssh_host: str = " " # add server name
    ssh_port: int = " " # add server port
    ssh_username: str = " " # add server username
    ssh_password: Optional[str] = None

    remote_dataset_dir: str = " " # add path to the dataset with original recordings

    remote_annotations_dir: str = " " # add path to the annotation folder here

    remote_trimmed_wav_dir: str = " " # add path to the folder where you want the 2-second trimmed wavs to be stored as original IPI wavs 

    # Save processed fixed-IPI WAVs here
    remote_processed_wav_dir: str = " " # add folder where you want the complete standardized IPI wavs to be 

    allowed_extensions: Tuple[str, ...] = (".wav", ".WAV", ".mp3", ".MP3")

    # ---- Local machine ----
    audacity_exe: str = r"C:\Program Files\Audacity\Audacity.exe" # make sure this is where Audacity is in File Explorer
    audacity_open_args: Tuple[str, ...] = ()

    local_work_dir: str = " " # add folder path to local working directory like a waiting room for files

    audacity_pipe_timeout_sec: float = 30.0
    file_open_settle_sec: float = 3.0
    audacity_close_settle_sec: float = 1.5

    onset_label_name: str = "onset"
    target_ipi_sec: float = 0.050

    # Labels with duration are treated as protected and should be kept intact where possible.
 
    max_crossfade_sec: float = 0.0  

    min_background_patch_sec: float = 0.003
    random_seed: int = 1337

    trimmed_duration_sec: float = 2.0
    treat_existing_remote_output_as_done: bool = True

    min_output_duration_sec: float = 2.0


CONFIG = Config()
# 
TO_PIPE = r"\\.\pipe\ToSrvPipe" # This is the default name Audacity uses for the pipe to receive commands.
FROM_PIPE = r"\\.\pipe\FromSrvPipe" 
EOL = "\r\n\0"

# 
class AudacityPipeClient:
    def __init__(self, timeout: float = 30.0): # Try to open the write pipe with retries, since Audacity may take a moment to create it on startup.
        self.to_pipe = None
        self.reply = ""
        self.reply_ready = threading.Event()
        self.reader_pipe_broken = threading.Event()

        self._open_write_pipe(timeout=timeout)
        self._start_reader_thread()

    def _open_write_pipe(self, timeout: float = 30.0): 
        start = time.time() 
        last_error = None

        while time.time() - start < timeout:
            try:
                self.to_pipe = open(TO_PIPE, "w", newline="")
                return
            except Exception as e:
                last_error = e
                time.sleep(0.5) # Wait a bit before retrying, to give Audacity time to start and create the pipe.

        raise RuntimeError(
            f"Could not open Audacity write pipe after {timeout:.1f}s: {last_error}\n"
            "Make sure Audacity is running locally and mod-script-pipe is enabled and all previous Audacity windows are closed"
        )

    def _start_reader_thread(self): # We need a separate thread to read from Audacity's FromSrvPipe because it blocks until a message is available, and we want to be able to detect if the pipe breaks (for example if Audacity is closed) while waiting for a reply.
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self): # This runs in a separate thread and continuously reads from Audacity's FromSrvPipe.
        try:
            with open(FROM_PIPE, "r", newline="") as from_pipe:
                message = ""
                while True:
                    line = from_pipe.readline()

                    if line == "":
                        self.reader_pipe_broken.set()
                        break

                    if line == "\n":
                        self.reply = message
                        self.reply_ready.set()
                        message = ""
                    else:
                        message += line
        except Exception:
            self.reader_pipe_broken.set()

    def write(self, command: str): # if pipe broke it raises an error.
        if self.reader_pipe_broken.is_set():
            raise RuntimeError("Audacity read pipe is broken. Audacity may have closed.")

        self.reply = ""
        self.reply_ready.clear()
        self.to_pipe.write(command + EOL)
        self.to_pipe.flush()

    def request(self, command: str, timeout: float = 10.0) -> str: # Send a command and wait for the reply, with timeout and error handling.
        self.write(command)

        start = time.time()
        while not self.reply_ready.is_set():
            if self.reader_pipe_broken.is_set():
                raise RuntimeError("Audacity read pipe broke while waiting for a reply.")
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for reply to: {command}")
            time.sleep(0.05)

        return self.reply

    def close(self): # Close the write pipe and signal the reader thread to stop by breaking the read pipe (for example by closing Audacity).
        try:
            if self.to_pipe:
                self.to_pipe.close()
        except Exception:
            pass


class AudacitySession:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.client: Optional[AudacityPipeClient] = None

    def open_file(self, local_audio_path: Path):
        self.close(force=True)
        self.proc = launch_audacity_with_file(self.cfg, local_audio_path)
        self.client = AudacityPipeClient(timeout=self.cfg.audacity_pipe_timeout_sec)

    def request(self, command: str, timeout: float = 10.0) -> str:
        if self.client is None:
            raise RuntimeError("Audacity pipe client is not available.")
        return self.client.request(command, timeout=timeout)

    def close(self, force: bool = True):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

        if force:
            force_close_audacity_windows(self.cfg)

        self.proc = None


class StateStore:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.data = self._load()

    def _load(self) -> Dict:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "completed": {},
            "skipped": {},
            "skipped_folders": {},
            "last_updated": None,
        }

    def save(self):
        self.data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def is_completed(self, remote_path: str) -> bool:
        return remote_path in self.data.get("completed", {})

    def mark_completed(self, remote_path: str, metadata: Dict):
        self.data.setdefault("completed", {})[remote_path] = metadata
        self.data.get("skipped", {}).pop(remote_path, None)
        self.save()

    def mark_skipped(self, remote_path: str, metadata: Dict):
        self.data.setdefault("skipped", {})[remote_path] = metadata
        self.save()

    def mark_folder_skipped(self, remote_folder_path: str, metadata: Dict):
        folder_key = posixpath.normpath(remote_folder_path)
        self.data.setdefault("skipped_folders", {})[folder_key] = metadata
        self.save()

    def is_in_skipped_folder(self, remote_path: str) -> bool:
        remote_path = posixpath.normpath(remote_path)
        for skipped_folder in self.data.get("skipped_folders", {}):
            skipped_folder = posixpath.normpath(skipped_folder)
            if remote_path == skipped_folder or remote_path.startswith(skipped_folder + "/"):
                return True
        return False

    def is_skipped(self, remote_path: str) -> bool:
        return remote_path in self.data.get("skipped", {})


def connect_ssh(cfg: Config) -> Tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": cfg.ssh_host,
        "port": cfg.ssh_port,
        "username": cfg.ssh_username,
    }
    if cfg.ssh_password:
        connect_kwargs["password"] = cfg.ssh_password

    ssh.connect(**connect_kwargs)
    sftp = ssh.open_sftp()
    return ssh, sftp


def sftp_exists(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        sftp.stat(remote_path)
        return True
    except (FileNotFoundError, OSError):
        return False


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str):
    remote_dir = posixpath.normpath(remote_dir)
    if remote_dir in ("", "/"):
        return

    parts = []
    current = remote_dir
    while current not in ("", "/"):
        parts.append(current)
        current = posixpath.dirname(current)

    for path in reversed(parts):
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def list_remote_audio_files(
    sftp: paramiko.SFTPClient,
    root_dir: str,
    allowed_extensions: Tuple[str, ...]
) -> List[str]:
    results: List[str] = []

    def walk(path: str):
        for entry in sftp.listdir_attr(path):
            remote_path = posixpath.join(path, entry.filename)
            is_dir = (entry.st_mode & 0o170000) == 0o040000
            if is_dir:
                walk(remote_path)
            else:
                if entry.filename.endswith(allowed_extensions):
                    results.append(remote_path)

    walk(root_dir)
    results.sort()
    return results


def remote_output_path_for_audio(remote_audio_path: str, cfg: Config) -> str:
    rel = PurePosixPath(remote_audio_path).relative_to(PurePosixPath(cfg.remote_dataset_dir))
    return str(PurePosixPath(cfg.remote_annotations_dir) / rel.with_suffix(".labels.json"))


def remote_trimmed_wav_path_for_audio(remote_audio_path: str, cfg: Config) -> str:
    stem = PurePosixPath(remote_audio_path).stem
    return str(PurePosixPath(cfg.remote_trimmed_wav_dir) / f"{stem}.wav")


def remote_processed_wav_path_for_audio(remote_audio_path: str, cfg: Config) -> str:
    stem = PurePosixPath(remote_audio_path).stem
    return str(PurePosixPath(cfg.remote_processed_wav_dir) / f"{stem}.wav")


def download_one_file(
    sftp: paramiko.SFTPClient,
    remote_path: str,
    local_root: Path,
    remote_dataset_dir: str
) -> Path:
    rel = PurePosixPath(remote_path).relative_to(PurePosixPath(remote_dataset_dir))
    local_path = local_root / Path(*rel.parts)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    return local_path


def upload_json(
    sftp: paramiko.SFTPClient,
    data: Dict,
    remote_json_path: str,
    local_tmp_json_path: Path
):
    local_tmp_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    ensure_remote_dir(sftp, posixpath.dirname(remote_json_path))
    sftp.put(str(local_tmp_json_path), remote_json_path)


def upload_wav(
    sftp: paramiko.SFTPClient,
    local_wav_path: Path,
    remote_wav_path: str
):
    ensure_remote_dir(sftp, posixpath.dirname(remote_wav_path))
    sftp.put(str(local_wav_path), remote_wav_path)


def parse_label_response(raw_response: str):
    end_index = raw_response.rfind("]")
    if end_index == -1:
        raise ValueError("No JSON array found in Audacity response.")
    return json.loads(raw_response[:end_index + 1])


def flatten_audacity_labels(parsed_labels) -> List[Dict]:
    flat: List[Dict] = []
    if not isinstance(parsed_labels, list):
        return flat

    for track_item in parsed_labels:
        if not isinstance(track_item, list) or len(track_item) < 2:
            continue

        track_index = track_item[0]
        track_labels = track_item[1]
        if not isinstance(track_labels, list):
            continue

        for label in track_labels:
            if not isinstance(label, list) or len(label) < 3:
                continue
            try:
                flat.append({
                    "track_index": int(track_index),
                    "start": float(label[0]),
                    "end": float(label[1]),
                    "text": str(label[2]),
                })
            except Exception:
                continue

    return flat


def extract_onset_times(flat_labels: List[Dict], onset_label_name: str) -> List[float]:
    onset_name = onset_label_name.strip().lower()
    onsets = [
        float(item["start"])
        for item in flat_labels
        if item["text"].strip().lower() == onset_name
    ]
    return sorted(set(onsets))


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def extract_protected_intervals(flat_labels: List[Dict]) -> List[Tuple[float, float]]:
    
    intervals = []
    for item in flat_labels:
        start = float(item["start"])
        end = float(item["end"])
        if end > start:
            intervals.append((start, end))
    return merge_intervals(intervals)


def extract_label_count(parsed_labels) -> int:
    count = 0
    if isinstance(parsed_labels, list):
        for item in parsed_labels:
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], list):
                count += len(item[1])
    return count


def time_to_sample(t: float, sr: int, max_n: int) -> int:
    return max(0, min(int(round(t * sr)), max_n))


def complement_intervals(
    protected_intervals_sec: List[Tuple[float, float]],
    total_duration_sec: float,
    min_len_sec: float
) -> List[Tuple[float, float]]:
    safe = []
    cursor = 0.0

    for start, end in protected_intervals_sec:
        if start - cursor >= min_len_sec:
            safe.append((cursor, start))
        cursor = max(cursor, end)

    if total_duration_sec - cursor >= min_len_sec:
        safe.append((cursor, total_duration_sec))

    return safe


def choose_random_background_patch(
    audio: np.ndarray,
    sr: int,
    safe_intervals_sec: List[Tuple[float, float]],
    patch_len_samples: int,
    rng: random.Random,
    min_background_patch_sec: float
) -> np.ndarray:
    channels = audio.shape[1]
    min_patch_samples = max(1, int(round(min_background_patch_sec * sr)))
    patch_len_samples = max(1, patch_len_samples)

    candidates = []
    for start_sec, end_sec in safe_intervals_sec:
        start_s = int(round(start_sec * sr))
        end_s = int(round(end_sec * sr))
        if end_s - start_s >= max(min_patch_samples, 1):
            candidates.append((start_s, end_s))

    if not candidates:
        return np.zeros((patch_len_samples, channels), dtype=audio.dtype)

    long_enough = [c for c in candidates if (c[1] - c[0]) >= patch_len_samples]
    if long_enough:
        start_s, end_s = rng.choice(long_enough)
        max_offset = (end_s - start_s) - patch_len_samples
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0
        return audio[start_s + offset:start_s + offset + patch_len_samples].copy()

    pieces = []
    remaining = patch_len_samples
    while remaining > 0:
        start_s, end_s = rng.choice(candidates)
        avail = end_s - start_s
        take = min(avail, remaining)
        max_offset = avail - take
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0
        pieces.append(audio[start_s + offset:start_s + offset + take].copy())
        remaining -= take

    return np.vstack(pieces)


def intersect_intervals_samples(
    intervals_samples: List[Tuple[int, int]],
    win_start: int,
    win_end: int
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for a, b in intervals_samples:
        lo = max(a, win_start)
        hi = min(b, win_end)
        if hi > lo:
            out.append((lo - win_start, hi - win_start))
    return merge_intervals(out)


def reduce_interval_by_trimming_removable(
    interval_audio: np.ndarray,
    protected_rel_intervals: List[Tuple[int, int]],
    target_len: int
) -> Tuple[np.ndarray, Dict]:

    current_len = len(interval_audio)
    if target_len >= current_len:
        return interval_audio.copy(), {
            "mode": "unchanged_or_expand",
            "removed_samples": 0,
            "removed_from_removable_only": True,
            "removable_available_samples": current_len - sum(b - a for a, b in protected_rel_intervals),
        }

    excess = current_len - target_len
    protected_rel_intervals = merge_intervals(protected_rel_intervals)

    segments: List[Dict] = []
    cursor = 0
    for start, end in protected_rel_intervals:
        if start > cursor:
            segments.append({
                "kind": "removable",
                "start": cursor,
                "end": start,
                "keep_start": cursor,
                "keep_end": start,
            })
        segments.append({
            "kind": "protected",
            "start": start,
            "end": end,
            "keep_start": start,
            "keep_end": end,
        })
        cursor = end

    if cursor < current_len:
        segments.append({
            "kind": "removable",
            "start": cursor,
            "end": current_len,
            "keep_start": cursor,
            "keep_end": current_len,
        })

    removable_available = sum(
        seg["end"] - seg["start"]
        for seg in segments
        if seg["kind"] == "removable"
    )

    remaining_excess = excess

    # Remove from removable segments, starting from the end of the interval.
    for seg in reversed(segments):
        if remaining_excess <= 0:
            break
        if seg["kind"] != "removable":
            continue

        seg_len = seg["keep_end"] - seg["keep_start"]
        if seg_len <= 0:
            continue

        take = min(seg_len, remaining_excess)
        seg["keep_end"] -= take
        remaining_excess -= take

    kept_pieces = []
    for seg in segments:
        if seg["keep_end"] > seg["keep_start"]:
            kept_pieces.append(interval_audio[seg["keep_start"]:seg["keep_end"]])

    reduced = np.vstack(kept_pieces) if kept_pieces else np.zeros((0, interval_audio.shape[1]), dtype=interval_audio.dtype)

    # Fallback if removable material was not enough.
    removed_from_removable_only = (remaining_excess == 0)
    if len(reduced) > target_len:
        reduced = reduced[:target_len].copy()
    elif len(reduced) < target_len:
        # This should only happen if fallback over-trimmed or there was no material left.
        # Pad with zeros to stay exact.
        pad = np.zeros((target_len - len(reduced), interval_audio.shape[1]), dtype=interval_audio.dtype)
        reduced = np.vstack([reduced, pad])

    return reduced, {
        "mode": "compressed",
        "removed_samples": excess,
        "removed_from_removable_only": removed_from_removable_only,
        "removable_available_samples": removable_available,
    }


def ensure_min_duration_with_background(
    audio_out: np.ndarray,
    sr: int,
    min_duration_sec: float,
    source_audio: np.ndarray,
    safe_intervals_sec: List[Tuple[float, float]],
    rng: random.Random,
    min_background_patch_sec: float,
) -> Tuple[np.ndarray, int]:
    target_samples = max(1, int(round(min_duration_sec * sr)))
    current_samples = len(audio_out)

    if current_samples >= target_samples:
        return audio_out, 0

    pad_len = target_samples - current_samples
    bg = choose_random_background_patch(
        audio=source_audio,
        sr=sr,
        safe_intervals_sec=safe_intervals_sec,
        patch_len_samples=pad_len,
        rng=rng,
        min_background_patch_sec=min_background_patch_sec,
    )
    return np.vstack([audio_out, bg]), pad_len


def build_fixed_ipi_preview(
    audio: np.ndarray,
    sr: int,
    onset_times_sec: List[float],
    protected_intervals_sec: List[Tuple[float, float]],
    target_ipi_sec: float,
    max_crossfade_sec: float,
    min_background_patch_sec: float,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict]:
  
    if audio.ndim == 1:
        audio = audio[:, None]

    n_samples, channels = audio.shape

    if len(onset_times_sec) == 0:
        raise ValueError("No onset labels found.")
    if len(onset_times_sec) == 1:
        return audio.copy(), {
            "note": "Only one onset found; preview left unchanged.",
            "onset_count": 1,
            "target_ipi_sec": target_ipi_sec,
        }

    onset_samples = [time_to_sample(t, sr, n_samples - 1) for t in onset_times_sec]
    onset_samples = sorted(set(onset_samples))

    target_ipi_samples = max(1, int(round(target_ipi_sec * sr)))

    protected_intervals_samples = [
        (time_to_sample(s, sr, n_samples), time_to_sample(e, sr, n_samples))
        for s, e in protected_intervals_sec
        if e > s
    ]
    protected_intervals_samples = merge_intervals(protected_intervals_samples)

    total_duration_sec = n_samples / sr
    safe_intervals_sec = complement_intervals(
        protected_intervals_sec=protected_intervals_sec,
        total_duration_sec=total_duration_sec,
        min_len_sec=min_background_patch_sec,
    )

    # Keep prefix before first onset unchanged.
    prefix = audio[:onset_samples[0]].copy()
    output_parts = [prefix]

    new_onset_positions = [len(prefix)]
    interval_summaries = []

    inserted_background_total = 0
    removed_excess_total = 0

    for i in range(len(onset_samples) - 1):
        src_start = onset_samples[i]
        src_end = onset_samples[i + 1]
        interval_audio = audio[src_start:src_end].copy()
        original_len = len(interval_audio)

        protected_rel = intersect_intervals_samples(
            intervals_samples=protected_intervals_samples,
            win_start=src_start,
            win_end=src_end,
        )

        if original_len == target_ipi_samples:
            interval_out = interval_audio
            op_meta = {
                "mode": "unchanged",
                "inserted_background_samples": 0,
                "removed_samples": 0,
                "removed_from_removable_only": True,
            }

        elif original_len < target_ipi_samples:
            pad_len = target_ipi_samples - original_len
            bg = choose_random_background_patch(
                audio=audio,
                sr=sr,
                safe_intervals_sec=safe_intervals_sec,
                patch_len_samples=pad_len,
                rng=rng,
                min_background_patch_sec=min_background_patch_sec,
            )
            interval_out = np.vstack([interval_audio, bg])
            inserted_background_total += pad_len
            op_meta = {
                "mode": "padded",
                "inserted_background_samples": pad_len,
                "removed_samples": 0,
                "removed_from_removable_only": True,
            }

        else:
            interval_out, reduce_meta = reduce_interval_by_trimming_removable(
                interval_audio=interval_audio,
                protected_rel_intervals=protected_rel,
                target_len=target_ipi_samples,
            )
            removed_excess_total += (original_len - target_ipi_samples)
            op_meta = {
                "mode": reduce_meta["mode"],
                "inserted_background_samples": 0,
                "removed_samples": reduce_meta["removed_samples"],
                "removed_from_removable_only": reduce_meta["removed_from_removable_only"],
                "removable_available_samples": reduce_meta["removable_available_samples"],
            }

        if len(interval_out) != target_ipi_samples:
            raise RuntimeError(
                f"Internal error: interval_out has length {len(interval_out)}, "
                f"expected {target_ipi_samples}"
            )

        output_parts.append(interval_out)
        new_onset_positions.append(new_onset_positions[0] + (i + 1) * target_ipi_samples)

        interval_summaries.append({
            "interval_index": i,
            "source_onset_a_sample": int(src_start),
            "source_onset_b_sample": int(src_end),
            "source_interval_samples": int(original_len),
            "target_interval_samples": int(target_ipi_samples),
            "new_onset_a_sample": int(new_onset_positions[i]),
            "new_onset_b_sample": int(new_onset_positions[i + 1]),
            **op_meta,
        })

    # Keep suffix after last onset unchanged.
    suffix = audio[onset_samples[-1]:].copy()
    output_parts.append(suffix)

    output = np.vstack(output_parts)

    output, tail_padding_samples = ensure_min_duration_with_background(
        audio_out=output,
        sr=sr,
        min_duration_sec=CONFIG.min_output_duration_sec,
        source_audio=audio,
        safe_intervals_sec=safe_intervals_sec,
        rng=rng,
        min_background_patch_sec=min_background_patch_sec,
    )

    actual_ipis_samples = [
        int(new_onset_positions[i] - new_onset_positions[i - 1])
        for i in range(1, len(new_onset_positions))
    ]
    actual_ipis_sec = [x / sr for x in actual_ipis_samples]
    new_onset_times_sec = [x / sr for x in new_onset_positions]

    metadata = {
        "onset_count": len(onset_samples),
        "target_ipi_sec": target_ipi_sec,
        "target_ipi_samples": target_ipi_samples,
        "new_onset_positions_samples": new_onset_positions,
        "new_onset_times_sec": new_onset_times_sec,
        "actual_ipis_samples": actual_ipis_samples,
        "actual_ipis_sec": actual_ipis_sec,
        "inserted_background_total_samples": inserted_background_total,
        "removed_excess_total_samples": removed_excess_total,
        "safe_background_intervals_count": len(safe_intervals_sec),
        "channels": channels,
        "sample_rate": sr,
        "interval_summaries": interval_summaries,
        "tail_padding_samples": tail_padding_samples,
        "tail_padding_sec": tail_padding_samples / sr,
        "final_duration_sec": len(output) / sr,
    }
    return output, metadata


def save_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def trim_audio_to_duration(
    local_input_path: Path,
    trimmed_output_path: Path,
    duration_sec: float,
    start_sec: float = 0.0,
) -> Dict:
    audio, sr = sf.read(str(local_input_path), always_2d=True)

    start_sample = max(0, int(round(start_sec * sr)))
    duration_samples = max(1, int(round(duration_sec * sr)))
    end_sample = min(len(audio), start_sample + duration_samples)

    trimmed_audio = audio[start_sample:end_sample].copy()

    if len(trimmed_audio) == 0:
        raise ValueError(
            f"No audio left to trim from {start_sec:.3f}s "
            f"(file duration: {len(audio) / sr:.3f}s)."
        )

    save_wav(trimmed_output_path, trimmed_audio, sr)

    return {
        "sample_rate": sr,
        "original_samples": int(len(audio)),
        "trimmed_samples": int(len(trimmed_audio)),
        "original_duration_sec": float(len(audio) / sr),
        "trimmed_duration_sec": float(len(trimmed_audio) / sr),
        "trim_start_sec": float(start_sample / sr),
        "trim_end_sec": float(end_sample / sr),
        "channels": int(trimmed_audio.shape[1]),
    }


def generate_preview_wav(
    local_audio_path: Path,
    parsed_labels,
    cfg: Config,
    preview_dir: Path,
) -> Tuple[Path, Dict]:
    flat_labels = flatten_audacity_labels(parsed_labels)
    onset_times_sec = extract_onset_times(flat_labels, cfg.onset_label_name)
    protected_intervals_sec = extract_protected_intervals(flat_labels)

    audio, sr = sf.read(str(local_audio_path), always_2d=True)
    rng = random.Random(cfg.random_seed)

    preview_audio, process_meta = build_fixed_ipi_preview(
        audio=audio,
        sr=sr,
        onset_times_sec=onset_times_sec,
        protected_intervals_sec=protected_intervals_sec,
        target_ipi_sec=cfg.target_ipi_sec,
        max_crossfade_sec=cfg.max_crossfade_sec,
        min_background_patch_sec=cfg.min_background_patch_sec,
        rng=rng,
    )

    preview_path = preview_dir / local_audio_path.name
    save_wav(preview_path, preview_audio, sr)

    preview_meta = {
        "preview_local_path": str(preview_path),
        "processing": process_meta,
        "onset_times_sec": onset_times_sec,
        "protected_intervals_sec": protected_intervals_sec,
    }
    return preview_path, preview_meta


def launch_audacity_with_file(cfg: Config, local_audio_path: Path) -> subprocess.Popen:
    if not os.path.isfile(cfg.audacity_exe):
        raise FileNotFoundError(f"Audacity executable not found: {cfg.audacity_exe}")

    print(f"\nOpening in Audacity:\n  {local_audio_path}")
    cmd = [cfg.audacity_exe, *cfg.audacity_open_args, str(local_audio_path)]
    proc = subprocess.Popen(cmd)
    time.sleep(cfg.file_open_settle_sec)
    return proc


def force_close_audacity_windows(cfg: Config):
    exe_name = os.path.basename(cfg.audacity_exe)
    try:
        subprocess.run(
            ["taskkill", "/IM", exe_name, "/F", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    finally:
        time.sleep(cfg.audacity_close_settle_sec)


def cleanup_local_file(path: Optional[Path]):
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"Warning: could not delete local file {path}: {e}")


def cleanup_empty_parents(path: Optional[Path], stop_at: Path):
    if path is None:
        return

    current = path.parent
    stop_at = stop_at.resolve()

    while True:
        try:
            if current.resolve() == stop_at:
                break
        except Exception:
            pass

        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def get_next_unannotated_file(
    sftp: paramiko.SFTPClient,
    state: StateStore,
    cfg: Config
) -> Optional[str]:
    all_audio = list_remote_audio_files(
        sftp=sftp,
        root_dir=cfg.remote_dataset_dir,
        allowed_extensions=cfg.allowed_extensions,
    )

    for remote_audio in all_audio:
        if state.is_completed(remote_audio):
            continue

        if state.is_skipped(remote_audio):
            continue

        if state.is_in_skipped_folder(remote_audio):
            continue

        remote_output = remote_output_path_for_audio(remote_audio, cfg)
        remote_trimmed = remote_trimmed_wav_path_for_audio(remote_audio, cfg)
        remote_processed = remote_processed_wav_path_for_audio(remote_audio, cfg)

        if cfg.treat_existing_remote_output_as_done:
            json_exists = sftp_exists(sftp, remote_output)
            trimmed_exists = sftp_exists(sftp, remote_trimmed)
            processed_exists = sftp_exists(sftp, remote_processed)

            if json_exists and trimmed_exists and processed_exists:
                state.mark_completed(remote_audio, {
                    "reason": "all_remote_outputs_already_exist",
                    "remote_output": remote_output,
                    "remote_trimmed_wav": remote_trimmed,
                    "remote_processed_wav": remote_processed,
                })
                continue

        return remote_audio

    return None


def fetch_labels_from_current_audacity(session: AudacitySession, timeout: float) -> Tuple[str, object, int]:
    raw_response = session.request("GetInfo: Type=Labels", timeout=timeout)
    parsed = parse_label_response(raw_response)
    label_count = extract_label_count(parsed)
    return raw_response, parsed, label_count


def annotate_one_file(
    cfg: Config,
    sftp: paramiko.SFTPClient,
    session: AudacitySession,
    state: StateStore,
    remote_audio_path: str,
    local_download_root: Path,
    local_output_root: Path,
    local_preview_root: Path,
    local_trimmed_root: Path,
) -> str:
    remote_output_json = remote_output_path_for_audio(remote_audio_path, cfg)
    remote_trimmed_wav = remote_trimmed_wav_path_for_audio(remote_audio_path, cfg)
    remote_processed_wav = remote_processed_wav_path_for_audio(remote_audio_path, cfg)
    remote_folder = posixpath.dirname(remote_audio_path)

    print(f"\nDownloading:\n  {remote_audio_path}")
    local_original_path = download_one_file(
        sftp=sftp,
        remote_path=remote_audio_path,
        local_root=local_download_root,
        remote_dataset_dir=cfg.remote_dataset_dir,
    )

    # ---- Sample rate check ----
    try:
        audio_tmp, sr_tmp = sf.read(str(local_original_path), always_2d=True)

        print(f"Sample rate: {sr_tmp}")

        if sr_tmp < 100000:
            print(f"Skipping file (sample rate too low: {sr_tmp} Hz)")

            state.mark_skipped(remote_audio_path, {
                "reason": "sample_rate_too_low",
                "sample_rate": int(sr_tmp),
                "skipped_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

            cleanup_local_file(local_original_path)
            cleanup_empty_parents(local_original_path, stop_at=local_download_root)

            return "continue"

    except Exception as e:
        print(f"Could not read sample rate, skipping file: {e}")

        state.mark_skipped(remote_audio_path, {
            "reason": "sample_rate_read_failed",
            "error": str(e),
            "skipped_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        cleanup_local_file(local_original_path)
        cleanup_empty_parents(local_original_path, stop_at=local_download_root)

        return "continue"

    print(f"Downloaded to:\n  {local_original_path}")

    rel = PurePosixPath(remote_audio_path).relative_to(PurePosixPath(cfg.remote_dataset_dir))
    local_trimmed_path = (local_trimmed_root / Path(*rel.parts)).with_suffix(".wav")
    local_trimmed_path.parent.mkdir(parents=True, exist_ok=True)

    preview_path: Optional[Path] = None
    cached_raw_response: Optional[str] = None
    cached_parsed = None
    cached_label_count: Optional[int] = None
    cached_preview_meta: Optional[Dict] = None
    segment_index = 0
    trim_meta: Optional[Dict] = None

    def prepare_current_segment() -> Dict:
        nonlocal trim_meta, preview_path
        nonlocal cached_raw_response, cached_parsed, cached_label_count, cached_preview_meta

        # Clear cached preview/labels because they belong to the previous segment
        cached_raw_response = None
        cached_parsed = None
        cached_label_count = None
        cached_preview_meta = None

        # Delete previous preview if it exists
        cleanup_local_file(preview_path)
        cleanup_empty_parents(preview_path, stop_at=local_preview_root)
        preview_path = None

        start_sec = segment_index * cfg.trimmed_duration_sec

        trim_meta = trim_audio_to_duration(
            local_input_path=local_original_path,
            trimmed_output_path=local_trimmed_path,
            duration_sec=cfg.trimmed_duration_sec,
            start_sec=start_sec,
        )

        session.open_file(local_trimmed_path)

        print(
            f"\nOpened segment {segment_index + 1} "
            f"({trim_meta['trim_start_sec']:.3f}s - {trim_meta['trim_end_sec']:.3f}s)"
        )

        return trim_meta

    try:
        trim_meta = prepare_current_segment()

        print(
            f"Trimmed to {trim_meta['trimmed_duration_sec']:.3f} s locally.\n"
            f"Will upload to server only when you press 'd'."
        )

        print("\nAnnotate the trimmed file in Audacity.")
        print("Then use one of these commands:")
        print("  s  -> read current labels, build preview WAV, close Audacity, open preview in Audacity")
        print("  2  -> discard this 2-second segment and load the next 2 seconds from the same file")
        print("  d  -> save labels JSON + current 2-second WAV + processed WAV to server, mark done")
        print("  k  -> skip this file, close Audacity, go to next file")
        print("  f  -> skip this folder, close Audacity, go to first file in next folder")
        print("  q  -> close Audacity and quit script")

        while True:
            key = input("\nEnter command (s/d/2/k/f/q): ").strip().lower()

            if key == "s":
                try:
                    raw_response, parsed, label_count = fetch_labels_from_current_audacity(
                        session=session,
                        timeout=10.0,
                    )

                    preview_rel = PurePosixPath(remote_audio_path).relative_to(PurePosixPath(cfg.remote_dataset_dir))
                    preview_dir = local_preview_root / Path(*preview_rel.parts[:-1])

                    preview_path, preview_meta = generate_preview_wav(
                        local_audio_path=local_trimmed_path,
                        parsed_labels=parsed,
                        cfg=cfg,
                        preview_dir=preview_dir,
                    )

                    cached_raw_response = raw_response
                    cached_parsed = parsed
                    cached_label_count = label_count
                    cached_preview_meta = preview_meta

                    print("\nRaw response:")
                    print(raw_response)
                    print(f"Parsed label count: {label_count}")
                    print(f"Preview WAV created:\n  {preview_path}")
                    print("New onset times in preview (sec):")
                    print(preview_meta["processing"].get("new_onset_times_sec", []))
                    print("Actual preview IPIs (sec):")
                    print(preview_meta["processing"].get("actual_ipis_sec", []))

                    session.open_file(preview_path)
                    print("Closed previous Audacity window(s) without saving and opened preview.")

                except Exception as e:
                    print(f"Could not generate preview from Audacity labels: {e}")

            elif key == "d":
                try:
                    if cached_raw_response is not None and cached_parsed is not None and cached_preview_meta is not None:
                        raw_response = cached_raw_response
                        parsed = cached_parsed
                        label_count = int(cached_label_count)
                        preview_meta = cached_preview_meta
                        if preview_path is None or not preview_path.exists():
                            preview_rel = PurePosixPath(remote_audio_path).relative_to(PurePosixPath(cfg.remote_dataset_dir))
                            preview_dir = local_preview_root / Path(*preview_rel.parts[:-1])
                            preview_path, preview_meta = generate_preview_wav(
                                local_audio_path=local_trimmed_path,
                                parsed_labels=parsed,
                                cfg=cfg,
                                preview_dir=preview_dir,
                            )
                    else:
                        raw_response, parsed, label_count = fetch_labels_from_current_audacity(
                            session=session,
                            timeout=10.0,
                        )
                        preview_rel = PurePosixPath(remote_audio_path).relative_to(PurePosixPath(cfg.remote_dataset_dir))
                        preview_dir = local_preview_root / Path(*preview_rel.parts[:-1])

                        preview_path, preview_meta = generate_preview_wav(
                            local_audio_path=local_trimmed_path,
                            parsed_labels=parsed,
                            cfg=cfg,
                            preview_dir=preview_dir,
                        )

                    metadata = {
                        "remote_audio_path": remote_audio_path,
                        "remote_trimmed_wav": remote_trimmed_wav,
                        "remote_output_json": remote_output_json,
                        "remote_processed_wav": remote_processed_wav,
                        "annotated_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "label_count": label_count,
                        "trim_info": trim_meta,
                        "segment_index": segment_index,
                        "segment_start_sec": trim_meta["trim_start_sec"],
                        "segment_end_sec": trim_meta["trim_end_sec"],
                        "audacity_labels_raw": raw_response,
                        "audacity_labels_parsed": parsed,
                        "preview_processing": preview_meta["processing"],
                        "onset_times_sec": preview_meta["onset_times_sec"],
                        "protected_intervals_sec": preview_meta["protected_intervals_sec"],
                    }

                    local_json_path = (local_output_root / Path(*rel.parts)).with_suffix(".labels.json")

                    # Upload WAVs first, JSON last, so JSON acts like the "done" marker.
                    upload_wav(
                        sftp=sftp,
                        local_wav_path=local_trimmed_path,
                        remote_wav_path=remote_trimmed_wav,
                    )

                    upload_wav(
                        sftp=sftp,
                        local_wav_path=preview_path,
                        remote_wav_path=remote_processed_wav,
                    )

                    upload_json(
                        sftp=sftp,
                        data=metadata,
                        remote_json_path=remote_output_json,
                        local_tmp_json_path=local_json_path,
                    )

                    state.mark_completed(remote_audio_path, {
                        "remote_trimmed_wav": remote_trimmed_wav,
                        "remote_output_json": remote_output_json,
                        "remote_processed_wav": remote_processed_wav,
                        "label_count": label_count,
                        "segment_index": segment_index,
                        "segment_start_sec": trim_meta["trim_start_sec"],
                        "segment_end_sec": trim_meta["trim_end_sec"],
                        "annotated_at_local_time": metadata["annotated_at_local_time"],
                    })

                    session.close(force=True)

                    print(f"\nSaved current 2-second WAV to server:\n  {remote_trimmed_wav}")
                    print(f"Saved processed WAV to server:\n  {remote_processed_wav}")
                    print(f"Saved labels JSON to server:\n  {remote_output_json}")
                    print(f"Marked complete. Label count: {label_count}")
                    print("Closed Audacity without saving.")
                    return "continue"

                except Exception as e:
                    print(f"Save failed: {e}")

            elif key == "2":
                try:
                    segment_index += 1
                    trim_meta = prepare_current_segment()

                    print(
                        f"Loaded next segment: "
                        f"{trim_meta['trim_start_sec']:.3f}s - {trim_meta['trim_end_sec']:.3f}s"
                    )

                except Exception as e:
                    segment_index -= 1
                    print(f"Could not load next 2-second segment: {e}")

            elif key == "k":
                state.mark_skipped(remote_audio_path, {
                    "skipped_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "file_skipped_by_user",
                })
                session.close(force=True)
                print("Skipped this file. Closed Audacity without saving.")
                return "continue"

            elif key == "f":
                state.mark_folder_skipped(remote_folder, {
                    "skipped_at_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "folder_skipped_by_user",
                })
                session.close(force=True)
                print(f"Skipped folder:\n  {remote_folder}")
                print("Closed Audacity without saving.")
                return "continue"

            elif key == "q":
                session.close(force=True)
                raise KeyboardInterrupt()

            else:
                print("Unknown command. Use s, d, 2, k, f, or q.")

    finally:
        cleanup_local_file(preview_path)
        cleanup_empty_parents(preview_path, stop_at=local_preview_root)

        cleanup_local_file(local_trimmed_path)
        cleanup_empty_parents(local_trimmed_path, stop_at=local_trimmed_root)

        cleanup_local_file(local_original_path)
        cleanup_empty_parents(local_original_path, stop_at=local_download_root)

        if local_original_path is not None:
            print(f"Deleted local original:\n  {local_original_path}")
        if local_trimmed_path is not None:
            print(f"Deleted local trimmed copy:\n  {local_trimmed_path}")
        if preview_path is not None:
            print(f"Deleted local preview:\n  {preview_path}")


def main():
    cfg = CONFIG

    local_work_dir = Path(cfg.local_work_dir)
    local_download_root = local_work_dir / "downloads"
    local_output_root = local_work_dir / "last_saved_json"
    local_preview_root = local_work_dir / "preview_wavs"
    local_trimmed_root = local_work_dir / "trimmed_wavs"
    state_path = local_work_dir / "annotation_state.json"

    local_work_dir.mkdir(parents=True, exist_ok=True)
    state = StateStore(state_path)

    if cfg.ssh_password is None:
        cfg.ssh_password = getpass(f"SSH password for {cfg.ssh_username}@{cfg.ssh_host}: ")

    print("Connecting to server...")
    ssh, sftp = connect_ssh(cfg)
    print("Connected.")

    session = AudacitySession(cfg)

    try:
        while True:
            remote_audio = get_next_unannotated_file(sftp, state, cfg)
            if remote_audio is None:
                print("\nNo more unannotated files found.")
                break

            print(f"\nNext file:\n  {remote_audio}")

            try:
                action = annotate_one_file(
                    cfg=cfg,
                    sftp=sftp,
                    session=session,
                    state=state,
                    remote_audio_path=remote_audio,
                    local_download_root=local_download_root,
                    local_output_root=local_output_root,
                    local_preview_root=local_preview_root,
                    local_trimmed_root=local_trimmed_root,
                )
                if action != "continue":
                    break
            except KeyboardInterrupt:
                print("\nStopping.")
                break

    finally:
        try:
            session.close(force=True)
        except Exception:
            pass
        try:
            sftp.close()
        except Exception:
            pass
        try:
            ssh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
