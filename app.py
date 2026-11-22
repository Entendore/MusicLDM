"""
MusicLDM Studio Pro - Complete Improved Version
A GUI for generating music/audio using the MusicLDM diffusion model.
"""

import sys
import os
import json
import random
import time
import shutil
import subprocess
import copy
import logging
import traceback
import inspect
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import scipy.io.wavfile
import scipy.signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog, QCheckBox, QGroupBox, QTabWidget, QPlainTextEdit,
    QStatusBar, QFormLayout, QComboBox, QSlider, QMenu, QScrollArea
)
from PySide6.QtCore import QThread, Signal, QObject, QUrl, Qt
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QFont, QCloseEvent, QAction, QDragEnterEvent, QDropEvent

import torch
from diffusers import MusicLDMPipeline


# ============================================================================
# CONSTANTS
# ============================================================================
class Const:
    """Centralized constants to avoid magic numbers."""
    SAMPLE_RATE_NATIVE = 16000          # MusicLDM native output sample rate
    SAMPLE_RATE_HIRES = 44100           # Upsampled output rate
    NORMALIZE_TARGET = 0.95             # Peak normalization target (avoid clipping)
    FADE_DURATION_SEC = 0.05            # Auto fade in/out duration
    DEFAULT_VOLUME = 0.9
    MAX_PROMPT_HISTORY = 50
    APP_NAME = "MusicLDM Studio Pro"
    SETTINGS_FILE = "musicldm_settings.json"
    LOG_FILE = "musicldm_studio.log"
    METADATA_EXT = ".meta.json"         # Sidecar metadata file extension
    NATIVE_CHUNK_DURATION = 10.24       # Approx native length of MusicLDM
    CHUNK_OVERLAP_SEC = 2.0            # Default crossfade overlap for long generations

    # Configurable crossfade (chunk stitching) bounds
    MIN_CROSSFADE_SEC = 0.5
    MAX_CROSSFADE_SEC = 8.0
    DEFAULT_CROSSFADE_SEC = 2.0

    # Loop crossfade (for seamless looping output)
    LOOP_CROSSFADE_SEC = 1.0

    # Multi-chunk loop: each chunk loops for this many seconds before stitching
    DEFAULT_LOOP_CHUNK_SEC = 10.24
    MIN_LOOP_CHUNK_SEC = 2.0

    # Seed algorithm options for chunk stitching
    SEED_ALGO_SEQUENTIAL = "sequential"
    SEED_ALGO_FIXED = "fixed"
    SEED_ALGO_RANDOM = "random"
    SEED_ALGO_HASH = "hash"

    SEED_ALGO_LABELS = {
        SEED_ALGO_SEQUENTIAL: "Sequential Offset",
        SEED_ALGO_FIXED: "Fixed (Same Seed)",
        SEED_ALGO_RANDOM: "Random Per Chunk",
        SEED_ALGO_HASH: "Hash-Based (Deterministic)",
    }


# ============================================================================
# AUDIO PROCESSING HELPERS
# ============================================================================
def apply_equal_power_fade(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    Apply equal-power fades (sin² curve) which produce fewer artifacts
    than linear fades when crossfading or removing clicks.
    """
    if len(audio) <= fade_samples * 2:
        return audio
    t_in = np.linspace(0, np.pi / 2, fade_samples)
    t_out = np.linspace(np.pi / 2, 0, fade_samples)
    fade_in = np.sin(t_in) ** 2
    fade_out = np.sin(t_out) ** 2
    audio = audio.copy()
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio


def apply_seamless_loop(audio: np.ndarray, loop_samples: int) -> np.ndarray:
    """
    Crossfade the tail of the audio with its head so the result loops
    seamlessly when played end-to-end. 
    The output is shorter than the input by `loop_samples` (the crossfaded 
    region is merged at the start and the original tail is dropped).
    """
    if loop_samples * 2 >= len(audio) or loop_samples <= 0:
        return audio
    tail = audio[-loop_samples:].copy()
    head = audio[:loop_samples].copy()
    t = np.linspace(0, np.pi / 2, loop_samples)
    fade_out = np.cos(t) ** 2  # 1 -> 0
    fade_in = np.sin(t) ** 2   # 0 -> 1
    crossfaded = tail * fade_out + head * fade_in
    middle = audio[loop_samples:-loop_samples]
    return np.concatenate([crossfaded, middle])


def normalize_audio(audio: np.ndarray, target: float = Const.NORMALIZE_TARGET) -> np.ndarray:
    """Peak-normalize audio to a target amplitude."""
    max_val = float(np.max(np.abs(audio)))
    if max_val > 1e-8:
        return (audio / max_val) * target
    return audio


def upsample_audio(audio: np.ndarray, target_rate: int) -> tuple[np.ndarray, int]:
    """Upsample audio using polyphase filtering (high quality)."""
    up = target_rate
    down = Const.SAMPLE_RATE_NATIVE
    return scipy.signal.resample_poly(audio, up, down), target_rate


def save_wav_with_metadata(
    path: str,
    audio: np.ndarray,
    sample_rate: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save WAV file and optional JSON sidecar with generation metadata."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    scipy.io.wavfile.write(path, rate=sample_rate, data=audio)
    if metadata:
        meta_path = os.path.splitext(path)[0] + Const.METADATA_EXT
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
        except OSError as e:
            logging.warning(f"Could not write metadata sidecar: {e}")


# ============================================================================
# WORKER
# ============================================================================
class MusicLDMWorker(QObject):
    """Worker that runs generation in a background QThread."""
    finished = Signal(str)
    batch_finished = Signal()
    batch_progress = Signal(int, int)
    error = Signal(str)
    status = Signal(str)
    progress = Signal(int)
    progress_max_update = Signal(int)
    log = Signal(str)
    eta_update = Signal(str)
    elapsed_update = Signal(str)
    vram_update = Signal(str, str)

    _shared_pipe: Optional[MusicLDMPipeline] = None
    _current_cache_dir: Optional[str] = None

    def __init__(self, mode: str, params: Dict[str, Any]):
        super().__init__()
        self.mode = mode
        self.params = params
        self._is_running = True
        self._is_paused = False
        self.start_time: Optional[float] = None
        self.total_steps = self.params.get("steps", 20)

    # ---- model loading ------------------------------------------------------
    def load_model(self) -> None:
        cache_dir = self.params.get("cache_dir", r"E:\cacheAI")

        if (MusicLDMWorker._shared_pipe is not None
                and MusicLDMWorker._current_cache_dir == cache_dir):
            self.log.emit("Model already loaded in memory.")
            return

        self.status.emit("Loading MusicLDM Model (first run may take a minute)...")
        self.log.emit(f"Using cache directory: {cache_dir}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                test_file = os.path.join(cache_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("ok")
                os.remove(test_file)
        except OSError as e:
            self.log.emit(f"Warning: cache dir not writable ({e}). Using default.")
            cache_dir = None

        try:
            pipe = MusicLDMPipeline.from_pretrained(
                "ucsd-reach/musicldm",
                cache_dir=cache_dir,
                torch_dtype=dtype,
            )
            pipe = pipe.to(device)
            if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                self.log.emit("Enabled CPU offload to reduce VRAM usage.")
            MusicLDMWorker._shared_pipe = pipe
            MusicLDMWorker._current_cache_dir = cache_dir
            self.log.emit(f"Model loaded successfully on {device.upper()}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    # ---- main entry ---------------------------------------------------------
    def run(self) -> None:
        try:
            self.load_model()
            if not self._is_running:
                return

            pipe = MusicLDMWorker._shared_pipe
            if pipe is None:
                raise RuntimeError("Pipeline failed to load.")

            if self.params.get("vae_slicing", False) and hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
                self.log.emit("VAE Slicing enabled to save VRAM.")

            if self.mode == "single":
                self._run_single()
            elif self.mode == "batch":
                self._run_batch()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- generation core ----------------------------------------------------
    def _build_metadata(self, prompt: str, negative_prompt: str,
                        idx: int, total: int, seed: int) -> Dict[str, Any]:
        return {
            "app": Const.APP_NAME,
            "generated_at": datetime.now().isoformat(),
            "model": "ucsd-reach/musicldm",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "variation_index": idx,
            "total_variations": total,
            "seed": seed,
            "duration_sec": self.params["duration"],
            "num_inference_steps": self.params["steps"],
            "guidance_scale": self.params["guidance"],
            "upsampled": self.params.get("upsample", False),
            "normalized": self.params.get("normalize", False),
            "auto_fade": self.params.get("auto_fade", True),
            "loop_audio": self.params.get("loop_audio", False),
            "loop_chunk_seconds": self.params.get("loop_chunk_seconds", Const.DEFAULT_LOOP_CHUNK_SEC),
            "crossfade_duration": self.params.get("crossfade_duration", Const.DEFAULT_CROSSFADE_SEC),
            "sample_rate": (Const.SAMPLE_RATE_HIRES
                            if self.params.get("upsample") else Const.SAMPLE_RATE_NATIVE),
            "seed_algorithm": self.params.get("seed_algorithm", Const.SEED_ALGO_SEQUENTIAL),
        }

    def _make_callback(self, chunk_index: int):
        """Creates a callback function for a specific chunk index to offset step count."""
        def callback_new(pipe, step, timestep, callback_kwargs):
            global_step = (chunk_index * self.params["steps"]) + step
            self._update_progress(global_step)
            return callback_kwargs

        def callback_legacy(step, timestep, latents):
            global_step = (chunk_index * self.params["steps"]) + step
            self._update_progress(global_step)
            return {}

        return callback_new, callback_legacy

    # ---- chunk seed computation --------------------------------------------
    def _compute_chunk_seed(self, base_seed: int, var_idx: int, chunk_idx: int) -> int:
        algorithm = self.params.get("seed_algorithm", Const.SEED_ALGO_SEQUENTIAL)

        if algorithm == Const.SEED_ALGO_FIXED:
            return base_seed % 1_000_000
        elif algorithm == Const.SEED_ALGO_RANDOM:
            return random.randint(0, 999999)
        elif algorithm == Const.SEED_ALGO_HASH:
            key = f"{base_seed}:{var_idx}:{chunk_idx}"
            h = hashlib.sha256(key.encode("utf-8")).hexdigest()
            return int(h[:8], 16) % 1_000_000
        else:  # SEED_ALGO_SEQUENTIAL (default)
            return (base_seed + var_idx * 1000 + chunk_idx * 100) % 1_000_000

    def _generate_audio(self, prompt: str, negative_prompt: str, output_path: str) -> List[str]:
        seed = self.params["seed"]
        variations = self.params.get("variations", 1)
        requested_duration = self.params["duration"]
        crossfade_duration = float(self.params.get(
            "crossfade_duration", Const.DEFAULT_CROSSFADE_SEC
        ))
        loop_audio = bool(self.params.get("loop_audio", False))
        loop_chunk_seconds = float(self.params.get(
            "loop_chunk_seconds", Const.DEFAULT_LOOP_CHUNK_SEC
        ))

        self.log.emit(f"Generating: '{prompt}'")
        self.log.emit(f"Using Seed: {seed}")
        self.log.emit(
            f"Parameters: Duration={requested_duration}s, "
            f"Steps={self.params['steps']}, Guidance={self.params['guidance']}, "
            f"Variations={variations}, Crossfade={crossfade_duration}s, "
            f"Loop={loop_audio}, LoopChunk={loop_chunk_seconds}s"
        )

        pipe = MusicLDMWorker._shared_pipe
        sig_params = inspect.signature(pipe.__call__).parameters
        supports_length = "audio_length_in_seconds" in sig_params

        needs_chunking = (
            not supports_length and requested_duration > Const.NATIVE_CHUNK_DURATION
        )

        # Setup progress bar limits for chunked vs single
        if needs_chunking:
            if loop_audio:
                # Multi-chunk loop: calculate number of loop sections
                if loop_chunk_seconds >= requested_duration:
                    # Single base chunk, tiled to full duration
                    self.total_steps = self.params["steps"]
                else:
                    # Multiple sections, each looping for loop_chunk_seconds
                    safe_overlap = min(
                        crossfade_duration, loop_chunk_seconds / 2
                    )
                    step_duration = loop_chunk_seconds - safe_overlap
                    num_loop_sections = int(np.ceil(
                        (requested_duration - safe_overlap) / step_duration
                    ))
                    self.total_steps = self.params["steps"] * num_loop_sections
            else:
                safe_overlap = min(crossfade_duration, Const.NATIVE_CHUNK_DURATION / 2)
                step_duration = Const.NATIVE_CHUNK_DURATION - safe_overlap
                num_chunks = int(np.ceil((requested_duration - safe_overlap) / step_duration))
                self.total_steps = self.params["steps"] * num_chunks
            self.progress_max_update.emit(self.total_steps)
        else:
            self.total_steps = self.params["steps"]
            self.progress_max_update.emit(self.total_steps)

        self.start_time = time.time()
        self.eta_update.emit("Calculating ETA...")

        paths: List[str] = []
        target_native_samples = int(requested_duration * Const.SAMPLE_RATE_NATIVE)

        for var_idx in range(variations):
            if needs_chunking:
                if loop_audio:
                    self.log.emit(
                        f"Loop Audio enabled (chunk={loop_chunk_seconds}s, "
                        f"Variation {var_idx+1}/{variations})..."
                    )
                    audio = self._generate_loop_audio(
                        prompt, negative_prompt, var_idx, requested_duration
                    )
                else:
                    algo_label = Const.SEED_ALGO_LABELS.get(
                        self.params.get("seed_algorithm", Const.SEED_ALGO_SEQUENTIAL),
                        Const.SEED_ALGO_LABELS[Const.SEED_ALGO_SEQUENTIAL]
                    )
                    self.log.emit(
                        f"Generating long audio via chunking "
                        f"(Variation {var_idx+1}/{variations}, Seed Algo: {algo_label}, "
                        f"Crossfade: {crossfade_duration}s)..."
                    )
                    audio = self._generate_chunked_audio(
                        prompt, negative_prompt, var_idx, requested_duration
                    )
            else:
                cb_new, cb_legacy = self._make_callback(0)
                kwargs: Dict[str, Any] = dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=self.params["steps"],
                    guidance_scale=self.params["guidance"],
                    num_waveforms_per_prompt=1,
                    generator=torch.Generator(device="cpu").manual_seed(seed + var_idx),
                )

                if supports_length:
                    kwargs["audio_length_in_seconds"] = requested_duration

                if "callback_on_step_end" in sig_params:
                    kwargs["callback_on_step_end"] = cb_new
                    kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                elif "callback" in sig_params:
                    kwargs["callback"] = cb_legacy
                    kwargs["callback_steps"] = 1

                audios = pipe(**kwargs).audios
                audio = audios[0].astype(np.float32)

                # Fallback: Trim or pad if unsupported length but short duration
                if not supports_length:
                    if len(audio) > target_native_samples:
                        audio = audio[:target_native_samples]
                    elif len(audio) < target_native_samples:
                        padding = target_native_samples - len(audio)
                        audio = np.pad(audio, (0, padding), 'constant')

            if self.params.get("normalize", False):
                audio = normalize_audio(audio)

            if self.params.get("upsample", False):
                self.log.emit(f"Upsampling variation {var_idx + 1} to "
                              f"{Const.SAMPLE_RATE_HIRES / 1000:.1f} kHz...")
                audio, sample_rate = upsample_audio(audio, Const.SAMPLE_RATE_HIRES)
            else:
                sample_rate = Const.SAMPLE_RATE_NATIVE

            # Apply final seamless loop crossfade to the exact requested duration
            if loop_audio:
                loop_fade_samples = int(Const.LOOP_CROSSFADE_SEC * sample_rate)
                if loop_fade_samples * 2 < len(audio):
                    self.log.emit(
                        f"Applying final seamless loop crossfade "
                        f"({Const.LOOP_CROSSFADE_SEC}s)..."
                    )
                    audio = apply_seamless_loop(audio, loop_fade_samples)
                else:
                    self.log.emit("Audio too short for final loop crossfade; skipping.")
            elif self.params.get("auto_fade", True):
                fade_samples = int(Const.FADE_DURATION_SEC * sample_rate)
                audio = apply_equal_power_fade(audio, fade_samples)

            if variations > 1:
                base, ext = os.path.splitext(output_path)
                save_path = f"{base}_v{var_idx + 1}{ext}"
            else:
                save_path = output_path

            metadata = self._build_metadata(prompt, negative_prompt, var_idx, variations, seed)
            save_wav_with_metadata(save_path, audio, sample_rate, metadata)
            paths.append(save_path)

        elapsed = time.time() - self.start_time
        self.log.emit(f"Saved {len(paths)} variation(s) in {elapsed:.1f}s.")
        return paths

    # ========================================================================
    # LOOP AUDIO GENERATION (single-chunk or multi-chunk)
    # ========================================================================
    def _generate_loop_audio(self, prompt: str, negative_prompt: str,
                             var_idx: int, requested_duration: float) -> np.ndarray:
        """
        Generates loop audio with optional multi-chunk stitching.

        If loop_chunk_seconds >= requested_duration:
            Generates a single base chunk and tiles it to reach the requested
            duration (original behavior).

        Otherwise (multi-chunk loop):
            Generates multiple base chunks. Each chunk is internally made
            seamlessly loopable, then tiled to fill 'loop_chunk_seconds'.
            The tiled sections are then stitched together with equal-power
            crossfades and gain matching. This provides variety across the
            loop while keeping each section internally smooth.
        """
        sr = Const.SAMPLE_RATE_NATIVE
        target_samples = int(requested_duration * sr)
        loop_fade_samples = int(Const.LOOP_CROSSFADE_SEC * sr)
        loop_chunk_seconds = float(self.params.get(
            "loop_chunk_seconds", Const.DEFAULT_LOOP_CHUNK_SEC
        ))
        loop_chunk_seconds = max(Const.MIN_LOOP_CHUNK_SEC, loop_chunk_seconds)

        pipe = MusicLDMWorker._shared_pipe
        sig_params = inspect.signature(pipe.__call__).parameters

        # --- Single chunk loop (original behavior) ---
        if loop_chunk_seconds >= requested_duration:
            return self._generate_single_loop_chunk(
                prompt, negative_prompt, var_idx, requested_duration,
                pipe, sig_params, sr, target_samples, loop_fade_samples
            )

        # --- Multi-chunk loop ---
        # Each section: generate base chunk -> seamless-loop it -> tile to
        # loop_chunk_seconds -> stitch with crossfade
        crossfade_duration = float(self.params.get(
            "crossfade_duration", Const.DEFAULT_CROSSFADE_SEC
        ))
        crossfade_duration = max(
            Const.MIN_CROSSFADE_SEC,
            min(crossfade_duration,
                min(Const.NATIVE_CHUNK_DURATION / 2, loop_chunk_seconds / 2))
        )

        overlap_samples = int(crossfade_duration * sr)
        step_samples = int((loop_chunk_seconds - crossfade_duration) * sr)
        section_target_samples = int(loop_chunk_seconds * sr)

        # Total output needs extra for the final end-to-end loop crossfade
        total_output_samples = target_samples + loop_fade_samples

        num_sections = int(np.ceil(
            (total_output_samples - overlap_samples) / step_samples
        ))
        if num_sections < 1:
            num_sections = 1

        algo_label = Const.SEED_ALGO_LABELS.get(
            self.params.get("seed_algorithm", Const.SEED_ALGO_SEQUENTIAL),
            Const.SEED_ALGO_LABELS[Const.SEED_ALGO_SEQUENTIAL]
        )
        self.log.emit(
            f"Multi-chunk loop: {num_sections} sections, each looping "
            f"~{loop_chunk_seconds:.2f}s (Crossfade: {crossfade_duration:.2f}s, "
            f"Seed Algo: {algo_label})..."
        )

        final_audio = np.zeros(total_output_samples, dtype=np.float32)
        write_pos = 0

        for s_idx in range(num_sections):
            if not self._is_running:
                raise RuntimeError("Generation cancelled by user.")

            chunk_seed = self._compute_chunk_seed(self.params["seed"], var_idx, s_idx)
            cb_new, cb_legacy = self._make_callback(s_idx)

            kwargs: Dict[str, Any] = dict(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=self.params["steps"],
                guidance_scale=self.params["guidance"],
                num_waveforms_per_prompt=1,
                generator=torch.Generator(device="cpu").manual_seed(chunk_seed),
            )

            if "callback_on_step_end" in sig_params:
                kwargs["callback_on_step_end"] = cb_new
                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            elif "callback" in sig_params:
                kwargs["callback"] = cb_legacy
                kwargs["callback_steps"] = 1

            self.log.emit(
                f"Generating loop section {s_idx+1}/{num_sections} "
                f"(Seed: {chunk_seed})..."
            )

            audios = pipe(**kwargs).audios
            base_chunk = audios[0].astype(np.float32)

            # 1. Make the base chunk seamlessly loopable internally
            if loop_fade_samples * 2 < len(base_chunk):
                base_chunk = apply_seamless_loop(base_chunk, loop_fade_samples)
            else:
                self.log.emit(
                    "Warning: Base chunk too short for internal loop crossfade."
                )

            if len(base_chunk) == 0:
                self.log.emit("Warning: Empty base chunk, skipping section.")
                continue

            # 2. Tile the looped base chunk to fill this section's duration
            #    For the last section, fill whatever remains of final_audio
            remaining_to_fill = total_output_samples - write_pos
            if s_idx > 0:
                remaining_to_fill += overlap_samples  # need crossfade input too

            needed = max(section_target_samples, remaining_to_fill)
            needed = max(needed, overlap_samples * 2)  # safety floor

            repeats = int(np.ceil(needed / len(base_chunk)))
            section_audio = np.tile(base_chunk, repeats)[:needed]

            # Safety: pad if too short for overlap
            if len(section_audio) < overlap_samples * 2:
                section_audio = np.pad(
                    section_audio,
                    (0, overlap_samples * 2 - len(section_audio)),
                    'constant'
                )

            # 3. Stitch into final_audio
            if s_idx == 0:
                copy_len = min(len(section_audio), total_output_samples - write_pos)
                final_audio[write_pos:write_pos + copy_len] = section_audio[:copy_len]
                write_pos += step_samples
            else:
                # 3a. Gain matching between overlapping regions
                out_overlap = final_audio[write_pos - overlap_samples:write_pos]
                in_overlap = section_audio[:overlap_samples]

                out_peak = np.max(np.abs(out_overlap))
                in_peak = np.max(np.abs(in_overlap))

                if in_peak > 1e-8 and out_peak > 1e-8:
                    scale_factor = out_peak / in_peak
                    section_audio *= scale_factor
                    in_overlap = section_audio[:overlap_samples]

                # 3b. Equal-power crossfade
                t = np.linspace(0, np.pi / 2, overlap_samples)
                fade_out = np.cos(t) ** 2
                fade_in = np.sin(t) ** 2

                crossfaded = (out_overlap * fade_out) + (in_overlap * fade_in)
                final_audio[write_pos - overlap_samples:write_pos] = crossfaded

                # 3c. Append remaining non-overlapped part
                remaining_section = section_audio[overlap_samples:]
                copy_len = min(len(remaining_section), total_output_samples - write_pos)
                if copy_len > 0:
                    final_audio[write_pos:write_pos + copy_len] = remaining_section[:copy_len]

                write_pos += step_samples

        self.log.emit(
            f"Stitched {num_sections} loop sections to reach "
            f"{requested_duration:.2f}s."
        )
        return final_audio

    def _generate_single_loop_chunk(self, prompt: str, negative_prompt: str,
                                     var_idx: int, requested_duration: float,
                                     pipe, sig_params, sr: int,
                                     target_samples: int,
                                     loop_fade_samples: int) -> np.ndarray:
        """
        Original single-chunk loop behavior: generate one base chunk,
        make it seamlessly loopable, then tile to reach the requested duration.
        """
        chunk_seed = self._compute_chunk_seed(self.params["seed"], var_idx, 0)
        cb_new, cb_legacy = self._make_callback(0)

        kwargs: Dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=self.params["steps"],
            guidance_scale=self.params["guidance"],
            num_waveforms_per_prompt=1,
            generator=torch.Generator(device="cpu").manual_seed(chunk_seed),
        )

        if "callback_on_step_end" in sig_params:
            kwargs["callback_on_step_end"] = cb_new
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        elif "callback" in sig_params:
            kwargs["callback"] = cb_legacy
            kwargs["callback_steps"] = 1

        self.log.emit(f"Generating base loop chunk (Seed: {chunk_seed})...")

        audios = pipe(**kwargs).audios
        chunk_audio = audios[0].astype(np.float32)

        # 1. Make the base chunk seamlessly loopable internally
        if loop_fade_samples * 2 < len(chunk_audio):
            self.log.emit(
                f"Applying seamless loop to base chunk "
                f"({Const.LOOP_CROSSFADE_SEC}s)..."
            )
            chunk_audio = apply_seamless_loop(chunk_audio, loop_fade_samples)
        else:
            self.log.emit("Warning: Base chunk too short for internal loop crossfade.")

        if len(chunk_audio) == 0:
            return np.zeros(target_samples, dtype=np.float32)

        # 2. Tile the looped chunk to reach target_samples (+ extra for final loop)
        required_len = target_samples + loop_fade_samples
        repeats = int(np.ceil(required_len / len(chunk_audio)))
        final_audio = np.tile(chunk_audio, repeats)
        final_audio = final_audio[:required_len]

        self.log.emit(
            f"Looped base chunk {repeats}x to reach {requested_duration:.2f}s."
        )
        return final_audio

    # ========================================================================
    # CHUNKED AUDIO GENERATION (non-loop, for long durations)
    # ========================================================================
    def _generate_chunked_audio(self, prompt: str, negative_prompt: str,
                                 var_idx: int, requested_duration: float) -> np.ndarray:
        """
        Generates audio in chunks and crossfades them for long durations.
        The crossfade duration is configurable via params['crossfade_duration'].
        """
        crossfade_duration = float(self.params.get(
            "crossfade_duration", Const.DEFAULT_CROSSFADE_SEC
        ))
        # Clamp to safe bounds
        crossfade_duration = max(
            Const.MIN_CROSSFADE_SEC,
            min(crossfade_duration, Const.NATIVE_CHUNK_DURATION / 2)
        )

        overlap_duration = crossfade_duration
        step_duration = Const.NATIVE_CHUNK_DURATION - overlap_duration
        sr = Const.SAMPLE_RATE_NATIVE

        overlap_samples = int(overlap_duration * sr)
        step_samples = int(step_duration * sr)
        target_samples = int(requested_duration * sr)

        num_chunks = int(np.ceil((target_samples - overlap_samples) / step_samples))
        if num_chunks < 1:
            num_chunks = 1

        algo_label = Const.SEED_ALGO_LABELS.get(
            self.params.get("seed_algorithm", Const.SEED_ALGO_SEQUENTIAL),
            Const.SEED_ALGO_LABELS[Const.SEED_ALGO_SEQUENTIAL]
        )
        self.log.emit(
            f"Stitching {num_chunks} chunks "
            f"(Crossfade: {overlap_duration:.2f}s, Step: {step_duration:.2f}s, "
            f"Seed Algo: {algo_label})..."
        )

        # Pre-allocate final audio array
        final_audio = np.zeros(target_samples, dtype=np.float32)
        write_pos = 0

        pipe = MusicLDMWorker._shared_pipe
        sig_params = inspect.signature(pipe.__call__).parameters

        for c_idx in range(num_chunks):
            if not self._is_running:
                raise RuntimeError("Generation cancelled by user.")

            chunk_seed = self._compute_chunk_seed(self.params["seed"], var_idx, c_idx)
            cb_new, cb_legacy = self._make_callback(c_idx)

            kwargs: Dict[str, Any] = dict(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=self.params["steps"],
                guidance_scale=self.params["guidance"],
                num_waveforms_per_prompt=1,
                generator=torch.Generator(device="cpu").manual_seed(chunk_seed),
            )

            if "callback_on_step_end" in sig_params:
                kwargs["callback_on_step_end"] = cb_new
                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            elif "callback" in sig_params:
                kwargs["callback"] = cb_legacy
                kwargs["callback_steps"] = 1

            self.log.emit(f"Generating chunk {c_idx+1}/{num_chunks} (Seed: {chunk_seed})...")

            audios = pipe(**kwargs).audios
            chunk_audio = audios[0].astype(np.float32)

            # Safety: pad if too short for overlap
            if len(chunk_audio) < overlap_samples * 2:
                chunk_audio = np.pad(
                    chunk_audio,
                    (0, overlap_samples * 2 - len(chunk_audio)),
                    'constant'
                )

            if c_idx == 0:
                copy_len = min(len(chunk_audio), target_samples - write_pos)
                final_audio[write_pos:write_pos + copy_len] = chunk_audio[:copy_len]
                write_pos += step_samples
            else:
                # 1. Gain Matching
                out_overlap = final_audio[write_pos - overlap_samples:write_pos]
                in_overlap = chunk_audio[:overlap_samples]

                out_peak = np.max(np.abs(out_overlap))
                in_peak = np.max(np.abs(in_overlap))

                if in_peak > 1e-8 and out_peak > 1e-8:
                    scale_factor = out_peak / in_peak
                    chunk_audio *= scale_factor
                    in_overlap = chunk_audio[:overlap_samples]

                # 2. Equal-power crossfade
                t = np.linspace(0, np.pi / 2, overlap_samples)
                fade_out = np.cos(t) ** 2
                fade_in = np.sin(t) ** 2

                crossfaded = (out_overlap * fade_out) + (in_overlap * fade_in)
                final_audio[write_pos - overlap_samples:write_pos] = crossfaded

                # 3. Append remaining non-overlapped part
                remaining_chunk = chunk_audio[overlap_samples:]
                copy_len = min(len(remaining_chunk), target_samples - write_pos)
                if copy_len > 0:
                    final_audio[write_pos:write_pos + copy_len] = remaining_chunk[:copy_len]

                write_pos += step_samples

        return final_audio

    def _update_progress(self, step: int) -> None:
        if not self._is_running:
            raise RuntimeError("Generation cancelled by user.")

        while self._is_paused and self._is_running:
            time.sleep(0.1)
        if not self._is_running:
            raise RuntimeError("Generation cancelled by user.")

        self.progress.emit(step + 1)
        self.status.emit(f"Diffusion Step {step + 1}/{self.total_steps}...")

        if self.start_time and step > 0:
            elapsed = time.time() - self.start_time
            steps_left = self.total_steps - (step + 1)
            eta_sec = (elapsed / (step + 1)) * steps_left
            self.eta_update.emit(f"ETA: {int(eta_sec)}s")
            self.elapsed_update.emit(f"Elapsed: {int(elapsed)}s")

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            self.vram_update.emit(f"{alloc:.2f} GB", f"{reserved:.2f} GB")

    # ---- single -------------------------------------------------------------
    def _run_single(self) -> None:
        self.status.emit("Starting single generation...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.params.get("single_out_dir", os.path.abspath("."))
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_loop" if self.params.get("loop_audio", False) else ""
        output_path = os.path.join(
            out_dir, f"musicldm_output_{timestamp}{suffix}.wav"
        )

        result_paths = self._generate_audio(
            self.params["prompt"], self.params["negative_prompt"], output_path
        )
        if self._is_running:
            self.status.emit("Generation Complete!")
            self.finished.emit(result_paths[0])

    # ---- batch --------------------------------------------------------------
    def _run_batch(self) -> None:
        self.status.emit("Starting batch generation...")
        prompts: List[str] = self.params.get("prompts", [])
        total = len(prompts)
        if total == 0:
            self.error.emit("No prompts to process.")
            return

        skip_existing = self.params.get("skip_existing", False)
        completed = 0

        for i, prompt in enumerate(prompts):
            if not self._is_running:
                break
            prompt = prompt.strip()
            if not prompt:
                continue

            while self._is_paused and self._is_running:
                time.sleep(0.2)
            if not self._is_running:
                break

            self.batch_progress.emit(i + 1, total)
            self.status.emit(f"Batch {i + 1}/{total}: {prompt[:40]}...")

            safe_name = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
            output_path = os.path.join(
                self.params["output_dir"], f"{i + 1:02d}_{safe_name[:30]}.wav"
            )

            if skip_existing and os.path.exists(output_path):
                self.log.emit(f"Skipping (exists): {output_path}")
                completed += 1
                continue

            if self.params.get("vary_seed_per_prompt", False):
                effective_seed = (self.params["seed"] + i) % 1_000_000
                self.params["seed"] = effective_seed
                self.log.emit(f"Effective seed for item {i + 1}: {effective_seed}")

            try:
                self._generate_audio(prompt, self.params.get("negative_prompt", ""), output_path)
                completed += 1
            except Exception as e:
                self.log.emit(f"Error on prompt '{prompt}': {e}")

        if self._is_running:
            self.batch_progress.emit(total, total)
            self.status.emit(f"Batch complete: {completed}/{total} succeeded.")
            self.batch_finished.emit()

    # ---- control ------------------------------------------------------------
    def cancel(self) -> None:
        self._is_running = False
        self._is_paused = False

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False


# ============================================================================
# MAIN WINDOW
# ============================================================================
class MusicLDMStudio(QMainWindow):
    DEFAULT_SETTINGS: Dict[str, Any] = {
        "cache_dir": r"E:\cacheAI",
        "single_out_dir": os.path.abspath("outputs"),
        "prompt_history": [],
        "single": {
            "prompt": "", "negative": "", "duration": 5.0, "steps": 20,
            "guidance": 3.5, "seed": 42, "random_seed": False, "upsample": True,
            "normalize": True, "variations": 1, "auto_fade": True, "auto_play": False,
            "seed_algorithm": Const.SEED_ALGO_SEQUENTIAL,
            "crossfade_duration": Const.DEFAULT_CROSSFADE_SEC,
            "loop_audio": False,
            "loop_chunk_seconds": Const.DEFAULT_LOOP_CHUNK_SEC,
        },
        "batch": {
            "file_path": "", "out_dir": "", "duration": 5.0, "steps": 20,
            "guidance": 3.5, "seed": 42, "random_seed": False, "upsample": True,
            "normalize": True, "variations": 1, "auto_fade": True,
            "skip_existing": False, "vary_seed_per_prompt": False,
            "seed_algorithm": Const.SEED_ALGO_SEQUENTIAL,
            "crossfade_duration": Const.DEFAULT_CROSSFADE_SEC,
            "loop_audio": False,
            "loop_chunk_seconds": Const.DEFAULT_LOOP_CHUNK_SEC,
        },
        "vae_slicing": False,
        "volume": Const.DEFAULT_VOLUME,
    }

    PRESETS: List[Dict[str, str]] = [
        {
            "label": "🌃 Synthwave 80s",
            "prompt": "A retro 1980s synthwave track with pulsing analog bass, bright synth leads, neon atmosphere, and a driving drum machine beat at 120 BPM"
        },
        {
            "label": "🎬 Epic Cinematic",
            "prompt": "An epic cinematic orchestral score with heavy brass, sweeping string sections, and thunderous timpani building to a massive crescendo"
        },
        {
            "label": "🎧 Lo-Fi Hip Hop",
            "prompt": "A chill lo-fi hip hop beat with dusty vinyl crackle, mellow Rhodes piano, lazy boom-bap drums, and a warm sub bass at 85 BPM"
        },
        {
            "label": "🎸 Acoustic Campfire",
            "prompt": "A warm acoustic guitar folk song with cheerful strumming, subtle fingerpicking, and a gentle campfire ambience in the background"
        },
        {
            "label": "🎷 Smooth Jazz Club",
            "prompt": "A smooth jazz saxophone solo with upright bass, brushed drum rhythms, and warm piano chords in a dimly lit, intimate club setting"
        },
        {
            "label": "🌌 Ambient Space",
            "prompt": "A deep ambient space drone with ethereal synthesizer pads, slow evolving textures, cosmic reverb, and no percussion"
        },
        {
            "label": "🥁 Techno Club",
            "prompt": "A driving techno dance track with a heavy four-on-the-floor kick, pulsing sub bass, dark synth stabs, and hi-hats at 128 BPM"
        },
        {
            "label": "🎹 Melancholy Piano",
            "prompt": "A melancholic solo classical piano piece in a minor key, expressive and delicate, with subtle room reverberation"
        },
        {
            "label": "🌴 Bossa Nova",
            "prompt": "A relaxing Bossa Nova track with nylon string guitar, soft brush percussion, double bass, and a gentle Latin rhythm"
        },
        {
            "label": "🤠 Country Western",
            "prompt": "An upbeat country western song with twangy Telecaster guitar, acoustic rhythm, fiddle, and a steady train-beat snare drum"
        }
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(Const.APP_NAME)
        self.resize(1000, 880)
        self.setAcceptDrops(True)
        self.setStyleSheet(self._stylesheet())

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[MusicLDMWorker] = None
        self.current_audio_path: Optional[str] = None

        self.settings_file = Const.SETTINGS_FILE
        self.settings = self._load_settings()

        self._setup_logger()

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(self.settings.get("volume", Const.DEFAULT_VOLUME))
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)

        self.init_ui()
        self._apply_settings_to_ui()
        self._setup_menus()

    # ---- logging ------------------------------------------------------------
    def _setup_logger(self) -> None:
        self.log_file = open(Const.LOG_FILE, "w", encoding="utf-8")
        logging.basicConfig(
            stream=self.log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self._write_log(f"--- {Const.APP_NAME} Log Initialized at {datetime.now()} ---")

    def _write_log(self, message: str) -> None:
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def append_log(self, message: str) -> None:
        self.log_window.appendPlainText(message)
        self._write_log(message)
        logging.info(message)

    # ---- settings -----------------------------------------------------------
    def _load_settings(self) -> Dict[str, Any]:
        defaults = copy.deepcopy(self.DEFAULT_SETTINGS)
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for key, default_val in defaults.items():
                    if key in loaded:
                        if isinstance(default_val, dict):
                            default_val.update(loaded[key])
                        else:
                            defaults[key] = loaded[key]
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Could not load settings: {e}")
        return defaults

    def _save_settings(self) -> None:
        self.settings["cache_dir"] = self.cache_dir_input.text()
        self.settings["single_out_dir"] = self.single_out_dir_input.text()
        self.settings["vae_slicing"] = self.vae_slicing_check.isChecked()
        self.settings["volume"] = self.volume_slider.value() / 100.0

        prompt = self.prompt_combo.currentText().strip()
        if prompt and prompt not in self.settings["prompt_history"]:
            self.settings["prompt_history"].insert(0, prompt)
            self.settings["prompt_history"] = self.settings["prompt_history"][:Const.MAX_PROMPT_HISTORY]

        self.settings["single"] = {
            "prompt": self.prompt_combo.currentText(),
            "negative": self.negative_input.text(),
            "duration": self.duration_spin.value(),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "seed": self.seed_spin.value(),
            "random_seed": self.random_seed_check.isChecked(),
            "upsample": self.upsample_check.isChecked(),
            "normalize": self.normalize_check.isChecked(),
            "variations": self.variations_spin.value(),
            "auto_fade": self.auto_fade_check.isChecked(),
            "auto_play": self.auto_play_check.isChecked(),
            "seed_algorithm": self.seed_algo_combo.currentData(),
            "crossfade_duration": self.crossfade_spin.value(),
            "loop_audio": self.loop_check.isChecked(),
            "loop_chunk_seconds": self.loop_chunk_spin.value(),
        }

        self.settings["batch"] = {
            "file_path": self.batch_file_path.text(),
            "out_dir": self.batch_out_dir.text(),
            "duration": self.batch_duration_spin.value(),
            "steps": self.batch_steps_spin.value(),
            "guidance": self.batch_guidance_spin.value(),
            "seed": self.batch_seed_spin.value(),
            "random_seed": self.batch_random_seed_check.isChecked(),
            "upsample": self.batch_upsample_check.isChecked(),
            "normalize": self.batch_normalize_check.isChecked(),
            "variations": self.batch_variations_spin.value(),
            "auto_fade": self.batch_auto_fade_check.isChecked(),
            "skip_existing": self.batch_skip_existing_check.isChecked(),
            "vary_seed_per_prompt": self.batch_vary_seed_check.isChecked(),
            "seed_algorithm": self.batch_seed_algo_combo.currentData(),
            "crossfade_duration": self.batch_crossfade_spin.value(),
            "loop_audio": self.batch_loop_check.isChecked(),
            "loop_chunk_seconds": self.batch_loop_chunk_spin.value(),
        }

        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
        except OSError as e:
            logging.error(f"Could not save settings: {e}")

    def _apply_settings_to_ui(self) -> None:
        s = self.settings
        self.cache_dir_input.setText(s["cache_dir"])
        self.single_out_dir_input.setText(s["single_out_dir"])
        self.vae_slicing_check.setChecked(s["vae_slicing"])
        self.volume_slider.setValue(int(s.get("volume", Const.DEFAULT_VOLUME) * 100))

        self.prompt_combo.clear()
        self.prompt_combo.addItems(s["prompt_history"])

        ui = s["single"]
        self.prompt_combo.setEditText(ui["prompt"])
        self.negative_input.setText(ui["negative"])
        self.duration_spin.setValue(ui["duration"])
        self.steps_spin.setValue(ui["steps"])
        self.guidance_spin.setValue(ui["guidance"])
        self.seed_spin.setValue(ui["seed"])
        self.random_seed_check.setChecked(ui["random_seed"])
        self.upsample_check.setChecked(ui["upsample"])
        self.normalize_check.setChecked(ui["normalize"])
        self.variations_spin.setValue(ui["variations"])
        self.auto_fade_check.setChecked(ui["auto_fade"])
        self.auto_play_check.setChecked(ui["auto_play"])
        self.crossfade_spin.setValue(ui.get("crossfade_duration", Const.DEFAULT_CROSSFADE_SEC))
        self.loop_check.setChecked(ui.get("loop_audio", False))
        self.loop_chunk_spin.setValue(ui.get("loop_chunk_seconds", Const.DEFAULT_LOOP_CHUNK_SEC))

        single_algo = ui.get("seed_algorithm", Const.SEED_ALGO_SEQUENTIAL)
        idx = self.seed_algo_combo.findData(single_algo)
        if idx >= 0:
            self.seed_algo_combo.setCurrentIndex(idx)

        bi = s["batch"]
        self.batch_file_path.setText(bi["file_path"])
        self.batch_out_dir.setText(bi["out_dir"])
        self.batch_duration_spin.setValue(bi["duration"])
        self.batch_steps_spin.setValue(bi["steps"])
        self.batch_guidance_spin.setValue(bi["guidance"])
        self.batch_seed_spin.setValue(bi["seed"])
        self.batch_random_seed_check.setChecked(bi["random_seed"])
        self.batch_upsample_check.setChecked(bi["upsample"])
        self.batch_normalize_check.setChecked(bi["normalize"])
        self.batch_variations_spin.setValue(bi["variations"])
        self.batch_auto_fade_check.setChecked(bi["auto_fade"])
        self.batch_skip_existing_check.setChecked(bi.get("skip_existing", False))
        self.batch_vary_seed_check.setChecked(bi.get("vary_seed_per_prompt", False))
        self.batch_crossfade_spin.setValue(bi.get("crossfade_duration", Const.DEFAULT_CROSSFADE_SEC))
        self.batch_loop_check.setChecked(bi.get("loop_audio", False))
        self.batch_loop_chunk_spin.setValue(bi.get("loop_chunk_seconds", Const.DEFAULT_LOOP_CHUNK_SEC))

        batch_algo = bi.get("seed_algorithm", Const.SEED_ALGO_SEQUENTIAL)
        idx = self.batch_seed_algo_combo.findData(batch_algo)
        if idx >= 0:
            self.batch_seed_algo_combo.setCurrentIndex(idx)

    # ---- close event --------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:
        thread_running = False
        if self.worker_thread is not None:
            try:
                thread_running = self.worker_thread.isRunning()
            except RuntimeError:
                self.worker_thread = None

        if thread_running:
            reply = QMessageBox.question(
                self, "Exit Confirmation",
                "A generation is currently running.\nCancel it and exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

            if self.worker is not None:
                try:
                    self.worker.cancel()
                except RuntimeError:
                    pass

            self.statusBar().showMessage("Cancelling and shutting down...")
            self.worker_thread.wait(5000)
            try:
                if self.worker_thread.isRunning():
                    self.worker_thread.terminate()
                    self.worker_thread.wait(1000)
            except RuntimeError:
                pass

        self._save_settings()

        if MusicLDMWorker._shared_pipe is not None:
            del MusicLDMWorker._shared_pipe
            MusicLDMWorker._shared_pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            self._write_log(f"--- Application closed safely at {datetime.now()} ---")
            if hasattr(self, "log_file") and not self.log_file.closed:
                self.log_file.close()
        except Exception as e:
            print(f"Error closing log: {e}", file=sys.stderr)

        event.accept()

    # ---- stylesheet ---------------------------------------------------------
    def _stylesheet(self) -> str:
        return """
            QMainWindow { background-color: #121212; }
            QTabWidget::pane { border: 1px solid #333; border-radius: 4px; top: -1px; }
            QTabBar::tab { background: #1e1e1e; color: #aaa; padding: 8px 20px;
                border: 1px solid #333; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #333; color: #fff; }
            QGroupBox { color: #ddd; font-weight: bold; border: 1px solid #333;
                border-radius: 6px; margin-top: 14px; padding: 16px 10px 10px 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QLabel { color: #aaa; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QComboBox {
                background-color: #1e1e1e; color: #fff; border: 1px solid #333;
                border-radius: 4px; padding: 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background: #1e1e1e; color: #fff;
                selection-background-color: #2979ff; }
            QPushButton { background-color: #2979ff; color: #fff; font-weight: bold;
                border: none; border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background-color: #448aff; }
            QPushButton:disabled { background-color: #333; color: #666; }
            QPushButton#danger { background-color: #ff1744; }
            QPushButton#danger:hover { background-color: #ff4569; }
            QPushButton#warn { background-color: #ff9100; }
            QPushButton#warn:hover { background-color: #ffab40; }
            QPushButton#preset { background-color: #333; color: #2979ff; font-weight: bold; }
            QPushButton#preset:hover { background-color: #444; color: #448aff; }
            QProgressBar { border: 1px solid #333; border-radius: 4px; text-align: center;
                color: #fff; background-color: #1e1e1e; height: 20px; }
            QProgressBar::chunk { background-color: #2979ff; border-radius: 3px; }
            QStatusBar { background-color: #1e1e1e; color: #aaa; }
            QCheckBox { color: #aaa; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px;
                border: 1px solid #555; background: #1e1e1e; }
            QCheckBox::indicator:checked { background: #2979ff; border: 1px solid #2979ff; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 6px;
                background: #1e1e1e; border-radius: 3px; }
            QSlider::handle:horizontal { background: #2979ff; width: 14px;
                margin: -5px 0; border-radius: 7px; }
            QScrollArea { border: none; background-color: transparent; }
        """

    # ---- UI building --------------------------------------------------------
    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)

        title = QLabel(f"🎵 {Const.APP_NAME}")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #fff;")
        main_layout.addWidget(title)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self._build_single_tab()
        self._build_batch_tab()
        self._build_settings_tab()

        prog_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Step %v / %m (%p%)")
        self.progress_bar.hide()
        prog_layout.addWidget(self.progress_bar, 4)

        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color: #2979ff; font-weight: bold;")
        self.eta_label.setFixedWidth(110)
        prog_layout.addWidget(self.eta_label)

        self.elapsed_label = QLabel("")
        self.elapsed_label.setStyleSheet("color: #aaa;")
        self.elapsed_label.setFixedWidth(120)
        prog_layout.addWidget(self.elapsed_label)

        self.vram_label = QLabel("")
        self.vram_label.setStyleSheet("color: #aaa;")
        self.vram_label.setFixedWidth(180)
        prog_layout.addWidget(self.vram_label)
        main_layout.addLayout(prog_layout)

        self.batch_indicator = QLabel("")
        self.batch_indicator.setStyleSheet("color: #ff9100; font-weight: bold;")
        self.batch_indicator.hide()
        main_layout.addWidget(self.batch_indicator)

        ctrl_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel Generation")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        self.cancel_btn.setEnabled(False)
        ctrl_layout.addWidget(self.cancel_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setObjectName("warn")
        self.pause_btn.clicked.connect(self.pause_generation)
        self.pause_btn.setEnabled(False)
        ctrl_layout.addWidget(self.pause_btn)

        self.resume_btn = QPushButton("▶ Resume")
        self.resume_btn.clicked.connect(self.resume_generation)
        self.resume_btn.setEnabled(False)
        ctrl_layout.addWidget(self.resume_btn)
        main_layout.addLayout(ctrl_layout)

        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout(playback_group)

        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        playback_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾 Save As...")
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)
        playback_layout.addWidget(self.save_btn)

        playback_layout.addWidget(QLabel("Vol:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(Const.DEFAULT_VOLUME * 100))
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        playback_layout.addWidget(self.volume_slider)

        main_layout.addWidget(playback_group)

        self.log_window = QPlainTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setMaximumHeight(140)
        self.log_window.setPlaceholderText(
            f"Generation logs appear here. Full log: {Const.LOG_FILE}"
        )
        main_layout.addWidget(self.log_window)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")

    def _setup_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        clear_history = QAction("Clear Prompt History", self)
        clear_history.triggered.connect(self.clear_prompt_history)
        file_menu.addAction(clear_history)

        open_log = QAction("Open Log File", self)
        open_log.triggered.connect(self.open_log_file)
        file_menu.addAction(open_log)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menubar.addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def _build_single_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Presets in a Scrollable Area ---
        presets_container = QWidget()
        presets_layout = QHBoxLayout(presets_container)
        presets_layout.setContentsMargins(0, 0, 0, 0)
        presets_layout.addWidget(QLabel("Quick Presets:"))

        for preset in self.PRESETS:
            btn = QPushButton(preset["label"])
            btn.setObjectName("preset")
            btn.setToolTip(preset["prompt"])
            btn.clicked.connect(
                lambda _=False, p=preset["prompt"]: self.prompt_combo.setEditText(p)
            )
            presets_layout.addWidget(btn)
        presets_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(presets_container)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(50)
        layout.addWidget(scroll)

        prompt_group = QGroupBox("Prompt Engineering")
        form = QFormLayout(prompt_group)

        self.prompt_combo = QComboBox()
        self.prompt_combo.setEditable(True)
        self.prompt_combo.setInsertPolicy(QComboBox.NoInsert)
        self.prompt_combo.lineEdit().setPlaceholderText(
            "e.g., A fast tempo electronic dance track with heavy bass"
        )
        form.addRow("Prompt History:", self.prompt_combo)

        self.negative_input = QLineEdit()
        self.negative_input.setPlaceholderText("e.g., low quality, noise, distortion")
        form.addRow("Negative:", self.negative_input)
        layout.addWidget(prompt_group)

        params_group = QGroupBox("Parameters")
        params_layout = QHBoxLayout(params_group)

        params_layout.addWidget(QLabel("Dur (s):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(2.0, 180.0)
        self.duration_spin.setToolTip("Length of generated audio in seconds. (>10s uses chunk stitching).")
        params_layout.addWidget(self.duration_spin)

        params_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 200)
        self.steps_spin.setToolTip("More steps = higher quality but slower.")
        params_layout.addWidget(self.steps_spin)

        params_layout.addWidget(QLabel("Guide:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 15.0)
        self.guidance_spin.setSingleStep(0.5)
        params_layout.addWidget(self.guidance_spin)

        params_layout.addWidget(QLabel("Variations:"))
        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 5)
        params_layout.addWidget(self.variations_spin)

        params_layout.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        params_layout.addWidget(self.seed_spin)

        self.random_seed_check = QCheckBox("Random")
        self.random_seed_check.setToolTip("Pick a random seed each run.")
        params_layout.addWidget(self.random_seed_check)

        self.upsample_check = QCheckBox("Upsample")
        self.upsample_check.setToolTip("Resample 16 kHz → 44.1 kHz.")
        params_layout.addWidget(self.upsample_check)

        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setToolTip("Peak-normalize audio to avoid clipping.")
        params_layout.addWidget(self.normalize_check)

        self.auto_fade_check = QCheckBox("Fade In/Out")
        self.auto_fade_check.setToolTip("Apply 50 ms equal-power fades to remove clicks.")
        params_layout.addWidget(self.auto_fade_check)

        self.auto_play_check = QCheckBox("Auto-Play")
        params_layout.addWidget(self.auto_play_check)
        layout.addWidget(params_group)

        # --- Chunk Stitching & Looping Controls ---
        stitch_group = QGroupBox("Chunk Stitching (for audio > 10s) & Looping")
        stitch_layout = QHBoxLayout(stitch_group)

        stitch_layout.addWidget(QLabel("Seed Algo:"))
        self.seed_algo_combo = QComboBox()
        self.seed_algo_combo.setToolTip(
            "Controls how seeds are assigned to each chunk when stitching long audio:\n\n"
            "• Sequential Offset: seed + var×1000 + chunk×100 (reproducible, linear)\n"
            "• Fixed (Same Seed): All chunks use the base seed (maximum consistency)\n"
            "• Random Per Chunk: Each chunk gets a fresh random seed (maximum variety)\n"
            "• Hash-Based: SHA-256 hash of seed:var:chunk (reproducible, well-distributed)"
        )
        for algo_key, algo_label in Const.SEED_ALGO_LABELS.items():
            self.seed_algo_combo.addItem(algo_label, algo_key)
        stitch_layout.addWidget(self.seed_algo_combo)

        stitch_layout.addWidget(QLabel("Crossfade (s):"))
        self.crossfade_spin = QDoubleSpinBox()
        self.crossfade_spin.setRange(Const.MIN_CROSSFADE_SEC, Const.MAX_CROSSFADE_SEC)
        self.crossfade_spin.setSingleStep(0.25)
        self.crossfade_spin.setDecimals(2)
        self.crossfade_spin.setToolTip(
            "Crossfade overlap duration between chunks during stitching.\n"
            "Longer crossfades produce smoother transitions but require more chunks\n"
            "(and thus more compute time) to cover the requested duration.\n"
            f"Range: {Const.MIN_CROSSFADE_SEC}s – {Const.MAX_CROSSFADE_SEC}s"
        )
        stitch_layout.addWidget(self.crossfade_spin)

        self.loop_check = QCheckBox("🔁 Loop Audio")
        self.loop_check.setToolTip(
            "Generates audio that loops seamlessly end-to-end.\n\n"
            "If Loop Chunk (s) >= total duration: a single base chunk is generated\n"
            "  and tiled to reach the requested duration.\n\n"
            "If Loop Chunk (s) < total duration: multiple base chunks are generated,\n"
            "  each tiled/looped for Loop Chunk seconds, then stitched together with\n"
            "  crossfades for variety while maintaining seamless looping."
        )
        stitch_layout.addWidget(self.loop_check)

        stitch_layout.addWidget(QLabel("Loop Chunk (s):"))
        self.loop_chunk_spin = QDoubleSpinBox()
        self.loop_chunk_spin.setRange(Const.MIN_LOOP_CHUNK_SEC, 180.0)
        self.loop_chunk_spin.setSingleStep(0.5)
        self.loop_chunk_spin.setDecimals(2)
        self.loop_chunk_spin.setToolTip(
            "When Loop Audio is enabled, each generated base chunk is tiled/looped\n"
            "to fill this duration, then chunks are stitched together with crossfades.\n\n"
            "• If this value >= total duration: a single chunk is looped (simple mode).\n"
            "• If this value < total duration: multiple chunks are generated, each\n"
            "  looping for this many seconds, then crossfaded together for variety.\n\n"
            "Lower values = more variety (changes more frequently).\n"
            "Higher values = more stable/repetitive sections.\n"
            f"Range: {Const.MIN_LOOP_CHUNK_SEC}s – 180s"
        )
        stitch_layout.addWidget(self.loop_chunk_spin)

        stitch_layout.addStretch()
        layout.addWidget(stitch_group)

        self.single_gen_btn = QPushButton("🎵 Generate Single Track")
        self.single_gen_btn.setMinimumHeight(40)
        self.single_gen_btn.clicked.connect(lambda: self.start_generation("single"))
        layout.addWidget(self.single_gen_btn)

        layout.addStretch()
        self.tabs.addTab(tab, "Single Generation")

    def _build_batch_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        input_group = QGroupBox("Batch Input")
        input_layout = QVBoxLayout(input_group)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Prompt File:"))
        self.batch_file_path = QLineEdit()
        self.batch_file_path.setPlaceholderText("Drop a .txt file here or click Browse")
        file_layout.addWidget(self.batch_file_path)

        browse_btn = QPushButton(" Browse")
        browse_btn.clicked.connect(self.browse_batch_file)
        file_layout.addWidget(browse_btn)

        edit_btn = QPushButton("✏️ Edit")
        edit_btn.setToolTip("Open prompt file in default text editor")
        edit_btn.clicked.connect(self.edit_batch_file)
        file_layout.addWidget(edit_btn)

        preview_btn = QPushButton("👁 Preview")
        preview_btn.setToolTip("Show first few prompts in the file")
        preview_btn.clicked.connect(self.preview_batch_file)
        file_layout.addWidget(preview_btn)
        input_layout.addLayout(file_layout)

        out_dir_layout = QHBoxLayout()
        out_dir_layout.addWidget(QLabel("Output Folder:"))
        self.batch_out_dir = QLineEdit()
        out_dir_layout.addWidget(self.batch_out_dir)
        dir_btn = QPushButton(" Browse")
        dir_btn.clicked.connect(self.browse_batch_dir)
        out_dir_layout.addWidget(dir_btn)
        input_layout.addLayout(out_dir_layout)
        layout.addWidget(input_group)

        params_group = QGroupBox("Shared Parameters")
        params_layout = QHBoxLayout(params_group)

        params_layout.addWidget(QLabel("Dur:"))
        self.batch_duration_spin = QDoubleSpinBox()
        self.batch_duration_spin.setRange(2.0, 180.0)
        self.batch_duration_spin.setToolTip("Length of generated audio in seconds. (>10s uses chunk stitching).")
        params_layout.addWidget(self.batch_duration_spin)

        params_layout.addWidget(QLabel("Steps:"))
        self.batch_steps_spin = QSpinBox()
        self.batch_steps_spin.setRange(10, 200)
        params_layout.addWidget(self.batch_steps_spin)

        params_layout.addWidget(QLabel("Guide:"))
        self.batch_guidance_spin = QDoubleSpinBox()
        self.batch_guidance_spin.setRange(1.0, 15.0)
        self.batch_guidance_spin.setSingleStep(0.5)
        params_layout.addWidget(self.batch_guidance_spin)

        params_layout.addWidget(QLabel("Vars:"))
        self.batch_variations_spin = QSpinBox()
        self.batch_variations_spin.setRange(1, 5)
        params_layout.addWidget(self.batch_variations_spin)

        params_layout.addWidget(QLabel("Seed:"))
        self.batch_seed_spin = QSpinBox()
        self.batch_seed_spin.setRange(0, 999999)
        params_layout.addWidget(self.batch_seed_spin)

        self.batch_random_seed_check = QCheckBox("Random")
        params_layout.addWidget(self.batch_random_seed_check)

        self.batch_upsample_check = QCheckBox("Upsample")
        params_layout.addWidget(self.batch_upsample_check)

        self.batch_normalize_check = QCheckBox("Normalize")
        params_layout.addWidget(self.batch_normalize_check)

        self.batch_auto_fade_check = QCheckBox("Fade In/Out")
        params_layout.addWidget(self.batch_auto_fade_check)

        self.batch_skip_existing_check = QCheckBox("Skip Existing")
        self.batch_skip_existing_check.setToolTip("Skip prompts whose output file already exists.")
        params_layout.addWidget(self.batch_skip_existing_check)

        self.batch_vary_seed_check = QCheckBox("Vary Seed/Prompt")
        self.batch_vary_seed_check.setToolTip("Offset seed by index for variety across the batch.")
        params_layout.addWidget(self.batch_vary_seed_check)
        layout.addWidget(params_group)

        # --- Chunk Stitching & Looping Controls (Batch) ---
        stitch_group = QGroupBox("Chunk Stitching (for audio > 10s) & Looping")
        stitch_layout = QHBoxLayout(stitch_group)

        stitch_layout.addWidget(QLabel("Seed Algo:"))
        self.batch_seed_algo_combo = QComboBox()
        self.batch_seed_algo_combo.setToolTip(
            "Controls how seeds are assigned to each chunk when stitching long audio:\n\n"
            "• Sequential Offset: seed + var×1000 + chunk×100 (reproducible, linear)\n"
            "• Fixed (Same Seed): All chunks use the base seed (maximum consistency)\n"
            "• Random Per Chunk: Each chunk gets a fresh random seed (maximum variety)\n"
            "• Hash-Based: SHA-256 hash of seed:var:chunk (reproducible, well-distributed)"
        )
        for algo_key, algo_label in Const.SEED_ALGO_LABELS.items():
            self.batch_seed_algo_combo.addItem(algo_label, algo_key)
        stitch_layout.addWidget(self.batch_seed_algo_combo)

        stitch_layout.addWidget(QLabel("Crossfade (s):"))
        self.batch_crossfade_spin = QDoubleSpinBox()
        self.batch_crossfade_spin.setRange(Const.MIN_CROSSFADE_SEC, Const.MAX_CROSSFADE_SEC)
        self.batch_crossfade_spin.setSingleStep(0.25)
        self.batch_crossfade_spin.setDecimals(2)
        self.batch_crossfade_spin.setToolTip(
            "Crossfade overlap duration between chunks during stitching.\n"
            "Longer crossfades produce smoother transitions but require more chunks."
        )
        stitch_layout.addWidget(self.batch_crossfade_spin)

        self.batch_loop_check = QCheckBox("🔁 Loop Audio")
        self.batch_loop_check.setToolTip(
            "Generates audio that loops seamlessly end-to-end.\n\n"
            "If Loop Chunk (s) >= total duration: a single base chunk is looped.\n"
            "If Loop Chunk (s) < total duration: multiple chunks are generated,\n"
            "  each looping for Loop Chunk seconds, then stitched together."
        )
        stitch_layout.addWidget(self.batch_loop_check)

        stitch_layout.addWidget(QLabel("Loop Chunk (s):"))
        self.batch_loop_chunk_spin = QDoubleSpinBox()
        self.batch_loop_chunk_spin.setRange(Const.MIN_LOOP_CHUNK_SEC, 180.0)
        self.batch_loop_chunk_spin.setSingleStep(0.5)
        self.batch_loop_chunk_spin.setDecimals(2)
        self.batch_loop_chunk_spin.setToolTip(
            "Each generated base chunk is tiled/looped to fill this duration,\n"
            "then chunks are stitched together with crossfades.\n"
            "Lower = more variety, Higher = more stable/repetitive."
        )
        stitch_layout.addWidget(self.batch_loop_chunk_spin)

        stitch_layout.addStretch()
        layout.addWidget(stitch_group)

        self.batch_gen_btn = QPushButton("⚙️ Generate Batch")
        self.batch_gen_btn.setMinimumHeight(40)
        self.batch_gen_btn.clicked.connect(lambda: self.start_generation("batch"))
        layout.addWidget(self.batch_gen_btn)

        layout.addStretch()
        self.tabs.addTab(tab, "Batch Generation")

    def _build_settings_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        cache_group = QGroupBox("Model Cache Directory")
        cache_layout = QHBoxLayout(cache_group)
        cache_layout.addWidget(QLabel("Path:"))
        self.cache_dir_input = QLineEdit()
        cache_layout.addWidget(self.cache_dir_input)
        browse_btn = QPushButton(" Browse")
        browse_btn.clicked.connect(self.browse_cache_dir)
        cache_layout.addWidget(browse_btn)
        layout.addWidget(cache_group)

        out_group = QGroupBox("Single Generation Output Folder")
        out_layout = QHBoxLayout(out_group)
        out_layout.addWidget(QLabel("Path:"))
        self.single_out_dir_input = QLineEdit()
        out_layout.addWidget(self.single_out_dir_input)
        browse_out_btn = QPushButton(" Browse")
        browse_out_btn.clicked.connect(self.browse_single_out_dir)
        out_layout.addWidget(browse_out_btn)
        layout.addWidget(out_group)

        opt_group = QGroupBox("System Optimizations")
        opt_layout = QVBoxLayout(opt_group)
        self.vae_slicing_check = QCheckBox("Enable VAE Slicing (slower but saves VRAM)")
        self.vae_slicing_check.setToolTip(
            "Process the VAE decoder in chunks. Useful for 4–6 GB GPUs."
        )
        opt_layout.addWidget(self.vae_slicing_check)
        layout.addWidget(opt_group)

        info_label = QLabel(
            "ℹ️ Hugging Face model weights are stored in the Cache Directory.\n"
            "VAE Slicing processes the audio decoder in chunks to prevent OOM on 4–6 GB GPUs.\n"
            "Long Audio: Durations > 10s are generated in ~10s chunks and crossfaded together.\n"
            "Crossfade Duration: Controls overlap length between chunks (longer = smoother but slower).\n"
            "Loop Audio: Generates one chunk and repeats it to reach target duration seamlessly.\n"
            "Loop Chunk (s): When < total duration, each chunk loops for this many seconds,\n"
            "  then chunks are stitched together with crossfades for variety.\n"
            "Seed Algorithm: Controls how seeds are assigned per chunk during stitching.\n"
            f"Log File: {Const.LOG_FILE} (in the application directory)\n"
            f"Metadata Sidecar: each generation writes a {Const.METADATA_EXT} file with parameters."
        )
        info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info_label)

        layout.addStretch()
        self.tabs.addTab(tab, "⚙️ Settings")

    # ---- menu actions -------------------------------------------------------
    def clear_prompt_history(self) -> None:
        reply = QMessageBox.question(
            self, "Clear History",
            "Clear all saved prompt history?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.settings["prompt_history"] = []
            self.prompt_combo.clear()
            self._save_settings()

    def open_log_file(self) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(Const.LOG_FILE)
            elif sys.platform == "darwin":
                subprocess.run(["open", Const.LOG_FILE])
            else:
                subprocess.run(["xdg-open", Const.LOG_FILE])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open log:\n{e}")

    def show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            f"<h3>{Const.APP_NAME}</h3>"
            "<p>A GUI for the MusicLDM text-to-audio diffusion model.</p>"
            "<p>Improved version with metadata sidecar, VRAM monitoring, "
            "equal-power fades, batch pause/resume, long-audio chunking, "
            "configurable crossfade duration, seamless loop output, "
            "multi-chunk loop stitching, configurable seed algorithms, and more.</p>"
        )

    # ---- browse helpers -----------------------------------------------------
    def browse_cache_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Model Cache Directory")
        if d:
            self.cache_dir_input.setText(d)

    def browse_single_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Single Output Directory")
        if d:
            self.single_out_dir_input.setText(d)

    def browse_batch_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select Prompt File", "", "Text Files (*.txt)")
        if f:
            self.batch_file_path.setText(f)

    def browse_batch_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.batch_out_dir.setText(d)

    def edit_batch_file(self) -> None:
        file_path = self.batch_file_path.text().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", "Please select a valid prompt file first.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", file_path])
            else:
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file:\n{e}")

    def preview_batch_file(self) -> None:
        file_path = self.batch_file_path.text().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", "Please select a valid prompt file first.")
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            preview = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines[:10]))
            if len(lines) > 10:
                preview += f"\n... and {len(lines) - 10} more."
            QMessageBox.information(self, f"Preview ({len(lines)} prompts)", preview)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Could not read file:\n{e}")

    # ---- drag & drop --------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(u.toLocalFile().lower().endswith(".txt") for u in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for u in event.mimeData().urls():
            path = u.toLocalFile()
            if path.lower().endswith(".txt"):
                self.batch_file_path.setText(path)
                self.tabs.setCurrentIndex(1)
                break

    # ---- generation ---------------------------------------------------------
    def start_generation(self, mode: str) -> None:
        self._save_settings()
        params = copy.deepcopy(self.settings)

        if mode == "single":
            prompt = self.prompt_combo.currentText().strip()
            if not prompt:
                QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
                return
            if self.random_seed_check.isChecked():
                params["seed"] = random.randint(0, 999999)
                self.seed_spin.setValue(params["seed"])
            else:
                params["seed"] = self.seed_spin.value()

            params["prompt"] = prompt
            params["negative_prompt"] = self.negative_input.text()
            params["duration"] = self.duration_spin.value()
            params["steps"] = self.steps_spin.value()
            params["guidance"] = self.guidance_spin.value()
            params["variations"] = self.variations_spin.value()
            params["upsample"] = self.upsample_check.isChecked()
            params["normalize"] = self.normalize_check.isChecked()
            params["auto_fade"] = self.auto_fade_check.isChecked()
            params["seed_algorithm"] = self.seed_algo_combo.currentData()
            params["crossfade_duration"] = self.crossfade_spin.value()
            params["loop_audio"] = self.loop_check.isChecked()
            params["loop_chunk_seconds"] = self.loop_chunk_spin.value()

        elif mode == "batch":
            file_path = self.batch_file_path.text().strip()
            out_dir = self.batch_out_dir.text().strip()
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(self, "Input Error", "Please select a valid prompt file.")
                return
            if not out_dir:
                QMessageBox.warning(self, "Input Error", "Please select an output directory.")
                return
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    prompts = [ln.strip() for ln in f if ln.strip()]
            except OSError as e:
                QMessageBox.critical(self, "Read Error", str(e))
                return
            if not prompts:
                QMessageBox.warning(self, "Input Error", "The selected file is empty.")
                return

            params["prompts"] = prompts
            params["output_dir"] = out_dir
            params["batch_total"] = len(prompts)

            if self.batch_random_seed_check.isChecked():
                params["seed"] = random.randint(0, 999999)
                self.batch_seed_spin.setValue(params["seed"])
            else:
                params["seed"] = self.batch_seed_spin.value()

            params["prompt"] = ""
            params["negative_prompt"] = ""
            params["duration"] = self.batch_duration_spin.value()
            params["steps"] = self.batch_steps_spin.value()
            params["guidance"] = self.batch_guidance_spin.value()
            params["variations"] = self.batch_variations_spin.value()
            params["upsample"] = self.batch_upsample_check.isChecked()
            params["normalize"] = self.batch_normalize_check.isChecked()
            params["auto_fade"] = self.batch_auto_fade_check.isChecked()
            params["skip_existing"] = self.batch_skip_existing_check.isChecked()
            params["vary_seed_per_prompt"] = self.batch_vary_seed_check.isChecked()
            params["seed_algorithm"] = self.batch_seed_algo_combo.currentData()
            params["crossfade_duration"] = self.batch_crossfade_spin.value()
            params["loop_audio"] = self.batch_loop_check.isChecked()
            params["loop_chunk_seconds"] = self.batch_loop_chunk_spin.value()

        self.single_gen_btn.setEnabled(False)
        self.batch_gen_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.batch_indicator.setVisible(mode == "batch")
        self.vram_label.setText("VRAM: -")

        self.progress_bar.setRange(0, self.steps_spin.value() if mode == "single" else 100)
        if mode == "single":
            self.progress_bar.setFormat("Step %v / %m (%p%)")
        else:
            self.progress_bar.setFormat("Batch: %p%")

        self.eta_label.setText("")
        self.elapsed_label.setText("")
        self.log_window.clear()
        self.statusBar().showMessage("Starting...")

        self.worker_thread = QThread()
        self.worker = MusicLDMWorker(mode, params)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.progress_max_update.connect(lambda max_val: self.progress_bar.setRange(0, max_val))
        self.worker.eta_update.connect(self.eta_label.setText)
        self.worker.elapsed_update.connect(self.elapsed_label.setText)
        self.worker.vram_update.connect(
            lambda a, r: self.vram_label.setText(f"VRAM: {a} / {r}")
        )
        self.worker.batch_progress.connect(self.on_batch_progress)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.batch_finished.connect(self.on_batch_finished)
        self.worker.error.connect(self.on_generation_error)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.batch_finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.batch_finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_batch_progress(self, current: int, total: int) -> None:
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
            self.batch_indicator.setText(f"Processing item {current} of {total}")

    def cancel_generation(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.statusBar().showMessage("Cancelling...")
            self.cancel_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)

    def pause_generation(self) -> None:
        if self.worker:
            self.worker.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)
            self.statusBar().showMessage("Paused. Click Resume to continue.")

    def resume_generation(self) -> None:
        if self.worker:
            self.worker.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)
            self.statusBar().showMessage("Resumed.")

    def on_generation_finished(self, file_path: str) -> None:
        self.progress_bar.hide()
        self.eta_label.setText("")
        self.elapsed_label.setText("")
        self.reset_ui_state()
        self.statusBar().showMessage(f"Finished: {file_path}")
        self.append_log(f"\nSuccess! Best variation saved to: {file_path}")

        self.current_audio_path = file_path
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        if self.auto_play_check.isChecked():
            self.play_audio()

    def on_batch_finished(self) -> None:
        self.progress_bar.hide()
        self.eta_label.setText("")
        self.elapsed_label.setText("")
        self.batch_indicator.setText("")
        self.batch_indicator.hide()
        self.reset_ui_state()
        self.statusBar().showMessage("Batch generation complete.")
        self.append_log("\nBatch process finished.")

    def on_generation_error(self, error_message: str) -> None:
        self.progress_bar.hide()
        self.eta_label.setText("")
        self.elapsed_label.setText("")
        self.batch_indicator.setText("")
        self.batch_indicator.hide()
        self.reset_ui_state()
        self.statusBar().showMessage("Error during generation.")
        self.append_log(f"\nERROR: {error_message}")
        if "cancelled" not in error_message.lower():
            QMessageBox.critical(self, "Generation Error", error_message)

    def reset_ui_state(self) -> None:
        self.single_gen_btn.setEnabled(True)
        self.batch_gen_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)

    # ---- playback -----------------------------------------------------------
    def play_audio(self) -> None:
        if self.current_audio_path:
            self.media_player.play()
            self.stop_btn.setEnabled(True)

    def stop_audio(self) -> None:
        self.media_player.stop()
        self.stop_btn.setEnabled(False)

    def on_media_status_changed(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.stop_btn.setEnabled(False)

    def on_volume_changed(self, value: int) -> None:
        self.audio_output.setVolume(value / 100.0)

    def save_audio(self) -> None:
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            return
        default_name = os.path.basename(self.current_audio_path)
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio File", default_name, "WAV Files (*.wav)"
        )
        if file_path:
            try:
                shutil.copy(self.current_audio_path, file_path)
                meta_src = os.path.splitext(self.current_audio_path)[0] + Const.METADATA_EXT
                if os.path.exists(meta_src):
                    meta_dst = os.path.splitext(file_path)[0] + Const.METADATA_EXT
                    shutil.copy(meta_src, meta_dst)
                self.statusBar().showMessage(f"Saved to: {file_path}")
            except OSError as e:
                QMessageBox.critical(self, "Save Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MusicLDMStudio()
    window.show()
    sys.exit(app.exec())