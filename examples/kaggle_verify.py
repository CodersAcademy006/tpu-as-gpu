"""Runs every example against the current notebook's hardware and writes one
results file. Meant to be the first (and only) cell run on a fresh Kaggle
TPU VM notebook — turns "go run the smoke tests" into one command instead
of four, and leaves a dated record in results/ instead of scrollback.

Usage (on Kaggle, repo checked out at /kaggle/working/tpu-as-gpu):
    !pip install -e . -q
    !python examples/kaggle_verify.py
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
RESULTS = REPO_ROOT / "results"


def _make_sample_wav(path: Path) -> None:
    """Synthesizes a 2s sine wave so the Whisper smoke test needs no
    downloaded audio file — just proves the pipeline runs end to end."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t).astype("float32")
    sf.write(path, tone, sr)


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return proc.returncode, (proc.stdout + proc.stderr)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    runs = [
        ("torch_xla smoke test", [sys.executable, str(EXAMPLES / "torch_xla/smoke_test.py")]),
        ("jax smoke test", [sys.executable, str(EXAMPLES / "jax/smoke_test.py")]),
        ("tensorflow smoke test", [sys.executable, str(EXAMPLES / "tensorflow/smoke_test.py")]),
    ]

    sample_wav = RESULTS / "_sample.wav"
    try:
        _make_sample_wav(sample_wav)
        runs.append((
            "whisper transcription on TPU",
            [sys.executable, str(EXAMPLES / "torch_xla/whisper_tpu.py"), "--audio", str(sample_wav)],
        ))
    except ImportError as e:
        print(f"skipping whisper run, missing dep: {e}")

    date = datetime.date.today().isoformat()
    out_path = RESULTS / f"kaggle_run_{date}.md"
    lines = [f"# Kaggle TPU run — {date}\n"]

    for label, cmd in runs:
        print(f"--- {label} ---")
        code, output = _run(cmd)
        print(output)
        status = "PASS" if code == 0 else f"FAIL (exit {code})"
        lines.append(f"## {label} — {status}\n\n```\n{output.strip()}\n```\n")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
