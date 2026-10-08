"""Whisper transcription on a Kaggle TPU, via transformers (not CTranslate2 —
CTranslate2 has no XLA backend, which is the whole reason this repo exists).

Usage: python whisper_tpu.py --audio clip.wav [--model openai/whisper-base]
"""
import argparse
import time

import librosa
import torch
import torch_xla.core.xla_model as xm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="openai/whisper-base")
    args = parser.parse_args()

    device = xm.xla_device()
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)

    audio, sr = librosa.load(args.audio, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    start = time.time()
    generated_ids = model.generate(input_features)
    xm.mark_step()
    elapsed = time.time() - start

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"transcribe time: {elapsed:.2f}s")
    print(f"text: {text}")


if __name__ == "__main__":
    main()
