#!/usr/bin/env python3.12
import os
import argparse
import datetime
import torch
import soundfile as sf
from omegaconf import OmegaConf
import hydra
from safetensors.torch import load_file

from f5_tts.model import CFM
from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    infer_process,
    load_vocoder,
    nfe_step,
    sway_sampling_coef,
    target_rms,
    speed,
    fix_duration,
)
from f5_tts.model.utils import get_tokenizer
from ema_pytorch import EMA

# -----------------------------
# argparse with defaults
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config",    type=str, default="Raon-OpenTTS-0.3B/config.yaml")
parser.add_argument("--vocab_path", type=str, default="Raon-OpenTTS-0.3B/vocab.txt")
parser.add_argument("--ckpt_dir",  type=str, default="Raon-OpenTTS-0.3B")
parser.add_argument("--ckpt_name", type=str, default="Raon-OpenTTS-0.3B-bf16.safetensors")
parser.add_argument("--ref_audio", type=str, default="ref.wav")
parser.add_argument("--ref_text",  type=str, default="ref.txt")
parser.add_argument("--gen_text",  type=str, default="gen.txt")
args = parser.parse_args()

# -----------------------------
# Read input files
# -----------------------------
for label, path in (("ref_audio", args.ref_audio),
                    ("ref_text",  args.ref_text),
                    ("gen_text",  args.gen_text)):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label} file: {path}")

with open(args.ref_text, "r", encoding="utf-8") as f:
    ref_text = f.read().strip()

with open(args.gen_text, "r", encoding="utf-8") as f:
    gen_text = f.read().strip()

print(f"> ref_audio : {args.ref_audio}")
print(f"> ref_text  : {ref_text}")
print(f"> gen_text  : {gen_text}")

# -----------------------------
# Device
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# -----------------------------
# Load config
# -----------------------------
model_cfg = OmegaConf.load(args.config)

model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch
mel_spec_cfg = model_cfg.model.mel_spec

# Tokenizer (uses --vocab_path)
tokenizer_name = model_cfg.model.tokenizer
char_map, vocab_size = get_tokenizer(args.vocab_path, tokenizer_name)
vocab_char_map = char_map

print(f"> Vocab size: {vocab_size}")

# -----------------------------
# Create model
# -----------------------------
model = CFM(
    transformer=model_cls(
        **model_arc,
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_cfg.n_mel_channels,
    ),
    mel_spec_kwargs=mel_spec_cfg,
    vocab_char_map=vocab_char_map,
).to(device)

ema = EMA(model, include_online_model=False).to(device)

# -----------------------------
# Load checkpoint
# -----------------------------
ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
print(f"> Loading checkpoint: {ckpt_path}")

if ckpt_path.endswith(".safetensors"):
    state = load_file(ckpt_path)
    model.load_state_dict(state)
    print("> Weights loaded directly from safetensors.")
else:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ema_model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'ema_model_state_dict'.")
    ema.load_state_dict(ckpt["ema_model_state_dict"])
    for key, param in ema.ema_model.state_dict().items():
        model.state_dict()[key].copy_(param)
    print("> EMA weights applied to CFM.")

model.eval()
print("> EMA weights applied to CFM.")

# -----------------------------
# Load vocoder
# -----------------------------
vocoder = load_vocoder(
    vocoder_name=model_cfg.model.mel_spec.mel_spec_type,
    is_local=model_cfg.model.vocoder.is_local,
    local_path=model_cfg.model.vocoder.local_path,
)

# -----------------------------
# Inference
# -----------------------------
now = datetime.datetime.now()
out_dir = now.strftime("%y%m%d")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, now.strftime("%H%M%S") + ".wav")

audio_segment, final_sr, _ = infer_process(
    args.ref_audio,
    ref_text.lower(),
    gen_text.lower(),
    model,
    vocoder,
    mel_spec_type=model_cfg.model.mel_spec.mel_spec_type,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
)

sf.write(out_path, audio_segment, final_sr)
print(f"Saved: {out_path}")
