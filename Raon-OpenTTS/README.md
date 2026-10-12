TTS code copied from <https://github.com/krafton-ai/Raon-OpenTTS>.

## Inferrence procedure

```
Input text
   ↓
vocab.txt lookup table
   ↓
token IDs    Reference Audio
    ↓        ↓
    Raon Model
        ↓
Mel-Spectrogram
        ↓
HiFi-GAN Vocoder
        ↓
Audio Waveform (.wav)
```

The lookup table and configuration file are downloaded from <https://huggingface.co/drbaph/Raon-OpenTTS-comfyui> and commited to git:

```sh
# download text-to-token lookup table (delete the last 4 lines afte downloading)
hf download drbaph/Raon-OpenTTS-comfyui Raon-OpenTTS-0.3B/vocab.txt --local-dir .
hf download drbaph/Raon-OpenTTS-comfyui Raon-OpenTTS-0.3B/config.yaml --local-dir .
```

The 53 MB vocoder and ~680 MB model can be downloaded on demand:

```sh
# download vocoder
hf download drbaph/Raon-OpenTTS-comfyui tts-hifigan-libritts-16kHz/generator.ckpt --local-dir pretrained_models
# download model
hf download drbaph/Raon-OpenTTS-comfyui Raon-OpenTTS-0.3B/Raon-OpenTTS-0.3B-bf16.safetensors --local-dir .
```

## Usage

```sh
brew install python@3.12
python3.12 -m venv python/3.12
source python/3.12/bin/activate
git clone https://github.com/jintonic/tts
cd tts/raon
pip install -r requirements.txt
# change text in gen.txt
./infer.py
deactivate
```

