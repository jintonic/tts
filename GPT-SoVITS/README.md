## Prepare conda based python

```sh
brew install --cask miniforge
source /opt/homebrew/Caskroom/miniforge/base/bin/activate
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
```

## Install GPT-SoVITS

```sh
git clone --depth=1 https://github.com/RVC-Boss/GPT-SoVITS
cd GPT-SoVITS
# add the following to requirements.txt
# starlette<0.40
# fastapi<0.110
# resampy
bash install.sh --device MPS --source HF
# directly run the generation page:
python GPT_SoVITS/inference_webui.py
```

## Add proper noun pronunciations

Add proper noun pronunciations in `GPT_SoVITS/text/engdict-hot.rep` using https://en.wikipedia.org/wiki/ARPABET. All words should be capitalized. But when using them, use small letters. For example,

```
GEANTFOUR JH IY AO2 N T F AO1 R
PHYSINO F AH0 Z IY1 N OW
```

After each modification, delete cache file `GPT_SoVITS/text/engdict_cache.pickle` and re-run `python GPT_SoVITS/inference_webui.py`

