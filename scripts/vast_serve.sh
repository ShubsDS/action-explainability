#!/usr/bin/env bash
# Runs ON the rented machine (4 GPUs, 80 GB each or more; 450 GB of disk). Not run locally.
#
# Downloads the base model and the two step 2 adapters from Hugging Face, checks the adapters'
# sha256, converts them with tinker-cookbook's `build_lora_adapter` (the same commit upstream
# pins), and starts vLLM with both adapters loaded. The server listens on 127.0.0.1 only: reach
# it through an SSH tunnel (ssh -L 8000:127.0.0.1:8000 ...).
#
#   bash vast_serve.sh setup     # install, download, convert (idempotent)
#   bash vast_serve.sh serve     # start vLLM in the foreground
#
# The server's API key is read from $VLLM_API_KEY or from the file $WORK/.vllm_key (mode 600),
# and handed to vLLM through its environment, never on a command line.
#
# Every step is skipped when its output already exists, so a rerun resumes.
set -euo pipefail

WORK=${WORK:-/workspace/td}
NGPU=${NGPU:-4}
BASE=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
BASE_REV=2dc98e2afe4face0e4ce40972a915c45368bd34a
COOKBOOK=ac227aed68a0c9f75f06900de65c3ff17fbd8ade   # upstream/pyproject.toml pins this commit
VLLM_VERSION=0.18.1                                   # the model card's pin; the cookbook tested >=0.18.0
MAX_MODEL_LEN=${MAX_MODEL_LEN:-72000}                 # more than the 65,536 total (prompt + sampled) a request may ask for

# name on the server | Hugging Face repo | sha256 of adapter_model.safetensors (from the Hub)
ADAPTERS=(
  "anti_s1_step30|barbonara/corin-nemotron-super-anti-s1-rl-step30|26484c32647a784d0c0dbcf2f43e424f795645021ff0ee4e432dd7178c544365"
  "anti_s2_step60|barbonara/corin-nemotron-super-anti-s2-rl-step60|7ab0c8b9c1c3755f4b1107728ffb6884ca617f01f2d57bc1323856cea8ef49dc"
)

setup() {
  mkdir -p "$WORK"/{raw,peft}
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  df -h "$WORK" | tail -1

  # uv brings its own Python, so the machine's image only needs the NVIDIA driver.
  command -v git > /dev/null && command -v curl > /dev/null || { apt-get update -qq; apt-get install -y -qq git curl > /dev/null; }
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
  if [ ! -x "$WORK/venv-vllm/bin/vllm" ]; then
    echo "$(date +%H:%M) installing vLLM $VLLM_VERSION"
    uv venv --quiet --python 3.12 "$WORK/venv-vllm"
    uv pip install --quiet --python "$WORK/venv-vllm/bin/python" "vllm==$VLLM_VERSION" huggingface_hub hf_transfer
  fi
  # A second environment: the cookbook's own dependencies conflict with vLLM's pins.
  if [ ! -x "$WORK/venv-convert/bin/python" ]; then
    echo "$(date +%H:%M) installing the converter"
    uv venv --quiet --python 3.12 "$WORK/venv-convert"
    uv pip install --quiet --python "$WORK/venv-convert/bin/python" torch safetensors \
      "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@$COOKBOOK"
  fi
  HF="$WORK/venv-vllm/bin/hf"
  export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1

  echo "$(date +%H:%M) base model (247 GB)"
  "$HF" download "$BASE" --revision "$BASE_REV" --local-dir "$WORK/base" > "$WORK/download.log" 2>&1
  du -sh "$WORK/base" | cut -f1

  for row in "${ADAPTERS[@]}"; do
    IFS='|' read -r name repo sha <<< "$row"
    echo "$(date +%H:%M) adapter $name"
    "$HF" download "$repo" adapter_config.json adapter_model.safetensors --local-dir "$WORK/raw/$name" >> "$WORK/download.log" 2>&1
    got=$(sha256sum "$WORK/raw/$name/adapter_model.safetensors" | cut -d' ' -f1)
    [ "$got" = "$sha" ] || { echo "sha256 mismatch for $name: $got"; exit 1; }
    if [ ! -f "$WORK/peft/$name/adapter_config.json" ]; then
      rm -rf "$WORK/peft/$name"
      "$WORK/venv-convert/bin/python" -c "
from tinker_cookbook import weights
weights.build_lora_adapter(base_model='$WORK/base', adapter_path='$WORK/raw/$name', output_path='$WORK/peft/$name')"
    fi
    "$WORK/venv-convert/bin/python" -c "
import json; c = json.load(open('$WORK/peft/$name/adapter_config.json'))
print('  converted: r', c['r'], 'alpha', c['lora_alpha'], 'rank_pattern', c.get('rank_pattern'), 'targets', len(c['target_modules']))"
    du -sh "$WORK/peft/$name" | cut -f1
  done
  echo "$(date +%H:%M) setup done"
}

serve() {
  if [ -z "${VLLM_API_KEY:-}" ] && [ -f "$WORK/.vllm_key" ]; then VLLM_API_KEY=$(cat "$WORK/.vllm_key"); fi
  : "${VLLM_API_KEY:?set VLLM_API_KEY to the same value as TD_VLLM_API_KEY in the repo .env file}"
  export VLLM_API_KEY   # vLLM reads its API key from this variable
  mods=()
  for row in "${ADAPTERS[@]}"; do
    IFS='|' read -r name _ _ <<< "$row"
    mods+=("$name=$WORK/peft/$name")
  done
  # --generation-config vllm: do not take sampling defaults from the model's generation_config.json;
  # every request states temperature 1, top_p 1, top_k off, as on Tinker.
  # --max-lora-rank 16: the converter fuses two rank-8 Mamba projections into one rank-16 target.
  exec "$WORK/venv-vllm/bin/vllm" serve "$WORK/base" \
    --served-model-name base \
    --host 127.0.0.1 --port 8000 \
    --trust-remote-code --dtype bfloat16 \
    --tensor-parallel-size "$NGPU" \
    --max-model-len "$MAX_MODEL_LEN" \
    --generation-config vllm \
    --enable-lora --max-loras 2 --max-lora-rank 16 \
    --lora-modules "${mods[@]}"
}

case "${1:-}" in
  setup) setup ;;
  serve) serve ;;
  *) echo "usage: $0 setup|serve"; exit 2 ;;
esac
