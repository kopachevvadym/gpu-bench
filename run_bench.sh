#!/usr/bin/env bash
# Бенчмарк для тестового завдання UKRLAW: NCCL між двома GPU і Wan 2.2 I2V A14B на 1 та 2 GPU.
# Запуск на орендованому вузлі з 2 GPU (≥ 80 ГБ кожна), Ubuntu, CUDA-драйвер уже встановлено.
# Потрібно ≥ 250 ГБ вільного диска: ваги Wan 2.2 I2V A14B важать понад 100 ГБ.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

STEPS="${STEPS:-20}"          # 20 кроків замість 40, щоб вкластися в бюджет; однаково для 1 і 2 GPU
SIZE="${SIZE:-1280*720}"
WORK="${WORK:-$HOME/bench}"
PROMPT="${PROMPT:-A lawyer in a modern Kyiv office explains a document to a client, soft daylight, calm camera movement}"
mkdir -p "$WORK/logs" && cd "$WORK"

echo "== 0. Залізо і драйвер"
nvidia-smi | tee logs/nvidia-smi.txt
nvidia-smi topo -m | tee logs/topo.txt
python3 -c "import platform;print(platform.platform())" | tee logs/os.txt

echo "== 1. Фоновий запис памʼяті, потужності й температури (раз на секунду)"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -l 1 > logs/gpu.csv &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "== 2. nccl-tests: all_reduce між двома GPU"
if [ ! -d nccl-tests ]; then git clone --depth 1 https://github.com/NVIDIA/nccl-tests; fi
(cd nccl-tests && make -j > /dev/null)
./nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 2 | tee logs/nccl_all_reduce_2gpu.log \
  || echo "nccl-tests failed: системна NCCL з образу може не підтримувати Blackwell"
echo "== 2b. all_reduce через NCCL з PyTorch (саме її використовує Wan)"
torchrun --nproc_per_node=2 "$SCRIPT_DIR/ar.py" 2>&1 | grep "NCCL OK" | tee logs/torch_nccl_allreduce.log || true

echo "== 3. Wan 2.2: код і ваги"
if [ ! -d Wan2.2 ]; then
  git clone --depth 1 https://github.com/Wan-Video/Wan2.2
  # speech2video і animate тягнуть decord, якого немає в requirements; для I2V вони не потрібні
  (cd Wan2.2 && git apply "$SCRIPT_DIR/patches/wan_optional_imports.diff")
fi
pip install -q -U wheel ninja packaging
pip install -q flash-attn --no-build-isolation   # без цього pip збирає flash-attn без torch і падає
pip install -q -r Wan2.2/requirements.txt
pip install -q "huggingface_hub[cli]"
if [ ! -d Wan2.2-I2V-A14B ]; then
  hf download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B \
    || huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
fi
IMAGE="$(ls Wan2.2/examples/*i2v* 2>/dev/null | head -1)"
echo "Вхідне зображення: $IMAGE"

run () { # $1 = мітка, далі команда
  local tag="$1"; shift
  local t0=$(date +%s)
  echo "START $tag $(date -Is)" >> "$WORK"/logs/marks.txt
  "$@" 2>&1 | tee "$WORK/logs/$tag.log"
  local t1=$(date +%s)
  echo "END $tag $(date -Is)" >> "$WORK"/logs/marks.txt
  echo "$tag,$((t1 - t0))" >> "$WORK"/logs/wall_seconds.csv
}

echo "== 4. Wan 2.2 I2V на 1 GPU"
cd Wan2.2
run wan_1gpu env CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 generate.py \
  --task i2v-A14B --size "$SIZE" --ckpt_dir ../Wan2.2-I2V-A14B --image "../$IMAGE" \
  --sample_steps "$STEPS" --prompt "$PROMPT" --save_file ../logs/wan_1gpu.mp4

echo "== 5. Wan 2.2 I2V на 2 GPU, FSDP + Ulysses sequence parallel"
run wan_2gpu torchrun --nproc_per_node=2 generate.py \
  --task i2v-A14B --size "$SIZE" --ckpt_dir ../Wan2.2-I2V-A14B --image "../$IMAGE" \
  --sample_steps "$STEPS" --dit_fsdp --t5_fsdp --ulysses_size 2 \
  --prompt "$PROMPT" --save_file ../logs/wan_2gpu.mp4
cd ..

kill $SMI_PID 2>/dev/null || true
python3 "$SCRIPT_DIR/summarize.py" logs | tee logs/results.md
echo "Готово. Скопіюй папку $WORK/logs у репозиторій (без .mp4, якщо вони великі)."
