#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/Mfkovaleski/Pixel-depth/pixel-depth"
NYU_ROOT="/home/Mfkovaleski/dados/comparacao-pixel-depth"
DATA_DIR="${DATA_DIR:-$NYU_ROOT/da3_nyu}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/pixel-full-da3-seed42}"

NYU_MAT="${NYU_MAT:-$NYU_ROOT/downloads/nyu_depth_v2_labeled.mat}"
SPLIT_JSON="${SPLIT_JSON:-$NYU_ROOT/split.json}"

cd "$PROJECT"
source .venv/bin/activate
PYTHON="$PROJECT/.venv/bin/python"

if [[ ! -f "$DATA_DIR/train.csv" || ! -f "$DATA_DIR/val.csv" || ! -f "$DATA_DIR/test.csv" ]]; then
  if [[ ! -f "$NYU_MAT" || ! -f "$SPLIT_JSON" ]]; then
    echo "ERRO: faltam os manifests e os arquivos NYUv2 para gerá-los." >&2
    echo "Esperado: $NYU_MAT e $SPLIT_JSON" >&2
    exit 2
  fi
  echo "Preparando NPZs/manifests NYUv2 em $DATA_DIR" >&2
  "$PYTHON" scripts/prepare_nyu_da3.py \
    --labeled-mat "$NYU_MAT" \
    --split-json "$SPLIT_JSON" \
    --output-dir "$DATA_DIR"
fi

for split in train val test; do
  manifest="$DATA_DIR/${split}.csv"
  if [[ ! -f "$manifest" ]]; then
    echo "ERRO: manifesto não encontrado: $manifest" >&2
    exit 2
  fi
done

# Gera pseudo-labels do teacher para treino
PYTHONPATH="/home/Mfkovaleski/Pixel-depth/src" \
"$PYTHON" scripts/generate_teacher_labels.py \
  --manifest "$DATA_DIR/train.csv" \
  --output-manifest "$DATA_DIR/train_full.csv" \
  --output-dir "$DATA_DIR/train_teacher_npz" \
  --model depth-anything/DA3MONO-LARGE \
  --device cuda \
  --process-res 504

# Gera pseudo-labels do teacher para validação
PYTHONPATH="/home/Mfkovaleski/Pixel-depth/src" \
"$PYTHON" scripts/generate_teacher_labels.py \
  --manifest "$DATA_DIR/val.csv" \
  --output-manifest "$DATA_DIR/val_full.csv" \
  --output-dir "$DATA_DIR/val_teacher_npz" \
  --model depth-anything/DA3MONO-LARGE \
  --device cuda \
  --process-res 504

# Treinamento DA3-style: depth + ray + teacher.
# NYUv2 não possui poses multiview; portanto não há supervisão de câmera.
"$PYTHON" -m pixel_depth.cli train \
  --full-da3 \
  --config configs/small.json \
  --train-manifest "$DATA_DIR/train_full.csv" \
  --val-manifest "$DATA_DIR/val_full.csv" \
  --output "$RUN_DIR" \
  --epochs 20 \
  --batch-size 1 \
  --workers 4 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --camera-weight 0.0 \
  --seed 42 \
  --device cuda

# Avaliação
"$PYTHON" -m pixel_depth.cli evaluate \
  --checkpoint "$RUN_DIR/last.pt" \
  --manifest "$DATA_DIR/test.csv" \
  --output "$RUN_DIR/test.json" \
  --device cuda
