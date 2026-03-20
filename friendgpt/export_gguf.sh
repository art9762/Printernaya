#!/bin/bash
# ============================================================
# FriendGPT — Экспорт обученной модели в GGUF
# ============================================================
#
# GGUF — универсальный формат для llama.cpp, Ollama, LM Studio.
# Работает на Mac, Windows, Linux без GPU.
#
# Использование:
#   # Шаг 1: Слить LoRA с базовой моделью
#   python train_cuda.py merge \
#       --adapter-dir friends_data/models/Рядовой\ Табуретка/lora_cuda/final_adapter \
#       --output-dir friends_data/models/Рядовой\ Табуретка/merged \
#       --model meta-llama/Meta-Llama-3.1-8B-Instruct
#
#   # Шаг 2: Конвертировать в GGUF
#   bash export_gguf.sh friends_data/models/Рядовой\ Табуретка/merged
#
#   # Или всё сразу:
#   bash export_gguf.sh friends_data/models/Рядовой\ Табуретка/merged --quantize q4_k_m
#
# Доступные квантизации:
#   q4_k_m  — 4.8 GB, хороший баланс (рекомендуется)
#   q5_k_m  — 5.7 GB, чуть лучше качество
#   q8_0    — 8.5 GB, минимальная потеря качества
#   f16     — 16 GB, без квантизации

set -euo pipefail

# --- Цвета ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
err()  { echo -e "${RED}✗${NC} $1"; exit 1; }

# --- Аргументы ---
MODEL_DIR="${1:?Укажите путь к merged модели: bash export_gguf.sh <model_dir>}"
QUANT="${2:-q4_k_m}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"

# Определяем имя выходного файла
MODEL_NAME=$(basename "$MODEL_DIR")
OUTPUT_F16="${MODEL_DIR}/${MODEL_NAME}-f16.gguf"
OUTPUT_QUANT="${MODEL_DIR}/${MODEL_NAME}-${QUANT}.gguf"

log "Модель: $MODEL_DIR"
log "Квантизация: $QUANT"
log "Выход: $OUTPUT_QUANT"
echo ""

# --- Проверяем/клонируем llama.cpp ---
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    log "Клонирую llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi

# --- Устанавливаем Python-зависимости для конвертации ---
log "Проверяю зависимости для конвертации..."
pip install -q gguf sentencepiece protobuf transformers 2>/dev/null || true

# --- Конвертация HF → GGUF (f16) ---
CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    err "Скрипт конвертации не найден: $CONVERT_SCRIPT"
fi

if [ ! -f "$OUTPUT_F16" ]; then
    log "Конвертация в GGUF (f16)..."
    python "$CONVERT_SCRIPT" "$MODEL_DIR" \
        --outfile "$OUTPUT_F16" \
        --outtype f16
    ok "F16 GGUF создан: $OUTPUT_F16"
else
    ok "F16 GGUF уже существует, пропускаю"
fi

# --- Квантизация ---
if [ "$QUANT" = "f16" ]; then
    ok "Квантизация не нужна (f16), готово!"
    ok "Файл: $OUTPUT_F16"
    exit 0
fi

# Проверяем наличие скомпилированного llama-quantize
QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/llama-quantize"
fi

if [ ! -f "$QUANTIZE_BIN" ]; then
    log "Компилирую llama.cpp (нужен cmake)..."
    cd "$LLAMA_CPP_DIR"

    # Определяем GPU support
    if command -v nvcc &> /dev/null; then
        log "CUDA обнаружена, компилирую с GPU поддержкой..."
        cmake -B build -DGGML_CUDA=ON
    else
        log "Компилирую без GPU (CPU only)..."
        cmake -B build
    fi

    cmake --build build --config Release -j$(nproc 2>/dev/null || echo 4)
    cd - > /dev/null

    QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
fi

if [ ! -f "$QUANTIZE_BIN" ]; then
    err "llama-quantize не найден. Скомпилируйте llama.cpp вручную."
fi

log "Квантизация: f16 → ${QUANT}..."
"$QUANTIZE_BIN" "$OUTPUT_F16" "$OUTPUT_QUANT" "$QUANT"

ok "Готово!"
echo ""
echo "=========================================="
echo -e "  Файл: ${GREEN}${OUTPUT_QUANT}${NC}"
echo -e "  Размер: $(du -h "$OUTPUT_QUANT" | cut -f1)"
echo "=========================================="
echo ""
echo "Как использовать:"
echo ""
echo "  # Через Ollama (рекомендуется):"
echo "  ollama create friend-${MODEL_NAME} -f Modelfile"
echo "  ollama run friend-${MODEL_NAME}"
echo ""
echo "  # Через llama.cpp:"
echo "  ${LLAMA_CPP_DIR}/build/bin/llama-cli -m ${OUTPUT_QUANT} -p \"Привет!\" -n 256"
echo ""
echo "  # На Mac через MLX:"
echo "  pip install mlx-lm"
echo "  # Импортируйте GGUF или используйте merged модель напрямую"
echo ""

# --- Создаём Modelfile для Ollama ---
MODELFILE="${MODEL_DIR}/Modelfile"
cat > "$MODELFILE" << HEREDOC
FROM ./${MODEL_NAME}-${QUANT}.gguf

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
HEREDOC

ok "Modelfile для Ollama создан: ${MODELFILE}"
