#!/bin/bash
# ============================================================
#  FriendGPT — Полная установка и обучение на NVIDIA GPU
# ============================================================
#
#  Этот скрипт делает ВСЁ:
#    1. Проверяет GPU, CUDA, Python
#    2. Создаёт venv и ставит зависимости
#    3. Логинится в Hugging Face (Llama — gated модель)
#    4. Обучает QLoRA адаптер
#    5. Сливает LoRA с базовой моделью
#    6. Конвертирует в GGUF (для Ollama / llama.cpp / Mac)
#
#  Использование:
#    git clone <repo_url> && cd friendgpt
#    # Положите friends_data/ с мака в эту папку
#    bash setup_and_train.sh "Рядовой Табуретка"
#
#  Или с параметрами:
#    bash setup_and_train.sh "Рядовой Табуретка" --epochs 5 --quant q5_k_m
#
# ============================================================

set -euo pipefail

# ======================== НАСТРОЙКИ ==========================

FRIEND_NAME="${1:?Укажите имя друга: bash setup_and_train.sh \"Имя Друга\"}"
shift  # Остальные аргументы

# Параметры по умолчанию
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LR="2e-5"
LORA_R=16
QUANT="q4_k_m"
RESUME=""
FLASH_ATTN=""
SKIP_INSTALL=""
SKIP_TRAIN=""
SKIP_MERGE=""
SKIP_GGUF=""
HF_TOKEN=""

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum)   GRAD_ACCUM="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --lora-r)       LORA_R="$2"; shift 2 ;;
        --quant)        QUANT="$2"; shift 2 ;;
        --resume)       RESUME="--resume"; shift ;;
        --flash-attn)   FLASH_ATTN="--flash-attn"; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-train)   SKIP_TRAIN=1; shift ;;
        --skip-merge)   SKIP_MERGE=1; shift ;;
        --skip-gguf)    SKIP_GGUF=1; shift ;;
        --hf-token)     HF_TOKEN="$2"; shift 2 ;;
        *) echo "Неизвестный аргумент: $1"; exit 1 ;;
    esac
done

# ======================== ЦВЕТА ==============================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()     { echo -e "${CYAN}${BOLD}[$(date +%H:%M:%S)]${NC} $1"; }
ok()      { echo -e "${GREEN}${BOLD}  ✓${NC} $1"; }
warn()    { echo -e "${YELLOW}${BOLD}  ⚠${NC} $1"; }
err()     { echo -e "${RED}${BOLD}  ✗${NC} $1"; }
header()  { echo -e "\n${BOLD}════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $1${NC}"; echo -e "${BOLD}════════════════════════════════════════════${NC}\n"; }

# ======================== ПУТИ ===============================

BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
FRIENDS_DATA="./friends_data"
ADAPTER_DIR="${FRIENDS_DATA}/models/${FRIEND_NAME}/lora_cuda/final_adapter"
MERGED_DIR="${FRIENDS_DATA}/models/${FRIEND_NAME}/merged"
GGUF_DIR="${MERGED_DIR}"
VENV_DIR="./venv_cuda"

# ======================== ФУНКЦИИ ============================

check_command() {
    if command -v "$1" &> /dev/null; then
        ok "$1 найден: $(command -v "$1")"
        return 0
    else
        err "$1 не найден"
        return 1
    fi
}

elapsed_since() {
    local start=$1
    local now=$(date +%s)
    local diff=$((now - start))
    local mins=$((diff / 60))
    local secs=$((diff % 60))
    echo "${mins}мин ${secs}сек"
}

# ============================================================
#  ЭТАП 0: Проверка системы
# ============================================================

header "Этап 0/5 — Проверка системы"

# Python
if ! check_command python3; then
    err "Установите Python 3.10+: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python версия: ${PYTHON_VERSION}"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    ok "Python >= 3.10"
else
    err "Нужен Python >= 3.10, у вас ${PYTHON_VERSION}"
    exit 1
fi

# NVIDIA GPU
if ! check_command nvidia-smi; then
    err "nvidia-smi не найден. Установите NVIDIA драйвера."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
log "GPU: ${GPU_NAME} (${GPU_VRAM} MB VRAM)"

# Датасет
DATASET_MERGED="${FRIENDS_DATA}/${FRIEND_NAME}/dataset_merged"
DATASET_SINGLE="${FRIENDS_DATA}/${FRIEND_NAME}/dataset"

if [ -d "$DATASET_MERGED" ]; then
    DATASET_DIR="$DATASET_MERGED"
    TRAIN_COUNT=$(wc -l < "${DATASET_DIR}/train.jsonl" 2>/dev/null || echo 0)
    ok "Датасет найден: ${DATASET_DIR} (${TRAIN_COUNT} примеров)"
elif [ -d "$DATASET_SINGLE" ]; then
    DATASET_DIR="$DATASET_SINGLE"
    TRAIN_COUNT=$(wc -l < "${DATASET_DIR}/train.jsonl" 2>/dev/null || echo 0)
    ok "Датасет найден: ${DATASET_DIR} (${TRAIN_COUNT} примеров)"
else
    err "Датасет для '${FRIEND_NAME}' не найден!"
    err "Ожидается: ${DATASET_MERGED}/train.jsonl"
    err ""
    err "Скопируйте папку friends_data/ с мака в корень проекта."
    exit 1
fi

# git
check_command git || true
# cmake (для llama.cpp)
check_command cmake || warn "cmake не найден — GGUF конвертация может не работать"

echo ""
log "Конфигурация:"
log "  Друг:         ${FRIEND_NAME}"
log "  Модель:       ${BASE_MODEL}"
log "  Эпохи:        ${EPOCHS}"
log "  Batch:        ${BATCH_SIZE} × ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM)) эффективный"
log "  LoRA r:       ${LORA_R}"
log "  Квантизация:  ${QUANT}"
log "  Датасет:      ${TRAIN_COUNT} примеров"
echo ""
read -p "Начинаем? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]?$ ]]; then
    log "Отменено."
    exit 0
fi

TOTAL_START=$(date +%s)

# ============================================================
#  ЭТАП 1: Установка зависимостей
# ============================================================

if [ -z "$SKIP_INSTALL" ]; then
    header "Этап 1/5 — Установка зависимостей"
    STEP_START=$(date +%s)

    # Создаём venv
    if [ ! -d "$VENV_DIR" ]; then
        log "Создаю виртуальное окружение..."
        python3 -m venv "$VENV_DIR"
        ok "venv создан: ${VENV_DIR}"
    else
        ok "venv уже существует"
    fi

    # Активируем
    source "${VENV_DIR}/bin/activate"
    ok "venv активирован"

    # Обновляем pip
    log "Обновляю pip..."
    pip install --upgrade pip -q

    # PyTorch с CUDA
    log "Устанавливаю PyTorch с CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q

    # Проверяем CUDA в PyTorch
    if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        ok "PyTorch + CUDA работает"
    else
        err "PyTorch не видит CUDA! Проверьте версию CUDA драйвера."
        err "Попробуйте: pip install torch --index-url https://download.pytorch.org/whl/cu124"
        exit 1
    fi

    # Остальные зависимости
    log "Устанавливаю ML зависимости (transformers, peft, trl, bitsandbytes)..."
    pip install -r requirements_cuda.txt -q
    ok "Все зависимости установлены"

    # Flash Attention (опционально)
    if [ -n "$FLASH_ATTN" ]; then
        log "Устанавливаю Flash Attention 2..."
        pip install flash-attn --no-build-isolation -q || warn "Flash Attention не удалось установить (не критично)"
    fi

    ok "Этап 1 завершён за $(elapsed_since $STEP_START)"
else
    source "${VENV_DIR}/bin/activate" 2>/dev/null || true
    warn "Установка пропущена (--skip-install)"
fi

# ============================================================
#  ЭТАП 2: Hugging Face авторизация
# ============================================================

header "Этап 2/5 — Hugging Face авторизация"

# Llama — gated модель, нужен токен
if [ -n "$HF_TOKEN" ]; then
    log "Используем переданный HF токен"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    ok "Токен установлен"
elif python3 -c "from huggingface_hub import HfFolder; t=HfFolder.get_token(); assert t" 2>/dev/null; then
    ok "Уже авторизованы в Hugging Face"
else
    warn "Llama 3.1 — gated модель. Нужна авторизация."
    echo ""
    echo "  1. Зарегистрируйтесь: https://huggingface.co/join"
    echo "  2. Примите лицензию: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "  3. Создайте токен: https://huggingface.co/settings/tokens"
    echo ""
    log "Запускаю huggingface-cli login..."
    huggingface-cli login
fi

# ============================================================
#  ЭТАП 3: Обучение QLoRA
# ============================================================

if [ -z "$SKIP_TRAIN" ]; then
    header "Этап 3/5 — Обучение QLoRA (${EPOCHS} эпох)"
    STEP_START=$(date +%s)

    HF_TOKEN_FLAG=""
    if [ -n "$HF_TOKEN" ]; then
        HF_TOKEN_FLAG="--hf-token ${HF_TOKEN}"
    fi

    python3 train_cuda.py train \
        --friend "$FRIEND_NAME" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --lora-r "$LORA_R" \
        --model "$BASE_MODEL" \
        $RESUME \
        $FLASH_ATTN \
        $HF_TOKEN_FLAG

    if [ -d "$ADAPTER_DIR" ]; then
        ok "Адаптер сохранён: ${ADAPTER_DIR}"
    else
        err "Адаптер не найден после обучения!"
        exit 1
    fi

    ok "Этап 3 завершён за $(elapsed_since $STEP_START)"
else
    warn "Обучение пропущено (--skip-train)"
fi

# ============================================================
#  ЭТАП 4: Merge LoRA + базовая модель
# ============================================================

if [ -z "$SKIP_MERGE" ]; then
    header "Этап 4/5 — Слияние LoRA с базовой моделью"
    STEP_START=$(date +%s)

    HF_TOKEN_FLAG=""
    if [ -n "$HF_TOKEN" ]; then
        HF_TOKEN_FLAG="--hf-token ${HF_TOKEN}"
    fi

    python3 train_cuda.py merge \
        --adapter-dir "$ADAPTER_DIR" \
        --output-dir "$MERGED_DIR" \
        --model "$BASE_MODEL" \
        $HF_TOKEN_FLAG

    ok "Merged модель: ${MERGED_DIR}"
    ok "Этап 4 завершён за $(elapsed_since $STEP_START)"
else
    warn "Merge пропущен (--skip-merge)"
fi

# ============================================================
#  ЭТАП 5: Конвертация в GGUF
# ============================================================

if [ -z "$SKIP_GGUF" ]; then
    header "Этап 5/5 — Конвертация в GGUF (${QUANT})"
    STEP_START=$(date +%s)

    bash export_gguf.sh "$MERGED_DIR" "$QUANT"

    ok "Этап 5 завершён за $(elapsed_since $STEP_START)"
else
    warn "GGUF конвертация пропущена (--skip-gguf)"
fi

# ============================================================
#  ГОТОВО
# ============================================================

GGUF_FILE="${MERGED_DIR}/$(basename "$MERGED_DIR")-${QUANT}.gguf"

header "Готово!"
echo ""
log "Общее время: $(elapsed_since $TOTAL_START)"
echo ""

if [ -f "$GGUF_FILE" ]; then
    GGUF_SIZE=$(du -h "$GGUF_FILE" | cut -f1)
    echo -e "  ${GREEN}${BOLD}GGUF файл:${NC} ${GGUF_FILE}"
    echo -e "  ${GREEN}${BOLD}Размер:${NC}    ${GGUF_SIZE}"
else
    echo -e "  ${GREEN}${BOLD}Merged модель:${NC} ${MERGED_DIR}"
fi

echo ""
echo -e "  ${BOLD}Как использовать:${NC}"
echo ""
echo "  # На Mac через Ollama:"
echo "  scp -r ${GGUF_FILE} mac:~/friendgpt/"
echo "  ollama create friend-$(echo "$FRIEND_NAME" | tr ' ' '-') -f Modelfile"
echo "  ollama run friend-$(echo "$FRIEND_NAME" | tr ' ' '-')"
echo ""
echo "  # Через llama.cpp:"
echo "  ./llama.cpp/build/bin/llama-cli -m ${GGUF_FILE} -p 'Привет!' -n 256"
echo ""
echo "  # Или просто скопируйте .gguf на мак и загрузите в LM Studio"
echo ""
