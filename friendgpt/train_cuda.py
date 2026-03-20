#!/usr/bin/env python3
"""
FriendGPT — CUDA Training Script
=================================
Обучение LoRA/QLoRA на NVIDIA GPU (RTX 4070 Ti, 12GB VRAM).
Использует тот же формат датасетов что и основной FriendGPT.

Использование:
    python train_cuda.py --friend "Рядовой Табуретка" --epochs 3
    python train_cuda.py --friend "Рядовой Табуретка" --epochs 3 --resume
    python train_cuda.py --data-dir ./my_dataset --output-dir ./my_adapter

После обучения запустите export_gguf.sh для конвертации в GGUF формат,
который можно использовать на Mac, Linux, Windows через llama.cpp / Ollama.
"""

import argparse
import json
import os
import sys
import signal
import time
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ============================================================
# Конфигурация по умолчанию (оптимизировано для RTX 4070 Ti 12GB)
# ============================================================

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4  # Эффективный batch_size = 4 * 4 = 16
DEFAULT_LR = 2e-5
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_SAVE_STEPS = 500
DEFAULT_LOGGING_STEPS = 50

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ============================================================
# Утилиты
# ============================================================

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log(msg, color=Colors.CYAN):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.BOLD}[{timestamp}]{Colors.RESET} {color}{msg}{Colors.RESET}")


def log_ok(msg):
    log(f"✓ {msg}", Colors.GREEN)


def log_warn(msg):
    log(f"⚠ {msg}", Colors.YELLOW)


def log_err(msg):
    log(f"✗ {msg}", Colors.RED)


def check_gpu():
    """Проверяет наличие и параметры GPU."""
    if not torch.cuda.is_available():
        log_err("CUDA не доступна! Убедитесь что установлены NVIDIA драйвера и CUDA toolkit.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    log(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    return vram_gb


def load_jsonl_dataset(data_dir: Path) -> tuple[Dataset, Dataset]:
    """Загружает train.jsonl и valid.jsonl из директории FriendGPT."""
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if not train_path.exists():
        log_err(f"Не найден файл: {train_path}")
        sys.exit(1)

    def read_jsonl(path):
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    train_data = read_jsonl(train_path)
    log(f"Train: {len(train_data)} примеров из {train_path}")

    valid_data = []
    if valid_path.exists():
        valid_data = read_jsonl(valid_path)
        log(f"Valid: {len(valid_data)} примеров из {valid_path}")
    else:
        log_warn("valid.jsonl не найден, валидация будет пропущена")

    return Dataset.from_list(train_data), Dataset.from_list(valid_data) if valid_data else None


def format_chat(example, tokenizer):
    """
    Конвертирует пример из ChatML формата FriendGPT в строку
    с использованием chat template токенизатора.

    Формат входных данных (FriendGPT JSONL):
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ============================================================
# Callback для красивого вывода и безопасного прерывания
# ============================================================

class FriendGPTCallback(TrainerCallback):
    """Callback для логирования прогресса и обработки Ctrl+C."""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None
        self.interrupted = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        log(f"Начинаем обучение: {self.total_steps} шагов")
        log(f"Чекпоинт каждые {args.save_steps} шагов")
        log(f"Ctrl+C для безопасной остановки\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            elapsed = time.time() - self.start_time
            step = state.global_step
            pct = (step / self.total_steps) * 100

            # Оценка оставшегося времени
            if step > 0:
                secs_per_step = elapsed / step
                remaining = secs_per_step * (self.total_steps - step)
                eta_h = int(remaining // 3600)
                eta_m = int((remaining % 3600) // 60)
                eta_str = f"{eta_h}ч {eta_m}мин" if eta_h > 0 else f"{eta_m}мин"
            else:
                eta_str = "..."

            loss = logs["loss"]
            lr = logs.get("learning_rate", 0)
            log(f"Шаг {step}/{self.total_steps} ({pct:.1f}%) | loss: {loss:.4f} | lr: {lr:.2e} | ETA: {eta_str}")

    def on_save(self, args, state, control, **kwargs):
        log_ok(f"Чекпоинт сохранён (шаг {state.global_step})")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        log_ok(f"Обучение завершено за {hours}ч {mins}мин")


# ============================================================
# Основной пайплайн обучения
# ============================================================

def train(args):
    # --- Проверка GPU ---
    vram_gb = check_gpu()

    # Автоматическая подстройка batch_size под VRAM
    batch_size = args.batch_size
    if vram_gb < 10:
        batch_size = min(batch_size, 2)
        log_warn(f"Мало VRAM ({vram_gb:.0f}GB), batch_size снижен до {batch_size}")
    elif vram_gb < 8:
        batch_size = 1
        log_warn(f"Критически мало VRAM ({vram_gb:.0f}GB), batch_size = 1")

    # --- Определяем пути ---
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.friend:
        data_dir = Path("friends_data") / args.friend / "dataset_merged"
        if not data_dir.exists():
            # Пробуем личный датасет
            data_dir = Path("friends_data") / args.friend / "dataset"
            if not data_dir.exists():
                log_err(f"Датасет не найден для '{args.friend}'")
                log_err(f"Сначала импортируйте чат: python cli.py import ...")
                sys.exit(1)
    else:
        log_err("Укажите --friend или --data-dir")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.friend:
        output_dir = Path("friends_data") / "models" / args.friend / "lora_cuda"
    else:
        output_dir = Path("output_lora")

    output_dir.mkdir(parents=True, exist_ok=True)

    friend_label = args.friend or data_dir.name
    log(f"\n{'='*60}")
    log(f"  FriendGPT CUDA Training")
    log(f"  Друг: {friend_label}")
    log(f"  Модель: {args.model}")
    log(f"  Данные: {data_dir}")
    log(f"  Выход: {output_dir}")
    log(f"  Эпохи: {args.epochs}, batch: {batch_size}×{args.grad_accum}")
    log(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    log(f"{'='*60}\n")

    # --- Загружаем датасет ---
    log("Загрузка датасета...")
    train_dataset, valid_dataset = load_jsonl_dataset(data_dir)

    # --- Загружаем токенизатор ---
    log("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Форматируем датасет ---
    log("Форматирование данных через chat template...")
    train_dataset = train_dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="Formatting train",
    )
    if valid_dataset:
        valid_dataset = valid_dataset.map(
            lambda x: format_chat(x, tokenizer),
            remove_columns=valid_dataset.column_names,
            desc="Formatting valid",
        )

    # --- QLoRA: 4-bit квантизация ---
    log("Загрузка модели с 4-bit квантизацией (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=args.hf_token,
        attn_implementation="flash_attention_2" if args.flash_attn else "sdpa",
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    log_ok(f"Модель загружена: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B параметров")

    # --- LoRA конфигурация ---
    log("Настройка LoRA адаптера...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log_ok(f"LoRA: {trainable/1e6:.1f}M обучаемых / {total/1e9:.1f}B всего ({100*trainable/total:.2f}%)")

    # --- Вычисляем шаги ---
    steps_per_epoch = len(train_dataset) // (batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(100, total_steps // 10)

    log(f"Шагов на эпоху: {steps_per_epoch}, всего: {total_steps}")

    # --- Проверяем resume ---
    resume_from = None
    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            resume_from = str(checkpoints[-1])
            log_ok(f"Продолжаем с чекпоинта: {resume_from}")
        else:
            log_warn("Чекпоинты не найдены, начинаем сначала")

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,  # Хранить 3 последних чекпоинта
        eval_strategy="steps" if valid_dataset else "no",
        eval_steps=args.save_steps if valid_dataset else None,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        max_steps=-1,
        group_by_length=True,
        dataloader_num_workers=4,
        report_to="none",
        seed=42,
        optim="paged_adamw_8bit",  # Экономит VRAM
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # --- Trainer ---
    callback = FriendGPTCallback(total_steps)

    # Используем response template для Llama 3.1
    # Обучаем ТОЛЬКО на ответах assistant, не на промпте
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        max_seq_length=args.max_seq_len,
        packing=False,
        callbacks=[callback],
    )

    # --- Обработка Ctrl+C ---
    original_sigint = signal.getsignal(signal.SIGINT)
    ctrl_c_count = 0

    def graceful_shutdown(signum, frame):
        nonlocal ctrl_c_count
        ctrl_c_count += 1
        if ctrl_c_count == 1:
            log_warn("\nCtrl+C: сохраняю чекпоинт и останавливаюсь...")
            log_warn("(нажмите ещё раз для немедленной остановки)")
            trainer.control.should_save = True
            trainer.control.should_training_stop = True
        else:
            log_err("\nПринудительная остановка!")
            signal.signal(signal.SIGINT, original_sigint)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, graceful_shutdown)

    # --- Обучение ---
    log("Запуск обучения...")
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        log_warn("Тренировка прервана")

    # --- Сохраняем финальный адаптер ---
    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    log("Сохранение финального адаптера...")
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    # Сохраняем метаданные
    metadata = {
        "friend_name": friend_label,
        "base_model": args.model,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_examples": len(train_dataset),
        "trained_at": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(final_adapter_dir / "friendgpt_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log_ok(f"\nАдаптер сохранён в: {final_adapter_dir}")
    log_ok(f"Для конвертации в GGUF запустите:")
    log(f"  bash export_gguf.sh {args.model} {final_adapter_dir}")

    # Восстанавливаем обработчик
    signal.signal(signal.SIGINT, original_sigint)


# ============================================================
# Merge — слияние LoRA с базовой моделью (full precision)
# ============================================================

def merge(args):
    """Сливает LoRA адаптер с базовой моделью."""
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    if not adapter_dir.exists():
        log_err(f"Адаптер не найден: {adapter_dir}")
        sys.exit(1)

    log(f"Загрузка базовой модели: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        token=args.hf_token,
    )

    log(f"Загрузка LoRA адаптера: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    log("Слияние LoRA с базовой моделью...")
    model = model.merge_and_unload()

    log(f"Сохранение объединённой модели в: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    tokenizer.save_pretrained(str(output_dir))

    log_ok(f"Модель сохранена в: {output_dir}")
    log_ok(f"Для конвертации в GGUF запустите:")
    log(f"  bash export_gguf.sh --merged {output_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FriendGPT CUDA Training — обучение на NVIDIA GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Команда")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Обучить LoRA адаптер")
    p_train.add_argument("--friend", type=str, help="Имя друга (из FriendGPT)")
    p_train.add_argument("--data-dir", type=str, help="Путь к датасету (вместо --friend)")
    p_train.add_argument("--output-dir", type=str, help="Куда сохранить результат")
    p_train.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Базовая модель (default: {DEFAULT_MODEL})")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_train.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p_train.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    p_train.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p_train.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    p_train.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    p_train.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    p_train.add_argument("--resume", action="store_true", help="Продолжить с последнего чекпоинта")
    p_train.add_argument("--flash-attn", action="store_true", help="Использовать Flash Attention 2")
    p_train.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (для gated моделей)")

    # --- merge ---
    p_merge = subparsers.add_parser("merge", help="Слить LoRA с базовой моделью")
    p_merge.add_argument("--adapter-dir", type=str, required=True, help="Путь к LoRA адаптеру")
    p_merge.add_argument("--output-dir", type=str, required=True, help="Куда сохранить merged модель")
    p_merge.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p_merge.add_argument("--hf-token", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        train(args)
    elif args.command == "merge":
        merge(args)


if __name__ == "__main__":
    main()
