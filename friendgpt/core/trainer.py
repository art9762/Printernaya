"""
Модуль для LoRA fine-tuning с использованием mlx-lm библиотеки.

Оптимизирован для Apple Silicon (M4 Pro 24GB) с использованием mlx-lm,
фреймворка машинного обучения для Mac.
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TrainingConfig:
    """Конфигурация для тренировки LoRA адаптера."""

    def __init__(
        self,
        model_name: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        batch_size: int = 1,
        lora_layers: int = 8,
        learning_rate: float = 1e-5,
        iters: Optional[int] = None,
    ):
        """
        Инициализация конфигурации тренировки.

        Args:
            model_name: Название модели из Hugging Face Hub
            batch_size: Размер батча (оптимальный для M4 Pro: 1)
            lora_layers: Количество слоев для LoRA адаптации
            learning_rate: Скорость обучения
            iters: Количество итераций (вычисляется автоматически если None)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.lora_layers = lora_layers
        self.learning_rate = learning_rate
        self.iters = iters

    def calculate_iters(self, dataset_size: int, epochs: int = 3) -> int:
        """
        Вычисляет количество итераций для достижения нужного числа эпох.

        Args:
            dataset_size: Размер датасета (количество примеров)
            epochs: Количество эпох для прохода (по умолчанию 3)

        Returns:
            Количество итераций
        """
        if dataset_size == 0:
            return 100
        iters = max(100, (dataset_size // self.batch_size) * epochs)
        logger.info(f"Вычислено итераций: {iters} для {dataset_size} примеров, {epochs} эпох")
        return iters

    def to_dict(self) -> dict:
        """Преобразует конфигурацию в словарь."""
        return {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'lora_layers': self.lora_layers,
            'learning_rate': self.learning_rate,
            'iters': self.iters,
        }


class Trainer:
    """Класс для управления тренировкой LoRA адаптера через mlx_lm."""

    def __init__(
        self,
        data_dir: Path | str,
        adapter_path: Path | str,
        model_name: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        batch_size: int = 1,
        lora_layers: int = 8,
        learning_rate: float = 1e-5,
    ):
        """
        Инициализация Trainer.

        Args:
            data_dir: Директория с тренировочными данными (JSONL файлы)
            adapter_path: Путь для сохранения обученного адаптера
            model_name: Название базовой модели
            batch_size: Размер батча
            lora_layers: Количество слоев для LoRA
            learning_rate: Скорость обучения
        """
        self.data_dir = Path(data_dir)
        self.adapter_path = Path(adapter_path)
        self.model_name = model_name

        # Создаём директории если их нет
        self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.config = TrainingConfig(
            model_name=model_name,
            batch_size=batch_size,
            lora_layers=lora_layers,
            learning_rate=learning_rate,
        )

        self.training_loss: list[float] = []
        self.is_training = False

    def _count_dataset_samples(self) -> int:
        """Подсчитывает количество примеров в датасете."""
        count = 0
        jsonl_files = list(self.data_dir.glob("*.jsonl"))

        if not jsonl_files:
            logger.warning(f"JSONL файлы не найдены в {self.data_dir}")
            return 0

        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    count += sum(1 for _ in f)
            except Exception as e:
                logger.error(f"Ошибка при чтении {jsonl_file}: {e}")

        logger.info(f"Найдено {count} примеров в датасете")
        return count

    def _parse_loss_from_output(self, output: str) -> Optional[float]:
        """
        Парсит значение loss из вывода тренировки.

        Ищет строки вида:
        - "loss: 2.345"
        - "Loss: 2.345"
        - "step 10 loss 2.345"
        """
        # Попытка найти loss в различных форматах
        patterns = [
            r'loss[:\s]+([0-9.]+)',
            r'Loss[:\s]+([0-9.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def train(self, epochs: int = 3) -> bool:
        """
        Запускает тренировку LoRA адаптера.

        Args:
            epochs: Количество эпох для тренировки

        Returns:
            True если тренировка прошла успешно, False иначе
        """
        if self.is_training:
            logger.warning("Тренировка уже в процессе")
            return False

        logger.info(f"Начинаем тренировку для модели: {self.model_name}")
        logger.info(f"Данные: {self.data_dir}")
        logger.info(f"Адаптер будет сохранён в: {self.adapter_path}")

        # Подсчитываем датасет и вычисляем итерации
        dataset_size = self._count_dataset_samples()
        if dataset_size == 0:
            logger.error("Датасет пуст или не найден")
            return False

        iters = self.config.calculate_iters(dataset_size, epochs)
        self.config.iters = iters

        # Формируем команду для mlx_lm.lora
        cmd = [
            sys.executable,
            '-m', 'mlx_lm.lora',
            '--model', self.model_name,
            '--data', str(self.data_dir),
            '--train',
            '--batch-size', str(self.config.batch_size),
            '--lora-layers', str(self.config.lora_layers),
            '--iters', str(iters),
            '--learning-rate', str(self.config.learning_rate),
            '--adapter-path', str(self.adapter_path),
        ]

        logger.info(f"Команда: {' '.join(cmd)}")

        self.is_training = True
        self.training_loss.clear()

        try:
            # Запускаем тренировку с потоковым выводом
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Построчный вывод
            )

            # Обрабатываем вывод в реальном времени
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[mlx_lm] {line}")

                    # Пытаемся распарсить loss
                    loss = self._parse_loss_from_output(line)
                    if loss is not None:
                        self.training_loss.append(loss)

            returncode = process.wait()

            if returncode != 0:
                logger.error(f"Тренировка завершилась с ошибкой (код {returncode})")
                self.is_training = False
                return False

            logger.info("Тренировка завершена успешно")
            logger.info(
                f"Средний loss: {sum(self.training_loss) / len(self.training_loss):.4f}"
                if self.training_loss else "Loss значения не найдены"
            )

            self.is_training = False
            return True

        except Exception as e:
            logger.error(f"Ошибка при тренировке: {e}")
            self.is_training = False
            return False

    def evaluate(self, val_data_dir: Optional[Path | str] = None) -> Optional[float]:
        """
        Оценивает loss на валидационном наборе данных.

        Args:
            val_data_dir: Директория с валидационными данными.
                         Если None, используется data_dir/val

        Returns:
            Значение loss или None в случае ошибки
        """
        if val_data_dir is None:
            val_data_dir = self.data_dir / 'val'

        val_data_dir = Path(val_data_dir)

        if not val_data_dir.exists():
            logger.warning(f"Валидационная директория не найдена: {val_data_dir}")
            return None

        logger.info(f"Оцениваем на валидационных данных: {val_data_dir}")

        cmd = [
            sys.executable,
            '-m', 'mlx_lm.lora',
            '--model', self.model_name,
            '--data', str(val_data_dir),
            '--adapter-path', str(self.adapter_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"Ошибка при оценке: {result.stderr}")
                return None

            # Парсим loss из вывода
            loss = None
            for line in result.stdout.split('\n'):
                parsed_loss = self._parse_loss_from_output(line)
                if parsed_loss is not None:
                    loss = parsed_loss
                    break

            if loss is not None:
                logger.info(f"Валидационный loss: {loss:.4f}")
            else:
                logger.warning("Не удалось распарсить loss из вывода")

            return loss

        except subprocess.TimeoutExpired:
            logger.error("Оценка превышила timeout (300 сек)")
            return None
        except Exception as e:
            logger.error(f"Ошибка при оценке: {e}")
            return None

    def fuse_adapter(self, output_path: Path | str) -> bool:
        """
        Объединяет LoRA адаптер с базовой моделью для ускорения инференса.

        Args:
            output_path: Путь для сохранения объединённой модели

        Returns:
            True если объединение успешно, False иначе
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Объединяем адаптер с базовой моделью")
        logger.info(f"Адаптер: {self.adapter_path}")
        logger.info(f"Выходной путь: {output_path}")

        cmd = [
            sys.executable,
            '-m', 'mlx_lm.fuse',
            '--model', self.model_name,
            '--adapter-path', str(self.adapter_path),
            '--save-path', str(output_path),
        ]

        logger.info(f"Команда: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"Ошибка при объединении: {result.stderr}")
                return False

            logger.info("Адаптер успешно объединён с базовой моделью")
            logger.info(result.stdout)
            return True

        except subprocess.TimeoutExpired:
            logger.error("Объединение превышило timeout (600 сек)")
            return False
        except Exception as e:
            logger.error(f"Ошибка при объединении: {e}")
            return False

    def get_loss_history(self) -> list[float]:
        """Возвращает историю loss значений во время тренировки."""
        return self.training_loss.copy()


class FriendModel:
    """Класс для управления моделью друга (базовая модель + адаптер + личность)."""

    def __init__(
        self,
        friend_name: str,
        base_model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        models_dir: Path | str = "models",
        configs_dir: Path | str = "configs",
    ):
        """
        Инициализация FriendModel.

        Args:
            friend_name: Имя друга (используется для файлов и папок)
            base_model: Название базовой модели
            models_dir: Директория для моделей друзей
            configs_dir: Директория для конфигураций
        """
        self.friend_name = friend_name
        self.base_model = base_model

        self.models_dir = Path(models_dir)
        self.configs_dir = Path(configs_dir)

        # Пути для моделей друга
        self.friend_models_dir = self.models_dir / friend_name
        self.adapters_dir = self.friend_models_dir / "adapters"
        self.fused_model_dir = self.friend_models_dir / "fused"
        self.config_path = self.configs_dir / f"{friend_name}.yaml"

        # Создаём необходимые директории
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.friend_models_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.fused_model_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # Состояние модели
        self._personality_profile: dict = {}
        self._is_trained = False

        # Загружаем конфигурацию если она существует
        self._load_config()

    def _load_config(self) -> None:
        """Загружает конфигурацию из YAML файла."""
        if not self.config_path.exists():
            logger.info(f"Конфигурация для {self.friend_name} не найдена, создаём новую")
            self._save_config()
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            self._personality_profile = config.get('personality_profile', {})
            self._is_trained = config.get('is_trained', False)
            logger.info(f"Конфигурация загружена для {self.friend_name}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")

    def _save_config(self) -> None:
        """Сохраняет конфигурацию в YAML файл."""
        config = {
            'friend_name': self.friend_name,
            'base_model': self.base_model,
            'is_trained': self._is_trained,
            'personality_profile': self._personality_profile,
            'adapter_path': str(self.get_adapter_path()),
            'fused_model_path': str(self.get_fused_model_path()),
        }

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Конфигурация сохранена для {self.friend_name}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации: {e}")

    def set_personality_profile(self, profile: dict) -> None:
        """
        Устанавливает профиль личности друга.

        Args:
            profile: Словарь с параметрами личности (черты, интересы и т.д.)
        """
        self._personality_profile = profile
        self._save_config()
        logger.info(f"Профиль личности установлен для {self.friend_name}")

    def get_personality_profile(self) -> dict:
        """Возвращает профиль личности друга."""
        return self._personality_profile.copy()

    def get_adapter_path(self) -> Path:
        """Возвращает путь к адаптеру."""
        return self.adapters_dir / "adapter_weights"

    def get_fused_model_path(self) -> Path:
        """Возвращает путь к объединённой модели."""
        return self.fused_model_dir / "model"

    def get_model_path(self) -> Path:
        """
        Возвращает путь к моделе (объединённой если существует, иначе базовой).

        Returns:
            Путь к объединённой модели если она существует, иначе название базовой модели
        """
        fused_path = self.get_fused_model_path()

        # Проверяем наличие объединённой модели
        if fused_path.exists():
            logger.info(f"Используем объединённую модель для {self.friend_name}")
            return fused_path

        if self._is_trained and self.get_adapter_path().exists():
            logger.info(f"Объединённая модель не найдена, используем базовую с адаптером")
            return self.get_adapter_path()

        logger.info(f"Модель не обучена, используем базовую: {self.base_model}")
        return Path(self.base_model)

    def is_trained(self) -> bool:
        """Возвращает True если модель была обучена."""
        return self._is_trained

    def train(
        self,
        data_dir: Path | str,
        epochs: int = 3,
        batch_size: int = 1,
        lora_layers: int = 8,
        learning_rate: float = 1e-5,
    ) -> bool:
        """
        Обучает LoRA адаптер для этого друга.

        Args:
            data_dir: Директория с тренировочными данными
            epochs: Количество эпох
            batch_size: Размер батча
            lora_layers: Количество слоев для LoRA
            learning_rate: Скорость обучения

        Returns:
            True если тренировка успешна, False иначе
        """
        logger.info(f"Начинаем тренировку для друга: {self.friend_name}")

        trainer = Trainer(
            data_dir=data_dir,
            adapter_path=self.get_adapter_path(),
            model_name=self.base_model,
            batch_size=batch_size,
            lora_layers=lora_layers,
            learning_rate=learning_rate,
        )

        success = trainer.train(epochs=epochs)

        if success:
            self._is_trained = True
            self._save_config()
            logger.info(f"Тренировка для {self.friend_name} завершена успешно")
        else:
            logger.error(f"Тренировка для {self.friend_name} не удалась")

        return success

    def fuse(self) -> bool:
        """
        Объединяет LoRA адаптер с базовой моделью.

        Returns:
            True если объединение успешно, False иначе
        """
        if not self._is_trained:
            logger.error(f"Модель для {self.friend_name} не обучена, объединение невозможно")
            return False

        logger.info(f"Объединяем адаптер для {self.friend_name}")

        trainer = Trainer(
            data_dir=Path.home() / "temp",  # Временная директория (не используется для fuse)
            adapter_path=self.get_adapter_path(),
            model_name=self.base_model,
        )

        success = trainer.fuse_adapter(output_path=self.get_fused_model_path())

        if success:
            logger.info(f"Адаптер объединён успешно для {self.friend_name}")
        else:
            logger.error(f"Ошибка при объединении адаптера для {self.friend_name}")

        return success

    def __repr__(self) -> str:
        """Строковое представление FriendModel."""
        status = "обучена" if self._is_trained else "не обучена"
        return (
            f"FriendModel(name={self.friend_name}, "
            f"status={status}, "
            f"model={self.base_model})"
        )
