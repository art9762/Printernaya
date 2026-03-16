"""
Модуль для построения обучающих наборов данных из распарсенных Telegram диалогов.

Преобразует список ConversationTurn в формат для fine-tuning LoRA моделей MLX-LM.
Применяет фильтрацию, разделение на train/val и аугментацию данных.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json
from pathlib import Path
from collections import defaultdict

from .telegram_parser import TelegramParser, ConversationTurn, PersonalityProfile


@dataclass
class TrainingExample:
    """Пример для обучения в формате ChatML."""
    messages: list[dict]  # [{"role": "system"|"user"|"assistant", "content": "..."}, ...]

    def to_dict(self) -> dict:
        """Преобразовать в словарь для JSONL."""
        return {"messages": self.messages}


@dataclass
class DatasetStats:
    """Статистика построенного датасета."""
    total_examples: int = 0
    train_examples: int = 0
    valid_examples: int = 0
    avg_exchange_length: float = 0.0
    avg_context_length: float = 0.0
    avg_reply_length: float = 0.0
    min_reply_length: int = 0
    max_reply_length: int = 0
    excluded_short_replies: int = 0
    excluded_time_gaps: int = 0

    def __str__(self) -> str:
        """Форматированный вывод статистики."""
        lines = [
            "=== Статистика датасета ===",
            f"Всего примеров: {self.total_examples}",
            f"  - Обучение (90%): {self.train_examples}",
            f"  - Валидация (10%): {self.valid_examples}",
            f"Средняя длина обмена: {self.avg_exchange_length:.1f} сообщений",
            f"Средняя длина контекста: {self.avg_context_length:.1f} токенов",
            f"Средняя длина ответа: {self.avg_reply_length:.1f} токенов",
            f"Диапазон длины ответа: {self.min_reply_length}-{self.max_reply_length}",
            f"Исключено (короткие ответы): {self.excluded_short_replies}",
            f"Исключено (большой временной разрыв): {self.excluded_time_gaps}",
        ]
        return "\n".join(lines)


class DatasetBuilder:
    """Построитель обучающего датасета из диалогов Telegram."""

    # Минимальное количество слов в ответе для включения в датасет
    MIN_REPLY_WORDS = 3

    # Максимальный временной разрыв между сообщениями (часы)
    MAX_TIME_GAP_HOURS = 24

    # Размер контекста (количество предыдущих сообщений)
    CONTEXT_SIZE = 5

    # Процент примеров для обучения
    TRAIN_SPLIT = 0.9

    def __init__(
        self,
        context_size: int = CONTEXT_SIZE,
        min_reply_words: int = MIN_REPLY_WORDS,
        max_time_gap_hours: int = MAX_TIME_GAP_HOURS,
        train_split: float = TRAIN_SPLIT,
    ):
        """
        Инициализировать построитель датасета.

        Args:
            context_size: Количество предыдущих сообщений для контекста
            min_reply_words: Минимальная длина ответа в словах
            max_time_gap_hours: Максимальный временной разрыв между сообщениями
            train_split: Доля примеров для обучения (0.0-1.0)
        """
        self.context_size = context_size
        self.min_reply_words = min_reply_words
        self.max_time_gap_hours = max_time_gap_hours
        self.train_split = train_split
        self.stats = DatasetStats()

    def build_dataset(
        self,
        turns: list[ConversationTurn],
        friend_name: str,
        personality_profile: Optional[dict] = None,
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        """
        Построить датасет из диалога.

        Args:
            turns: Список ConversationTurn из распарсенного диалога
            friend_name: Имя друга для фильтрации сообщений
            personality_profile: Профиль личности друга для генерации системного промпта
                                 (если None, будет извлечен из turns)

        Returns:
            Кортеж (train_examples, valid_examples)
        """
        # Извлечь профиль личности, если не предоставлен
        if personality_profile is None:
            parser = TelegramParser()
            personality_profile = parser.extract_personality_profile(turns, friend_name)

        # Сегментировать диалог на отдельные беседы по временным разрывам
        conversations = self._segment_by_time_gaps(turns)

        # Генерировать примеры для обучения
        all_examples = []
        for conversation in conversations:
            examples = self._create_examples_from_conversation(
                conversation, friend_name, personality_profile
            )
            all_examples.extend(examples)

        # Разделить на train/val
        train_examples, valid_examples = self._split_train_val(all_examples)

        # Обновить статистику
        self._update_stats(all_examples, train_examples, valid_examples)

        return train_examples, valid_examples

    def _segment_by_time_gaps(self, turns: list[ConversationTurn]) -> list[list[ConversationTurn]]:
        """
        Сегментировать диалог на отдельные беседы на основе временных разрывов.

        Args:
            turns: Список всех сообщений

        Returns:
            Список сегментов (каждый сегмент - это список ConversationTurn)
        """
        if not turns:
            return []

        conversations = []
        current_conversation = [turns[0]]
        max_gap = timedelta(hours=self.max_time_gap_hours)

        for i in range(1, len(turns)):
            time_diff = turns[i].timestamp - turns[i - 1].timestamp

            # Если разрыв больше максимального, начать новую беседу
            if time_diff > max_gap:
                conversations.append(current_conversation)
                current_conversation = [turns[i]]
                self.stats.excluded_time_gaps += 1
            else:
                current_conversation.append(turns[i])

        # Добавить последнюю беседу
        if current_conversation:
            conversations.append(current_conversation)

        return conversations

    def _create_examples_from_conversation(
        self,
        conversation: list[ConversationTurn],
        friend_name: str,
        personality_profile: dict,
    ) -> list[TrainingExample]:
        """
        Создать примеры для обучения из одной беседы.

        Используется подход скользящего окна: берем контекст (последние N сообщений)
        как user input, ответ друга как assistant output.

        Args:
            conversation: Список сообщений одной беседы
            friend_name: Имя друга для фильтрации его сообщений
            personality_profile: Профиль личности друга

        Returns:
            Список примеров для обучения
        """
        examples = []

        # Получить список индексов сообщений друга
        friend_message_indices = [
            i for i, turn in enumerate(conversation)
            if turn.name == friend_name
        ]

        # Для каждого сообщения друга создать пример
        for friend_idx in friend_message_indices:
            friend_turn = conversation[friend_idx]

            # Проверить минимальную длину ответа
            reply_words = len(friend_turn.text.split())
            if reply_words < self.min_reply_words:
                self.stats.excluded_short_replies += 1
                continue

            # Собрать контекст: сообщения перед ответом друга
            context_start = max(0, friend_idx - self.context_size)
            context_turns = conversation[context_start:friend_idx]

            # Создать пример
            example = self._create_example(
                context_turns,
                friend_turn,
                personality_profile,
            )

            if example:
                examples.append(example)

                # Создать варианты с аугментированными системными промптами
                augmented = self._create_augmented_examples(
                    context_turns,
                    friend_turn,
                    personality_profile,
                )
                examples.extend(augmented)

        return examples

    def _create_example(
        self,
        context_turns: list[ConversationTurn],
        friend_turn: ConversationTurn,
        personality_profile: dict,
    ) -> Optional[TrainingExample]:
        """
        Создать один пример обучения в формате ChatML.

        Args:
            context_turns: Контекст сообщений перед ответом друга
            friend_turn: Сообщение друга (assistant output)
            personality_profile: Профиль личности друга

        Returns:
            TrainingExample или None если контекст пуст
        """
        if not context_turns:
            return None

        messages = []

        # Добавить системный промпт
        system_prompt = self._generate_system_prompt(
            friend_turn.name,
            personality_profile,
        )
        messages.append({"role": "system", "content": system_prompt})

        # Добавить контекст диалога
        for turn in context_turns:
            role = "assistant" if turn.name == friend_turn.name else "user"
            messages.append({"role": role, "content": turn.text})

        # Добавить ответ друга
        messages.append({"role": "assistant", "content": friend_turn.text})

        return TrainingExample(messages=messages)

    def _create_augmented_examples(
        self,
        context_turns: list[ConversationTurn],
        friend_turn: ConversationTurn,
        personality_profile: dict,
        num_augmentations: int = 2,
    ) -> list[TrainingExample]:
        """
        Создать варианты примера с аугментированными системными промптами.

        Для улучшения робастности модели используются слегка измененные
        формулировки системного промпта.

        Args:
            context_turns: Контекст сообщений
            friend_turn: Сообщение друга
            personality_profile: Профиль личности
            num_augmentations: Количество вариантов аугментации

        Returns:
            Список примеров с аугментированными системными промптами
        """
        augmented = []

        for variant_idx in range(num_augmentations):
            messages = []

            # Генерировать вариант системного промпта
            system_prompt = self._generate_augmented_system_prompt(
                friend_turn.name,
                personality_profile,
                variant_idx,
            )
            messages.append({"role": "system", "content": system_prompt})

            # Добавить контекст и ответ (как в обычном примере)
            for turn in context_turns:
                role = "assistant" if turn.name == friend_turn.name else "user"
                messages.append({"role": role, "content": turn.text})

            messages.append({"role": "assistant", "content": friend_turn.text})

            augmented.append(TrainingExample(messages=messages))

        return augmented

    def _generate_system_prompt(
        self,
        friend_name: str,
        personality_profile: dict,
    ) -> str:
        """
        Генерировать системный промпт на основе профиля личности.

        Args:
            friend_name: Имя друга
            personality_profile: Профиль с характеристиками личности

        Returns:
            Системный промпт для ChatML формата
        """
        traits = personality_profile.get("traits", [])
        interests = personality_profile.get("interests", [])
        communication_style = personality_profile.get("communication_style", "friendly")

        prompt_parts = [
            f"Ты воплощаешь роль {friend_name}.",
        ]

        if traits:
            traits_str = ", ".join(traits[:5])  # Ограничить количество трейтов
            prompt_parts.append(f"Твои основные черты характера: {traits_str}.")

        if interests:
            interests_str = ", ".join(interests[:5])
            prompt_parts.append(f"Тебе интересны: {interests_str}.")

        prompt_parts.append(
            f"Общайся {communication_style} тоном, как {friend_name} в Telegram диалогах."
        )

        return " ".join(prompt_parts)

    def _generate_augmented_system_prompt(
        self,
        friend_name: str,
        personality_profile: dict,
        variant_idx: int,
    ) -> str:
        """
        Генерировать вариант системного промпта для аугментации.

        Args:
            friend_name: Имя друга
            personality_profile: Профиль личности
            variant_idx: Индекс варианта (определяет способ генерации)

        Returns:
            Аугментированный системный промпт
        """
        traits = personality_profile.get("traits", [])
        interests = personality_profile.get("interests", [])

        if variant_idx == 0:
            # Вариант 1: Более минимальный промпт
            prompt_parts = [f"Ты {friend_name} из Telegram."]
            if traits:
                traits_str = ", ".join(traits[:3])
                prompt_parts.append(f"Ты {traits_str}.")
            return " ".join(prompt_parts)

        else:  # variant_idx >= 1
            # Вариант 2: Более подробный промпт с контекстом
            prompt_parts = [
                f"{friend_name} - это человек, отвечающий в Telegram чатах."
            ]
            if traits and interests:
                prompt_parts.append(
                    f"Он интересуется {', '.join(interests[:3])} и известен своей "
                    f"{traits[0] if traits else 'уникальной'} личностью."
                )
            prompt_parts.append("Отвечай естественно и аутентично.")
            return " ".join(prompt_parts)

    def _split_train_val(
        self,
        examples: list[TrainingExample],
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        """
        Разделить примеры на train и validation наборы.

        Args:
            examples: Список всех примеров

        Returns:
            Кортеж (train_examples, valid_examples)
        """
        split_idx = int(len(examples) * self.train_split)
        train_examples = examples[:split_idx]
        valid_examples = examples[split_idx:]
        return train_examples, valid_examples

    def _update_stats(
        self,
        all_examples: list[TrainingExample],
        train_examples: list[TrainingExample],
        valid_examples: list[TrainingExample],
    ) -> None:
        """
        Обновить статистику датасета.

        Args:
            all_examples: Все примеры
            train_examples: Примеры для обучения
            valid_examples: Примеры для валидации
        """
        self.stats.total_examples = len(all_examples)
        self.stats.train_examples = len(train_examples)
        self.stats.valid_examples = len(valid_examples)

        if all_examples:
            # Посчитать статистику по длинам
            context_lengths = []
            reply_lengths = []

            for example in all_examples:
                # Пропустить системный промпт
                context_msgs = [m for m in example.messages[1:-1]]
                reply_msg = example.messages[-1]["content"]

                # Длина контекста в словах (примерная метрика)
                context_words = sum(
                    len(m["content"].split()) for m in context_msgs
                )
                context_lengths.append(context_words)

                # Длина ответа
                reply_words = len(reply_msg.split())
                reply_lengths.append(reply_words)

            self.stats.avg_context_length = (
                sum(context_lengths) / len(context_lengths)
                if context_lengths else 0.0
            )
            self.stats.avg_reply_length = (
                sum(reply_lengths) / len(reply_lengths)
                if reply_lengths else 0.0
            )
            self.stats.min_reply_length = min(reply_lengths) if reply_lengths else 0
            self.stats.max_reply_length = max(reply_lengths) if reply_lengths else 0
            self.stats.avg_exchange_length = (
                (len(all_examples) + sum(len(ex.messages) for ex in all_examples))
                / len(all_examples)
                if all_examples else 0.0
            )

    def save_dataset(
        self,
        train_examples: list[TrainingExample],
        valid_examples: list[TrainingExample],
        output_dir: Path,
    ) -> None:
        """
        Сохранить датасет в JSONL файлы для MLX-LM.

        Args:
            train_examples: Примеры для обучения
            valid_examples: Примеры для валидации
            output_dir: Директория для сохранения файлов
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Сохранить обучающий датасет
        train_path = output_dir / "train.jsonl"
        with open(train_path, "w", encoding="utf-8") as f:
            for example in train_examples:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

        # Сохранить валидационный датасет
        valid_path = output_dir / "valid.jsonl"
        with open(valid_path, "w", encoding="utf-8") as f:
            for example in valid_examples:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

        print(f"✓ Датасет сохранен в {output_dir}")
        print(f"  - train.jsonl ({len(train_examples)} примеров)")
        print(f"  - valid.jsonl ({len(valid_examples)} примеров)")

    def get_stats(self) -> DatasetStats:
        """Получить статистику построенного датасета."""
        return self.stats
