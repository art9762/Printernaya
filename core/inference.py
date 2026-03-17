"""
Модуль для запуска инференса (генерации ответов) из обученных моделей друзей используя MLX.

Содержит классы для управления инференсом отдельного друга, групповым чатом и загруженными моделями.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from collections import deque

from mlx_lm import load, generate


@dataclass
class FriendResponse:
    """Ответ одного друга в групповом чате."""
    name: str
    text: str
    delay_seconds: float


class FriendEngine:
    """
    Движок инференса для отдельного друга.

    Загружает обученную модель (fused или base+adapter), генерирует ответы
    и поддерживает контекст разговора.
    """

    # Параметры генерации по умолчанию
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.95
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_REPETITION_PENALTY = 1.1

    # Размер окна истории сообщений (скользящее окно)
    MAX_HISTORY_SIZE = 20

    def __init__(
        self,
        name: str,
        model_path: str,
        adapter_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    ):
        """
        Инициализирует движок инференса для друга.

        Args:
            name: Имя друга
            model_path: Путь к базовой модели
            adapter_path: Путь к адаптеру (если используется), None для fused модели
            system_prompt: Системный промпт с описанием личности друга
            temperature: Температура для генерации
            top_p: Top-p значение для sampling
            max_tokens: Максимум токенов в ответе
            repetition_penalty: Штраф за повторения
        """
        self.name = name
        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.system_prompt = system_prompt or ""

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty

        # История сообщений (скользящее окно)
        self.conversation_history: deque = deque(maxlen=self.MAX_HISTORY_SIZE)

        # Модель и токенизер (ленивая загрузка)
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Загружает модель и токенизер при первом использовании."""
        if self._model_loaded:
            return

        # Определяем адаптер, если нужен
        adapter_file = str(self.adapter_path) if self.adapter_path else None

        # Загружаем модель используя mlx_lm
        self.model, self.tokenizer = load(
            model_name=str(self.model_path),
            adapter_file=adapter_file,
        )

        self._model_loaded = True

    def generate_response(self, messages: list[dict]) -> str:
        """
        Генерирует ответ на основе истории сообщений.

        Args:
            messages: Список сообщений [{"role": "user/assistant", "content": "..."}]

        Returns:
            Сгенерированный ответ от друга
        """
        # Загружаем модель если нужно
        self._load_model()

        # Строим промпт с системным сообщением и историей
        prompt = self._build_prompt(messages)

        # Генерируем ответ
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            verbose=False,
        )

        # Очищаем промпт от ответа, чтобы получить только сгенерированную часть
        generated_text = response.strip()
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def chat(self, user_message: str) -> str:
        """
        Отправляет сообщение и получает ответ, поддерживая историю разговора.

        Args:
            user_message: Сообщение от пользователя

        Returns:
            Ответ от друга
        """
        # Добавляем сообщение пользователя в историю
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Генерируем ответ
        response = self.generate_response(list(self.conversation_history))

        # Добавляем ответ ассистента в историю
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def reset(self) -> None:
        """Очищает историю разговора."""
        self.conversation_history.clear()

    def _build_prompt(self, messages: list[dict]) -> str:
        """
        Строит промпт из системного сообщения и истории.

        Args:
            messages: История сообщений

        Returns:
            Форматированный промпт для модели
        """
        prompt_parts = []

        # Добавляем системный промпт если есть
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}\n")

        # Добавляем историю сообщений
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")

        # Добавляем начало ответа ассистента
        prompt_parts.append("Assistant: ")

        return "".join(prompt_parts)

    def unload_model(self) -> None:
        """Выгружает модель из памяти для освобождения ресурсов."""
        self.model = None
        self.tokenizer = None
        self._model_loaded = False


class GroupChat:
    """
    Управляет групповым чатом с несколькими друзьями.

    Каждый друг может ответить на сообщение в зависимости от релевантности,
    друзья также могут реагировать на ответы друг друга.
    """

    # Минимальный релевантность для автоматического ответа
    MIN_RELEVANCE_SCORE = 0.3

    # Минимальный и максимальный случайный delay между ответами (в секундах)
    MIN_DELAY = 0.5
    MAX_DELAY = 3.0

    def __init__(self, friends: dict[str, FriendEngine]):
        """
        Инициализирует групповой чат.

        Args:
            friends: Словарь {имя: FriendEngine}
        """
        self.friends = friends
        self.shared_history: deque = deque(maxlen=50)

    def send_message(self, text: str, author: str = "User") -> list[FriendResponse]:
        """
        Отправляет сообщение и получает ответы от друзей.

        Args:
            text: Текст сообщения
            author: Автор сообщения (по умолчанию "User")

        Returns:
            Список ответов от друзей (FriendResponse)
        """
        # Добавляем сообщение в общую историю
        self.shared_history.append({
            "role": "user" if author == "User" else "assistant",
            "content": f"{author}: {text}",
            "author": author,
        })

        responses = []

        # Каждый друг решает, нужно ли ему ответить
        for friend_name, friend_engine in self.friends.items():
            # Вычисляем релевантность по ключевым словам
            relevance_score = self._calculate_relevance(text, friend_engine)

            # Решаем, отвечает ли друг
            if relevance_score > self.MIN_RELEVANCE_SCORE or random.random() < 0.2:
                # Генерируем случайный delay для имитации естественного чата
                delay = random.uniform(self.MIN_DELAY, self.MAX_DELAY)

                # Подготавливаем сообщения для друга (последние 10 из общей истории)
                recent_messages = list(self.shared_history)[-10:]
                # Очищаем от имён авторов для обычного формата
                clean_messages = [
                    {
                        "role": msg["role"],
                        "content": msg["content"].replace(f"{msg['author']}: ", "", 1)
                    }
                    for msg in recent_messages
                ]

                # Генерируем ответ
                friend_response_text = friend_engine.generate_response(clean_messages)

                # Добавляем ответ в общую историю
                self.shared_history.append({
                    "role": "assistant",
                    "content": f"{friend_name}: {friend_response_text}",
                    "author": friend_name,
                })

                responses.append(FriendResponse(
                    name=friend_name,
                    text=friend_response_text,
                    delay_seconds=delay,
                ))

        return responses

    def _calculate_relevance(self, text: str, friend_engine: FriendEngine) -> float:
        """
        Вычисляет релевантность сообщения для друга на основе ключевых слов.

        Args:
            text: Текст сообщения
            friend_engine: Движок друга

        Returns:
            Оценка релевантности от 0 до 1
        """
        # Извлекаем ключевые слова из истории друга
        keywords = self._extract_keywords_from_history(friend_engine)

        # Подсчитываем совпадения
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)

        # Нормализуем к диапазону [0, 1]
        if not keywords:
            return 0.5  # Нейтральная релевантность если нет истории

        relevance = min(matches / len(keywords), 1.0)
        return relevance

    def _extract_keywords_from_history(
        self,
        friend_engine: FriendEngine,
        min_length: int = 3,
    ) -> list[str]:
        """
        Извлекает ключевые слова из истории друга.

        Args:
            friend_engine: Движок друга
            min_length: Минимальная длина слова для учёта

        Returns:
            Список ключевых слов
        """
        keywords = []

        for msg in friend_engine.conversation_history:
            content = msg.get("content", "")
            # Простой парсинг: разбиваем на слова, удаляем пунктуацию
            words = content.lower().split()
            for word in words:
                # Удаляем пунктуацию с концов
                word = word.strip(".,!?;:")
                if len(word) >= min_length:
                    keywords.append(word)

        # Возвращаем топ-20 уникальных слов по частоте
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(20)]

    def reset(self) -> None:
        """Очищает историю группового чата и всех друзей."""
        self.shared_history.clear()
        for friend_engine in self.friends.values():
            friend_engine.reset()


class ModelManager:
    """
    Синглтон для управления загруженными моделями.

    Реализует ленивую загрузку моделей и освобождение памяти.
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        """Реализует паттерн синглтон."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Инициализирует менеджер моделей."""
        if self._initialized:
            return

        self._friends: dict[str, FriendEngine] = {}
        self._initialized = True

    def register_friend(
        self,
        name: str,
        model_path: str,
        adapter_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> FriendEngine:
        """
        Регистрирует нового друга с его моделью.

        Args:
            name: Имя друга
            model_path: Путь к модели
            adapter_path: Путь к адаптеру (если есть)
            system_prompt: Системный промпт с описанием личности
            **kwargs: Дополнительные параметры для FriendEngine

        Returns:
            Экземпляр FriendEngine для друга
        """
        if name in self._friends:
            return self._friends[name]

        friend = FriendEngine(
            name=name,
            model_path=model_path,
            adapter_path=adapter_path,
            system_prompt=system_prompt,
            **kwargs
        )

        self._friends[name] = friend
        return friend

    def get_friend(self, name: str) -> Optional[FriendEngine]:
        """
        Получает движок друга по имени.

        Args:
            name: Имя друга

        Returns:
            FriendEngine или None если друг не зарегистрирован
        """
        return self._friends.get(name)

    def list_friends(self) -> list[str]:
        """
        Список всех зарегистрированных друзей.

        Returns:
            Список имён друзей
        """
        return list(self._friends.keys())

    def unload_friend(self, name: str) -> bool:
        """
        Выгружает модель друга из памяти.

        Args:
            name: Имя друга

        Returns:
            True если друг был выгружен, False если его нет
        """
        if name not in self._friends:
            return False

        self._friends[name].unload_model()
        return True

    def unload_all(self) -> None:
        """Выгружает все модели из памяти."""
        for friend in self._friends.values():
            friend.unload_model()

    def reset_all_histories(self) -> None:
        """Очищает историю разговоров всех друзей."""
        for friend in self._friends.values():
            friend.reset()
