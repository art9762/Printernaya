"""
Модуль для парсинга JSON-экспортов Telegram Desktop.

Парсит формат экспорта из Telegram Desktop (result.json) и извлекает
структурированные данные разговоров для последующего использования в обучении.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict, Counter
import statistics
from html.parser import HTMLParser
import glob


@dataclass
class ConversationTurn:
    """Один ход в разговоре."""
    role: str  # "user", "assistant", "system"
    name: str  # Имя участника
    text: str  # Текст сообщения
    timestamp: datetime  # Время сообщения


@dataclass
class Message:
    """Структурированное сообщение из Telegram."""
    from_name: str  # Имя отправителя
    from_id: str  # ID отправителя
    date: datetime  # Время отправления
    text: str  # Чистый текст сообщения
    is_forward: bool  # Переадресованное ли сообщение
    is_media: bool  # Содержит ли медиа
    is_service: bool  # Сервисное ли сообщение
    original_text: str = ""  # Оригинальный текст перед очисткой


@dataclass
class PersonalityProfile:
    """Профиль личности на основе истории чата."""
    frequently_used_words: dict[str, int]  # Слово -> количество
    emoji_patterns: dict[str, int]  # Эмодзи -> количество
    average_message_length: float  # Средняя длина сообщения
    median_message_length: float  # Медианная длина сообщения
    response_time_stats: dict[str, float]  # Статистика времени ответа (mean, median, std)
    slang_expressions: list[tuple[str, int]]  # (выражение, количество) отсортированные
    message_count: int  # Общее количество сообщений
    unique_words: int  # Количество уникальных слов


class TelegramParser:
    """Парсер JSON-экспортов Telegram Desktop."""

    def __init__(self, strip_links: bool = True):
        """
        Инициализация парсера.

        Args:
            strip_links: Удалять ли ссылки из текста при очистке
        """
        self.strip_links = strip_links
        self.messages: list[Message] = []
        self.participants: set[str] = set()

    def parse_file(self, json_path: Path | str) -> list[Message]:
        """
        Парсит JSON-файл экспорта Telegram.

        Args:
            json_path: Путь к файлу result.json

        Returns:
            Список структурированных сообщений
        """
        json_path = Path(json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.messages = []
        self.participants = set()

        # Извлекаем сообщения из массива "messages"
        if isinstance(data, dict) and 'messages' in data:
            messages_raw = data['messages']
        elif isinstance(data, list):
            messages_raw = data
        else:
            raise ValueError("Неожиданный формат JSON экспорта")

        for msg_raw in messages_raw:
            msg = self._parse_message(msg_raw)
            if msg:
                self.messages.append(msg)
                self.participants.add(msg.from_name)

        # Сортируем по времени
        self.messages.sort(key=lambda m: m.date)
        return self.messages

    def parse_auto(self, path: Path | str) -> list[Message]:
        """
        Автоматически определяет формат и парсит экспорт (JSON или HTML).

        Args:
            path: Путь к файлу или папке с экспортом

        Returns:
            Список структурированных сообщений
        """
        format_type = auto_detect_format(str(path))

        if format_type == 'json':
            return self.parse_file(path)
        elif format_type == 'html':
            html_parser = TelegramHTMLParser(strip_links=self.strip_links)
            path_obj = Path(path)

            # Если это директория, парсим все файлы в ней
            if path_obj.is_dir():
                messages = html_parser.feed_directory(path_obj)
            else:
                # Если это файл, парсим его
                messages = html_parser.feed_file(path_obj)

            # Копируем результаты в основной парсер
            self.messages = messages
            self.participants = html_parser.participants

            return self.messages
        else:
            raise ValueError(f"Неизвестный формат: {format_type}")

    def _parse_message(self, msg_raw: dict[str, Any]) -> Optional[Message]:
        """
        Парсит отдельное сообщение из JSON.

        Args:
            msg_raw: Словарь с данными сообщения

        Returns:
            Message или None если это сервисное сообщение
        """
        # Проверяем наличие обязательных полей
        if 'from' not in msg_raw or 'date' not in msg_raw:
            return None

        from_name = msg_raw.get('from', '')
        from_id = str(msg_raw.get('from_id', ''))

        # Парсим дату (в ISO 8601 формате: "2023-01-15T12:34:56")
        try:
            date_str = msg_raw.get('date', '')
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

        # Проверяем на сервисные сообщения
        is_service = msg_raw.get('type', '') == 'service' or 'action' in msg_raw

        # Если это сервисное сообщение и нет текста, пропускаем
        if is_service and 'text' not in msg_raw:
            return None

        # Извлекаем текст (может быть строка или список)
        original_text, is_media = self._extract_text(msg_raw.get('text', ''))

        # Очищаем текст
        text = self._clean_text(original_text)

        # Если после очистки текст пуст, пропускаем
        if not text:
            return None

        # Проверяем на переадресованные сообщения
        is_forward = 'forwarded_from' in msg_raw or msg_raw.get('type', '') == 'message' and 'forwarded' in msg_raw

        return Message(
            from_name=from_name,
            from_id=from_id,
            date=date,
            text=text,
            is_forward=is_forward,
            is_media=is_media,
            is_service=is_service,
            original_text=original_text
        )

    def _extract_text(self, text_field: Any) -> tuple[str, bool]:
        """
        Извлекает текст из поля text (может быть строка или список объектов).

        Args:
            text_field: Поле text из JSON

        Returns:
            Кортеж (текст, содержит_ли_медиа)
        """
        is_media = False

        if isinstance(text_field, str):
            return text_field, is_media

        if isinstance(text_field, list):
            text_parts = []
            for item in text_field:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    # Объект с "type" и "text"
                    if item.get('type') == 'link':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'mention':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'hashtag':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'emoji':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') in ('photo', 'video', 'sticker', 'audio', 'file'):
                        is_media = True
                    else:
                        text_parts.append(item.get('text', ''))

            return ''.join(text_parts), is_media

        return '', is_media

    def _clean_text(self, text: str) -> str:
        """
        Очищает текст сообщения.

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        # Удаляем ссылки если требуется
        if self.strip_links:
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)

        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)

        # Удаляем ведущие/завершающие пробелы
        text = text.strip()

        return text

    def extract_conversation_pairs(
        self,
        participant1: str,
        participant2: str,
        merge_threshold_seconds: int = 120
    ) -> list[tuple[Message, Message]]:
        """
        Извлекает пары (сообщение1 -> ответ2) для двух участников.

        Args:
            participant1: Имя первого участника (обычно пользователь)
            participant2: Имя второго участника (обычно друг)
            merge_threshold_seconds: Порог для объединения последовательных сообщений (в секундах)

        Returns:
            Список кортежей (сообщение от participant1, ответ от participant2)
        """
        # Фильтруем только сообщения этих двух участников
        filtered_msgs = [
            m for m in self.messages
            if m.from_name in (participant1, participant2)
            and not m.is_service
            and not m.is_forward
        ]

        # Объединяем последовательные сообщения от одного человека
        merged_msgs = self._merge_consecutive_messages(filtered_msgs, merge_threshold_seconds)

        pairs = []
        i = 0
        while i < len(merged_msgs) - 1:
            current = merged_msgs[i]
            next_msg = merged_msgs[i + 1]

            # Ищем пару: сообщение от p1, ответ от p2
            if current.from_name == participant1 and next_msg.from_name == participant2:
                pairs.append((current, next_msg))
                i += 2
            else:
                i += 1

        return pairs

    def _merge_consecutive_messages(
        self,
        messages: list[Message],
        threshold_seconds: int
    ) -> list[Message]:
        """
        Объединяет последовательные сообщения от одного человека в один.

        Args:
            messages: Список сообщений
            threshold_seconds: Максимальная разница во времени для объединения

        Returns:
            Новый список сообщений с объединенными подряд идущими
        """
        if not messages:
            return []

        merged = []
        current_group = [messages[0]]

        for i in range(1, len(messages)):
            msg = messages[i]
            prev = messages[i - 1]

            # Проверяем: одинаковый ли отправитель и близко ли по времени
            if (msg.from_name == prev.from_name and
                (msg.date - prev.date).total_seconds() <= threshold_seconds):
                current_group.append(msg)
            else:
                # Завершаем группу
                merged.append(self._merge_message_group(current_group))
                current_group = [msg]

        # Добавляем последнюю группу
        if current_group:
            merged.append(self._merge_message_group(current_group))

        return merged

    def _merge_message_group(self, messages: list[Message]) -> Message:
        """
        Объединяет группу сообщений в одно.

        Args:
            messages: Список сообщений для объединения

        Returns:
            Одно объединенное сообщение
        """
        # Используем первое и последнее сообщение для метаданных
        first = messages[0]
        last = messages[-1]

        # Объединяем текст с переносами строк
        merged_text = '\n'.join(m.text for m in messages)

        return Message(
            from_name=first.from_name,
            from_id=first.from_id,
            date=first.date,
            text=merged_text,
            is_forward=any(m.is_forward for m in messages),
            is_media=any(m.is_media for m in messages),
            is_service=any(m.is_service for m in messages),
            original_text=first.original_text
        )

    def build_training_dataset(
        self,
        participant1: str,
        participant2: str,
        system_prompt: Optional[str] = None
    ) -> list[dict[str, str]]:
        """
        Создает датасет в формате для обучения (system/user/assistant).

        Args:
            participant1: Имя первого участника (роль user)
            participant2: Имя второго участника (роль assistant)
            system_prompt: Пользовательский системный промпт

        Returns:
            Список диалогов в формате OpenAI API
        """
        pairs = self.extract_conversation_pairs(participant1, participant2)

        if not system_prompt:
            system_prompt = f"Ты {participant2}. Отвечай естественно и дружелюбно."

        dataset = []

        for user_msg, assistant_msg in pairs:
            turn = {
                "system": system_prompt,
                "user": user_msg.text,
                "assistant": assistant_msg.text
            }
            dataset.append(turn)

        return dataset

    def extract_personality_profile(
        self,
        participant: str,
        exclude_words: Optional[list[str]] = None
    ) -> PersonalityProfile:
        """
        Извлекает профиль личности участника на основе его сообщений.

        Args:
            participant: Имя участника
            exclude_words: Список слов для исключения из анализа (стопслова)

        Returns:
            PersonalityProfile с анализом привычек сообщений
        """
        # Фильтруем сообщения участника
        user_messages = [
            m for m in self.messages
            if m.from_name == participant
            and not m.is_service
            and not m.is_forward
        ]

        if not user_messages:
            return PersonalityProfile(
                frequently_used_words={},
                emoji_patterns={},
                average_message_length=0,
                median_message_length=0,
                response_time_stats={},
                slang_expressions=[],
                message_count=0,
                unique_words=0
            )

        # Анализируем слова
        exclude_words = exclude_words or self._get_default_stopwords()
        word_counter = Counter()
        message_lengths = []

        for msg in user_messages:
            # Извлекаем слова
            words = re.findall(r'\w+', msg.text.lower())
            message_lengths.append(len(msg.text))

            for word in words:
                if len(word) > 1 and word not in exclude_words:
                    word_counter[word] += 1

        # Анализируем эмодзи
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+",
            flags=re.UNICODE
        )

        emoji_counter = Counter()
        for msg in user_messages:
            emojis = emoji_pattern.findall(msg.text)
            emoji_counter.update(emojis)

        # Вычисляем статистику длины сообщений
        avg_length = statistics.mean(message_lengths) if message_lengths else 0
        median_length = statistics.median(message_lengths) if message_lengths else 0

        # Анализируем время ответов
        response_times = self._calculate_response_times(participant)

        # Извлекаем сленг и выражения (слова длиной > 3 символов, часто используемые)
        slang_expressions = [
            (word, count) for word, count in word_counter.most_common(20)
            if len(word) > 3
        ]

        return PersonalityProfile(
            frequently_used_words=dict(word_counter.most_common(50)),
            emoji_patterns=dict(emoji_counter.most_common(20)),
            average_message_length=avg_length,
            median_message_length=median_length,
            response_time_stats=response_times,
            slang_expressions=slang_expressions,
            message_count=len(user_messages),
            unique_words=len(word_counter)
        )

    def _calculate_response_times(self, participant: str) -> dict[str, float]:
        """
        Вычисляет статистику времени ответов участника.

        Args:
            participant: Имя участника

        Returns:
            Словарь со статистикой (mean, median, std_dev)
        """
        response_times = []

        for i, msg in enumerate(self.messages):
            # Если это сообщение от участника и есть предыдущее от другого
            if msg.from_name == participant and i > 0:
                prev_msg = self.messages[i - 1]
                if prev_msg.from_name != participant:
                    # Время ответа в минутах
                    response_time = (msg.date - prev_msg.date).total_seconds() / 60
                    if response_time > 0:  # Игнорируем одновременные сообщения
                        response_times.append(response_time)

        if not response_times:
            return {
                'mean': 0,
                'median': 0,
                'std_dev': 0
            }

        return {
            'mean': statistics.mean(response_times),
            'median': statistics.median(response_times),
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }

    def _get_default_stopwords(self) -> set[str]:
        """
        Возвращает набор стопслов для русского и английского языков.

        Returns:
            Множество стопслов
        """
        russian_stopwords = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'а', 'то',
            'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы',
            'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'ему',
            'еще', 'при', 'как', 'а', 'это', 'эти', 'нибудь', 'ибо', 'либо',
            'для', 'ж', 'ц', 'щ', 'х'
        }

        english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'am', 'was', 'were', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'what', 'which', 'who', 'why', 'how'
        }

        return russian_stopwords | english_stopwords

    def get_statistics(self) -> dict[str, Any]:
        """
        Возвращает общую статистику по сообщениям.

        Returns:
            Словарь со статистикой
        """
        total_messages = len(self.messages)
        messages_by_participant = defaultdict(int)
        media_count = 0
        forward_count = 0
        service_count = 0

        for msg in self.messages:
            messages_by_participant[msg.from_name] += 1
            if msg.is_media:
                media_count += 1
            if msg.is_forward:
                forward_count += 1
            if msg.is_service:
                service_count += 1

        date_range = (
            f"{self.messages[0].date.date()} - {self.messages[-1].date.date()}"
            if self.messages else "N/A"
        )

        return {
            'total_messages': total_messages,
            'participants': list(self.participants),
            'participant_count': len(self.participants),
            'messages_by_participant': dict(messages_by_participant),
            'media_messages': media_count,
            'forward_messages': forward_count,
            'service_messages': service_count,
            'date_range': date_range
        }


class TelegramHTMLParser(HTMLParser):
    """
    Парсер HTML-экспортов Telegram Desktop.

    Обрабатывает файлы messages.html, messages2.html и т.д., расположенные
    в папке экспорта. Использует встроенный html.parser (без внешних зависимостей).
    """

    def __init__(self, strip_links: bool = True):
        """
        Инициализация HTML-парсера.

        Args:
            strip_links: Удалять ли ссылки из текста при очистке
        """
        super().__init__()
        self.strip_links = strip_links
        self.messages: list[Message] = []
        self.participants: set[str] = set()

        # Состояние парсинга
        self._current_message: dict[str, Any] = {}
        self._current_text_parts: list[str] = []
        self._in_message = False
        self._in_body = False
        self._in_from_name = False
        self._in_date = False
        self._in_text = False
        self._in_forwarded = False
        self._in_media = False
        self._last_from_name: Optional[str] = None
        self._is_service = False

    def feed_directory(self, directory_path: Path | str) -> list[Message]:
        """
        Парсит все HTML-файлы сообщений в папке экспорта.

        Args:
            directory_path: Путь к папке экспорта (содержит messages.html, messages2.html и т.д.)

        Returns:
            Список структурированных сообщений
        """
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            raise ValueError(f"Путь {directory_path} не является директорией")

        # Собираем все файлы messages*.html и сортируем по номеру
        message_files = sorted(
            directory_path.glob('messages*.html'),
            key=lambda p: (
                int(re.search(r'\d+', p.stem).group(0))
                if re.search(r'\d+', p.stem)
                else 0
            )
        )

        if not message_files:
            raise ValueError(f"Не найдены файлы messages*.html в {directory_path}")

        # Парсим каждый файл по очереди
        for html_file in message_files:
            with open(html_file, 'r', encoding='utf-8') as f:
                self.feed(f.read())

        # Сортируем по времени
        self.messages.sort(key=lambda m: m.date)
        return self.messages

    def feed_file(self, file_path: Path | str) -> list[Message]:
        """
        Парсит один HTML-файл сообщений.

        Args:
            file_path: Путь к файлу messages.html

        Returns:
            Список структурированных сообщений
        """
        file_path = Path(file_path)

        if not file_path.is_file():
            raise ValueError(f"Файл {file_path} не существует")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.feed(f.read())

        # Сортируем по времени
        self.messages.sort(key=lambda m: m.date)
        return self.messages

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        """Обработчик открывающего тега."""
        attrs_dict = dict(attrs)

        # Проверяем наличие сообщения
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class'].split()

            # Начало нового сообщения
            if 'message' in classes:
                # Сохраняем предыдущее сообщение, если оно есть
                if self._current_message:
                    self._save_message()

                # Инициализируем новое сообщение
                self._current_message = {
                    'id': attrs_dict.get('id', ''),
                    'is_service': 'service' in classes,
                    'is_joined': 'joined' in classes,
                }
                self._in_message = True
                self._current_text_parts = []
                self._is_service = 'service' in classes

        # В теле сообщения
        elif tag == 'div' and self._in_message:
            if 'class' in attrs_dict:
                classes = attrs_dict['class'].split()

                if 'body' in classes:
                    self._in_body = True
                elif 'from_name' in classes:
                    self._in_from_name = True
                elif 'date' in classes and 'details' in classes:
                    self._in_date = True
                    # Парсим дату из атрибута title
                    if 'title' in attrs_dict:
                        self._parse_date(attrs_dict['title'])
                elif 'text' in classes and not self._in_media:
                    self._in_text = True
                elif 'forwarded' in classes:
                    self._in_forwarded = True
                elif 'media_wrap' in classes:
                    self._in_media = True
                    self._current_message['is_media'] = True

        # Ссылки в тексте
        elif tag == 'a' and self._in_text:
            if 'href' in attrs_dict:
                self._current_text_parts.append(attrs_dict.get('title', attrs_dict['href']))

    def handle_endtag(self, tag: str) -> None:
        """Обработчик закрывающего тега."""
        if tag == 'div':
            if self._in_from_name:
                self._in_from_name = False
            elif self._in_date:
                self._in_date = False
            elif self._in_text:
                self._in_text = False
            elif self._in_body:
                self._in_body = False
            elif self._in_message:
                self._in_message = False
                # Сохраняем последнее сообщение в конце блока
                if self._current_message:
                    self._save_message()

    def handle_data(self, data: str) -> None:
        """Обработчик текстовых данных."""
        # Очищаем от пробелов в начале/конце
        data = data.strip()

        if not data:
            return

        if self._in_from_name:
            # Сохраняем имя отправителя
            self._current_message['from_name'] = data
            self._last_from_name = data
        elif self._in_date:
            # Дата уже парсена из атрибута title
            pass
        elif self._in_text:
            # Добавляем текст
            self._current_text_parts.append(data)

    def _parse_date(self, date_str: str) -> None:
        """
        Парсит дату из атрибута title формата "DD.MM.YYYY HH:MM:SS UTC±HH:MM".

        Args:
            date_str: Строка с датой
        """
        try:
            # Извлекаем дату и время, игнорируя UTC offset
            # Формат: "01.03.2024 15:30:45 UTC+03:00"
            match = re.match(
                r'(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})',
                date_str
            )
            if match:
                day, month, year, hour, minute, second = map(int, match.groups())
                self._current_message['date'] = datetime(year, month, day, hour, minute, second)
        except (ValueError, AttributeError):
            pass

    def _save_message(self) -> None:
        """Сохраняет текущее сообщение в список если оно валидно."""
        if not self._current_message:
            return

        # Пропускаем сервисные сообщения
        if self._current_message.get('is_service', False):
            return

        # Проверяем наличие обязательных полей
        if 'date' not in self._current_message:
            return

        # Используем последнее известное имя (для "joined" сообщений)
        from_name = self._current_message.get('from_name', self._last_from_name or '')

        if not from_name:
            return

        # Собираем текст
        text_content = ''.join(self._current_text_parts).strip()

        if not text_content:
            return

        # Очищаем текст
        cleaned_text = self._clean_text(text_content)

        if not cleaned_text:
            return

        # Создаем сообщение
        msg = Message(
            from_name=from_name,
            from_id='',  # HTML экспорт не содержит ID
            date=self._current_message['date'],
            text=cleaned_text,
            is_forward=self._in_forwarded,
            is_media=self._current_message.get('is_media', False),
            is_service=False,
            original_text=text_content
        )

        self.messages.append(msg)
        self.participants.add(from_name)

    def _clean_text(self, text: str) -> str:
        """
        Очищает текст сообщения.

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        # Удаляем ссылки если требуется
        if self.strip_links:
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'www\.\S+', '', text)

        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)

        # Удаляем ведущие/завершающие пробелы
        text = text.strip()

        return text


def auto_detect_format(path: str) -> str:
    """
    Автоматически определяет формат экспорта (JSON или HTML).

    Args:
        path: Путь к файлу или папке

    Returns:
        'json' или 'html'

    Raises:
        ValueError: Если формат не может быть определен
    """
    path_obj = Path(path)

    # Проверяем расширение файла
    if str(path).endswith('.json'):
        return 'json'
    elif str(path).endswith('.html'):
        return 'html'

    # Проверяем если это директория с HTML экспортом
    if path_obj.is_dir():
        if (path_obj / 'messages.html').exists():
            return 'html'

    raise ValueError(
        f"Не удалось определить формат экспорта для '{path}'. "
        "Ожидается .json файл, .html файл или папка с messages.html"
    )


# Примеры использования и вспомогательные функции

def load_and_analyze(json_path: str, participant1: str, participant2: str) -> None:
    """
    Пример: Загружает экспорт и анализирует диалог двух участников.

    Args:
        json_path: Путь к файлу result.json
        participant1: Имя первого участника
        participant2: Имя второго участника
    """
    parser = TelegramParser(strip_links=True)
    parser.parse_file(json_path)

    print(f"Загружено сообщений: {len(parser.messages)}")
    print(f"Участников: {len(parser.participants)}")
    print(f"Имена участников: {parser.participants}")
    print()

    # Статистика
    stats = parser.get_statistics()
    print(f"Статистика: {stats}")
    print()

    # Пары сообщений
    pairs = parser.extract_conversation_pairs(participant1, participant2)
    print(f"Найдено пар сообщений: {len(pairs)}")
    if pairs:
        print(f"Первая пара:")
        print(f"  {participant1}: {pairs[0][0].text[:100]}")
        print(f"  {participant2}: {pairs[0][1].text[:100]}")
    print()

    # Профиль личности
    profile1 = parser.extract_personality_profile(participant1)
    profile2 = parser.extract_personality_profile(participant2)

    print(f"Профиль {participant1}:")
    print(f"  Сообщений: {profile1.message_count}")
    print(f"  Среднее длина сообщения: {profile1.average_message_length:.1f}")
    print(f"  Топ слова: {list(profile1.frequently_used_words.items())[:5]}")
    print(f"  Топ эмодзи: {list(profile1.emoji_patterns.items())[:5]}")
    print()

    print(f"Профиль {participant2}:")
    print(f"  Сообщений: {profile2.message_count}")
    print(f"  Среднее длина сообщения: {profile2.average_message_length:.1f}")
    print(f"  Топ слова: {list(profile2.frequently_used_words.items())[:5]}")
    print(f"  Топ эмодзи: {list(profile2.emoji_patterns.items())[:5]}")


if __name__ == "__main__":
    # Пример использования:
    # load_and_analyze("path/to/result.json", "User", "Friend")
    pass
