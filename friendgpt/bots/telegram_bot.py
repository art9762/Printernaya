#!/usr/bin/env python3
"""
Модуль для создания Telegram ботов, которые действуют как обученные друзья.
Каждый друг получает своего собственного бота.

Требует:
- python-telegram-bot v20+
- Конфигурация в configs/bots.yaml
- FriendEngine и GroupChat из core.inference
"""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from telegram import Update, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

from core.inference import FriendEngine, GroupChat


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FriendBot:
    """
    Обёртка для одного Telegram бота одного друга.
    Генерирует ответы в стиле друга и управляет разговором.
    """

    def __init__(self, friend_name: str, token: str, friend_engine: FriendEngine):
        """
        Инициализация бота друга.

        Args:
            friend_name: Имя друга
            token: Telegram bot token
            friend_engine: FriendEngine для генерации ответов
        """
        self.friend_name = friend_name
        self.token = token
        self.engine = friend_engine
        self.app: Optional[Application] = None
        self._conversations: Dict[int, list] = {}  # chat_id -> история сообщений

    async def initialize(self) -> None:
        """Инициализация приложения и обработчиков."""
        self.app = Application.builder().token(self.token).build()

        # Обработчики команд
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("reset", self._handle_reset))
        self.app.add_handler(CommandHandler("profile", self._handle_profile))
        self.app.add_handler(CommandHandler("style", self._handle_style))

        # Обработчик текстовых сообщений
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Обработчик ошибок
        self.app.add_error_handler(self._handle_error)

        logger.info(f"FriendBot '{self.friend_name}' инициализирован")

    async def start(self) -> None:
        """Запуск бота."""
        if self.app:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            logger.info(f"FriendBot '{self.friend_name}' запущен")

    async def stop(self) -> None:
        """Остановка бота."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info(f"FriendBot '{self.friend_name}' остановлен")

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /start.
        Отправляет приветствие в стиле друга.
        """
        chat_id = update.effective_chat.id
        user_name = update.effective_user.first_name or "друг"

        # Инициализируем историю для этого чата если её нет
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []

        # Генерируем приветствие через FriendEngine используя chat()
        greeting_prompt = f"Привет, {user_name}! Напиши дружеское приветствие."
        greeting = self.engine.chat(greeting_prompt)

        await update.message.reply_text(greeting)
        logger.info(f"[{self.friend_name}] Приветствие отправлено пользователю {user_name}")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик текстовых сообщений.
        Генерирует ответ используя FriendEngine.
        """
        chat_id = update.effective_chat.id
        user_message = update.message.text
        user_name = update.effective_user.first_name or "друг"

        # Инициализируем историю если её нет
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []

        try:
            # Добавляем сообщение в историю
            self._conversations[chat_id].append({
                "role": "user",
                "name": user_name,
                "content": user_message,
                "timestamp": datetime.now()
            })

            # Отправляем действие "печать" для имитации задержки
            await update.message.chat.send_action(ChatAction.TYPING)

            # Генерируем ответ через FriendEngine используя chat()
            response = self.engine.chat(user_message)

            # Имитируем задержку печати (в зависимости от длины ответа)
            typing_delay = min(len(response) / 100, 3.0)  # макс 3 секунды
            await asyncio.sleep(typing_delay)

            # Добавляем ответ в историю
            self._conversations[chat_id].append({
                "role": "assistant",
                "name": self.friend_name,
                "content": response,
                "timestamp": datetime.now()
            })

            # Отправляем ответ
            await update.message.reply_text(response)
            logger.info(
                f"[{self.friend_name}] Ответ отправлен пользователю {user_name}: "
                f"{response[:50]}..."
            )

        except Exception as e:
            logger.error(f"[{self.friend_name}] Ошибка при обработке сообщения: {e}")
            await update.message.reply_text(
                f"Извини, что-то пошло не так. Попробуй ещё раз."
            )

    async def _handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /reset.
        Очищает историю разговора.
        """
        chat_id = update.effective_chat.id
        self._conversations[chat_id] = []

        await update.message.reply_text(
            f"История разговора очищена! Давайте начнём с чистого листа. 😊"
        )
        logger.info(f"[{self.friend_name}] История разговора очищена для чата {chat_id}")

    async def _handle_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /profile.
        Показывает профиль и характер друга.
        """
        try:
            # Получаем системный промпт как описание профиля
            profile = self.engine.system_prompt or f"Я {self.friend_name}"
            profile_text = f"👤 <b>Профиль {self.friend_name}</b>\n\n"

            if isinstance(profile, dict):
                for key, value in profile.items():
                    profile_text += f"<b>{key}:</b> {value}\n"
            else:
                profile_text += str(profile)

            await update.message.reply_text(profile_text, parse_mode="HTML")
            logger.info(f"[{self.friend_name}] Профиль отправлен")

        except Exception as e:
            logger.error(f"[{self.friend_name}] Ошибка при получении профиля: {e}")
            await update.message.reply_text(
                f"Не смогу показать профиль прямо сейчас."
            )

    async def _handle_style(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /style.
        Показывает примеры типичных фраз друга из истории разговора.
        """
        try:
            style_text = f"💬 <b>Стиль общения {self.friend_name}</b>\n\n"

            # Получаем примеры из истории разговора
            assistant_messages = [
                msg.get("content", "")
                for msg in self.engine.conversation_history
                if msg.get("role") == "assistant"
            ]

            if assistant_messages:
                # Показываем последние 3 ответа
                for i, example in enumerate(assistant_messages[-3:], 1):
                    style_text += f"{i}. \"{example}\"\n\n"
            else:
                style_text += f"История разговора пуста. Напиши мне, чтобы увидеть примеры!"

            await update.message.reply_text(style_text, parse_mode="HTML")
            logger.info(f"[{self.friend_name}] Примеры стиля отправлены")

        except Exception as e:
            logger.error(f"[{self.friend_name}] Ошибка при получении примеров стиля: {e}")
            await update.message.reply_text(
                f"Не смогу показать примеры прямо сейчас."
            )

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик ошибок."""
        logger.error(f"[{self.friend_name}] Обновление {update} вызвало ошибку {context.error}")


class GroupBot:
    """
    Бот для группового чата, где все друзья отвечают последовательно.
    Использует GroupChat из core.inference.
    """

    def __init__(self, token: str, friends: list, group_chat: GroupChat):
        """
        Инициализация группового бота.

        Args:
            token: Telegram bot token
            friends: Список имён друзей
            group_chat: GroupChat для генерации ответов
        """
        self.token = token
        self.friends = friends
        self.group_chat = group_chat
        self.app: Optional[Application] = None
        self._conversations: Dict[int, list] = {}  # chat_id -> история сообщений

        # Эмодзи для каждого друга
        self.friend_emojis = {
            "Alice": "👩",
            "Bob": "👨",
            "Charlie": "🧑",
            "Diana": "👩‍🦰",
            "Eve": "👩‍💼",
        }

    async def initialize(self) -> None:
        """Инициализация приложения и обработчиков."""
        self.app = Application.builder().token(self.token).build()

        # Обработчики команд
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("reset", self._handle_reset))

        # Обработчик текстовых сообщений
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Обработчик ошибок
        self.app.add_error_handler(self._handle_error)

        logger.info("GroupBot инициализирован")

    async def start(self) -> None:
        """Запуск группового бота."""
        if self.app:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            logger.info("GroupBot запущен")

    async def stop(self) -> None:
        """Остановка группового бота."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("GroupBot остановлен")

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /start для группового чата."""
        chat_id = update.effective_chat.id

        if chat_id not in self._conversations:
            self._conversations[chat_id] = []

        friends_list = ", ".join(self.friends)
        greeting = (
            f"Привет! 👋 Это групповой чат с {friends_list}. "
            f"Напишите что-нибудь, и все друзья ответят по очереди!"
        )

        await update.message.reply_text(greeting)
        logger.info(f"GroupBot: Приветствие отправлено")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик сообщений для группового чата.
        Все друзья отвечают последовательно.
        """
        chat_id = update.effective_chat.id
        user_message = update.message.text
        user_name = update.effective_user.first_name or "друг"

        if chat_id not in self._conversations:
            self._conversations[chat_id] = []

        try:
            # Добавляем сообщение в историю
            self._conversations[chat_id].append({
                "role": "user",
                "name": user_name,
                "content": user_message,
                "timestamp": datetime.now()
            })

            # Отправляем действие "печать"
            await update.message.chat.send_action(ChatAction.TYPING)

            # Генерируем ответы от всех друзей через GroupChat используя send_message()
            # Метод send_message возвращает list[FriendResponse]
            friend_responses = self.group_chat.send_message(user_message, author=user_name)

            # Отправляем ответ каждого друга с его именем и эмодзи
            for friend_response in friend_responses:
                emoji = self.friend_emojis.get(friend_response.name, "🤖")
                friend_message = f"{emoji} <b>{friend_response.name}:</b>\n{friend_response.text}"

                # Имитируем задержку печати
                typing_delay = min(len(friend_response.text) / 100, 3.0)
                await asyncio.sleep(typing_delay)

                # Отправляем ответ
                await update.message.reply_text(friend_message, parse_mode="HTML")

                # Добавляем ответ в историю
                self._conversations[chat_id].append({
                    "role": "assistant",
                    "name": friend_response.name,
                    "content": friend_response.text,
                    "timestamp": datetime.now()
                })

                logger.info(
                    f"GroupBot [{friend_response.name}]: Ответ отправлен - {friend_response.text[:50]}..."
                )

                # Пауза соответствует задержке из друга
                await asyncio.sleep(friend_response.delay_seconds)

        except Exception as e:
            logger.error(f"GroupBot: Ошибка при обработке сообщения: {e}")
            await update.message.reply_text(
                "Извините, что-то пошло не так в групповом чате. Попробуйте ещё раз."
            )

    async def _handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /reset для группового чата."""
        chat_id = update.effective_chat.id
        self._conversations[chat_id] = []

        await update.message.reply_text(
            "История группового разговора очищена! Начнём заново! 🔄"
        )
        logger.info(f"GroupBot: История разговора очищена для чата {chat_id}")

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик ошибок."""
        logger.error(f"GroupBot: Обновление {update} вызвало ошибку {context.error}")


class BotManager:
    """
    Менеджер для управления несколькими ботами друзей одновременно.
    Читает конфигурацию из configs/bots.yaml.
    """

    def __init__(self, config_path: str = "configs/bots.yaml"):
        """
        Инициализация менеджера ботов.

        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.friend_bots: Dict[str, FriendBot] = {}
        self.group_bot: Optional[GroupBot] = None
        self._running_tasks = []

        logger.info(f"BotManager инициализирован с конфигурацией {config_path}")

    def _load_config(self) -> dict:
        """
        Загрузка конфигурации из YAML файла.

        Returns:
            Словарь с конфигурацией
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Конфигурация не найдена: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                raise ValueError("Конфигурация пуста")

            logger.info(f"Конфигурация загружена из {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise

    async def add_bot(self, friend_name: str, token: str) -> None:
        """
        Добавление нового бота друга.

        Args:
            friend_name: Имя друга
            token: Telegram bot token
        """
        try:
            # Получаем конфигурацию друга из config если она есть
            friend_config = self.config.get("bots", {}).get(friend_name, {})
            model_path = friend_config.get("model_path", f"models/{friend_name}")
            adapter_path = friend_config.get("adapter_path")
            system_prompt = friend_config.get("system_prompt", f"You are {friend_name}")

            # Создаём FriendEngine для этого друга с необходимыми параметрами
            friend_engine = FriendEngine(
                name=friend_name,
                model_path=model_path,
                adapter_path=adapter_path,
                system_prompt=system_prompt
            )

            # Создаём FriendBot
            bot = FriendBot(friend_name, token, friend_engine)
            await bot.initialize()

            self.friend_bots[friend_name] = bot
            logger.info(f"Бот друга '{friend_name}' добавлен")

        except Exception as e:
            logger.error(f"Ошибка при добавлении бота '{friend_name}': {e}")
            raise

    async def start_all(self) -> None:
        """Запуск всех ботов."""
        try:
            # Загружаем боты друзей из конфигурации
            bots_config = self.config.get("bots", {})

            for friend_name, bot_config in bots_config.items():
                if friend_name != "group_bot":  # Пропускаем групповой бот на этом этапе
                    token = bot_config.get("token")
                    if token:
                        await self.add_bot(friend_name, token)

            # Запускаем все боты друзей
            for friend_name, bot in self.friend_bots.items():
                task = asyncio.create_task(bot.start())
                self._running_tasks.append(task)
                logger.info(f"Бот '{friend_name}' запущен")

            # Загружаем и запускаем групповой бот если конфигурирован
            group_config = self.config.get("group_bot", {})
            if group_config:
                token = group_config.get("token")
                friends = group_config.get("friends", list(self.friend_bots.keys()))

                if token and friends:
                    try:
                        group_chat = GroupChat(friends)
                        self.group_bot = GroupBot(token, friends, group_chat)
                        await self.group_bot.initialize()

                        task = asyncio.create_task(self.group_bot.start())
                        self._running_tasks.append(task)
                        logger.info("Групповой бот запущен")

                    except Exception as e:
                        logger.error(f"Ошибка при запуске группового бота: {e}")

            logger.info(
                f"Запущено {len(self.friend_bots)} ботов друзей и "
                f"{'1' if self.group_bot else '0'} групповой бот"
            )

            # Ждём завершения задач
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Ошибка при запуске ботов: {e}")
            raise

    async def stop_all(self) -> None:
        """Остановка всех ботов."""
        try:
            # Останавливаем все боты друзей
            for friend_name, bot in self.friend_bots.items():
                await bot.stop()
                logger.info(f"Бот '{friend_name}' остановлен")

            # Останавливаем групповой бот если он работает
            if self.group_bot:
                await self.group_bot.stop()
                logger.info("Групповой бот остановлен")

            # Отменяем все задачи
            for task in self._running_tasks:
                if not task.done():
                    task.cancel()

            logger.info("Все боты остановлены")

        except Exception as e:
            logger.error(f"Ошибка при остановке ботов: {e}")
            raise


async def main():
    """
    Основная функция для запуска ботов через CLI.

    Использование:
        python -m friendgpt.bots.telegram_bot [--config configs/bots.yaml]
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Запуск Telegram ботов для друзей из friendgpt"
    )
    parser.add_argument(
        "--config",
        default="configs/bots.yaml",
        help="Путь к файлу конфигурации (по умолчанию: configs/bots.yaml)"
    )

    args = parser.parse_args()

    try:
        logger.info("="*60)
        logger.info("Запуск FriendGPT Telegram Bot Manager")
        logger.info("="*60)

        manager = BotManager(config_path=args.config)

        # Запускаем все боты
        await manager.start_all()

    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания")
        await manager.stop_all()

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
