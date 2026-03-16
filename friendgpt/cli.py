#!/usr/bin/env python3
"""
FriendGPT CLI - интерфейс командной строки для чата с обученными моделями друзей.
Позволяет импортировать чаты Telegram, обучать модели, сливать адаптеры и общаться.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Импорт лёгких модулей ядра (парсер и датасет)
try:
    from core.telegram_parser import TelegramParser
    from core.dataset_builder import DatasetBuilder
except ImportError:
    print("Ошибка: не удалось импортировать модули core/. Убедитесь, что проект установлен правильно.")
    sys.exit(1)

# Тяжёлые модули (mlx_lm) загружаются лениво — только при train/fuse/chat
FriendModel = None
FriendEngine = None


def _ensure_ml_imports():
    """Ленивый импорт модулей, требующих mlx_lm."""
    global FriendModel, FriendEngine
    if FriendModel is None:
        try:
            from core.trainer import FriendModel as _FM
            from core.inference import FriendEngine as _FE
            FriendModel = _FM
            FriendEngine = _FE
        except ImportError as e:
            print(f"Ошибка: для этой команды нужен mlx_lm. Установите: pip install mlx mlx-lm")
            print(f"Детали: {e}")
            sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False


@dataclass
class ColorScheme:
    """Схема окраски для терминала."""

    user: str
    friend: str
    system: str
    error: str
    success: str
    reset: str

    @staticmethod
    def create_default() -> 'ColorScheme':
        """Создаёт цветовую схему по умолчанию."""
        if HAS_COLORAMA:
            return ColorScheme(
                user=Fore.CYAN,
                friend=Fore.GREEN,
                system=Fore.YELLOW,
                error=Fore.RED,
                success=Fore.GREEN,
                reset=Style.RESET_ALL
            )
        else:
            # ANSI коды
            return ColorScheme(
                user='\033[36m',
                friend='\033[32m',
                system='\033[33m',
                error='\033[31m',
                success='\033[32m',
                reset='\033[0m'
            )


class FriendManager:
    """Менеджер для управления данными друзей."""

    def __init__(self, data_dir: str = "./friends_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        self.colors = ColorScheme.create_default()
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Загружает метаданные о друзьях."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Сохраняет метаданные о друзьях."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def get_friend_dir(self, friend_name: str) -> Path:
        """Получает директорию друга."""
        return self.data_dir / friend_name

    def friend_exists(self, friend_name: str) -> bool:
        """Проверяет существование друга."""
        return friend_name in self.metadata

    def add_friend(self, friend_name: str, status: str = "imported") -> None:
        """Добавляет нового друга (или обновляет существующего)."""
        if friend_name not in self.metadata:
            self.metadata[friend_name] = {
                "status": status,
                "sources": [],  # Список источников данных
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        else:
            # Друг уже есть — сбрасываем статус на imported (датасет обновился)
            self.metadata[friend_name]["status"] = status
            self.metadata[friend_name]["updated_at"] = datetime.now().isoformat()
        self._save_metadata()

    def add_source(self, friend_name: str, source_type: str, source_path: str,
                   dataset_dir: str, num_examples: int = 0) -> None:
        """
        Добавляет источник данных для друга.

        Args:
            friend_name: Имя друга
            source_type: Тип источника ("personal" или "group")
            source_path: Путь к исходному экспорту
            dataset_dir: Имя поддиректории с датасетом (напр. "dataset_personal_0")
            num_examples: Количество примеров в источнике
        """
        if friend_name not in self.metadata:
            self.add_friend(friend_name)

        # Инициализируем sources если нет (обратная совместимость)
        if "sources" not in self.metadata[friend_name]:
            self.metadata[friend_name]["sources"] = []

        self.metadata[friend_name]["sources"].append({
            "type": source_type,
            "source_path": source_path,
            "dataset_dir": dataset_dir,
            "num_examples": num_examples,
            "added_at": datetime.now().isoformat()
        })
        self.metadata[friend_name]["updated_at"] = datetime.now().isoformat()
        self._save_metadata()

    def get_sources(self, friend_name: str) -> list:
        """Возвращает список источников данных друга."""
        if friend_name in self.metadata:
            return self.metadata[friend_name].get("sources", [])
        return []

    def get_dataset_dirs(self, friend_name: str) -> List[Path]:
        """Возвращает пути ко всем директориям с датасетами друга."""
        friend_dir = self.get_friend_dir(friend_name)
        dirs = []
        for source in self.get_sources(friend_name):
            d = friend_dir / source["dataset_dir"]
            if d.exists():
                dirs.append(d)
        return dirs

    def update_friend_status(self, friend_name: str, status: str) -> None:
        """Обновляет статус друга."""
        if friend_name in self.metadata:
            self.metadata[friend_name]["status"] = status
            self.metadata[friend_name]["updated_at"] = datetime.now().isoformat()
            self._save_metadata()

    def get_friend_status(self, friend_name: str) -> Optional[str]:
        """Получает статус друга."""
        if friend_name in self.metadata:
            return self.metadata[friend_name]["status"]
        return None

    def list_friends(self) -> Dict[str, Any]:
        """Возвращает словарь всех друзей и их информации."""
        result = {}
        for name, info in self.metadata.items():
            sources = info.get("sources", [])
            source_types = [s["type"] for s in sources]
            result[name] = {
                "status": info["status"],
                "sources": source_types,
                "num_sources": len(sources),
            }
        return result


class ChatInterface:
    """Интерфейс для интерактивного чата."""

    def __init__(self, friend_name: str, inference_engine: FriendEngine, colors: ColorScheme):
        self.friend_name = friend_name
        self.inference = inference_engine
        self.colors = colors
        self.history: List[Dict[str, str]] = []

    def _print_colored(self, name: str, message: str, color: str) -> None:
        """Печатает сообщение с цветом."""
        print(f"{color}{name}:{self.colors.reset} {message}")

    def _simulate_typing(self, duration: float = 1.0) -> None:
        """Имитирует печать на клавиатуре."""
        dots = [".", "..", "..."]
        start_time = time.time()
        idx = 0

        while time.time() - start_time < duration:
            print(f"\r{self.friend_name} {self.colors.system}печатает{dots[idx % 3]}{self.colors.reset}", end="", flush=True)
            time.sleep(0.2)
            idx += 1

        print("\r" + " " * 50 + "\r", end="", flush=True)

    def _handle_command(self, command: str) -> bool:
        """
        Обрабатывает команды в чате.
        Возвращает False если нужно выйти, True в остальных случаях.
        """
        if command == "/quit" or command == "/exit":
            self._print_colored(self.colors.system, "До встречи!", self.colors.system)
            return False
        elif command == "/reset":
            self.history.clear()
            self._print_colored(self.colors.system, "История очищена", self.colors.system)
            return True
        elif command == "/save":
            self._save_conversation()
            return True
        elif command == "/help":
            self._print_help()
            return True
        else:
            self._print_colored(self.colors.error, f"Неизвестная команда: {command}", self.colors.error)
            return True

    def _save_conversation(self) -> None:
        """Сохраняет историю разговора в файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{self.friend_name}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        self._print_colored(self.colors.system, f"Разговор сохранён в {filename}", self.colors.system)

    def _print_help(self) -> None:
        """Выводит справку по командам."""
        help_text = """
{system}Доступные команды:{reset}
  /reset  - Очистить историю чата
  /save   - Сохранить разговор в файл
  /quit   - Выйти из чата
  /help   - Показать эту справку
        """.format(system=self.colors.system, reset=self.colors.reset)
        print(help_text)

    def run(self) -> None:
        """Запускает интерактивный чат."""
        self._print_colored(
            self.colors.system,
            f"Чат с {self.friend_name}. Напишите /help для справки.",
            self.colors.system
        )

        while True:
            try:
                user_input = input(f"{self.colors.user}Вы:{self.colors.reset} ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Добавляем в историю
                self.history.append({"role": "user", "content": user_input})

                # Имитируем печать
                self._simulate_typing(duration=0.5)

                # Получаем ответ
                response = self.inference.chat(user_input)

                # Выводим ответ
                self._print_colored(
                    self.friend_name,
                    response,
                    self.colors.friend
                )

                # Добавляем в историю
                self.history.append({"role": "friend", "content": response})

            except KeyboardInterrupt:
                print()
                self._print_colored(self.colors.system, "Чат прерван", self.colors.system)
                break
            except EOFError:
                break


class GroupChatInterface:
    """Интерфейс для группового чата с несколькими друзьями."""

    def __init__(self, friend_names: List[str], inference_engines: Dict[str, FriendEngine], colors: ColorScheme):
        self.friend_names = friend_names
        self.inference = inference_engines
        self.colors = colors
        self.history: List[Dict[str, str]] = []
        # Генерируем цвета для каждого друга
        self.friend_colors = self._generate_friend_colors()

    def _generate_friend_colors(self) -> Dict[str, str]:
        """Генерирует различные цвета для каждого друга."""
        if HAS_COLORAMA:
            color_list = [Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.YELLOW]
        else:
            color_list = ['\033[32m', '\033[34m', '\033[35m', '\033[36m', '\033[33m']

        return {name: color_list[i % len(color_list)] for i, name in enumerate(self.friend_names)}

    def _print_colored(self, name: str, message: str, color: str) -> None:
        """Печатает сообщение с цветом."""
        print(f"{color}{name}:{self.colors.reset} {message}")

    def _simulate_typing(self, duration: float = 0.5) -> None:
        """Имитирует печать на клавиатуре."""
        time.sleep(duration)

    def _handle_command(self, command: str) -> bool:
        """Обрабатывает команды в групповом чате."""
        if command in ("/quit", "/exit"):
            self._print_colored(self.colors.system, "До встречи!", self.colors.system)
            return False
        elif command == "/reset":
            self.history.clear()
            self._print_colored(self.colors.system, "История очищена", self.colors.system)
            return True
        elif command == "/save":
            self._save_conversation()
            return True
        elif command == "/help":
            self._print_help()
            return True
        else:
            self._print_colored(self.colors.error, f"Неизвестная команда: {command}", self.colors.error)
            return True

    def _save_conversation(self) -> None:
        """Сохраняет историю разговора в файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        friend_list = "_".join(self.friend_names)
        filename = f"group_chat_{friend_list}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        self._print_colored(self.colors.system, f"Разговор сохранён в {filename}", self.colors.system)

    def _print_help(self) -> None:
        """Выводит справку по командам."""
        help_text = f"""
{self.colors.system}Доступные команды:{self.colors.reset}
  /reset  - Очистить историю чата
  /save   - Сохранить разговор в файл
  /quit   - Выйти из чата
  /help   - Показать эту справку
        """
        print(help_text)

    def run(self) -> None:
        """Запускает интерактивный групповой чат."""
        friends_str = ", ".join(self.friend_names)
        self._print_colored(
            self.colors.system,
            f"Групповой чат с {friends_str}. Напишите /help для справки.",
            self.colors.system
        )

        while True:
            try:
                user_input = input(f"{self.colors.user}Вы:{self.colors.reset} ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Добавляем в историю
                self.history.append({"role": "user", "content": user_input})

                # Получаем ответы от всех друзей
                for friend_name in self.friend_names:
                    self._simulate_typing(duration=0.3)

                    response = self.inference[friend_name].chat(user_input)

                    self._print_colored(
                        friend_name,
                        response,
                        self.friend_colors[friend_name]
                    )

                    # Добавляем в историю
                    self.history.append({"role": friend_name, "content": response})

                print()  # Пустая строка для читаемости

            except KeyboardInterrupt:
                print()
                self._print_colored(self.colors.system, "Чат прерван", self.colors.system)
                break
            except EOFError:
                break


class FriendGPTCLI:
    """Основной класс CLI приложения."""

    def __init__(self):
        self.colors = ColorScheme.create_default()
        self.manager = FriendManager()

    def cmd_import(self, args: argparse.Namespace) -> None:
        """Команда: импортировать личный чат Telegram (автоопределение формата)."""
        print(f"{self.colors.system}Импорт личного чата из {args.path}...{self.colors.reset}")

        if not Path(args.path).exists():
            print(f"{self.colors.error}Ошибка: файл/папка не найдены{self.colors.reset}")
            return

        try:
            # Парсим файл Telegram с автоматическим определением формата
            parser = TelegramParser()
            messages = parser.parse_auto(args.path)
            print(f"{self.colors.success}✓ Спарсено {len(messages)} сообщений{self.colors.reset}")

            # Показываем участников
            stats = parser.get_statistics()
            msg_by = stats.get('messages_by_participant', {})
            print(f"\n{self.colors.system}Участники чата:{self.colors.reset}")
            for name, count in sorted(msg_by.items(), key=lambda x: -x[1]):
                marker = " ← друг" if name == args.name else ""
                print(f"  {name}: {count} сообщений{marker}")

            if args.name not in msg_by:
                print(f"\n{self.colors.error}Имя «{args.name}» не найдено в чате!{self.colors.reset}")
                print(f"  Используйте одно из имён выше с флагом --name")
                return

            # Конвертируем Message → ConversationTurn
            turns = parser.messages_to_turns()

            # Определяем уникальный номер источника
            existing_sources = self.manager.get_sources(args.name)
            source_idx = len(existing_sources)
            dataset_subdir = f"dataset_personal_{source_idx}"

            # Строим датасет
            builder = DatasetBuilder()
            train_examples, valid_examples = builder.build_dataset(
                turns=turns,
                friend_name=args.name
            )

            # Сохраняем датасет в поддиректорию
            friend_dir = self.manager.get_friend_dir(args.name)
            friend_dir.mkdir(exist_ok=True)
            builder.save_dataset(train_examples, valid_examples, friend_dir / dataset_subdir)

            # Регистрируем источник
            dataset_stats = builder.get_stats()
            self.manager.add_friend(args.name, status="imported")
            self.manager.add_source(
                friend_name=args.name,
                source_type="personal",
                source_path=str(args.path),
                dataset_dir=dataset_subdir,
                num_examples=dataset_stats.total_examples
            )

            # Выводим статистику
            print(f"\n{self.colors.success}Статистика датасета (личный чат):{self.colors.reset}")
            print(f"  Примеры для обучения: {dataset_stats.train_examples}")
            print(f"  Примеры для валидации: {dataset_stats.valid_examples}")
            print(f"  Средняя длина ответа: {dataset_stats.avg_reply_length:.1f} слов")

            total_sources = len(self.manager.get_sources(args.name))
            if total_sources > 1:
                print(f"\n{self.colors.system}У {args.name} теперь {total_sources} источников данных.{self.colors.reset}")
                print(f"  При обучении (train) все источники будут объединены автоматически.")

            print(f"\n{self.colors.success}✓ Личный чат импортирован!{self.colors.reset}")

        except Exception as e:
            print(f"{self.colors.error}Ошибка: {e}{self.colors.reset}")

    def cmd_import_group(self, args: argparse.Namespace) -> None:
        """
        Команда: импортировать групповой чат Telegram.

        Автоматически определяет всех участников. Если --friends не указан,
        создает датасеты для всех участников (кроме --exclude).
        Датасеты добавляются к существующим источникам каждого друга.
        """
        print(f"{self.colors.system}Импорт группового чата из {args.path}...{self.colors.reset}")

        if not Path(args.path).exists():
            print(f"{self.colors.error}Ошибка: файл/папка не найдены{self.colors.reset}")
            return

        try:
            # Парсим файл группового чата с автоматическим определением формата
            parser = TelegramParser()
            messages = parser.parse_auto(args.path)
            print(f"{self.colors.success}✓ Спарсено {len(messages)} сообщений{self.colors.reset}")

            # Конвертируем Message → ConversationTurn
            turns = parser.messages_to_turns()

            # Автодетект всех участников
            all_members = sorted(set(turn.name for turn in turns))
            stats = parser.get_statistics()
            msg_by_member = stats.get('messages_by_participant', {})

            print(f"\n{self.colors.system}Участники группового чата ({len(all_members)}):{self.colors.reset}")
            for member in all_members:
                count = msg_by_member.get(member, 0)
                print(f"  - {member} ({count} сообщений)")

            # Определяем список друзей для обучения
            if hasattr(args, 'friends') and args.friends:
                friends = args.friends
                # Проверяем, что все указанные друзья есть в сообщениях
                missing = set(friends) - set(all_members)
                if missing:
                    print(f"{self.colors.error}Не найдены в чате: {', '.join(missing)}{self.colors.reset}")
                    return
            else:
                # Берём всех участников, исключая указанных в --exclude
                exclude = set(args.exclude) if hasattr(args, 'exclude') and args.exclude else set()
                friends = [m for m in all_members if m not in exclude]

            print(f"\n{self.colors.system}Создание датасетов для: {', '.join(friends)}{self.colors.reset}")

            # Для каждого друга строим свой датасет (добавляется к имеющимся)
            for friend_name in friends:
                print(f"\n  Обработка {friend_name}...")

                # Определяем уникальный номер источника
                existing_sources = self.manager.get_sources(friend_name)
                source_idx = len(existing_sources)
                dataset_subdir = f"dataset_group_{source_idx}"

                # Строим групповой датасет
                builder = DatasetBuilder()
                train_examples, valid_examples = builder.build_group_dataset(
                    turns=turns,
                    friend_name=friend_name,
                    group_members=all_members
                )

                # Сохраняем датасет
                friend_dir = self.manager.get_friend_dir(friend_name)
                friend_dir.mkdir(exist_ok=True)
                builder.save_dataset(train_examples, valid_examples, friend_dir / dataset_subdir)

                # Регистрируем источник
                dataset_stats = builder.get_stats()
                self.manager.add_friend(friend_name, status="imported")
                self.manager.add_source(
                    friend_name=friend_name,
                    source_type="group",
                    source_path=str(args.path),
                    dataset_dir=dataset_subdir,
                    num_examples=dataset_stats.total_examples
                )

                print(f"  {self.colors.success}✓ {friend_name}: "
                      f"{dataset_stats.train_examples} train / "
                      f"{dataset_stats.valid_examples} valid "
                      f"(ответ ~{dataset_stats.avg_reply_length:.0f} слов){self.colors.reset}")

                total_sources = len(self.manager.get_sources(friend_name))
                if total_sources > 1:
                    print(f"    Всего источников: {total_sources} (при train будут объединены)")

            print(f"\n{self.colors.success}✓ Групповой чат импортирован!{self.colors.reset}")

        except Exception as e:
            print(f"{self.colors.error}Ошибка: {e}{self.colors.reset}")

    def cmd_train(self, args: argparse.Namespace) -> None:
        """
        Команда: обучить модель на данных друга.

        Автоматически объединяет все источники (личные + групповые чаты)
        в один датасет перед обучением.
        """
        _ensure_ml_imports()
        friend_name = args.friend_name

        if not self.manager.friend_exists(friend_name):
            print(f"{self.colors.error}Ошибка: друг '{friend_name}' не найден{self.colors.reset}")
            return

        base_model = args.model or "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        epochs = args.epochs or 3

        print(f"{self.colors.system}Обучение модели для {friend_name}...{self.colors.reset}")
        print(f"  Базовая модель: {base_model}")
        print(f"  Эпохи: {epochs}")

        try:
            friend_dir = self.manager.get_friend_dir(friend_name)

            # Собираем все директории с датасетами
            dataset_dirs = self.manager.get_dataset_dirs(friend_name)

            if not dataset_dirs:
                print(f"{self.colors.error}Ошибка: датасеты не найдены для {friend_name}{self.colors.reset}")
                return

            # Выводим информацию об источниках
            sources = self.manager.get_sources(friend_name)
            print(f"\n{self.colors.system}Источники данных ({len(sources)}):{self.colors.reset}")
            for src in sources:
                src_type = "личный" if src["type"] == "personal" else "групповой"
                print(f"  - {src_type} чат ({src['num_examples']} примеров)")

            # Объединяем датасеты если их несколько
            if len(dataset_dirs) > 1:
                print(f"\n{self.colors.system}Объединение {len(dataset_dirs)} датасетов...{self.colors.reset}")
                merged_dir = friend_dir / "dataset_merged"
                train_count, valid_count = DatasetBuilder.merge_datasets(
                    dataset_dirs, merged_dir, shuffle=True
                )
                dataset_dir = merged_dir
                print(f"  Итого: {train_count} train + {valid_count} valid примеров")
            else:
                dataset_dir = dataset_dirs[0]

            # Создаём модель друга
            friend_model = FriendModel(
                friend_name=friend_name,
                base_model=base_model,
                models_dir=friend_dir.parent / "models",
                configs_dir=friend_dir.parent / "configs"
            )

            # Обучаем адаптер
            print(f"\nОбучение LoRA адаптера ({epochs} эпох)...")
            success = friend_model.train(
                data_dir=dataset_dir,
                epochs=epochs,
                batch_size=1,
                lora_layers=8,
                learning_rate=1e-5
            )

            if not success:
                print(f"{self.colors.error}Ошибка: тренировка не удалась{self.colors.reset}")
                return

            # Обновляем статус
            self.manager.update_friend_status(friend_name, "trained")

            print(f"\n{self.colors.success}✓ Модель успешно обучена!{self.colors.reset}")
            print(f"  Адаптер сохранен в: {friend_model.get_adapter_path()}")

        except Exception as e:
            print(f"{self.colors.error}Ошибка: {e}{self.colors.reset}")

    def cmd_fuse(self, args: argparse.Namespace) -> None:
        """Команда: слить адаптер с базовой моделью."""
        _ensure_ml_imports()
        friend_name = args.friend_name

        status = self.manager.get_friend_status(friend_name)
        if status != "trained":
            print(f"{self.colors.error}Ошибка: модель для {friend_name} ещё не обучена{self.colors.reset}")
            return

        print(f"{self.colors.system}Слияние адаптера для {friend_name}...{self.colors.reset}")

        try:
            friend_dir = self.manager.get_friend_dir(friend_name)

            # Загружаем модель друга
            friend_model = FriendModel(
                friend_name=friend_name,
                models_dir=friend_dir.parent / "models",
                configs_dir=friend_dir.parent / "configs"
            )

            # Сливаем адаптер с базовой моделью
            success = friend_model.fuse()

            if not success:
                print(f"{self.colors.error}Ошибка: слияние не удалось{self.colors.reset}")
                return

            # Обновляем статус
            self.manager.update_friend_status(friend_name, "fused")

            print(f"{self.colors.success}✓ Адаптер успешно слит!{self.colors.reset}")
            print(f"  Сохранен в: {friend_model.get_fused_model_path()}")

        except Exception as e:
            print(f"{self.colors.error}Ошибка: {e}{self.colors.reset}")

    def cmd_chat(self, args: argparse.Namespace) -> None:
        """Команда: начать чат с другом."""
        _ensure_ml_imports()
        friend_name = args.friend_name

        status = self.manager.get_friend_status(friend_name)
        if not status or status not in ("trained", "fused"):
            print(f"{self.colors.error}Ошибка: модель для {friend_name} не готова к чату{self.colors.reset}")
            return

        try:
            friend_dir = self.manager.get_friend_dir(friend_name)

            # Загружаем модель друга
            friend_model = FriendModel(
                friend_name=friend_name,
                models_dir=friend_dir.parent / "models",
                configs_dir=friend_dir.parent / "configs"
            )

            # Получаем путь к модели (объединенной или с адаптером)
            model_path = friend_model.get_model_path()

            if not model_path.exists():
                print(f"{self.colors.error}Ошибка: модель не найдена{self.colors.reset}")
                return

            # Получаем системный промпт
            system_prompt = f"Ты {friend_name}. Отвечай естественно и дружелюбно."

            # Создаём инжин
            inference = FriendEngine(
                name=friend_name,
                model_path=str(model_path),
                system_prompt=system_prompt
            )

            # Запускаем чат
            chat = ChatInterface(friend_name, inference, self.colors)
            chat.run()

        except Exception as e:
            print(f"{self.colors.error}Ошибка: {e}{self.colors.reset}")

    def cmd_group(self, args: argparse.Namespace) -> None:
        """Команда: начать групповой чат."""
        _ensure_ml_imports()
        friends = args.friends

        # Проверяем, что все друзья существуют и готовы
        for friend_name in friends:
            status = self.manager.get_friend_status(friend_name)
            if not status or status not in ("trained", "fused"):
                print(f"{self.colors.error}Ошибка: модель для {friend_name} не готова к чату{self.colors.reset}")
                return

        try:
            # Создаём инжины для всех друзей
            inference_engines = {}
            for friend_name in friends:
                friend_dir = self.manager.get_friend_dir(friend_name)

                # Загружаем модель друга
                friend_model = FriendModel(
                    friend_name=friend_name,
                    models_dir=friend_dir.parent / "models",
                    configs_dir=friend_dir.parent / "configs"
                )

                # Получаем путь к модели
                model_path = friend_model.get_model_path()

                if not model_path.exists():
                    print(f"{self.colors.error}Ошибка: модель не найдена для {friend_name}{self.colors.reset}")
                    return

                # Создаём инжин для друга
                system_prompt = f"Ты {friend_name}. Отвечай естественно и дружелюбно."
                inference_engines[friend_name] = FriendEngine(
                    name=friend_name,
                    model_path=str(model_path),
                    system_prompt=system_prompt
                )

            # Запускаем групповой чат
            group_chat = GroupChatInterface(friends, inference_engines, self.colors)
            group_chat.run()

        except Exception as e:
            print(f"{self.colors.error}Ошибка: {e}{self.colors.reset}")

    def cmd_list(self, args: argparse.Namespace) -> None:
        """Команда: список всех друзей."""
        friends = self.manager.list_friends()

        if not friends:
            print(f"{self.colors.system}Друзья не найдены. Импортируйте чат Telegram.{self.colors.reset}")
            return

        print(f"\n{self.colors.system}Список друзей:{self.colors.reset}")
        print("-" * 60)

        for name, info in friends.items():
            status = info["status"]
            sources = info["sources"]

            # Определяем символ статуса
            if status == "imported":
                symbol = "📦"
                color = self.colors.system
            elif status == "trained":
                symbol = "🎓"
                color = self.colors.system
            elif status == "fused":
                symbol = "✨"
                color = self.colors.success
            else:
                symbol = "❓"
                color = self.colors.error

            # Формируем описание источников
            personal_count = sources.count("personal")
            group_count = sources.count("group")
            source_parts = []
            if personal_count:
                source_parts.append(f"{personal_count} личн.")
            if group_count:
                source_parts.append(f"{group_count} груп.")
            source_str = ", ".join(source_parts) if source_parts else "нет данных"

            print(f"  {symbol} {color}{name:<20}{self.colors.reset} [{status}] ({source_str})")

        print("-" * 60)
        print(f"Всего друзей: {len(friends)}")
        print()

    def cmd_profile(self, args: argparse.Namespace) -> None:
        """Команда: показать профиль друга."""
        friend_name = args.friend_name

        if not self.manager.friend_exists(friend_name):
            print(f"{self.colors.error}Ошибка: друг '{friend_name}' не найден{self.colors.reset}")
            return

        status = self.manager.get_friend_status(friend_name)

        try:
            friend_dir = self.manager.get_friend_dir(friend_name)
            dataset_path = friend_dir / "dataset.json"

            print(f"\n{self.colors.system}Профиль: {friend_name}{self.colors.reset}")
            print("-" * 50)
            print(f"  Статус: {status}")

            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)

                if isinstance(dataset, dict) and "stats" in dataset:
                    stats = dataset["stats"]
                    print(f"  Сообщений: {stats.get('total_examples', 0)}")
                    print(f"  Средняя длина: {stats.get('avg_message_length', 0):.1f} символов")
                    print(f"  Уникальные слова: {stats.get('unique_words', 0)}")
                    print(f"  Период: {stats.get('first_message_date', 'N/A')} — {stats.get('last_message_date', 'N/A')}")

            print("-" * 50)
            print()

        except Exception as e:
            print(f"{self.colors.error}Ошибка: {e}{self.colors.reset}")

    def main(self) -> None:
        """Основная функция CLI."""
        parser = argparse.ArgumentParser(
            description="FriendGPT - чат с обученными моделями друзей",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Примеры использования:
  # Импорт личного чата (JSON или HTML — автоопределение)
  python cli.py import result.json --name Иван
  python cli.py import chat_export/ --name Иван

  # Импорт группового чата (автодетект участников)
  python cli.py import-group group_export/              # все участники
  python cli.py import-group group.html --friends Вася Петя
  python cli.py import-group group.html --exclude БотСпам

  # Один друг из личного + группового чата (данные объединяются)
  python cli.py import dm_with_vasya/ --name Вася
  python cli.py import-group group_chat/ --friends Вася
  python cli.py train Вася   # обучится на обоих источниках

  # Обучение, слияние, чат
  python cli.py train Иван --epochs 5
  python cli.py fuse Иван
  python cli.py chat Иван
  python cli.py group Иван Мария
  python cli.py list
  python cli.py profile Иван
            """
        )

        subparsers = parser.add_subparsers(dest="command", help="Команды")

        # Команда import
        import_parser = subparsers.add_parser("import", help="Импортировать чат Telegram")
        import_parser.add_argument("path", help="Путь к result.json или result.html из экспорта Telegram")
        import_parser.add_argument("--name", required=True, help="Имя друга")
        import_parser.set_defaults(func=self.cmd_import)

        # Команда import-group
        import_group_parser = subparsers.add_parser("import-group", help="Импортировать групповой чат Telegram")
        import_group_parser.add_argument("path", help="Путь к экспорту группового чата (JSON/HTML/папка)")
        import_group_parser.add_argument("--friends", nargs="+", help="Имена друзей для обучения (по умолчанию — все участники)")
        import_group_parser.add_argument("--exclude", nargs="+", help="Участники, которых НЕ нужно обучать (напр. боты, вы сами)")
        import_group_parser.set_defaults(func=self.cmd_import_group)

        # Команда train
        train_parser = subparsers.add_parser("train", help="Обучить модель друга")
        train_parser.add_argument("friend_name", help="Имя друга")
        train_parser.add_argument("--model", default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", help="Базовая модель (default: Meta-Llama-3.1-8B)")
        train_parser.add_argument("--epochs", type=int, default=3, help="Количество эпох обучения")
        train_parser.set_defaults(func=self.cmd_train)

        # Команда fuse
        fuse_parser = subparsers.add_parser("fuse", help="Слить адаптер с базовой моделью")
        fuse_parser.add_argument("friend_name", help="Имя друга")
        fuse_parser.set_defaults(func=self.cmd_fuse)

        # Команда chat
        chat_parser = subparsers.add_parser("chat", help="Чат с другом")
        chat_parser.add_argument("friend_name", help="Имя друга")
        chat_parser.set_defaults(func=self.cmd_chat)

        # Команда group
        group_parser = subparsers.add_parser("group", help="Групповой чат")
        group_parser.add_argument("friends", nargs="+", help="Имена друзей")
        group_parser.set_defaults(func=self.cmd_group)

        # Команда list
        list_parser = subparsers.add_parser("list", help="Список всех друзей")
        list_parser.set_defaults(func=self.cmd_list)

        # Команда profile
        profile_parser = subparsers.add_parser("profile", help="Профиль друга")
        profile_parser.add_argument("friend_name", help="Имя друга")
        profile_parser.set_defaults(func=self.cmd_profile)

        # Парсим аргументы
        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Выполняем команду
        if hasattr(args, 'func'):
            args.func(args)


def main() -> None:
    """Точка входа в приложение."""
    cli = FriendGPTCLI()
    cli.main()


if __name__ == "__main__":
    main()
