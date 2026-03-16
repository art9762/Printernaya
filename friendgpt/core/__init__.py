# FriendGPT Core — ядро проекта
from .telegram_parser import TelegramParser, ConversationTurn
from .dataset_builder import DatasetBuilder
from .trainer import Trainer, FriendModel
from .inference import FriendEngine, GroupChat, ModelManager

__all__ = [
    "TelegramParser",
    "ConversationTurn",
    "DatasetBuilder",
    "Trainer",
    "FriendModel",
    "FriendEngine",
    "GroupChat",
    "ModelManager",
]
