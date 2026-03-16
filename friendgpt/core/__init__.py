# FriendGPT Core — ядро проекта
# Ленивые импорты: тяжёлые модули (mlx_lm) загружаются только при обращении
from .telegram_parser import TelegramParser, ConversationTurn, Message, PersonalityProfile
from .dataset_builder import DatasetBuilder

# Trainer и Inference требуют mlx_lm — импортируются лениво


def __getattr__(name):
    """Ленивый импорт тяжёлых модулей."""
    if name in ("Trainer", "FriendModel"):
        from .trainer import Trainer, FriendModel
        return {"Trainer": Trainer, "FriendModel": FriendModel}[name]
    elif name in ("FriendEngine", "GroupChat", "ModelManager"):
        from .inference import FriendEngine, GroupChat, ModelManager
        return {"FriendEngine": FriendEngine, "GroupChat": GroupChat, "ModelManager": ModelManager}[name]
    raise AttributeError(f"module 'core' has no attribute {name!r}")


__all__ = [
    "TelegramParser",
    "ConversationTurn",
    "Message",
    "PersonalityProfile",
    "DatasetBuilder",
    "Trainer",
    "FriendModel",
    "FriendEngine",
    "GroupChat",
    "ModelManager",
]
