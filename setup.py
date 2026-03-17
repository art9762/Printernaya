from setuptools import setup, find_packages

setup(
    name="friendgpt",
    version="0.1.0",
    description="Обучи LLM на чатах друзей из Telegram и общайся с ними",
    author="Artem",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "mlx>=0.16.0",
        "mlx-lm>=0.19.0",
        "python-telegram-bot>=20.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
    ],
    entry_points={
        "console_scripts": [
            "friendgpt=friendgpt.cli:main",
        ],
    },
)
