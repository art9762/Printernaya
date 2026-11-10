from setuptools import find_packages, setup

setup(
    name="friendgpt",
    version="0.1.0",
    description="Обучи LLM на чатах друзей из Telegram и общайся с ними",
    author="Artem",
    packages=find_packages(),
    py_modules=["cli"],
    python_requires=">=3.11",
    install_requires=[
        "beautifulsoup4>=4.12",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "mlx": ["mlx>=0.16.0", "mlx-lm>=0.19.0"],
        "telegram": ["python-telegram-bot>=20.0"],
    },
    entry_points={
        "console_scripts": [
            "friendgpt=friendgpt.cli:main",
        ],
    },
)
