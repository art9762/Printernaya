# Printernaya (FriendGPT)

A Telegram bot capable of imitating your friends using machine learning and language models fine-tuned on chat histories.

## Features

- **Chat Parsing**: Extracts and formats messages from Telegram chat exports to build a dataset.
- **Model Fine-Tuning**: Uses MLX to fine-tune language models based on specific chat histories.
- **Inference & Generation**: Generates messages in the style of trained personalities.
- **Telegram Bot Integration**: A functional bot that responds as the fine-tuned persona in real-time.

## Project Structure

- `core/`: Contains core logic for dataset building, model training, and inference.
  - `telegram_parser.py`: Parses exported Telegram chats.
  - `dataset_builder.py`: Prepares parsed data into formats suitable for training.
  - `trainer.py`: Fine-tunes the language models.
  - `inference.py`: Handles message generation from trained models.
- `bots/`: Contains logic for running the Telegram bot.
- `configs/`: YAML configuration files.
- `cli.py`: Command-line interface for running different parts of the pipeline.

## Getting Started

### Prerequisites

Ensure you have Python 3 installed. This project relies on the MLX framework, specifically optimized for Apple Silicon (M1/M2/M3).

### Installation

1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

The `cli.py` provides commands to manage the full lifecycle of the bot:

- Parse Telegram data
- Build training datasets
- Train models
- Run the Telegram bot

Use `python cli.py --help` for specific commands and usage options.

## License

See `LICENSE` file for details.
