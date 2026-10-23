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

Use Python **3.11 or newer**. Importing JSON/HTML exports and building datasets works on macOS, Windows and Linux without a GPU or model download. MLX training and inference require an Apple Silicon Mac. The separate NVIDIA scripts in `friendgpt/` use their own `requirements_cuda.txt`.

### Installation

1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows PowerShell: .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   python -m pip install .
   ```

For development, `python -m pip install -r requirements.txt` installs the project in editable mode.
For MLX training/inference on Apple Silicon, install `python -m pip install '.[mlx]'`.
For the Telegram integration as well, use `python -m pip install '.[mlx,telegram]'`.
The portable installation includes the HTML parser dependency; it does not install MLX, PyTorch or Telegram.

### Usage

The `cli.py` provides commands to manage the full lifecycle of the bot:

- Parse Telegram data
- Build training datasets
- Train models
- Run the Telegram bot

The installed `friendgpt` command and `python -m friendgpt` work outside the checkout. `python cli.py` remains available from the source tree. Help commands do not create a data directory.

```bash
friendgpt --help
friendgpt --data-dir ./friends_data import ./result.json --name "Alice"
friendgpt --data-dir ./friends_data import ./html_export/ --name "Alice"
friendgpt --data-dir ./friends_data import-group ./group.json --friends "Alice" "Bob"
friendgpt --data-dir ./friends_data list
```

Place `--data-dir` **before** the subcommand; it defaults to `./friends_data` relative to your current directory. Use the exact participant name from the export. A participant name must fit in a single directory name, without `/`, `\`, `:`, or a trailing dot/space. HTML directories must contain `messages.html` (and may contain subsequent `messages2.html`, etc.); JSON exports are supplied as a file.

Each successful import creates `train.jsonl` and `valid.jsonl` under a numbered source directory, for example `friends_data/Alice/dataset_personal_0/`, and registers it in `metadata.json`. Repeated imports add sources. Personal and group imports return a nonzero exit status for invalid input, unknown participants, an empty participant selection, or no usable training examples. Replies need at least three words and prior context. A failed group import may leave earlier successfully imported participants registered; inspect `list` before retrying.

Only import chat exports you are authorized to use. Dataset preparation runs locally. Training and live Telegram operation are separate actions.

### Validation

The regression suite creates fabricated JSON and HTML exports in temporary directories. It never reads chat history, loads models, trains, or contacts Telegram.

```bash
python -m unittest discover -s tests -v
python -m pip wheel --no-deps . --wheel-dir wheelhouse
python tests/check_wheel.py wheelhouse/friendgpt-0.1.0-py3-none-any.whl
```

The wheel check installs into a fresh virtual environment and repeats the suite from outside the repository, including the installed console command. It downloads ordinary Python dependencies during installation. CI runs these checks on Linux, Windows and macOS. These checks cover the offline import workflow; they do not establish model quality or live Telegram behavior.

## License

See `LICENSE` file for details.
