"""Exercise the CLI using fabricated exports only, without MLX or Telegram."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLED = os.environ.get("FRIENDGPT_TEST_INSTALLED") == "1"


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workdir = Path(self.directory.name)
        self.data = self.workdir / "nested" / "datasets"
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        self.environment["PYTHONIOENCODING"] = "utf-8"
        if not INSTALLED:
            self.environment["PYTHONPATH"] = str(ROOT)

    def run_python(self, *arguments):
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=self.workdir,
            env=self.environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
        )

    def run_cli(self, *arguments):
        return self.run_python("-m", "friendgpt", *arguments)

    def write_export(self, name="Alice", reply="A fabricated reply for testing"):
        messages = []
        for index in range(12):
            messages.extend([
                {"id": index * 2, "type": "message", "from": "Bob",
                 "date": f"2026-01-01T12:{index:02d}:00", "text": "A synthetic question"},
                {"id": index * 2 + 1, "type": "message", "from": name,
                 "date": f"2026-01-01T12:{index:02d}:30", "text": reply},
            ])
        export = self.workdir / "result.json"
        export.write_text(json.dumps({"messages": messages}), encoding="utf-8")
        return export

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_dataset(self, name="Alice", source="dataset_personal_0"):
        folder = self.data / name / source
        for split in ("train", "valid"):
            rows = (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(rows, split)
            for row in rows:
                example = json.loads(row)
                self.assertEqual(example["messages"][-1]["role"], "assistant")
        metadata = json.loads((self.data / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata[name]["status"], "imported")
        return metadata

    def test_module_help_does_not_create_data(self):
        self.assert_success(self.run_cli("--help"))
        self.assertEqual(list(self.workdir.iterdir()), [])

    def test_no_command_does_not_create_data(self):
        self.assert_success(self.run_cli())
        self.assertEqual(list(self.workdir.iterdir()), [])

    def test_subcommand_help_does_not_create_data(self):
        self.assert_success(self.run_cli("import", "--help"))
        self.assertEqual(list(self.workdir.iterdir()), [])

    def test_cli_does_not_import_model_or_bot_backends(self):
        result = self.run_python(
            "-c", "import friendgpt.cli, sys; "
            "assert not any(n.split('.')[0] in {'mlx', 'mlx_lm', 'telegram', 'torch'} "
            "for n in sys.modules)",
        )
        self.assert_success(result)

    @unittest.skipIf(INSTALLED, "legacy script is source-tree only")
    def test_legacy_script_help(self):
        self.assert_success(self.run_python(str(ROOT / "cli.py"), "--help"))
        self.assertEqual(list(self.workdir.iterdir()), [])

    @unittest.skipUnless(INSTALLED, "console entry is checked from installed wheel")
    def test_console_entry_help(self):
        script = Path(sys.executable).parent / ("friendgpt.exe" if os.name == "nt" else "friendgpt")
        result = subprocess.run(
            [str(script), "--help"], cwd=self.workdir, env=self.environment,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assert_success(result)
        self.assertEqual(list(self.workdir.iterdir()), [])

    def test_json_import_and_repeated_sources(self):
        export = self.write_export(name="Синтетическая Алиса")
        for index in range(2):
            result = self.run_cli("--data-dir", str(self.data), "import", str(export),
                                  "--name", "Синтетическая Алиса")
            self.assert_success(result)
            metadata = self.assert_dataset("Синтетическая Алиса", f"dataset_personal_{index}")
            self.assertEqual(len(metadata["Синтетическая Алиса"]["sources"]), index + 1)
        self.assert_success(self.run_cli("--data-dir", str(self.data), "list"))

    def test_html_directory_import(self):
        folder = self.workdir / "html_export"
        folder.mkdir()
        html = []
        for index in range(12):
            for sender, second, text in (("Bob", 0, "A synthetic question"),
                                         ("Alice", 30, "A fabricated reply for testing")):
                html.append(
                    f'<div class="message default"><div class="body">'
                    f'<div class="from_name">{sender}</div>'
                    f'<div class="date details" title="01.01.2026 12:{index:02d}:{second:02d} UTC+03:00"></div>'
                    f'<div class="text">{text}</div></div></div>'
                )
        (folder / "messages.html").write_text("\n".join(html), encoding="utf-8")
        result = self.run_cli("--data-dir", str(self.data), "import", str(folder), "--name", "Alice")
        self.assert_success(result)
        self.assert_dataset()

    def test_group_import(self):
        export = self.write_export()
        result = self.run_cli("--data-dir", str(self.data), "import-group", str(export), "--friends", "Alice")
        self.assert_success(result)
        metadata = self.assert_dataset(source="dataset_group_0")
        self.assertEqual(metadata["Alice"]["sources"][0]["type"], "group")

    def test_failed_imports_return_nonzero_without_registering_source(self):
        export = self.write_export()
        for arguments in (("import", "missing.json", "--name", "Alice"),
                          ("import", str(export), "--name", "Missing participant"),
                          ("import-group", str(export), "--friends", "Missing participant"),
                          ("import-group", str(export), "--exclude", "Alice", "Bob")):
            with self.subTest(arguments=arguments):
                result = self.run_cli("--data-dir", str(self.data), *arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse((self.data / "metadata.json").exists())

    def test_malformed_export_returns_nonzero(self):
        export = self.workdir / "invalid.json"
        export.write_text("{", encoding="utf-8")
        result = self.run_cli("--data-dir", str(self.data), "import", str(export), "--name", "Alice")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.data / "metadata.json").exists())

    def test_empty_training_set_is_not_registered(self):
        export = self.write_export(reply="ok")
        result = self.run_cli("--data-dir", str(self.data), "import", str(export), "--name", "Alice")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.data / "metadata.json").exists())

    def test_participant_name_cannot_escape_data_directory(self):
        for name in ("../escaped", "..\\escaped", "C:escaped", ".. "):
            with self.subTest(name=name):
                export = self.write_export(name=name)
                result = self.run_cli("--data-dir", str(self.data), "import", str(export), "--name", name)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.data.parent / "escaped").exists())
                self.assertFalse((self.data / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
