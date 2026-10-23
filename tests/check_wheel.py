"""Install a built wheel into a clean venv and rerun synthetic CLI tests."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main():
    wheel = Path(sys.argv[1]).resolve()
    tests = Path(__file__).with_name("test_cli.py").resolve()
    with tempfile.TemporaryDirectory() as temporary:
        environment = Path(temporary) / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", str(wheel)], check=True)
        variables = os.environ.copy()
        variables.pop("PYTHONPATH", None)
        variables["FRIENDGPT_TEST_INSTALLED"] = "1"
        subprocess.run([str(python), str(tests), "-v"], cwd=temporary, env=variables, check=True)


if __name__ == "__main__":
    main()
