from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "fixtures"


def load_runtime_env() -> None:
    """Load repository-local environment variables once for CLI and NIM usage."""
    load_dotenv(REPO_ROOT / ".env", override=False)
