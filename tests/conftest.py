import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# tests never touch the demo audit DB
os.environ["APP_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="itta-test-"), "test.sqlite")
os.environ.setdefault("PII_ENGINE", "presidio")
os.environ["LLM_PROVIDER"] = "offline"  # unit tests never call a paid API
