"""Pytest configuration — adds project root to sys.path for clean imports."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so imports like `from profiler.analyzer import ...` work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
