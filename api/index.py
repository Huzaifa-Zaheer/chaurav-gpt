"""Vercel serverless entry point.

Vercel detects this file and routes /api/* requests to the FastAPI app.
In Vercel's runtime, requests arrive with path /api/..., so we mount
all routes under the /api prefix here.
"""
import os
import sys
from pathlib import Path

# Ensure the project root is on the Python path so `backend.app` is importable.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Signal that we're running in Vercel so the app factory can adapt.
os.environ.setdefault("VERCEL", "1")

from backend.app.main import create_app  # noqa: E402

app = create_app()
