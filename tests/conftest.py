import os, sys
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:1")  # unreachable -> in-memory fallback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
