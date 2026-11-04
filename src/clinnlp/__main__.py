"""CLI: python -m clinnlp "text" | python -m clinnlp < note.txt"""
import json, sys

from .pipeline import Pipeline

text = " ".join(sys.argv[1:]) or sys.stdin.read()
print(json.dumps(Pipeline().extract(text), indent=2))
