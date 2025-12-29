from pathlib import Path

c = Path(__file__).resolve().parent

static = c / "static"

if static.exists():
    print("Static directory exists.")
