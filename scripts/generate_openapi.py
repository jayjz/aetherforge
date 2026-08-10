import json
import sys
import os
import argparse

# Add src to path so we can import the app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force mock engine so it doesn't crash trying to load models during schema dump
os.environ["AETHER_ENGINE"] = "mock"
from src.server import app

def dump_schema(output_path: str):
    schema = app.openapi()
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"✅ OpenAPI schema dumped to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="openapi.json")
    args = parser.parse_args()
    dump_schema(args.out)