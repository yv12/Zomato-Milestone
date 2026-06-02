import sys
import os

# Resolve absolute paths and append to python search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from app.ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    run_ingestion()
