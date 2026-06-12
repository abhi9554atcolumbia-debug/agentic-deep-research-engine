"""
Command-line entry point for the full research pipeline.

The Streamlit app in app.py is the recommended demo interface, but this
keeps `python main.py "question"` working for quick terminal runs.
"""

import sys
from main_v2 import run_pipeline


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Compare three low-cost methods for improving air quality in classrooms."

    run_pipeline(query)
