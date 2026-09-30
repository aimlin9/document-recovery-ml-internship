"""
Shared project paths and external-tool configuration.

The scripts were originally written to run from a single flat folder
(corpus, schema, model and outputs all side by side). The repository is
now organised into data/, schema/, results/ and scripts/, so every script
resolves its files from the project root through this module instead of
relying on the current working directory. That means any script can be
run as `python scripts/<name>.py` from the project root, or from inside
scripts/, and find the same files.
"""

import os
import shutil

import pytesseract

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCHEMA_DIR = os.path.join(PROJECT_ROOT, "schema")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")  # scratch images, not tracked in git

CORPUS_PATH = os.path.join(DATA_DIR, "sentence_corpus.jsonl")
MASKED_EXAMPLES_PATH = os.path.join(DATA_DIR, "masked_examples.jsonl")
MANIFEST_PATH = os.path.join(DATA_DIR, "held_out_manifest.json")
SCHEMA_PATH = os.path.join(SCHEMA_DIR, "document_schema.json")
FINE_TUNED_MODEL_PATH = os.environ.get(
    "ROBERTA_MODEL_PATH", os.path.join(PROJECT_ROOT, "roberta-finetuned-final")
)


def configure_tesseract():
    """Points pytesseract at a Tesseract binary. Order: the TESSERACT_CMD
    environment variable, then whatever is on PATH, then the default
    Windows install location. Previously the Windows path was hardcoded in
    every script, which broke the pipeline on Linux/macOS."""
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            pytesseract.pytesseract.tesseract_cmd = c
            return c
    return pytesseract.pytesseract.tesseract_cmd  # let pytesseract raise its own clear error


configure_tesseract()


def rel(path):
    """Project-relative, forward-slash path for writing into JSON outputs,
    so documents never embed one machine's absolute directory layout."""
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
