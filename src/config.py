"""Centralised configuration for the BT4222 legal analytics pipeline.

All scripts import paths and model names from here.
Run all scripts from the project root
"""
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# =============================================================================
# API
# =============================================================================
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-5.1")
AUDIT_MODEL: str      = os.getenv("AUDIT_MODEL",      "gpt-5.4-mini")
LABEL_MODEL: str      = os.getenv("LABEL_MODEL",      "gpt-5.1")

# =============================================================================
# Paths
# =============================================================================
# Inputs
PDF_INPUT_ALL = Path("Data/PDFs/ALL")
STATS_INPUT   = Path("Data/PDFs/ALL")
AUDIT_INPUT   = Path("Data/Processed/testouput")

# Outputs
EXTRACTION_OUTPUT = Path("Data/Processed/FinalAudited")
AUDIT_OUTPUT      = Path("Data/Processed/FinalAudited")
CSV_OUTPUT        = Path("Data/court_cases.csv")


# Ensure the shared output directory exists on first import
EXTRACTION_OUTPUT.mkdir(parents=True, exist_ok=True)
