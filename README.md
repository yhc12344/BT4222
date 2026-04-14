# Predicting Outcomes of Singapore Corporate Law Appeals

This repository contains an end-to-end legal analytics workflow for turning Singapore corporate law judgments into structured training data and predicting defendant-side outcomes.

The current codebase is organized around four practical stages:

1. Extract party-level facts and labels from judgment PDFs with GPT-based parsers.
2. Audit those rows for leakage, label consistency, and missing representation details.
3. Flatten audited JSON into a modeling table.
4. Train and evaluate predictive models such as XGBoost, HAN, and BERT-based baselines.

## Overview of project

This project is a legal analytics pipeline for Singapore corporate law judgments. It starts from judgment PDFs, converts them into structured party-level records, improves those records through a dedicated audit stage, and prepares a modeling dataset for downstream prediction tasks.

The workflow combines large language model extraction, rule-based quality control, direct label recovery from judgments, and notebook-based machine learning experiments. In practical terms, the repo is designed to support both dataset construction and model evaluation.

## Project objective

The modeling task in this repo is best understood as binary defendant-side risk prediction:

- `Liable` means the defendant-side row or case is treated as a positive outcome.
- `Not Liable` means the defendant-side row or case is treated as a negative outcome.

## Features

- GPT-based extraction of structured legal rows from judgment PDFs
- support for party-level facts, issues, rules, applications, and conclusions
- counterclaim-aware extraction in the categorized pipeline
- leakage detection and scrubbing to reduce hindsight contamination
- row-level auditing against the source judgment
- case-level label consistency checks between plaintiff and defendant sides
- label recovery and validation directly from PDFs through `label_checker.py`
- conversion from audited JSON into a modeling-ready CSV
- notebook-based evaluation at both row level and case level

## Benefits of using this project

- reduces manual effort in turning long judgments into structured training data
- makes dataset construction repeatable instead of one-off and manual
- improves trust in extracted data through a dedicated audit stage
- keeps the workflow modular so extraction, auditing, and modeling can be run separately
- supports comparison across multiple model families such as XGBoost, HAN, and BERT-based baselines
- preserves intermediate artifacts, which makes debugging and reporting easier

## Repository map

| Path | Role |
| --- | --- |
| `src/extract_case_rows_baseline.py` | Main extraction script used for standard judgments. |
| `src/extract_case_rows_categorized.py` | Special-case extractor used for structurally complex judgments such as multi-party disputes and counterclaims. |
| `src/audit_case_rows.py` | Second-pass auditor that checks each extracted row against the source PDF and enforces case-level consistency. |
| `src/label_checker.py` | Utility that extracts plaintiff/defendant outcome labels directly from a judgment PDF and supports the audit stage. |
| `src/json_to_df.py` | Converts audited JSON into a flat tabular dataset for modeling. |
| `src/XGBoost.ipynb` | Gradient-boosted baseline experiments and test-set evaluation. |
| `src/HAN.ipynb` | Hierarchical Attention Network experiments with row-level and case-level evaluation. |
| `src/BERT_Classifier.ipynb` | Transformer-based classifier experiments. |
| `src/config.py` | Shared configuration for models, API keys, and pipeline paths. |

## Data

The dataset in this project is built from Singapore court judgment PDFs and the structured artifacts generated from them during extraction, audit, and flattening.

You can find the source judgments on the Singapore Courts judgments portal here:

- https://www.judiciary.gov.sg/judgments/judgments-case-summaries

Because the raw judgment PDFs are not intended to be stored in this GitHub repository, you should download them separately and place them into `Data/PDFs/ALL`.

The judgment subset in this project was collected using the following catchwords:

| Category | Sub-Category | Specific Topic |
| --- | --- | --- |
| Companies | Directors | Appointment |
| Companies | Directors | Disqualification |
| Companies | Directors | Duties |
| Companies | Directors | Liabilities |
| Companies | Directors | Powers |
| Companies | Directors | Removal |
| Companies | Directors | Remuneration |

Within this repository, the main data locations are:

- `Data/PDFs/ALL` for source judgment PDFs
- `Data/Processed/FinalAudited` for extracted and audited JSON files
- `Data/Label_Checks` for standalone label-check outputs from `src/label_checker.py`
- `Data/court_cases.csv` for the final modeling dataset

## Setup

### Base environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Additional packages used by the extraction and modeling notebooks

The checked-in `requirements.txt` covers part of the utility stack, but the notebooks and GPT extraction scripts also rely on packages such as:

- `openai`
- `pdfplumber`
- `scikit-learn`
- `xgboost`
- `torch`
- `transformers`
- `matplotlib`
- `tqdm`

If those are not already installed in your environment, install them before running the notebooks.

### API configuration

The GPT extraction and audit scripts expect an OpenAI API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

You can also store it in `.env`, which is already supported by the extraction and audit scripts.

## Methodology

The following scripts and Jupyter notebooks form the methodology for this project.

### Step 1. Structured legal data extraction

**Main script:** `src/extract_case_rows_baseline.py`  
**Special-case extractor:** `src/extract_case_rows_categorized.py`

This stage converts raw judgment PDFs into structured legal rows that can be used for downstream analysis and prediction.

The main extraction workflow uses `src/extract_case_rows_baseline.py` to process the majority of judgments and generate the standard structured JSON used in the pipeline.

The categorized extractor, `src/extract_case_rows_categorized.py`, is used for special-case judgments where additional structural handling is needed, such as:

- multi-party cases
- counterclaims
- more explicit separation of `results` and `counterclaims`
- richer role normalization and structural resolution

Both extraction scripts read judgment PDFs from `Data/PDFs/ALL` and use GPT-based prompting to extract:

- party-level facts
- issue, rule, and application fields
- outcome labels
- metadata such as case number, coram, judge, and tribunal

### Step 2. Audit and repair extracted case rows

**Main script:** `src/audit_case_rows.py`

After extraction, the generated JSON rows are passed through an audit stage to improve data quality before modelling.

As currently configured in `src/config.py`, the audit script reads extracted JSON from `Data/Processed/FinalAudited`, matches each one to its corresponding judgment PDF, and writes the audited result back to `Data/Processed/FinalAudited`.

This script performs:

- row-level factual auditing against the source text
- leakage reduction for `Facts`, `Issue`, `Rule`, and `Application`
- fact-type and fact-date normalization
- plaintiff-defendant label consistency checks at case level
- recovery of missing counsel and law-firm information from the judgment header

This step is important because it reduces noisy or hindsight-contaminated features and improves the reliability of the final dataset.

### Step 3. Label validation and recovery

**Helper utility:** `src/label_checker.py`

In addition to the main audit logic, this project includes a label-checking utility that extracts plaintiff and defendant outcome labels directly from the judgment text.

This utility is used to:

- validate extracted labels
- recover missing labels where possible
- improve consistency between the extracted dataset and the original judgment

This provides an extra quality-control layer before the data is flattened for modelling.

### Step 4. Convert audited JSON into a modelling dataset

**Main script:** `src/json_to_df.py`
**File path:** `src/json_to_df.py`
**File path:** `src/json_to_df.py`

Once the audited JSON files are finalized, they are converted into a flat tabular dataset suitable for machine learning.

This script:

- reads audited JSON files from the configured audit output folder
- pairs plaintiff and defendant rows within each case
- combines the relevant text fields and metadata
- exports the final modelling table to `Data/court_cases.csv`

This CSV serves as the bridge between the legal data pipeline and the model training stage.

### Step 5. Exploratory analysis and model training

**Main notebooks:**

- `src/EDA.ipynb`
- `src/XGBoost.ipynb`
- `src/HAN.ipynb`
- `src/BERT_Classifier.ipynb`

After building the modelling dataset, we evaluate multiple approaches for defendant-side outcome prediction.

The notebooks are used for:

- exploratory data analysis
- feature inspection and preparation
- training baseline and neural models
- row-level and case-level evaluation
- comparing performance across model families

The main model families explored in this project are:

- XGBoost
- Hierarchical Attention Network (HAN)
- BERT-based classifier baselines



### Step 6. Evaluation and business interpretation

The final stage of the project focuses on evaluating predictive performance at both:

- row level
- case level

This distinction is important because row-level predictions reflect individual plaintiff-defendant pair decisions, while case-level predictions better represent the real operational decision unit for legal triage.

The evaluation stage is used to:

- compare model performance across architectures
- measure false positives and false negatives
- study the trade-off between recall and precision
- assess whether the system is more suitable for risk flagging, prioritization, or automated screening

## Documentation

For the full report documentation, see: [Google Doc](https://docs.google.com/document/d/1a1Mn4E9Vj_cmh-Qfrr0vBKh78H0SktNtOzRdeu1L24Y/edit?usp=sharing)

Project files and related materials are also available here: [Google Drive folder](https://drive.google.com/drive/folders/1qIGVH9jPR5h9CMDQZgGw6m1Rzc_6F-PS?usp=sharing)

Project files and related materials are also available here: [Google Drive folder](https://drive.google.com/drive/folders/1qIGVH9jPR5h9CMDQZgGw6m1Rzc_6F-PS?usp=sharing)
