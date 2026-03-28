# Data Extraction Pipeline (Feature Generation Stage)

## 1. Objective

The first stage of the pipeline focuses on transforming **raw legal judgments (PDFs)** into a **structured dataset** suitable for machine learning.

### Machine Learning Objective

Input (X): Pre-dispute structured facts and features  
Output (y): Legal outcome (e.g., Liable / Not Liable)

To achieve this, the extraction process must ensure:
- Clear separation between features (facts) and target (outcome)
- Structured representation of legal information
- Consistency across different cases

---

## 2. Core Idea: Structured Extraction using LLMs

Legal judgments are unstructured, narrative documents written in hindsight.

The extraction stage uses an LLM to convert:

Unstructured Legal Text → Structured JSON (ML-ready format)

Each case is decomposed into party-level rows, where each row represents one learning instance.

---

## 3. Party-Level Decomposition

Instead of treating one case as one datapoint:

Each party = one datapoint

This enables:
- Increased dataset size
- More granular supervision
- Better alignment with prediction tasks

Example:
One case with 3 defendants → 3 training samples

---

## 4. Prompting Strategy for Extraction

### 4.1 IRAC-Based Prompting

The extraction prompt follows the legal IRAC framework:

- Facts → observable events and conduct  
- Issue → legal question  
- Rule → governing law  
- Application → reasoning applied  
- Label → final outcome  

This acts as feature engineering at the prompting stage.

---

### 4.2 Schema-Constrained Output

The model is forced to return a strict JSON format:

{
  "Facts": [...],
  "Issue": "...",
  "Rule": "...",
  "Application": "...",
  "Label": "..."
}

Benefits:
- Ensures consistency
- Enables deterministic parsing
- Produces ML-ready data

---

### 4.3 Typed Fact Extraction

Each fact is structured with:
- Fact_Type
- Fact_Date
- Text

This enables:
- Temporal features  
- Event sequencing  
- Structured variables  

---

## 5. Feature Engineering Perspective (BT4222)

This stage performs automated feature engineering.

### Transformation Pipeline

Raw Judgment → Structured JSON → Features

---

### 5.1 Types of Features Generated

#### A. Text Features (NLP)
From:
- Facts
- Issue
- Application

Methods:
- TF-IDF
- Embeddings

---

#### B. Structured Features
From facts:
- Contracts
- Roles
- Financial transactions
- Relationships

Derived features:
- Presence of fiduciary duty  
- Number of transactions  
- Contract existence  

---

#### C. Temporal Features
From Fact_Date:
- Event ordering  
- Time gaps  
- Pre-dispute filtering  

---

### 5.2 Target Variable (y)

Role,Permitted Substantive Labels
Plaintiff,"Claim Allowed, Claim Dismissed, Unknown"
Defendant,"Liable, Not Liable, Partially Liable, Unknown"
Third Party / Procedural,Unknown (unless a substantive outcome is determined)
---

## 6. Limitations of Initial Extraction

### 6.1 Imperfect Feature/Target Separation

The extraction attempts separation but cannot fully guarantee it.

---

### 6.2 Data Leakage Risk

Legal judgments include hindsight information such as:
- Judicial reasoning  
- Verdict language  
- Trial evidence  

Examples:
- "The court held..."
- "The defendant was liable..."

---

### 6.3 Impact on ML Models

If leakage is present:
- Model learns shortcut patterns  
- Training accuracy becomes artificially high  
- Real-world performance drops  

---

## 7. Why Extraction Alone is Not Enough

Extraction provides:
- structured data  
- feature candidates  

But still contains:
- leakage  
- hallucinations  
- inconsistencies  

This motivates the need for an audit stage.

---

## 8. Summary

The extraction stage:
- converts raw judgments into structured JSON  
- decomposes cases into party-level samples  
- applies IRAC-based prompting  
- generates ML-ready features  

However, due to limitations, it requires a second-stage audit pipeline.

---
