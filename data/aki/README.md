# AKI Biomarker Dataset - WITH VERIFIED AKI DIAGNOSIS

12.09 from /Users/yangh/Desktop/LLM-RPGN/MIMIc-IV3.1/version1209_aki_verified

## Dataset Description
High-quality biomarker panel for middle-aged patients with CONFIRMED acute kidney injury (AKI).

## CRITICAL DIFFERENCE FROM PREVIOUS VERSION
⚠️ This dataset includes ONLY patients with verified AKI diagnosis using:
- ICD-10 diagnosis codes (N17.* - Acute kidney failure), OR
- ICD-9 diagnosis codes (584.* equivalents), OR
- KDIGO clinical criteria (creatinine-based)

## AKI Identification Methods

### 1. ICD Diagnosis Codes
- ICD-10: N17.0 through N17.9 (all acute kidney failure codes)
- ICD-9: 584.x codes (acute kidney failure)

### 2. KDIGO Clinical Criteria
Based on serum creatinine changes:
- ≥0.3 mg/dL increase within 48 hours, OR
- ≥1.5× baseline creatinine within 7 days

## Selection Criteria

### Patient Selection
- **Age range**: 50-64 years at admission
- **AKI diagnosis**: CONFIRMED by ICD codes and/or KDIGO criteria
- **Time window**: First 168 hours (7 days) from admission
- **Minimum timepoints**: ≥20 unique timepoints per admission

### Timepoint Quality Requirements
Each timepoint must have **≥2 of the following 3 core biomarkers**:
- Creatinine
- BUN (Blood Urea Nitrogen)
- Potassium

### Additional Biomarkers (when available)
- Bicarbonate
- Lactate
- pH
- Sodium

## AKI Diagnosis Statistics
- **Admissions**: 353
  - ICD codes only: 66,343
  - KDIGO criteria only: 8,852
  - Both ICD + KDIGO: 6,859

## Dataset Statistics
- **Patients**: 345
- **Admissions**: 353
- **Timepoints**: 8,249
- **Mean age**: 57.4 ± 4.4 years

## Biomarker Coverage
- creatinine: 8,043 measurements (97.5%)
- bun: 8,173 measurements (99.1%)
- potassium: 6,794 measurements (82.4%)
- bicarbonate: 11 measurements (0.1%)
- lactate: 284 measurements (3.4%)
- ph: 360 measurements (4.4%)
- sodium: 6,210 measurements (75.3%)

## Quality Assurance
✅ ALL patients have CONFIRMED AKI diagnosis (ICD and/or KDIGO)
✅ ALL timepoints have ≥2 core biomarkers present
✅ ALL admissions have ≥20 unique timepoints
✅ Only middle-aged patients (50-64 years)
✅ Only measurements within first 168 hours

## Files
- `aki_biomarkers_wide_verified.csv`: Full wide format with demographics and AKI flags
- `aki_biomarkers_simple_verified.csv`: Simplified (IDs + time + biomarkers + AKI method)
- `patients_aki_verified.csv`: Patient demographics
- `admissions_aki_verified.csv`: Admission details with AKI classification
- `aki_diagnosis_details.csv`: Detailed AKI diagnosis information (ICD codes, KDIGO criteria)
- `summary_aki_verified.csv`: Per-admission summary statistics
- `patient_summary_aki_verified.csv`: Per-patient summary statistics
- `README.txt`: This file

## Data Format
Wide format: each row = one timepoint
- One row per unique (subject_id, hadm_id, hours_from_admission) combination
- Columns: IDs, time variables, biomarker measurements, demographics, AKI flags

## AKI Classification Columns
- `aki_icd`: Boolean - AKI diagnosed by ICD codes
- `aki_kdigo`: Boolean - AKI diagnosed by KDIGO criteria
- `aki_method`: String - 'ICD', 'KDIGO', or 'Both'

## Extraction Date
2025-12-09 18:39:40

## Source
MIMIC-IV v3.1
Input directory: /Users/yangh/Desktop/LLM-RPGN/MIMIc-IV3.1/all_uncompressed/
Output directory: /Users/yangh/Desktop/LLM-RPGN/MIMIc-IV3.1/version1209_aki_verified/

## Why This Matters for ODE Modeling
This dataset contains ACTUAL AKI progression, not just kidney-related biomarkers.
The ODE systems will model true disease dynamics rather than normal kidney function.
