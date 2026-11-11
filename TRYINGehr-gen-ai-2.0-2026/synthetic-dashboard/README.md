# EHR Gen AI 2.0 — patient comparison prototype

A synthetic-data prototype for query-driven cohort selection, patient similarity, and record review.

> **Safety and scope:** The included dataset is fictional and MIMIC-inspired. It is not official MIMIC-IV, is not the OMOP CDM, and must not be used for clinical decisions. This prototype is a software baseline, not a validated clinical system.

## What changed in this revision

- Allergy information distinguishes `KNOWN`, `NO_KNOWN_ALLERGIES`, `UNKNOWN`, and `NOT_DOCUMENTED`. A missing row is not treated as a negative allergy history.
- Medication orders (`prescriptions.csv`) and administration events (`medication_administrations.csv`) are separate tables.
- Admissions have explicit `outcome_status` and `outcome_recorded` fields. A discharge destination is not interpreted as successful treatment; current arrivals remain `PENDING`.
- Query-driven cohort filters support conditions, medications, measurements, profile groups, and date windows.
- Medication-based similarity can use orders or administrations; Jaccard and IDF-weighted Jaccard scores are shown together.
- Patient comparison is organized into color-coded tabs for conditions, measurements, allergies, and medications, with the selected new and historical patients shown side by side.
- Every historical patient has an explicit final coded diagnosis linked to that encounter's diagnosis record. The historical view includes a styled timeline of assessments, procedure and medication events, simulated response checks, complications, and disposition.
- Synthetic response statuses and rationale notes are mock scenario labels only. Causality and actual treatment effectiveness are not established, and the dashboard does not produce treatment recommendations or plans.
- The independent pandas baseline checks all supported combinations of profile, condition, medication, measurement, and date-window filters.
- A hand-specified synthetic pair is used to compare Jaccard and IDF rankings. The benchmark also reports condition and medication diversity for new patients.
- `evaluate.py` runs all 20 questions in `eval_questions.json` and writes `EVALUATION_REPORT.md` and `evaluation_results.json`, including expected/retrieved IDs and supporting record samples.
- Cohort filters are unranked; their P@5/R@5 uses the first five IDs in ascending ID order and is reported separately from the known-pair ranking metrics.
- The add-review-item dialog is synthetic-only, session-only, manually entered, and not a clinical recommendation or a write to patient records.
- The Streamlit UI is kept as a standalone baseline. A separate adapter to another dashboard should be added only after confirming the target dashboard's inputs, outputs, and interface.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
python generate_synthetic_data.py
python -m unittest discover -s tests -v
python evaluate.py
streamlit run app.py
```

## GitHub upload

Add the project files to the repository and commit them. Do not upload real patient data, credentials, `.env` files, or virtual environments. The generated `synthetic_mimic_100/` directory is ignored by `.gitignore` and can be recreated with the generator.

Suggested commit message:

`feat: add query-driven synthetic EHR comparison baseline`

## Known limitations to address next

1. The query parser is currently a structured UI, not a natural-language clinical question parser.
2. Condition records in this synthetic schema do not have their own event timestamp; date windows are applied to dated medication and measurement evidence.
3. There is no converter or agentic-search implementation in this workspace or project archive, so this repository cannot run an end-to-end converter/agent test. No OMOP adapter is included; its need depends on the missing pipeline's input contract.
4. No RxNorm, SNOMED CT, or LOINC vocabulary resources or MIMIC-IV demo dataset are bundled or configured locally. Confirm authorized resources and intended vocabulary mappings before integrating them.
5. The benchmark evaluates a deterministic synthetic structured-search baseline, not natural-language question answering. Its P@5/R@5 scores use explicit query fixtures and relevant patient IDs; results do not generalize to a clinical retrieval system.
6. Local Streamlit is verified. The supplied hosted Streamlit URL redirects unauthenticated access to Streamlit auth, so its app content could not be verified in this session.
7. Historical records do not include a clinician-authored treatment plan. Generated course notes, diagnoses, orders, administrations, procedures, rationales, response statuses, and outcomes are fictional/descriptive context only; they do not establish treatment effectiveness or causality. This software does not generate care recommendations for a new patient.
