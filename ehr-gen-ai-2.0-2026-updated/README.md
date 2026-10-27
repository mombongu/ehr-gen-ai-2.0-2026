# EHR Gen AI 2.0 — patient comparison prototype

A synthetic-data prototype for query-driven cohort selection, patient similarity, and record review.

> **Safety and scope:** The included dataset is fictional and MIMIC-inspired. It is not official MIMIC-IV, is not the OMOP CDM, and must not be used for clinical decisions. This prototype is a software baseline, not a validated clinical system.

## What changed in this revision

- Allergy information distinguishes `KNOWN`, `NO_KNOWN_ALLERGIES`, `UNKNOWN`, and `NOT_DOCUMENTED`. A missing row is not treated as a negative allergy history.
- Medication orders (`prescriptions.csv`) and administration events (`medication_administrations.csv`) are separate tables.
- Admissions have explicit `outcome_status` and `outcome_recorded` fields. A discharge destination is not interpreted as successful treatment; current arrivals remain `PENDING`.
- Query-driven cohort filters support conditions, medications, measurements, profile groups, and date windows.
- Medication-based similarity can use orders or administrations; condition and measurement features are integrated into one feature-overlap score.
- A pandas reference check validates profile-only cohort membership.
- `tests/test_retrieval.py` contains automated checks; `eval_questions.json` defines 20 representative questions, expected behavior, and supporting tables.
- The Streamlit UI is kept as a standalone baseline. A separate adapter to another dashboard should be added only after confirming the target dashboard's inputs, outputs, and interface.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
python generate_synthetic_data.py
python -m unittest discover -s tests -v
streamlit run app.py
```

## GitHub upload

Add the project files to the repository and commit them. Do not upload real patient data, credentials, `.env` files, or virtual environments. The generated `synthetic_mimic_100/` directory is ignored by `.gitignore` and can be recreated with the generator.

Suggested commit message:

`feat: add query-driven synthetic EHR comparison baseline`

## Known limitations to address next

1. The query parser is currently a structured UI, not a natural-language clinical question parser.
2. Condition records in this synthetic schema do not have their own event timestamp; date windows are applied to dated medication and measurement evidence.
3. The current independent validator fully checks profile-only cohort membership. Extend it with independently written SQL/pandas reference implementations for every filter combination before relying on query results.
4. The evaluation JSON specifies expected behavior and supporting tables; exact answer fixtures should be added for each generated dataset seed and tested automatically.
5. Similarity is simple Jaccard overlap. It is not a clinical relevance score, outcome predictor, or treatment-effectiveness measure.
6. Dashboard integration is intentionally deferred until the baseline is tested and the existing dashboard's contract is known.
