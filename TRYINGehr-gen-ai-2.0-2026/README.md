# EHR Gen AI 2.0 (2026)

Course project work on an EHR search agent: similar-patient retrieval, an OMOP-to-LA-CDM data pipeline, and a synthetic testbed with known answers.

> All synthetic data here is fictional. Do not commit MIMIC-IV demo rows or any credentialed data to this repo (`.gitignore` blocks the data folders).

## Layout

| Folder | What it is | Data it uses |
|---|---|---|
| `synthetic-dashboard/` | Streamlit patient-similarity app, pandas retrieval (Jaccard and IDF-weighted Jaccard), 20 evaluation questions, unit tests, evaluation report | Generated synthetic MIMIC-inspired CSVs (not OMOP) |
| `pipeline/` | `translate_synthea.py` (Synthea export to OMOP-style CSVs) and `omop_to_lacdm.py` (OMOP-style CSVs to the LA-CDM train/val/test format) | Omer's Synthea export |
| `notebooks/` | `MIMIC_OMOP.ipynb`: MIMIC-IV demo OMOP audit (concept naming, disease-group counts) and early condition-overlap Streamlit apps | MIMIC-IV demo OMOP (credentialed; not included) |
| `omop-agent/` | Omer's single-patient OMOP search agent (Llama, Flask, React). Added as a git submodule, see below | MIMIC-IV demo OMOP parquet |
| `archive/` | First version of the synthetic generator, kept for reference | n/a |

## Add Omer's repo as a submodule

Run once from the repo root, so his code and history stay his:

```bash
git submodule add https://github.com/OmerKahveci03/OMOP-EHR-Agent omop-agent
git commit -m "Add omop-agent submodule"
```

## Run the synthetic dashboard

```bash
cd synthetic-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_synthetic_data.py
python -m unittest discover -s tests -v
python evaluate.py
streamlit run app.py
```

## Run the LA-CDM pipeline

See `pipeline/README.md`. Before running it, note:

- `omop_to_lacdm.py` needs `test_costs.csv` (not in this repo yet; add it to `pipeline/`).
- Its default `--min_count 200` will drop every class on a small Synthea export (for example, 328 patients with 4 classes). Lower it, e.g. `--min_count 10`.
- It reads `measurement_concept_map.csv` and `*_concept_name` columns, which `translate_synthea.py` creates. The raw MIMIC-IV demo does not have these (it uses concept IDs plus `2b_concept.csv`), so it needs a small mapping step first.

## How the parts connect (planned)

1. Cross-patient similarity (Jaccard, IDF-weighted, concept filters) on OMOP columns, to feed evidence into the search agent.
2. Evidence flags for reliability (truncated rows, unmapped concept IDs) for a critic agent.
3. Evaluation in LA-CDM terms: diagnosis accuracy, tests ordered, test cost, with retrieval precision and recall at k as a secondary check.
