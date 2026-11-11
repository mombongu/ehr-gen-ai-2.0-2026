# Structured Retrieval Evaluation

> Synthetic software test only; not a clinical validation. This evaluates the structured retrieval baseline, not an LLM agent or converter.

- Questions correct: 20/20 (100.0%)
- Macro precision@5: 0.9000 across 16 retrieval questions
- Macro recall@5: 0.3119 across 16 retrieval questions
- Cohort/filter queries: macro P@5 1.0000, R@5 0.2136 across 14 questions
- Known-pair ranking queries: macro P@5 0.2000, R@5 1.0000 across 2 questions

| ID | Expected answer | Correct | P@k | R@k | Expected / retrieved |
|---|---|---:|---:|---:|---:|
| Q01 | Historical patients with at least one of new patient 20001's recorded diagnosis codes. | yes | 1.0000 | 0.0725 | 69 / 69 |
| Q02 | Historical patients with a metformin prescription order. | yes | 1.0000 | 0.2778 | 18 / 18 |
| Q03 | Historical patients with a recorded metformin administration. | yes | 1.0000 | 0.2941 | 17 / 17 |
| Q04 | Patients with a glucose measurement from 2023-07-01 through 2025-06-01. | yes | 1.0000 | 0.2500 | 20 / 20 |
| Q05 | Patients with a creatinine measurement from 2023-07-01 through 2025-06-01. | yes | 1.0000 | 0.0500 | 100 / 100 |
| Q06 | Patients whose profile_group is diabetes. | yes | 1.0000 | 0.2500 | 20 / 20 |
| Q07 | No. An absent allergy row means the status is not documented by this table, not that the patient has no known allergies. | yes | N/A | N/A | 0 / 0 |
| Q08 | Only patients with an explicit UNKNOWN allergy status. | yes | 1.0000 | 0.3846 | 13 / 13 |
| Q09 | Historical admissions with outcome_recorded=1; show outcome_status independently from discharge_location. | yes | 1.0000 | 0.0556 | 90 / 90 |
| Q10 | Current arrivals are PENDING with outcome_recorded=0 until an outcome is recorded. | yes | 1.0000 | 0.5000 | 10 / 10 |
| Q11 | Known similar pair: new patient 20001, historical patient 20011. | yes | 0.2000 | 1.0000 | 1 / 5 |
| Q12 | Known similar pair: new patient 20001, historical patient 20011. | yes | 0.2000 | 1.0000 | 1 / 5 |
| Q13 | Patients satisfying both the E11.9 diagnosis and metformin-order constraints. | yes | 1.0000 | 0.2500 | 20 / 20 |
| Q14 | The renal-profile cohort with membership checked against the independent pandas baseline. | yes | 1.0000 | 0.2500 | 20 / 20 |
| Q15 | No matching cohort or feature set; show an explicit empty-result state. | yes | N/A | N/A | 0 / 0 |
| Q16 | At most 5 ranked candidates, all drawn from the historical cohort. | yes | 1.0000 | 0.0556 | 90 / 5 |
| Q17 | Patients with an oxygen-saturation measurement in lab or chart events, with units and timestamps in supporting rows. | yes | 1.0000 | 0.0500 | 100 / 100 |
| Q18 | Shared feature evidence for 20001 and 20011, plus the score recomputed from the feature sets. | yes | N/A | N/A | 1 / 1 |
| Q19 | No. Discharge destination does not establish a successful outcome; use the explicit outcome_status. | yes | N/A | N/A | 15 / 15 |
| Q20 | Medication and measurement evidence is restricted to the inclusive date window; undated diagnoses remain unfiltered. | yes | 1.0000 | 0.2500 | 20 / 20 |

## New-patient diversity

- Patients: 10
- Conditions: {'distinct_across_new_patients': 20, 'mean_distinct_per_patient': 4.1, 'minimum_distinct_per_patient': 3, 'maximum_distinct_per_patient': 5}
- Medication orders: {'distinct_across_new_patients': 22, 'mean_distinct_per_patient': 5.5, 'minimum_distinct_per_patient': 4, 'maximum_distinct_per_patient': 6}

## Known-pair ranking comparison

| Question | Evidence basis | Method | Pair rank | P@k | R@k |
|---|---|---|---:|---:|---:|
| Q11 | orders | jaccard | 1 | 0.2000 | 1.0000 |
| Q11 | orders | idf | 1 | 0.2000 | 1.0000 |
| Q12 | administrations | jaccard | 1 | 0.2000 | 1.0000 |
| Q12 | administrations | idf | 1 | 0.2000 | 1.0000 |

Supporting expected and retrieved patient IDs, evidence-row samples, and per-question checks are in [evaluation_results.json](./evaluation_results.json).

For cohort/filter questions, P@k and R@k use the first k returned IDs in ascending subject-ID order because this baseline does not rank cohort results. These values measure exact set retrieval at a cutoff, not ranking quality. Only Q11 and Q12 are ranked similarity evaluations.

IDF weighting left the hand-specified pair at rank 1 for both order and administration evidence (the same as unweighted Jaccard). It changed some other top-five candidates, but one known pair is not enough to conclude that IDF improves retrieval.

## Scope limits

- `precision@k` and `recall@k` measure patient-ID retrieval against source-table predicates and the explicit known-pair fixture; conceptual questions are assessed with correctness checks and have no ranked-list metric.
- Questions with an empty expected result or non-ranking answer use N/A for precision/recall.
- Local Streamlit responded with HTTP 200 and the add-review-item dialog was submitted successfully. The supplied hosted URL redirected unauthenticated access with HTTP 303 to Streamlit auth, so its app content could not be verified in this session.
- The repository contains no converter, agentic-search implementation, OMOP adapter, or RxNorm/SNOMED CT/LOINC vocabulary package. The results do not certify those components.
- Synthetic diagnosis/event and treatment-plan limitations remain; this is not clinical decision support.
