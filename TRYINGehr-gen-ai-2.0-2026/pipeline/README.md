# OMOP → LA-CDM adapter

Everything needed to run the LA-CDM replication (Bani-Harouni et al., ICLR 2026,
arXiv:2506.13474) on **OMOP CDM data** instead of MIMIC-CDM.

The paper's method, in one line: two agents share one LLM — a **hypothesis
agent** that emits `Hypothesis: X, Confidence: 0-10`, and a **decision agent**
that ReAct-style-picks either `Test: <name>` or `Diagnosis: <condition>` — and
the environment reveals only the tests the agent asks for, replying
`not available.` when the patient never had it. Training is a 3-phase cycle:
decision-agent RL (GRPO) → hypothesis-agent SFT → confidence-calibration RL
(GRPO), 100 steps each, repeating. Rewards: `r_pos`/`r_neg` for correct/incorrect
diagnosis, `r_invalid` for format violations, minus normalized test cost.

## Why an adapter is the whole job

The paper's pipeline is *not* MIMIC-specific except for one file. `src/`
(environment, GRPO trainer, rewards, metrics) talks to a `Dataset` that returns
`{patient_id, patient_history, condition, test_results}`. So if we emit the same
three CSVs the paper's loader reads, we inherit the entire method unchanged and
only swap the data. That is the portability claim in your milestone report,
made concrete.

## Files

| File | What it is |
|---|---|
| `omop_to_lacdm.py` | OMOP CSVs → `train/val/test.csv` + `lab_test_mapping.csv` + `test_costs.csv` |
| `test_costs.csv` | Per-test cost in USD, from the paper's Appendix Table 4 (BIDMC charges) |
| `make_test_fixture.py` | Generates a tiny OMOP fixture so you can test the converter in 5 seconds |

## Usage

```bash
# 1. point at a folder of OMOP CSV exports
#    needs: measurement.csv, measurement_concept_map.csv,
#           condition_occurrence.csv, procedure_occurrence.csv
#    optional: note.csv (preferred source of patient history)

# 2. convert
python omop_to_lacdm.py \
    --data_dir  /blue/data/.../synthea_omop \
    --output_dir data \
    --costs test_costs.csv \
    --min_count 200 \
    --max_classes 4

# 3. sanity-check before spending GPU hours
python -c "import pandas as pd; d=pd.read_csv('data/train.csv'); print(d.shape); print(d.Label.value_counts())"
```

Then, in a clone of the paper's repo, point the configs at this data:

```yaml
# configs/model/defaults.yaml
model_name: "/blue/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf"

# configs/dataset/defaults.yaml
train_data_file: data/train.csv
val_data_file:   data/val.csv
test_data_file:  data/test.csv
lab_tests:       [Hemoglobin A1c, Lipid Panel, Renal Function Panel, Liver Function Panel, Complete Blood Count, Urinalysis]
imaging_tests:   [CT, MRI, Ultrasound, Radiograph]
other_tests:     [Physical Examination]
disease_list:    [<your 4 OMOP condition names, exactly as in data/*.csv>]
```

Note `disease_list` and the test lists are interpolated straight into the agent
prompts, so they must match the strings in your CSVs character for character.

## What this does and does not preserve

Preserved exactly: the environment mechanics, the two-agent loop, all four
reward functions, GRPO, the cyclic loss schedule, ECE / class-accuracy /
micro-macro-F1 / avg-test-cost metrics.

Changed, unavoidably:

1. **Test names.** The paper's tests are MIMIC lab *panels* (CBC, BMP, CMP…)
   resolved through `lab_test_mapping.csv`. OMOP has no panels — it has
   individual concept-level measurements. `TEST_CATALOG` in the converter maps
   LOINC concept names onto panel-like bundles by regex. Edit that table to
   match what your extract actually contains; it is the single knob that matters
   most.
2. **Imaging has no report text.** MIMIC-CDM gives you radiology *report
   findings*, which is most of what makes imaging informative. Synthea emits
   procedure codes with no report text, so imaging columns record *what was
   done and when*. Expect weaker results on any task that leans on imaging, and
   say so in your writeup rather than letting it look like a negative result.
3. **Test costs are the paper's, not yours.** Appendix Table 4 is BIDMC prices.
   Re-price from your own institution's charge master if you want the cost axis
   to mean anything.
4. **Label definition.** The paper inherits MIMIC-CDM's four-way diagnosis
   label. Here the label is the *most recent* condition concept per patient,
   filtered to classes with ≥ `--min_count` patients. **This is the number to
   sanity-check first** — if Synthea's disease modules produce lopsided
   prevalence, your 4 classes may be trivially separable and every accuracy
   number will be meaningless. Check class balance and per-class feature overlap
   before training anything.

## Suggested order of work

1. Convert 1,000 patients, print class balance + how often each test is
   non-`not available`. If most test columns are empty, your regex catalog is
   wrong — fix before proceeding.
2. Evaluate the **untrained** agent (LA-CDM ZS row of Table 1) on the test set.
   This is your cheapest real result and the anchor every later number compares
   against.
3. SFT the hypothesis agent only, confirm hypothesis accuracy moves.
4. Add the two GRPO phases.