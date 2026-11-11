"""Convert an OMOP CDM extract into the flat format LA-CDM expects.

LA-CDM's `MIMICCDMDataset` reads three CSVs (train/val/test) where every row is
one patient and every column is one *test* the agent may request, plus a
`lab_test_mapping.csv` that maps a test label to the set of source identifiers
that count as that test.

This script produces exactly that from OMOP `measurement`, `procedure_occurrence`
and `note` tables. Nothing here is MIMIC-specific: the only inputs are the OMOP
tables plus two config files you control (`test_costs.csv`, and the panel
definition block below).

Output columns, matching the paper's loader:
    Patient ID, Patient History Summary, <test 1..n>, Label
"""
from __future__ import annotations

import argparse
import ast
import os
import re
from collections import defaultdict

import pandas as pd

# --------------------------------------------------------------------------
# Test catalogue.
#
# A "test" is a named, requestable bundle of results. In OMOP we identify
# results by (concept_id, concept_name) in `measurement_concept_map`, or by
# `procedure_concept_name` in `procedure_occurrence`. Anything with an empty
# result string is reported to the agent as "not available." -- exactly what
# the paper's environment does when a test was never performed.
#
# `regex` is matched case-insensitively against concept names. Order matters:
# the first match wins, so put specific panels before generic ones.
# --------------------------------------------------------------------------
TEST_CATALOG = [
    # name,                     regex on concept name / procedure name
    ("Hemoglobin A1c",          r"hba1c|hemoglobin a1c|glycohemoglobin|glycated hemoglobin"),
    ("Lipid Panel",             r"lipid panel|lipid profile|cholesterol.*triglyceride|^lipid"),
    ("Renal Function Panel",    r"renal function panel|renal function|creatinine.*bun|basic metabolic panel|^\bbmp\b"),
    ("Liver Function Panel",    r"hepatic function|liver function|liver panel|comprehensive metabolic panel|^\bcmp\b"),
    ("Complete Blood Count",    r"complete blood count|cbc|complete blood"),
    ("Urinalysis",              r"urinalysis|urine"),
    ("CT",                      r"\bct\b|computed tomography|cat scan"),
    ("MRI",                     r"\bmri\b|magnetic resonance|mr "),
    ("Ultrasound",              r"ultrasound|ultrasonography|doppler|echograph"),
    ("Radiograph",              r"radiograph|x-?ray|plain film"),
    ("Physical Examination",     r"physical exam|physical therapy note|admission note|history and physical"),
]

RESULT_COL = "result_value"
NUMERIC = re.compile(r"^-?\d+(\.\d+)?")


def _fmt_value(value: float | str, unit: str | None) -> str:
    """Render one measurement the way the paper renders lab values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if NUMERIC.match(text):
        number = float(text)
        text = f"{number:g}"
        if unit:
            text = f"{text} {unit}"
    return text


def load_catalog(catalog_path: str) -> list[tuple[str, str]]:
    if catalog_path:
        rows = []
        with open(catalog_path) as handle:
            header = handle.readline()
            if "regex" in header:
                handle.seek(0)
                frame = pd.read_csv(handle, dtype=str).fillna("")
            else:
                frame = pd.DataFrame(
                    [line.rstrip("\n") for line in handle if line.strip()],
                    columns=["name", "regex"],
                )
        for _, row in frame.iterrows():
            rows.append((row["name"], str(row["regex"])))
        return rows
    return TEST_CATALOG


def build_test_cost_map(costs_path: str) -> dict[str, float]:
    """Costs are normalised to sum to 1.0 (the paper scales them so that
    requesting every test exactly cancels a correct-diagnosis reward)."""
    frame = pd.read_csv(costs_path)
    raw = dict(zip(frame["test_name"], frame["cost_usd"].astype(float)))
    total = sum(raw.values())
    return {name: cost / total for name, cost in raw.items()}


def _first_per_test(group: pd.DataFrame) -> str:
    """Paper detail: if a test was recorded several times during the stay only
    the first value is kept, to simulate early diagnosis after admission."""
    group = group.sort_values("_ord")
    parts = []
    for _, row in group.iterrows():
        value = _fmt_value(row[RESULT_COL], row.get("unit_source_value"))
        if not value:
            continue
        label = row.get("concept_name") or ""
        parts.append(f"{label}: {value}" if label else value)
    if not parts:
        return ""
    return ", ".join(parts) + "\n"


def load_measurements(data_dir: str, catalog: list[tuple[str, str]]) -> pd.DataFrame:
    """One row per (patient, test)."""
    concept_map = pd.read_csv(os.path.join(data_dir, "measurement_concept_map.csv"))
    concept_map["concept_name"] = concept_map["concept_name"].fillna("")
    # OMOP's measurement_concept_map keys on `concept_id`; measurement.csv
    # refers to it via `measurement_concept_id`. Rename to line the merge up.
    concept_map = concept_map.rename(columns={"concept_id": "measurement_concept_id"})
    measurement = pd.read_csv(
        os.path.join(data_dir, "measurement.csv"),
        usecols=[
            "person_id",
            "measurement_concept_id",
            "measurement_date",
            "value_as_number",
            "value_as_string",
            "unit_source_value",
        ],
    )
    measurement = measurement.merge(concept_map, on="measurement_concept_id", how="left")
    measurement[RESULT_COL] = measurement["value_as_number"].combine_first(
        measurement["value_as_string"]
    )

    compiled = [(name, re.compile(rgx, re.I)) for name, rgx in catalog]
    measurement["test_name"] = None
    for name, pattern in compiled:
        hit = measurement["test_name"].isna() & measurement["concept_name"].str.contains(pattern)
        measurement.loc[hit, "test_name"] = name

    measurement = measurement[measurement["test_name"].notna()].copy()
    measurement["_ord"] = measurement.groupby(
        ["person_id", "test_name"]
    ).cumcount()
    measurement = measurement.sort_values(["person_id", "test_name", "_ord"])

    per_test = (
        measurement.groupby(["person_id", "test_name"])
        .apply(_first_per_test, include_groups=False)
        .rename("result")
        .reset_index()
    )
    return per_test[per_test["result"].str.len() > 0]


def load_procedures(data_dir: str, catalog: list[tuple[str, str]]) -> pd.DataFrame:
    """Imaging and procedures act as requestable tests whose 'result' is the
    procedure/visit record. Synthea has no radiology report text, so we record
    what was done and when -- that is a real limitation, flagged in the README.
    """
    path = os.path.join(data_dir, "procedure_occurrence.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["person_id", "test_name", "result"])
    procedures = pd.read_csv(
        path,
        usecols=["person_id", "procedure_concept_name", "procedure_date"],
    )
    procedures["procedure_concept_name"] = procedures["procedure_concept_name"].fillna("")

    compiled = [(name, re.compile(rgx, re.I)) for name, rgx in catalog]
    procedures["test_name"] = None
    for name, pattern in compiled:
        hit = procedures["test_name"].isna() & procedures[
            "procedure_concept_name"
        ].str.contains(pattern)
        procedures.loc[hit, "test_name"] = name

    procedures = procedures[procedures["test_name"].notna()].copy()
    procedures["_ord"] = procedures.groupby(["person_id", "test_name"]).cumcount()
    procedures = procedures.sort_values(["person_id", "test_name", "_ord"])

    def render(group: pd.DataFrame) -> str:
        names = ", ".join(dict.fromkeys(group["procedure_concept_name"]))
        return f"{names} (first recorded {group['procedure_date'].iloc[0][:10]})\n"

    return (
        procedures.groupby(["person_id", "test_name"])
        .apply(render, include_groups=False)
        .rename("result")
        .reset_index()
    )


def _label_series(frame: pd.DataFrame) -> pd.Series:
    """Prefer the human-readable concept name; some OMOP extracts (and
    Synthea's condition table in certain versions) only carry IDs."""
    name = frame["condition_concept_name"]
    fallback = frame["condition_concept_id"].astype(str)
    return name.where(name.notna() & (name.astype(str).str.strip() != ""), fallback)


def load_history(data_dir: str) -> pd.DataFrame:
    """Initial patient state p0. Preference order:
    1. first clinical note (NOTE table), 2. observation/condition summary.
    """
    note_path = os.path.join(data_dir, "note.csv")
    if os.path.exists(note_path):
        notes = pd.read_csv(
            note_path, usecols=["person_id", "note_date", "note_text"]
        ).sort_values("note_date")
        return (
            notes.groupby("person_id")
            .first()
            .reset_index()[["person_id", "note_text"]]
            .rename(columns={"note_text": "Patient History Summary"})
        )

    # Fallback: assemble a structured history from conditions + demographics.
    condition = pd.read_csv(
        os.path.join(data_dir, "condition_occurrence.csv"),
        usecols=[
            "person_id",
            "condition_concept_name",
            "condition_concept_id",
            "condition_start_date",
        ],
    ).dropna(subset=["condition_concept_name"])
    condition["label"] = _label_series(condition).astype(str)
    history = condition.groupby("person_id")["label"].apply(
        lambda names: "History: " + "; ".join(dict.fromkeys(names))
    )
    return history.rename("Patient History Summary").reset_index()


def load_labels(data_dir: str, min_count: int, max_classes: int | None) -> pd.DataFrame:
    """Index diagnosis = most recent condition concept, restricted to classes
    with at least `min_count` patients (paper: 4 diseases, 2,400 patients)."""
    condition = pd.read_csv(
        os.path.join(data_dir, "condition_occurrence.csv"),
        usecols=[
            "person_id",
            "condition_concept_name",
            "condition_concept_id",
            "condition_start_date",
        ],
    ).dropna(subset=["condition_concept_name"])
    condition["label"] = _label_series(condition).astype(str)
    latest = (
        condition.sort_values("condition_start_date")
        .groupby("person_id")
        .last()
        .reset_index()[["person_id", "label"]]
        .rename(columns={"label": "Label"})
    )
    counts = latest["Label"].value_counts()
    keep = counts[counts >= min_count].index.tolist()
    if max_classes:
        keep = counts[counts >= min_count].head(max_classes).index.tolist()
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"[labels] excluded {len(dropped)} classes below/over the cut")
    return latest[latest["Label"].isin(keep)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="folder of OMOP CSVs")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--test_catalog", default="", help="CSV of name,regex")
    parser.add_argument("--costs", default="test_costs.csv")
    parser.add_argument("--min_count", type=int, default=200)
    parser.add_argument("--max_classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=269)
    args = parser.parse_args()

    catalog = load_catalog(args.test_catalog)
    tests = [name for name, _ in catalog]
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/6] loading labels ...")
    labels = load_labels(args.data_dir, args.min_count, args.max_classes)
    print(f"      {labels['Label'].nunique()} classes: "
          f"{sorted(labels['Label'].unique())}")

    print("[2/6] loading measurement-derived tests ...")
    measurements = load_measurements(args.data_dir, catalog)
    print("[3/6] loading procedure/imaging-derived tests ...")
    procedures = load_procedures(args.data_dir, catalog)
    results = pd.concat([measurements, procedures], ignore_index=True)
    print(f"      {len(results)} (patient, test) result cells")

    print("[4/6] loading patient history ...")
    history = load_history(args.data_dir)

    # pivot measurements -> one row per patient, one column per test
    pivot = results.pivot_table(
        index="person_id", columns="test_name", values="result", aggfunc="first"
    ).reset_index()
    data = pivot.merge(history, on="person_id", how="left")
    data = data.merge(labels, on="person_id", how="inner")
    data["Patient History Summary"] = data["Patient History Summary"].fillna(
        "No summary available."
    )

    for test in tests:
        if test not in data.columns:
            data[test] = ""
        data[test] = data[test].fillna("not available.\n")

    data = data.rename(columns={"person_id": "Patient ID"})
    data = data[["Patient ID", "Patient History Summary"] + tests + ["Label"]]
    print(f"      {len(data)} patients, {len(tests)} requestable tests")

    print("[5/6] stratified 80/10/10 split (seed %d) ..." % args.seed)
    labels_only = data["Label"]
    buckets = {name: [] for name in ("train", "val", "test")}
    for _, group in data.groupby("Label"):
        # Stratified 80/10/10, mirroring the paper's split_dataset.py.
        shuffled = group.sample(frac=1.0, random_state=args.seed)
        n = len(shuffled)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        cut1, cut2 = n_train, n_train + n_val
        buckets["train"].extend(shuffled.index[:cut1])
        buckets["val"].extend(shuffled.index[cut1:cut2])
        buckets["test"].extend(shuffled.index[cut2:])

    for split, idx in buckets.items():
        subset = data.loc[sorted(idx)].reset_index(drop=True)
        subset.to_csv(os.path.join(args.output_dir, f"{split}.csv"), index=False)
        print(f"      {split}: {len(subset)} patients")

    print("[6/6] writing lab_test_mapping.csv and test_costs.csv ...")
    mapping = pd.DataFrame(
        {
            "label": tests,
            "corresponding_ids": [
                str([name]) for name in tests
            ],
        }
    )
    # The loader resolves ids through lab_test_mapping['itemid']; supply a
    # matching itemid column so the resolution path is identical to the paper's.
    mapping["itemid"] = range(1000, 1000 + len(tests))
    mapping.to_csv(os.path.join(args.output_dir, "lab_test_mapping.csv"), index=False)

    cost_map = build_test_cost_map(args.costs)
    missing = [t for t in tests if t not in cost_map]
    if missing:
        print(f"      WARNING no cost for {missing}; test-cost reward will skip them")
    pd.DataFrame(
        {"label": tests, "cost_usd": [cost_map.get(t, 0.0) for t in tests]}
    ).to_csv(os.path.join(args.output_dir, "test_costs.csv"), index=False)

    with open(os.path.join(args.output_dir, "class_list.txt"), "w") as handle:
        handle.write("\n".join(sorted(data["Label"].unique())))

    print("done ->", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()