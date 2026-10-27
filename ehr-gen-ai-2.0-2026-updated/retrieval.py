"""Query-driven cohort selection and patient similarity for synthetic EHR testing."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd

TABLES = [
    "patients", "admissions", "diagnoses_icd", "prescriptions",
    "medication_administrations", "procedures_icd", "labevents", "chartevents",
    "allergies", "discharge", "radiology", "edstays", "triage",
]

@dataclass
class RetrievalQuery:
    conditions: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    profile_group: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    medication_basis: str = "orders"  # "orders" or "administrations"
    top_k: int = 5

def load_dataset(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    data = {}
    for name in TABLES:
        path = root / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path).fillna("")
    required = {"patients", "admissions", "diagnoses_icd", "prescriptions", "allergies"}
    missing = required - data.keys()
    if missing:
        raise FileNotFoundError(f"Missing required dataset tables: {', '.join(sorted(missing))}")
    return data

def _within_window(frame: pd.DataFrame, date_column: str, query: RetrievalQuery) -> pd.DataFrame:
    if date_column not in frame.columns:
        return frame
    out = frame.copy()
    dates = pd.to_datetime(out[date_column], errors="coerce")
    if query.start_date:
        out = out[dates >= pd.Timestamp(query.start_date)]
        dates = pd.to_datetime(out[date_column], errors="coerce")
    if query.end_date:
        out = out[dates <= pd.Timestamp(query.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
    return out

def _patient_rows(data: dict[str, pd.DataFrame], table: str, pid, query: RetrievalQuery,
                  date_column: str | None = None) -> pd.DataFrame:
    frame = data.get(table, pd.DataFrame())
    if frame.empty or "subject_id" not in frame.columns:
        return frame
    frame = frame[frame["subject_id"].astype(str) == str(pid)]
    if date_column:
        frame = _within_window(frame, date_column, query)
    return frame

def patient_features(data: dict[str, pd.DataFrame], pid, query: RetrievalQuery | None = None) -> set[str]:
    query = query or RetrievalQuery()
    features: set[str] = set()
    for row in _patient_rows(data, "diagnoses_icd", pid, query).to_dict("records"):
        code = str(row.get("icd_code", "")).strip()
        if code and (not query.conditions or any(c.lower() in code.lower() or c.lower() in str(row.get("long_title", "")).lower() for c in query.conditions)):
            features.add("condition:" + code.lower())
    med_table = "medication_administrations" if query.medication_basis == "administrations" else "prescriptions"
    med_date = "charttime" if med_table == "medication_administrations" else "starttime"
    for row in _patient_rows(data, med_table, pid, query, med_date).to_dict("records"):
        med = str(row.get("drug", "")).strip().lower()
        if med and (not query.medications or any(m.lower() in med for m in query.medications)):
            features.add("medication:" + med)
    for table, date_col in [("labevents", "charttime"), ("chartevents", "charttime")]:
        for row in _patient_rows(data, table, pid, query, date_col).to_dict("records"):
            label = str(row.get("label", "")).strip().lower()
            if label and (not query.measurements or any(m.lower() in label for m in query.measurements)):
                features.add("measurement:" + label)
    # Allergy state is represented explicitly. Missing rows are NOT inferred to mean no allergies.
    allergy_rows = _patient_rows(data, "allergies", pid, query)
    if not allergy_rows.empty:
        for row in allergy_rows.to_dict("records"):
            status = str(row.get("status", "")).strip().upper()
            name = str(row.get("allergy_name", "")).strip().lower()
            if status == "KNOWN" and name:
                features.add("allergy:known:" + name)
            elif status in {"NO_KNOWN_ALLERGIES", "UNKNOWN", "NOT_DOCUMENTED"}:
                features.add("allergy_status:" + status.lower())
    return features

def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def rank_historical_patients(data: dict[str, pd.DataFrame], query_patient_id,
                             query: RetrievalQuery | None = None) -> pd.DataFrame:
    query = query or RetrievalQuery()
    patients = data["patients"].copy()
    historical = patients[patients["patient_status"].astype(str).str.upper() == "HISTORICAL"]
    q_features = patient_features(data, query_patient_id, query)
    results = []
    for _, candidate in historical.iterrows():
        pid = candidate["subject_id"]
        features = patient_features(data, pid, query)
        shared = sorted(q_features & features)
        results.append({
            "subject_id": pid,
            "profile_group": candidate.get("profile_group", ""),
            "similarity": jaccard_similarity(q_features, features),
            "shared_features": shared,
            "query_feature_count": len(q_features),
            "candidate_feature_count": len(features),
        })
    if not results:
        return pd.DataFrame(columns=["subject_id", "profile_group", "similarity", "shared_features"])
    return pd.DataFrame(results).sort_values(
        ["similarity", "subject_id"], ascending=[False, True]
    ).head(max(1, int(query.top_k))).reset_index(drop=True)

def select_cohort(data: dict[str, pd.DataFrame], query: RetrievalQuery) -> pd.DataFrame:
    """Select patient IDs matching query constraints; each supplied filter is ANDed."""
    patients = data["patients"].copy()
    if query.profile_group:
        patients = patients[patients["profile_group"].astype(str).str.lower() == query.profile_group.lower()]
    ids = set(patients["subject_id"].astype(str))
    if query.conditions:
        dx = data["diagnoses_icd"].copy()
        dx_text = dx.get("icd_code", "").astype(str) + " " + dx.get("long_title", "").astype(str)
        mask = pd.Series(False, index=dx.index)
        for term in query.conditions:
            mask |= dx_text.str.contains(term, case=False, regex=False, na=False)
        ids &= set(dx.loc[mask, "subject_id"].astype(str))
    if query.medications:
        table_name = "medication_administrations" if query.medication_basis == "administrations" else "prescriptions"
        meds = data.get(table_name, pd.DataFrame())
        if not meds.empty:
            if table_name == "medication_administrations":
                meds = _within_window(meds, "charttime", query)
            else:
                meds = _within_window(meds, "starttime", query)
            mask = pd.Series(False, index=meds.index)
            for term in query.medications:
                mask |= meds["drug"].astype(str).str.contains(term, case=False, regex=False, na=False)
            ids &= set(meds.loc[mask, "subject_id"].astype(str))
        else:
            ids = set()
    if query.measurements:
        matched = set()
        for table in ("labevents", "chartevents"):
            frame = data.get(table, pd.DataFrame())
            if frame.empty:
                continue
            frame = _within_window(frame, "charttime", query)
            mask = pd.Series(False, index=frame.index)
            for term in query.measurements:
                mask |= frame["label"].astype(str).str.contains(term, case=False, regex=False, na=False)
            matched |= set(frame.loc[mask, "subject_id"].astype(str))
        ids &= matched
    return patients[patients["subject_id"].astype(str).isin(ids)].sort_values("subject_id").reset_index(drop=True)

def validate_cohort_against_pandas(data: dict[str, pd.DataFrame], query: RetrievalQuery,
                                   returned: pd.DataFrame) -> dict:
    """Independent, simple pandas baseline for a profile-only cohort check."""
    if query.conditions or query.medications or query.measurements or query.start_date or query.end_date:
        return {"validated": False, "reason": "Reference validator currently supports profile-only queries."}
    expected = data["patients"]
    if query.profile_group:
        expected = expected[expected["profile_group"].astype(str).str.lower() == query.profile_group.lower()]
    expected_ids = set(expected["subject_id"].astype(str))
    actual_ids = set(returned["subject_id"].astype(str))
    return {"validated": expected_ids == actual_ids, "expected_count": len(expected_ids),
            "actual_count": len(actual_ids), "missing_ids": sorted(expected_ids - actual_ids),
            "unexpected_ids": sorted(actual_ids - expected_ids)}
