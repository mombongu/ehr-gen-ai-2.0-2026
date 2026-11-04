"""Query-driven cohort selection and patient similarity for synthetic EHR testing."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
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
    similarity_method: str = "jaccard"  # "jaccard" or "idf"
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
    constrained_categories = bool(
        query.conditions or query.medications or query.measurements
    )
    if not constrained_categories or query.conditions:
        for row in _patient_rows(data, "diagnoses_icd", pid, query).to_dict("records"):
            code = str(row.get("icd_code", "")).strip()
            if code and (
                not query.conditions
                or any(
                    term.lower() in code.lower()
                    or term.lower() in str(row.get("long_title", "")).lower()
                    for term in query.conditions
                )
            ):
                features.add("condition:" + code.lower())
    if not constrained_categories or query.medications:
        med_table = (
            "medication_administrations"
            if query.medication_basis == "administrations"
            else "prescriptions"
        )
        med_date = "charttime" if med_table == "medication_administrations" else "starttime"
        for row in _patient_rows(data, med_table, pid, query, med_date).to_dict("records"):
            med = str(row.get("drug", "")).strip().lower()
            if med and (
                not query.medications
                or any(term.lower() in med for term in query.medications)
            ):
                features.add("medication:" + med)
    if not constrained_categories or query.measurements:
        for table, date_col in [("labevents", "charttime"), ("chartevents", "charttime")]:
            for row in _patient_rows(data, table, pid, query, date_col).to_dict("records"):
                label = str(row.get("label", "")).strip().lower()
                if label and (
                    not query.measurements
                    or any(term.lower() in label for term in query.measurements)
                ):
                    features.add("measurement:" + label)
    if not constrained_categories:
        # Allergy state is explicit. Missing rows are NOT inferred as a negative history.
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


def idf_weights(
    data: dict[str, pd.DataFrame], query: RetrievalQuery | None = None
) -> dict[str, float]:
    """Return smoothed inverse document-frequency weights across patients."""
    patients = data["patients"]
    feature_query = query or RetrievalQuery()
    documents = [
        patient_features(data, pid, feature_query)
        for pid in patients["subject_id"].tolist()
    ]
    document_count = len(documents)
    if not document_count:
        return {}
    frequencies: dict[str, int] = {}
    for features in documents:
        for feature in features:
            frequencies[feature] = frequencies.get(feature, 0) + 1
    return {
        feature: math.log((1 + document_count) / (1 + frequency)) + 1
        for feature, frequency in frequencies.items()
    }


def idf_weighted_jaccard(
    a: set[str], b: set[str], weights: dict[str, float]
) -> float:
    """Compute Jaccard overlap weighted by each feature's inverse document frequency."""
    union = a | b
    denominator = sum(weights.get(feature, 1.0) for feature in union)
    if not denominator:
        return 0.0
    numerator = sum(weights.get(feature, 1.0) for feature in a & b)
    return numerator / denominator


def rank_historical_patients(data: dict[str, pd.DataFrame], query_patient_id,
                             query: RetrievalQuery | None = None) -> pd.DataFrame:
    query = query or RetrievalQuery()
    patients = data["patients"].copy()
    historical = patients[patients["patient_status"].astype(str).str.upper() == "HISTORICAL"]
    q_features = patient_features(data, query_patient_id, query)
    weights = idf_weights(data, query)
    results = []
    for _, candidate in historical.iterrows():
        pid = candidate["subject_id"]
        features = patient_features(data, pid, query)
        shared = sorted(q_features & features)
        jaccard = jaccard_similarity(q_features, features)
        idf_score = idf_weighted_jaccard(q_features, features, weights)
        results.append({
            "subject_id": pid,
            "profile_group": candidate.get("profile_group", ""),
            "similarity": idf_score if query.similarity_method == "idf" else jaccard,
            "similarity_jaccard": jaccard,
            "similarity_idf": idf_score,
            "shared_features": shared,
            "query_feature_count": len(q_features),
            "candidate_feature_count": len(features),
        })
    if not results:
        return pd.DataFrame(columns=[
            "subject_id", "profile_group", "similarity", "similarity_jaccard",
            "similarity_idf", "shared_features",
        ])
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
    """Independently recompute supported cohort filters with direct pandas operations."""
    expected = data["patients"]
    if query.profile_group:
        expected = expected[expected["profile_group"].astype(str).str.lower() == query.profile_group.lower()]

    expected_ids = set(expected["subject_id"].astype(str))

    if query.conditions:
        diagnoses = data["diagnoses_icd"]
        codes = diagnoses.get("icd_code", pd.Series("", index=diagnoses.index)).astype(str)
        titles = diagnoses.get("long_title", pd.Series("", index=diagnoses.index)).astype(str)
        matching = pd.Series(False, index=diagnoses.index)
        for term in query.conditions:
            matching |= codes.str.contains(term, case=False, regex=False, na=False)
            matching |= titles.str.contains(term, case=False, regex=False, na=False)
        expected_ids &= set(diagnoses.loc[matching, "subject_id"].astype(str))

    if query.medications:
        medication_table = (
            "medication_administrations"
            if query.medication_basis == "administrations"
            else "prescriptions"
        )
        medications = data.get(medication_table, pd.DataFrame()).copy()
        if not medications.empty:
            date_column = "charttime" if medication_table == "medication_administrations" else "starttime"
            if date_column in medications.columns and (query.start_date or query.end_date):
                dates = pd.to_datetime(medications[date_column], errors="coerce")
                if query.start_date:
                    medications = medications.loc[dates >= pd.Timestamp(query.start_date)]
                    dates = pd.to_datetime(medications[date_column], errors="coerce")
                if query.end_date:
                    upper = pd.Timestamp(query.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    medications = medications.loc[dates <= upper]
            drug_names = medications.get(
                "drug", pd.Series("", index=medications.index)
            ).astype(str)
            matching = pd.Series(False, index=medications.index)
            for term in query.medications:
                matching |= drug_names.str.contains(term, case=False, regex=False, na=False)
            expected_ids &= set(medications.loc[matching, "subject_id"].astype(str))
        else:
            expected_ids = set()

    if query.measurements:
        matching_ids: set[str] = set()
        for table_name in ("labevents", "chartevents"):
            measurements = data.get(table_name, pd.DataFrame()).copy()
            if measurements.empty:
                continue
            if "charttime" in measurements.columns and (query.start_date or query.end_date):
                dates = pd.to_datetime(measurements["charttime"], errors="coerce")
                if query.start_date:
                    measurements = measurements.loc[dates >= pd.Timestamp(query.start_date)]
                    dates = pd.to_datetime(measurements["charttime"], errors="coerce")
                if query.end_date:
                    upper = pd.Timestamp(query.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    measurements = measurements.loc[dates <= upper]
            labels = measurements.get(
                "label", pd.Series("", index=measurements.index)
            ).astype(str)
            matching = pd.Series(False, index=measurements.index)
            for term in query.measurements:
                matching |= labels.str.contains(term, case=False, regex=False, na=False)
            matching_ids |= set(measurements.loc[matching, "subject_id"].astype(str))
        expected_ids &= matching_ids

    actual_ids = set(returned["subject_id"].astype(str))
    return {"validated": expected_ids == actual_ids, "expected_count": len(expected_ids),
            "actual_count": len(actual_ids), "missing_ids": sorted(expected_ids - actual_ids),
            "unexpected_ids": sorted(actual_ids - expected_ids),
            "expected_ids": sorted(expected_ids), "actual_ids": sorted(actual_ids)}
