"""Streamlit UI for a synthetic EHR patient similarity prototype."""
from pathlib import Path

import pandas as pd
import streamlit as st

from retrieval import (
    RetrievalQuery,
    load_dataset,
    patient_features,
    rank_historical_patients,
    select_cohort,
    validate_cohort_against_pandas,
)

DATA = Path(__file__).parent / "synthetic_mimic_100"
st.set_page_config(page_title="Patient Similarity Explorer", layout="wide")

ITEM_CATEGORIES = ["Medication", "Procedure", "Treatment"]
ITEM_STATUSES = ["Draft", "Reviewed", "Entered in source system"]


@st.dialog("Add manually entered review item")
def add_review_item_dialog():
    st.caption(
        "For synthetic software testing only. This records your entry in this browser session; "
        "it is not a recommendation and is not saved to a patient record."
    )
    with st.form("add_review_item_form"):
        category = st.selectbox("Category", ITEM_CATEGORIES)
        name = st.text_input("Item name")
        status = st.selectbox("Review status", ITEM_STATUSES)
        notes = st.text_area("Notes", placeholder="Optional context")
        submitted = st.form_submit_button("Add item")
    if submitted:
        if not name.strip():
            st.warning("Enter an item name.")
            return
        items = st.session_state.get("review_items", []).copy()
        items.append({
            "Category": category,
            "Item": name.strip(),
            "Review status": status,
            "Notes": notes.strip(),
        })
        st.session_state["review_items"] = items
        st.rerun()


def format_list(values):
    values = [str(v).strip() for v in values if str(v).strip()]
    return values or ["None recorded"]


def _patient_records(frame, patient_id, date_column=None, query=None):
    if frame.empty or "subject_id" not in frame.columns:
        return frame
    records = frame[frame["subject_id"].astype(str) == str(patient_id)]
    if date_column and date_column in records.columns and query:
        dates = pd.to_datetime(records[date_column], errors="coerce")
        if query.start_date:
            records = records[dates >= pd.Timestamp(query.start_date)]
            dates = pd.to_datetime(records[date_column], errors="coerce")
        if query.end_date:
            end = pd.Timestamp(query.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            records = records[dates <= end]
    return records


def _shared_feature_value(data, patient_id, feature, query):
    category, _, value = feature.partition(":")
    if category == "condition":
        rows = _patient_records(data["diagnoses_icd"], patient_id)
        rows = rows[rows["icd_code"].astype(str).str.lower() == value]
        return "; ".join(
            f"{row['icd_code']} — {row.get('long_title', '')}".strip(" —")
            for row in rows.to_dict("records")
        ) or "No matching diagnosis record"

    if category == "medication":
        table = (
            "medication_administrations"
            if query.medication_basis == "administrations"
            else "prescriptions"
        )
        date_column = "charttime" if table == "medication_administrations" else "starttime"
        rows = _patient_records(data.get(table, pd.DataFrame()), patient_id, date_column, query)
        if not rows.empty:
            rows = rows[rows["drug"].astype(str).str.lower() == value]
        details = []
        for row in rows.to_dict("records"):
            dose = row.get("dose_administered", row.get("dose_val_rx", ""))
            unit = row.get("dose_unit", row.get("dose_unit_rx", ""))
            detail = " ".join(part for part in (str(dose).strip(), str(unit).strip()) if part)
            fields = [str(row.get("drug", ""))]
            if detail:
                fields.append(detail)
            for column in ("route", "frequency", "administration_status"):
                if str(row.get(column, "")).strip():
                    fields.append(str(row[column]).strip())
            details.append(", ".join(fields))
        return "; ".join(details) or "No matching medication record"

    if category == "measurement":
        details = []
        for table in ("labevents", "chartevents"):
            rows = _patient_records(data.get(table, pd.DataFrame()), patient_id, "charttime", query)
            if rows.empty:
                continue
            rows = rows[rows["label"].astype(str).str.lower() == value]
            for row in rows.to_dict("records"):
                measured = row.get("valuenum", row.get("value", ""))
                unit = row.get("valueuom", row.get("unit", ""))
                details.append(" ".join(part for part in (
                    str(row.get("label", "")),
                    str(measured).strip(),
                    str(unit).strip(),
                ) if part))
        return "; ".join(details) or "No matching measurement record"

    if category in {"allergy", "allergy_status"}:
        rows = _patient_records(data["allergies"], patient_id)
        if category == "allergy":
            _, _, allergy_name = value.partition(":")
            rows = rows[
                (rows["status"].astype(str).str.upper() == "KNOWN")
                & (rows["allergy_name"].astype(str).str.lower() == allergy_name)
            ]
            return "; ".join(
                f"{row['allergy_name']} ({row.get('reaction', '')})".strip(" ()")
                for row in rows.to_dict("records")
            ) or "No matching allergy record"
        rows = rows[rows["status"].astype(str).str.lower() == value]
        return "; ".join(rows["status"].astype(str).tolist()) or "No matching allergy status"

    return "Recorded feature"


if not DATA.exists():
    st.warning("Dataset not found. Run `python generate_synthetic_data.py` first.")
    st.stop()

try:
    data = load_dataset(DATA)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

patients = data["patients"]
new_patients = patients[patients["patient_status"].astype(str).str.upper() == "NEW"]
historical = patients[patients["patient_status"].astype(str).str.upper() == "HISTORICAL"]

st.markdown(
    """
    <style>
        :root {
            --sand: #cdc5b4;
            --rose: #B59DA4;
            --mocha: #6D3D14;
            --espresso: #551B14;
            --text: #2f1d17;
            --muted: rgba(47, 29, 23, 0.78);
            --shadow: 0 18px 38px rgba(85, 27, 20, 0.16);
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, var(--sand) 0%, var(--rose) 100%);
            color: var(--text);
        }

        .stApp { background: transparent; }
        .main .block-container { padding-top: 1.3rem; padding-bottom: 2.5rem; }
        h1, h2, h3, h4 { color: var(--espresso); }

        .hero {
            background: rgba(85, 27, 20, 0.90);
            color: white;
            border-radius: 26px;
            padding: 1.5rem 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }
        .hero-title {
            color: white !important;
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: -0.04em;
        }
        .hero-sub {
            color: rgba(255,255,255,0.8);
            margin-top: 0.6rem;
            font-size: 0.96rem;
        }
        .pill {
            display: inline-block;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.28);
            background: rgba(205, 197, 180, 0.15);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: white;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }
        .kpi-card {
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            backdrop-filter: blur(8px);
        }
        .kpi-label {
            color: rgba(255,255,255,0.82);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .kpi-value {
            margin-top: 0.5rem;
            font-size: 2rem;
            font-weight: 700;
            color: white;
        }
        .card {
            background: rgba(255,255,255,0.22);
            border: 1px solid rgba(85,27,20,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(85,27,20,0.08);
            margin-bottom: 1rem;
        }
        .card h3 { margin-top: 0; margin-bottom: 0.75rem; }
        .tiny-tag {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: rgba(85,27,20,0.12);
            color: var(--espresso);
            font-size: 0.7rem;
            font-weight: 700;
            margin-right: 0.35rem;
        }
        .tag {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(85,27,20,0.08);
            color: var(--espresso);
            font-size: 0.72rem;
            font-weight: 600;
        }
        .stSelectbox > div, .stDateInput > div, .stTextInput > div, .stMultiSelect > div {
            background: rgba(255,255,255,0.18);
            border-radius: 10px;
            color: white;
        }
        .stSelectbox label, .stTextInput label, .stDateInput label, .stMultiSelect label {
            color: rgba(255,255,255,0.92);
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--mocha), var(--espresso));
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            width: 100%;
        }
        .stButton > button:hover { filter: brightness(1.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Patient filters")
    profile_filter = st.multiselect(
        "Profile group",
        options=["Any", "diabetes", "cardiac", "respiratory", "infection", "renal"],
        default=["Any"],
    )
    condition_text = st.text_input("Relevant conditions", placeholder="e.g. diabetes, E11.9")
    medication_text = st.text_input("Relevant meds", placeholder="e.g. metformin")
    measurement_text = st.text_input("Relevant measurements", placeholder="e.g. glucose, creatinine")
    basis = st.selectbox("Medication evidence", ["orders", "administrations"])
    similarity_method = st.selectbox(
        "Similarity weighting",
        options=["jaccard", "idf"],
        format_func=lambda value: "IDF-weighted Jaccard" if value == "idf" else "Jaccard",
    )
    top_k = st.slider("Match count", 1, 20, 5)
    start_date = st.date_input("Start date", value=None)
    end_date = st.date_input("End date", value=None)
    st.divider()
    st.caption("Manual review notes are session-only and are not clinical recommendations.")
    if st.button("Add review item", width="stretch"):
        add_review_item_dialog()

review_items = st.session_state.get("review_items", [])
if review_items:
    with st.expander(f"Session review items ({len(review_items)})"):
        st.dataframe(pd.DataFrame(review_items), width="stretch", hide_index=True)

st.markdown(
    """
    <div class='hero'>
        <div>
          <div class='pill'>Synthetic EHR dashboard</div>
          <h1 class='hero-title'>Patient Similarity Explorer</h1>
          <div class='hero-sub'>Synthetic clinical records for software testing. Not real MIMIC data and not for clinical decision-making.</div>
        </div>
      <div class='kpi-grid'>
        <div class='kpi-card'>
          <div class='kpi-label'>Total patients</div>
          <div class='kpi-value'>{total}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>New arrivals</div>
          <div class='kpi-value'>{new}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Historical patients</div>
          <div class='kpi-value'>{historical_total}</div>
        </div>
      </div>
    </div>
    """.format(
        total=len(patients),
        new=len(new_patients),
        historical_total=len(historical),
    ),
    unsafe_allow_html=True,
)

if new_patients.empty:
    st.warning("No patients labelled NEW.")
    st.stop()

selected_id = st.selectbox("Select a new patient", new_patients["subject_id"].tolist(), index=0)
patient = patients[patients["subject_id"] == selected_id].iloc[0]

ed = data.get("edstays", pd.DataFrame())
triage = data.get("triage", pd.DataFrame())
ed_patient = _patient_records(ed, selected_id)
triage_patient = _patient_records(triage, selected_id)
arrival = ed_patient.iloc[0] if not ed_patient.empty else {}
triage_record = triage_patient.iloc[0] if not triage_patient.empty else {}
complaint = str(
    triage_record.get("chiefcomplaint", "")
    or arrival.get("chief_complaint", "")
).strip()

st.markdown(
    f"""
    <div class='card'>
        <h3>New patient</h3>
        <div><span class='tiny-tag'>Patient {selected_id}</span> <span class='tiny-tag'>Age (approx.): {patient['anchor_age']}</span> <span class='tiny-tag'>Sex field: {patient['gender']}</span> <span class='tiny-tag'>Profile label: {patient['profile_group']}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Current arrival")
with st.container(border=True):
    st.markdown(f"## {complaint or 'Chief complaint not recorded'}")
    if not triage_patient.empty:
        st.caption(f"Triage recorded: {triage_record.get('charttime', 'Time not recorded')}")
    arrival_col, status_col, acuity_col = st.columns(3)
    with arrival_col:
        st.metric("Arrival method", str(arrival.get("arrival_transport", "") or "Not recorded"))
    with status_col:
        current_status = str(arrival.get("disposition", "") or "Not recorded")
        st.metric("Current status", current_status.replace("_", " ").title())
    with acuity_col:
        acuity = triage_record.get("acuity", "")
        st.metric("Triage acuity", str(acuity) if str(acuity).strip() else "Not recorded")

    vital_columns = st.columns(5)
    vitals = [
        ("Heart rate", triage_record.get("heartrate", ""), "bpm"),
        ("Respiratory rate", triage_record.get("resprate", ""), "breaths/min"),
        ("Temperature", triage_record.get("temperature", ""), "°C"),
        ("Oxygen saturation", triage_record.get("o2sat", ""), "%"),
        ("Blood pressure", (
            f"{triage_record.get('sbp', '')}/{triage_record.get('dbp', '')}"
            if str(triage_record.get("sbp", "")).strip()
            and str(triage_record.get("dbp", "")).strip()
            else ""
        ), "mmHg"),
    ]
    for column, (label, value, unit) in zip(vital_columns, vitals):
        display_value = f"{value} {unit}".strip() if str(value).strip() else "Not recorded"
        column.metric(label, display_value)
st.caption("Fictional triage data for software testing; not a clinical assessment.")

query_profile = None if ("Any" in profile_filter or not profile_filter) else profile_filter[0]
query = RetrievalQuery(
    conditions=[part.strip() for part in condition_text.split(",") if part.strip()],
    medications=[part.strip() for part in medication_text.split(",") if part.strip()],
    measurements=[part.strip() for part in measurement_text.split(",") if part.strip()],
    profile_group=query_profile,
    start_date=start_date.isoformat() if start_date else None,
    end_date=end_date.isoformat() if end_date else None,
    medication_basis=basis,
    similarity_method=similarity_method,
    top_k=top_k,
)

cohort = select_cohort(data, query)
features = patient_features(data, selected_id, query)

st.subheader("Recorded conditions, medications, and allergies")
dx = data["diagnoses_icd"][data["diagnoses_icd"]["subject_id"].astype(str) == str(selected_id)]
meds = data["prescriptions"][data["prescriptions"]["subject_id"].astype(str) == str(selected_id)]
allergy = data["allergies"][data["allergies"]["subject_id"].astype(str) == str(selected_id)]

st.write("Conditions")
st.markdown(
    " ".join(f"<span class='tag'>{code}</span>" for code in dx["icd_code"].astype(str).tolist()),
    unsafe_allow_html=True,
)
st.write("Medications")
st.markdown(
    " ".join(f"<span class='tag'>{med}</span>" for med in meds["drug"].astype(str).tolist()),
    unsafe_allow_html=True,
)
st.write("Allergies")
if allergy.empty:
    st.write("None recorded")
else:
    st.markdown(
        " ".join(f"<span class='tag'>{a}</span>" for a in allergy["allergy_name"].astype(str).tolist()),
        unsafe_allow_html=True,
    )

st.subheader("Find similar historical patients")
if not features:
    st.info("No features match these filters. Broaden the query or review the date range.")
else:
    ranked = rank_historical_patients(data, selected_id, query)
    if ranked.empty:
        st.info("No historical candidates were found.")
    else:
        display = ranked[
            ["subject_id", "profile_group", "similarity", "similarity_jaccard", "similarity_idf"]
        ].copy()
        for column in ("similarity", "similarity_jaccard", "similarity_idf"):
            display[column] = (display[column] * 100).round(1).astype(str) + "%"
        display.columns = [
            "Subject ID", "Profile group", "Selected score",
            "Jaccard", "IDF-weighted Jaccard",
        ]

        st.write(f"Matching features found: {len(features)}")
        st.write(f"Number of historical patients: {len(ranked)}")
        st.dataframe(display, width="stretch")

        selected_historical = st.selectbox("Review a historical patient", ranked["subject_id"].tolist())
        match = ranked[ranked["subject_id"] == selected_historical].iloc[0]
        historical_patient = patients.loc[
            patients["subject_id"].astype(str) == str(selected_historical)
        ].iloc[0]

        st.markdown(
            f"""
            <div class='card'>
                <h3>Historical patient</h3>
                <div><span class='tiny-tag'>Patient {selected_historical}</span> <span class='tiny-tag'>Approx. age: {int(historical_patient['anchor_age'])}</span> <span class='tiny-tag'>Sex field: {historical_patient['gender']}</span> <span class='tiny-tag'>Similarity: {match['similarity'] * 100:.1f}%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        shared_features = match.get("shared_features", []) or []
        st.subheader("Side-by-side matching features")
        if shared_features:
            comparison = []
            for feature in shared_features:
                category, _, value = feature.partition(":")
                comparison.append({
                    "Category": category.replace("_", " ").title(),
                    "Shared feature": value.replace(":", " — "),
                    "New patient record": _shared_feature_value(data, selected_id, feature, query),
                    "Historical patient record": _shared_feature_value(
                        data, selected_historical, feature, query
                    ),
                })
            st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
        else:
            st.info("This patient has no shared recorded features with the selected new patient.")
        st.caption(
            "A shared measurement means the same measurement label was recorded; the measured values "
            "are shown for comparison and are not required to be equal."
        )

        st.subheader("Historical encounter and outcome")
        adm = _patient_records(data["admissions"], selected_historical)
        admission_columns = [
            column for column in (
                "hadm_id", "admittime", "dischtime", "admission_type",
                "admission_location", "discharge_location", "outcome_status",
                "outcome_recorded",
            ) if column in adm.columns
        ]
        if adm.empty:
            st.info("No historical admission record is available.")
        else:
            admission_display = adm[admission_columns].rename(columns={
                "hadm_id": "Admission ID",
                "admittime": "Admitted",
                "dischtime": "Discharged",
                "admission_type": "Admission type",
                "admission_location": "Admitted from",
                "discharge_location": "Discharged to",
                "outcome_status": "Recorded outcome",
                "outcome_recorded": "Outcome documented (1=yes, 0=no)",
            })
            st.dataframe(admission_display, width="stretch", hide_index=True)
        st.info(
            "The synthetic dataset does not include a historical chief-complaint field or a "
            "documented clinician-authored treatment plan. Diagnoses provide coded clinical context; "
            "orders, recorded administrations, and procedures below are separate evidence—not a plan "
            "or a recommendation for the new patient."
        )

        dx_hist = _patient_records(data["diagnoses_icd"], selected_historical)
        st.subheader("Recorded diagnoses (clinical context)")
        if dx_hist.empty:
            st.write("No diagnoses recorded.")
        else:
            diagnosis_columns = [
                column for column in ("hadm_id", "icd_code", "long_title")
                if column in dx_hist.columns
            ]
            st.dataframe(
                dx_hist[diagnosis_columns], width="stretch", hide_index=True
            )

        rx_hist = _patient_records(data["prescriptions"], selected_historical)
        st.subheader("Medication orders")
        if rx_hist.empty:
            st.write("No medication orders recorded.")
        else:
            order_columns = [
                column for column in (
                    "hadm_id", "starttime", "stoptime", "drug", "dose_val_rx",
                    "dose_unit_rx", "route", "frequency",
                ) if column in rx_hist.columns
            ]
            st.dataframe(rx_hist[order_columns], width="stretch", hide_index=True)

        administrations = _patient_records(
            data.get("medication_administrations", pd.DataFrame()), selected_historical
        )
        st.subheader("Medication administrations recorded as given")
        if administrations.empty:
            st.write("No medication administration events recorded.")
        else:
            administration_columns = [
                column for column in (
                    "hadm_id", "charttime", "drug", "dose_administered", "dose_unit",
                    "route", "frequency", "administration_status",
                ) if column in administrations.columns
            ]
            st.dataframe(
                administrations[administration_columns],
                width="stretch",
                hide_index=True,
            )

        proc_hist = _patient_records(data["procedures_icd"], selected_historical)
        with st.expander("Recorded procedures and other notes"):
            st.write("Procedures")
            st.dataframe(proc_hist, width="stretch", hide_index=True)
            for table, heading in (
                ("discharge", "Discharge notes"),
                ("radiology", "Radiology reports"),
            ):
                st.write(heading)
                rows = _patient_records(data.get(table, pd.DataFrame()), selected_historical)
                st.dataframe(rows, width="stretch", hide_index=True)

st.caption(
    "Jaccard and IDF-weighted Jaccard are experimental feature-overlap scores, not measures of "
    "treatment effectiveness or a basis for clinical action."
)

if not query.conditions and not query.medications and not query.measurements and (query.profile_group is not None):
    validation = validate_cohort_against_pandas(data, query, cohort)
    st.write("Profile-only pandas validation:", "PASS" if validation["validated"] else "FAIL", validation)
