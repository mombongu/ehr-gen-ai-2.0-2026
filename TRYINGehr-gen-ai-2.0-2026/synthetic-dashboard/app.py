"""Streamlit UI for a synthetic EHR patient similarity prototype."""
from html import escape
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


def _comparison_items(data, patient_id, category, query):
    if category == "Conditions":
        rows = _patient_records(data["diagnoses_icd"], patient_id)
        return [
            f"{row.get('icd_code', '')} — {row.get('long_title', '')}".strip(" —")
            for row in rows.to_dict("records")
        ] or ["No diagnosis records"]

    if category == "Measurements":
        items = []
        for table in ("labevents", "chartevents"):
            rows = _patient_records(
                data.get(table, pd.DataFrame()), patient_id, "charttime", query
            )
            for row in rows.to_dict("records"):
                value = row.get("valuenum", row.get("value", ""))
                unit = row.get("valueuom", row.get("unit", ""))
                items.append(" · ".join(part for part in (
                    str(row.get("charttime", "")).strip(),
                    str(row.get("label", "")).strip(),
                    str(value).strip(),
                    str(unit).strip(),
                ) if part))
        return items or ["No measurements recorded in this date range"]

    if category == "Allergies":
        rows = _patient_records(data["allergies"], patient_id)
        return [
            " · ".join(part for part in (
                str(row.get("status", "")).replace("_", " ").title(),
                str(row.get("allergy_name", "")).strip(),
                str(row.get("reaction", "")).strip(),
            ) if part)
            for row in rows.to_dict("records")
        ] or ["No allergy record (status not inferred)"]

    items = []
    orders = _patient_records(
        data.get("prescriptions", pd.DataFrame()), patient_id, "starttime", query
    )
    for row in orders.to_dict("records"):
        items.append(" · ".join(part for part in (
            "ORDER",
            str(row.get("starttime", "")).strip(),
            str(row.get("drug", "")).strip(),
            " ".join(str(part).strip() for part in (
                row.get("dose_val_rx", ""),
                row.get("dose_unit_rx", ""),
            ) if str(part).strip()),
            str(row.get("route", "")).strip(),
            str(row.get("frequency", "")).strip(),
        ) if part))
    administrations = _patient_records(
        data.get("medication_administrations", pd.DataFrame()),
        patient_id,
        "charttime",
        query,
    )
    for row in administrations.to_dict("records"):
        items.append(" · ".join(part for part in (
            "GIVEN",
            str(row.get("charttime", "")).strip(),
            str(row.get("drug", "")).strip(),
            " ".join(str(part).strip() for part in (
                row.get("dose_administered", ""),
                row.get("dose_unit", ""),
            ) if str(part).strip()),
            str(row.get("route", "")).strip(),
            str(row.get("administration_status", "")).strip(),
        ) if part))
    return items or ["No medication orders or administrations in this date range"]


def _render_comparison_card(title, patient_kind, items):
    values = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="compare-card {patient_kind}">
            <div class="compare-card-title">{escape(title)}</div>
            <ul>{values}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

# Render custom CSS/HTML directly with st.write to avoid markdown table parsing bugs
st.write(
        """
        <link href=\"https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap\" rel=\"stylesheet\" />
        <style>
            :root {
                --sand: #cdc5b4;
                --rose: #B59DA4;
                --mocha: #6D3D14;
                --espresso: #551B14;
                --text: #2f1d17;
                --muted: rgba(47, 29, 23, 0.78);
                --shadow: 0 18px 38px rgba(85, 27, 20, 0.16);
                --font: 'Poppins', sans-serif;
            }
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, var(--sand) 0%, var(--rose) 100%);
                color: var(--text);
                font-family: var(--font);
            }
            .stApp { background: transparent; }
            .main .block-container { padding-top: 1.3rem; padding-bottom: 2.5rem; }
            h1, h2, h3, h4 { color: var(--espresso); font-weight: 600; }
            .hero {
                background: linear-gradient(135deg, var(--mocha) 0%, var(--espresso) 100%);
                color: white;
                border-radius: 26px;
                padding: 1.5rem 1.5rem;
                box-shadow: var(--shadow);
                margin-bottom: 1.2rem;
            }
            .hero-title {
                color: white !important;
                margin: 0;
                font-size: 2.4rem;
                letter-spacing: -0.04em;
                font-weight: 600;
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
                background: linear-gradient(45deg, var(--rose) 0%, var(--sand) 100%);
                font-size: 0.72rem;
                color: var(--espresso);
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
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .card:hover {
                transform: translateY(-4px) scale(1.02);
                box-shadow: 0 14px 28px rgba(85,27,20,0.12);
            }
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
                border: 1px solid rgba(85,27,20,0.12);
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
            .compare-card {
                min-height: 180px;
                border-radius: 16px;
                padding: 1rem 1.15rem;
                margin: 0.35rem 0 1rem;
                border: 1px solid rgba(47,29,23,0.12);
                color: var(--text);
            }
            .compare-card.new {
                background: #e8f3ff;
                border-left: 6px solid #2878b5;
            }
            .compare-card.historical {
                background: #fff0df;
                border-left: 6px solid #b45b18;
            }
            .compare-card-title {
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .compare-card ul {
                margin: 0;
                padding-left: 1.25rem;
            }
            .compare-card li {
                margin-bottom: 0.45rem;
                overflow-wrap: anywhere;
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff;
                border-radius: 14px;
                border-color: rgba(85,27,20,0.14);
            }
            .st-key-historical-timeline-panel,
            .st-key-historical-timeline-panel [data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #ffffff !important;
                border-radius: 18px;
                border-color: rgba(85,27,20,0.18);
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

        st.subheader("Compare patient records")
        st.markdown(
            f":blue[NEW patient · {selected_id}]　 :orange[HISTORICAL patient · {selected_historical}]"
        )
        comparison_categories = st.tabs(
            ["Conditions", "Measurements", "Allergies", "Medications"]
        )
        for tab, category in zip(
            comparison_categories,
            ("Conditions", "Measurements", "Allergies", "Medications"),
        ):
            with tab:
                new_col, historical_col = st.columns(2)
                with new_col:
                    _render_comparison_card(
                        f"NEW patient · {selected_id}",
                        "new",
                        _comparison_items(data, selected_id, category, query),
                    )
                with historical_col:
                    _render_comparison_card(
                        f"HISTORICAL patient · {selected_historical}",
                        "historical",
                        _comparison_items(data, selected_historical, category, query),
                    )
        st.caption(
            "Medication orders are not proof of administration. The new patient's current "
            "encounter is still in progress; historical records are for comparison only."
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
            "This synthetic dataset has no clinician-authored treatment plan. Diagnoses, orders, "
            "administrations, procedures, and generated course notes are separate fictional records, "
            "not a plan or recommendation for the new patient."
        )

        administrations = _patient_records(
            data.get("medication_administrations", pd.DataFrame()), selected_historical
        )
        st.subheader("Historical hospital-course timeline")
        timeline_events = []

        course = _patient_records(
            data.get("hospital_course", pd.DataFrame()), selected_historical
        )
        dx_hist = _patient_records(data["diagnoses_icd"], selected_historical)
        for row in dx_hist.to_dict("records"):
            diagnosis_details = " — ".join(part for part in (
                str(row.get("icd_code", "")).strip(),
                str(row.get("long_title", "")).strip(),
            ) if part)
            timeline_events.append({
                "time": row.get("diagnosis_time", ""),
                "event": "Diagnosis recorded",
                "details": diagnosis_details or "Diagnosis details not recorded",
                "treatment": "",
                "rationale": "",
                "response_status": "",
                "causal_assessment": "",
                "response_interpretation": "",
            })
        final_diagnosis = course[
            course["event_type"] == "Final diagnosis"
        ] if "event_type" in course.columns else pd.DataFrame()
        if final_diagnosis.empty:
            st.info("No synthetic final diagnosis is recorded.")
        else:
            diagnosis = final_diagnosis.iloc[0]
            diagnosis_code = str(diagnosis.get("final_diagnosis_code", "")).strip()
            diagnosis_name = str(diagnosis.get("final_diagnosis", "")).strip()
            diagnosis_summary = " — ".join(
                part for part in (diagnosis_code, diagnosis_name) if part
            ) or str(diagnosis.get("summary", "Diagnosis details not recorded"))
            st.markdown(
                f"""
                <div class="compare-card historical">
                    <div class="compare-card-title">FINAL DIAGNOSIS · HISTORICAL PATIENT</div>
                    <div>{escape(diagnosis_summary)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for row in course.to_dict("records"):
            timeline_events.append({
                "time": row.get("event_time", ""),
                "event": row.get("event_type", "Hospital course note"),
                "details": row.get("summary", ""),
                "treatment": row.get("associated_treatment", ""),
                "rationale": row.get("rationale", ""),
                "response_status": row.get("response_status", ""),
                "causal_assessment": row.get("causal_assessment", ""),
                "response_interpretation": row.get("response_interpretation", ""),
            })

        procedures = _patient_records(
            data.get("procedures_icd", pd.DataFrame()), selected_historical
        )
        for row in procedures.to_dict("records"):
            timeline_events.append({
                "time": row.get("chartdate", ""),
                "event": "Procedure recorded",
                "details": row.get("procedure_name", "Procedure name not recorded"),
                "treatment": "",
                "rationale": row.get("rationale", ""),
                "response_status": "",
                "causal_assessment": "",
                "response_interpretation": "",
            })

        orders = _patient_records(
            data.get("prescriptions", pd.DataFrame()), selected_historical
        )
        for row in orders.to_dict("records"):
            order_details = " ".join(
                str(part).strip()
                for part in (
                    row.get("drug", ""),
                    " ".join(str(part).strip() for part in (
                        row.get("dose_val_rx", ""),
                        row.get("dose_unit_rx", ""),
                    ) if str(part).strip()),
                    row.get("route", ""),
                    row.get("frequency", ""),
                )
                if str(part).strip()
            )
            timeline_events.append({
                "time": row.get("starttime", ""),
                "event": "Medication order (not proof of administration)",
                "details": order_details or "Medication details not recorded",
                "treatment": "",
                "rationale": row.get("rationale", ""),
                "response_status": "",
                "causal_assessment": "",
                "response_interpretation": "",
            })

        for row in administrations.to_dict("records"):
            administration_details = " ".join(
                str(part).strip()
                for part in (
                    row.get("drug", ""),
                    " ".join(str(part).strip() for part in (
                        row.get("dose_administered", ""),
                        row.get("dose_unit", ""),
                    ) if str(part).strip()),
                    row.get("route", ""),
                    row.get("administration_status", ""),
                )
                if str(part).strip()
            )
            timeline_events.append({
                "time": row.get("charttime", ""),
                "event": "Medication administration",
                "details": administration_details or "Administration details not recorded",
                "treatment": "",
                "rationale": "",
                "response_status": "",
                "causal_assessment": "",
                "response_interpretation": "",
            })

        timeline_events = [
            event for event in timeline_events if str(event["details"]).strip()
        ]
        timeline_events.sort(
            key=lambda event: pd.to_datetime(event["time"], errors="coerce")
        )
        if not timeline_events:
            st.info("No dated course, procedure, or medication events are available.")
        else:
            with st.container(border=True, key="historical-timeline-panel"):
                st.caption(
                    f"{len(timeline_events)} fictional events, ordered by recorded time. "
                    "Event colors distinguish diagnoses, treatments, responses, complications, and outcome."
                )
                for event in timeline_events:
                    event_type = str(event["event"])
                    response_status = str(event["response_status"]).strip()
                    if event_type in {"Diagnosis recorded", "Final diagnosis"}:
                        color = "violet"
                    elif event_type == "Medication order (not proof of administration)":
                        color = "blue"
                    elif event_type == "Medication administration":
                        color = "green"
                    elif event_type == "Procedure recorded":
                        color = "orange"
                    elif event_type == "Response check":
                        color = "blue"
                    elif event_type == "Complication":
                        color = "red"
                    elif event_type == "Disposition and follow-up":
                        color = "green"
                    else:
                        color = "gray"

                    if event_type == "Disposition and follow-up":
                        if "against medical advice" in str(event["details"]).lower():
                            color = "red"
                        elif "transferred" in str(event["details"]).lower():
                            color = "orange"
                    prior_details = []
                    if str(event["treatment"]).strip():
                        recorded_before = [
                            earlier for earlier in timeline_events
                            if earlier is not event
                            and earlier["event"] in {
                                "Medication order (not proof of administration)",
                                "Medication administration",
                                "Procedure recorded",
                            }
                            and pd.notna(pd.to_datetime(earlier["time"], errors="coerce"))
                            and pd.to_datetime(earlier["time"], errors="coerce")
                            <= pd.to_datetime(event["time"], errors="coerce")
                        ]
                        prior_details = list(dict.fromkeys(
                            f"{earlier['event']}: {earlier['details']}"
                            for earlier in recorded_before
                        ))
                    with st.container(border=True):
                        marker, content = st.columns([0.12, 0.88])
                        marker.markdown(f":{color}[●]")
                        with content:
                            st.caption(str(event["time"]) or "Time not recorded")
                            st.markdown(f":{color}[**{event_type}**]")
                            st.write(str(event["details"]))
                            if response_status:
                                st.markdown(
                                    f":{color}[**Simulated response: {response_status}**]"
                                )

                            if str(event["treatment"]).strip():
                                if prior_details:
                                    st.caption("Recorded earlier in this encounter (temporal context only):")
                                    for detail in prior_details:
                                        st.write(f"- {detail}")
                                else:
                                    st.caption(f"Treatment context: {event['treatment']}")
                            if str(event["rationale"]).strip():
                                st.caption(f"Rationale: {event['rationale']}")
                            if str(event["causal_assessment"]).strip():
                                st.caption(f"Causal assessment: {event['causal_assessment']}")
                            if str(event["response_interpretation"]).strip():
                                st.caption(str(event["response_interpretation"]))
        st.caption(
            "This is simulated historical-pattern context for clinician review, not a suggested "
            "treatment plan. Simulated response labels and treatment timing do not establish "
            "effectiveness or causality. Clinicians must make independent decisions."
        )

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
