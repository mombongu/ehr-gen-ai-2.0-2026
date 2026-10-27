"""Streamlit UI for a synthetic EHR patient similarity prototype."""
from pathlib import Path

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
st.set_page_config(page_title="EHR Patient Comparison", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --sand: #cdc5b4;
            --rose: #B59DA4;
            --stone: #85756E;
            --mocha: #6D3D14;
            --espresso: #551B14;
            --panel: rgba(255, 255, 255, 0.28);
            --panel-strong: rgba(255, 255, 255, 0.42);
            --text: #2f1d17;
            --muted: rgba(47, 29, 23, 0.7);
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, var(--sand) 0%, var(--rose) 100%);
            color: var(--text);
        }

        .stApp {
            background: transparent;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            color: var(--espresso);
        }

        .dashboard-shell {
            background: rgba(85, 27, 20, 0.9);
            border-radius: 26px;
            padding: 1.5rem 1.5rem 1.1rem;
            box-shadow: 0 18px 42px rgba(85, 27, 20, 0.2);
            margin-bottom: 1.5rem;
        }

        .dashboard-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            color: white;
        }

        .eyebrow {
            margin: 0 0 0.35rem;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            opacity: 0.8;
        }

        .dashboard-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.2rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.8rem;
            background: rgba(205, 197, 180, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 999px;
            font-size: 0.8rem;
            color: white;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1.2rem;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 24px rgba(16, 8, 6, 0.08);
            backdrop-filter: blur(8px);
        }

        .metric-label {
            color: var(--mocha);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .metric-value {
            margin-top: 0.7rem;
            font-size: 2.15rem;
            font-weight: 700;
            color: var(--espresso);
        }

        .status-panel {
            background: rgba(255, 255, 255, 0.24);
            border: 1px solid rgba(85, 27, 20, 0.1);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-top: 1rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel-strong);
            border: 1px solid rgba(85, 27, 20, 0.08);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 10px 20px rgba(85, 27, 20, 0.06);
        }

        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
        }

        .stSelectbox > div, .stDateInput > div, .stTextInput > div {
            background: rgba(255, 255, 255, 0.32);
            border-radius: 12px;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--mocha), var(--espresso));
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("EHR Patient Comparison")
st.caption(
    "Synthetic records for software testing only. Not real patient data, not official MIMIC/OMOP, and not for clinical decisions."
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

st.markdown(
    """
    <div class="dashboard-shell">
        <div class="dashboard-header">
            <div>
                <p class="eyebrow">Synthetic EHR dashboard</p>
                <h1>Patient cohort overview</h1>
            </div>
            <div class="chip">Live patient comparison</div>
        </div>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total patients</div>
                <div class="metric-value">{total}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">New arrivals</div>
                <div class="metric-value">{new}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Historical patients</div>
                <div class="metric-value">{historical_total}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profiles</div>
                <div class="metric-value">{profiles}</div>
            </div>
        </div>
    </div>
    """.format(
        total=len(patients),
        new=len(new_patients),
        historical_total=len(historical),
        profiles=len(patients["profile_group"].dropna().unique()),
    ),
    unsafe_allow_html=True,
)

st.divider()
st.header("1. Select a query patient")
if new_patients.empty:
    st.warning("No patients labelled NEW.")
    st.stop()
selected_id = st.selectbox("New patient", new_patients["subject_id"].tolist())
patient = patients[patients["subject_id"] == selected_id].iloc[0]
st.write(f"**Patient {selected_id}** · approximate age {patient['anchor_age']} · profile {patient['profile_group']}")

with st.expander("Current arrival and triage"):
    for name in ("edstays", "triage"):
        if name in data:
            st.subheader(name.replace("_", " ").title())
            st.dataframe(data[name][data[name]["subject_id"].astype(str) == str(selected_id)], use_container_width=True)

with st.expander("Clinical records and allergy status"):
    for name, cols in [
        ("diagnoses_icd", ["icd_code", "long_title"]),
        ("prescriptions", ["drug", "dose_val_rx", "dose_unit_rx", "route", "starttime", "stoptime"]),
        ("medication_administrations", ["drug", "dose_administered", "dose_unit", "administration_status", "charttime"]),
        ("allergies", ["allergy_name", "reaction", "status"]),
        ("labevents", ["label", "valuenum", "valueuom", "charttime"]),
        ("chartevents", ["label", "valuenum", "valueuom", "charttime"]),
    ]:
        if name in data:
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            frame = data[name]
            frame = frame[frame["subject_id"].astype(str) == str(selected_id)]
            st.dataframe(frame[[col for col in cols if col in frame.columns]], use_container_width=True)
    allergy_rows = data["allergies"][data["allergies"]["subject_id"].astype(str) == str(selected_id)]
    if allergy_rows.empty:
        st.warning("No allergy record exists. This does not mean no known allergies.")
    elif (allergy_rows["status"] == "UNKNOWN").any():
        st.warning("Allergy status is recorded as UNKNOWN.")
    elif (allergy_rows["status"] == "NOT_DOCUMENTED").any():
        st.warning("Allergy information is NOT DOCUMENTED.")
    elif (allergy_rows["status"] == "NO_KNOWN_ALLERGIES").any():
        st.info("Record explicitly states NO KNOWN ALLERGIES.")

st.divider()
st.header("2. Define the comparison query")
col1, col2 = st.columns(2)
with col1:
    condition_text = st.text_input("Relevant conditions (comma-separated)", placeholder="e.g. diabetes, E11.9")
    medication_text = st.text_input("Relevant medications (comma-separated)", placeholder="e.g. metformin")
    measurement_text = st.text_input("Relevant measurements (comma-separated)", placeholder="e.g. glucose, creatinine")
with col2:
    basis = st.selectbox("Medication evidence", ["orders", "administrations"])
    profile = st.selectbox("Optional profile filter", ["Any", "diabetes", "cardiac", "respiratory", "infection", "renal"])
    start_date = st.date_input("Start date (optional)", value=None)
    end_date = st.date_input("End date (optional)", value=None)
top_k = st.slider("Number of historical matches", 1, 20, 5)


def split_terms(text):
    return [part.strip() for part in text.split(",") if part.strip()]


query = RetrievalQuery(
    conditions=split_terms(condition_text),
    medications=split_terms(medication_text),
    measurements=split_terms(measurement_text),
    profile_group=None if profile == "Any" else profile,
    start_date=start_date.isoformat() if start_date else None,
    end_date=end_date.isoformat() if end_date else None,
    medication_basis=basis,
    top_k=top_k,
)

st.subheader("Cohort matching the query")
cohort = select_cohort(data, query)
st.write(f"Matching patients: **{len(cohort)}**")
st.dataframe(cohort[["subject_id", "patient_status", "profile_group"]], use_container_width=True)
if not query.conditions and not query.medications and not query.measurements and (query.profile_group is not None):
    validation = validate_cohort_against_pandas(data, query, cohort)
    st.write("Profile-only pandas validation:", "PASS" if validation["validated"] else "FAIL", validation)

st.markdown(
    """
    <div class="status-panel">
        <strong>Query snapshot</strong><br>
        Medication basis: {basis} · Profile: {profile} · Match cap: {top_k}
    </div>
    """.format(basis=basis, profile=profile, top_k=top_k),
    unsafe_allow_html=True,
)

st.divider()
st.header("3. Rank similar historical patients")
features = patient_features(data, selected_id, query)
st.write(f"Query features available: **{len(features)}**")
if not features:
    st.info("No features match these filters. Broaden the query or review the selected time window.")
else:
    ranked = rank_historical_patients(data, selected_id, query)
    if ranked.empty:
        st.info("No historical candidates were found.")
    else:
        display = ranked[["subject_id", "profile_group", "similarity", "query_feature_count", "candidate_feature_count"]].copy()
        display["similarity"] = (display["similarity"] * 100).round(1).astype(str) + "%"
        st.dataframe(display, use_container_width=True)
        selected_historical = st.selectbox("Review a matched historical patient", ranked["subject_id"].tolist())
        match = ranked[ranked["subject_id"] == selected_historical].iloc[0]
        st.metric("Jaccard similarity", f"{match['similarity']:.1%}")
        st.write("Shared features")
        st.write(match["shared_features"] or ["No shared features"])
        row = patients[patients["subject_id"].astype(str) == str(selected_historical)].iloc[0]
        st.write(f"**Profile:** {row['profile_group']}")
        adm = data["admissions"]
        adm = adm[adm["subject_id"].astype(str) == str(selected_historical)]
        st.subheader("Encounter outcomes (explicit fields)")
        st.dataframe(adm[[c for c in ["hadm_id", "admittime", "dischtime", "discharge_location", "outcome_status", "outcome_recorded"] if c in adm.columns]], use_container_width=True)
        if "allergies" in data:
            st.subheader("Allergy records")
            st.dataframe(data["allergies"][data["allergies"]["subject_id"].astype(str) == str(selected_historical)], use_container_width=True)

st.caption(
    "Similarity is an experimental record-overlap score, not treatment effectiveness or a clinical recommendation. Time filters apply to dated medication and measurement evidence; condition records are not timestamped individually."
)
