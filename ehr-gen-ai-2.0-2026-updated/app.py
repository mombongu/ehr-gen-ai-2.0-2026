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


def format_list(values):
    values = [str(v).strip() for v in values if str(v).strip()]
    return values or ["None recorded"]


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
            --stone: #85756E;
            --mocha: #6D3D14;
            --espresso: #551B14;
            --cream: #f6f0ea;
            --panel: rgba(255,255,255,0.25);
            --panel-strong: rgba(255,255,255,0.38);
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
        .detail-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.5rem;
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
        .section-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.5rem;
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
    st.markdown('<div class="settings-title">Patient filters</div>', unsafe_allow_html=True)
    profile_filter = st.multiselect(
        "Profile group",
        options=["Any", "diabetes", "cardiac", "respiratory", "infection", "renal"],
        default=["Any"],
    )
    condition_text = st.text_input("Relevant conditions", placeholder="e.g. diabetes, E11.9")
    medication_text = st.text_input("Relevant meds", placeholder="e.g. metformin")
    measurement_text = st.text_input("Relevant measurements", placeholder="e.g. glucose, creatinine")
    basis = st.selectbox("Medication evidence", ["orders", "administrations"])
    top_k = st.slider("Match count", 1, 20, 5)
    start_date = st.date_input("Start date", value=None)
    end_date = st.date_input("End date", value=None)

st.markdown(
    """
    <div class='hero'>
      <div class='hero-row'>
        <div>
          <div class='pill'>Synthetic EHR dashboard</div>
          <h1 class='hero-title'>Patient Similarity Explorer</h1>
          <div class='hero-sub'>Synthetic clinical records for software testing. Not real MIMIC data and not for clinical decision-making.</div>
        </div>
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

st.markdown(
    f"""
    <div class='card'>
        <h3>New patient</h3>
        <div><span class='tiny-tag'>Patient {selected_id}</span> <span class='tiny-tag'>Age (approx.): {patient['anchor_age']}</span> <span class='tiny-tag'>Sex field: {patient['gender']}</span> <span class='tiny-tag'>Profile label: {patient['profile_group']}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

query_profile = None if ("Any" in profile_filter or not profile_filter) else profile_filter[0]
query = RetrievalQuery(
    conditions=[part.strip() for part in condition_text.split(",") if part.strip()],
    medications=[part.strip() for part in medication_text.split(",") if part.strip()],
    measurements=[part.strip() for part in measurement_text.split(",") if part.strip()],
    profile_group=query_profile,
    start_date=start_date.isoformat() if start_date else None,
    end_date=end_date.isoformat() if end_date else None,
    medication_basis=basis,
    top_k=top_k,
)

cohort = select_cohort(data, query)
features = patient_features(data, selected_id, query)

# patient detail panels
left_col, right_col = st.columns([1.65, 1.35])

with left_col:
    st.subheader("Current arrival and triage")
    ed = data.get("edstays", pd.DataFrame())
    triage = data.get("triage", pd.DataFrame())
    if not ed.empty:
        ed_patient = ed[ed["subject_id"].astype(str) == str(selected_id)]
        st.dataframe(ed_patient, use_container_width=True)
    if not triage.empty:
        triage_patient = triage[triage["subject_id"].astype(str) == str(selected_id)]
        st.dataframe(triage_patient, use_container_width=True)

with right_col:
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
        display = ranked[["subject_id", "profile_group", "similarity"]].copy()
        display["similarity"] = (display["similarity"] * 100).round(1).astype(str) + "%"
        display.columns = ["Subject ID", "Profile group", "Similarity"]

        st.write(f"Matching features found: {len(features)}")
        st.write(f"Number of historical patients: {len(ranked)}")
        st.dataframe(display, use_container_width=True)

        selected_historical = st.selectbox("Review a historical patient", ranked["subject_id"].tolist())
        match = ranked[ranked["subject_id"] == selected_historical].iloc[0]

        st.markdown(
            f"""
            <div class='card'>
                <h3>Historical patient</h3>
                <div><span class='tiny-tag'>Patient {selected_historical}</span> <span class='tiny-tag'>Similarity: {match['similarity'] * 100:.1f}%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("Shared matching features:")
        shared_features = match.get("shared_features", []) or ["No shared features"]
        st.markdown(" ".join(f"<span class='tag'>{feature}</span>" for feature in shared_features), unsafe_allow_html=True)

        with st.expander("Admissions"):
            adm = data["admissions"][data["admissions"]["subject_id"].astype(str) == str(selected_historical)]
            st.dataframe(adm, use_container_width=True)
        with st.expander("Diagnoses Icd"):
            dx_hist = data["diagnoses_icd"][data["diagnoses_icd"]["subject_id"].astype(str) == str(selected_historical)]
            st.dataframe(dx_hist, use_container_width=True)
        with st.expander("Prescriptions"):
            rx_hist = data["prescriptions"][data["prescriptions"]["subject_id"].astype(str) == str(selected_historical)]
            st.dataframe(rx_hist, use_container_width=True)
        with st.expander("Procedures Icd"):
            proc_hist = data["procedures_icd"][data["procedures_icd"]["subject_id"].astype(str) == str(selected_historical)]
            st.dataframe(proc_hist, use_container_width=True)
        with st.expander("Labevents"):
            labs_hist = data["labevents"][data["labevents"]["subject_id"].astype(str) == str(selected_historical)]
            st.dataframe(labs_hist, use_container_width=True)
        with st.expander("Discharge"):
            discharge_hist = data["discharge"][data["discharge"]["subject_id"].astype(str) == str(selected_historical)]
            st.dataframe(discharge_hist, use_container_width=True)
        with st.expander("Radiology"):
            rad_hist = data["radiology"][data["radiology"]["subject_id"].astype(str) == str(selected_historical)]
            st.dataframe(rad_hist, use_container_width=True)

st.caption(
    "Similarity is a simple Jaccard overlap across recorded conditions, medications, procedures, and allergies. It is an experimental retrieval score, not a measure of treatment effectiveness."
)

if not query.conditions and not query.medications and not query.measurements and (query.profile_group is not None):
    validation = validate_cohort_against_pandas(data, query, cohort)
    st.write("Profile-only pandas validation:", "PASS" if validation["validated"] else "FAIL", validation)
