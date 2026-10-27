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
st.set_page_config(page_title="EHR Care Dashboard", layout="wide")


STATUS_OPTIONS = ["Active", "Planned", "Completed", "Pending", "Hold"]
CATEGORY_MAP = {
    "Medication": "medications",
    "Procedure": "procedures",
    "Treatment": "treatments",
}


def initialize_care_plan():
    return {
        "medications": pd.DataFrame(
            [
                {"name": "Metformin", "dose": "500 mg", "status": "Active", "last_updated": "2026-10-27"},
                {"name": "Lisinopril", "dose": "10 mg", "status": "Active", "last_updated": "2026-10-27"},
            ]
        ),
        "procedures": pd.DataFrame(
            [
                {"name": "ECG", "status": "Completed", "date": "2026-10-26"},
                {"name": "Chest X-ray", "status": "Pending", "date": "2026-10-28"},
            ]
        ),
        "treatments": pd.DataFrame(
            [
                {"name": "IV fluids", "status": "Planned", "notes": "Monitor electrolytes"},
                {"name": "Observation", "status": "Active", "notes": "Repeat vitals q4h"},
            ]
        ),
    }


def summarize_status(df: pd.DataFrame) -> dict:
    if df.empty:
        return {status: 0 for status in STATUS_OPTIONS}
    counts = df["status"].fillna("Pending").value_counts().to_dict()
    summary = {status: counts.get(status, 0) for status in STATUS_OPTIONS}
    return summary


def add_new_item_dialog():
    if not hasattr(st, "dialog"):
        st.info("This Streamlit version does not support modal dialogs. Use the inline sidebar editor instead.")
        return

    @st.dialog("Add new care plan item")
    def _dialog():
        category = st.selectbox("Category", list(CATEGORY_MAP.keys()), index=0)
        key = CATEGORY_MAP[category]
        name = st.text_input("Name")

        if category == "Medication":
            dose = st.text_input("Dose")
            status = st.selectbox("Status", STATUS_OPTIONS, index=0)
            last_updated = st.text_input("Last updated", value=pd.Timestamp.today().strftime("%Y-%m-%d"))
            if st.form_submit_button("Add medication"):
                if not name.strip():
                    st.warning("Medication name is required.")
                    return
                new_df = st.session_state.care_plan[key].copy()
                new_df.loc[len(new_df)] = {
                    "name": name.strip(),
                    "dose": dose.strip() or "-",
                    "status": status,
                    "last_updated": last_updated,
                }
                st.session_state.care_plan[key] = new_df
                st.success(f"Added medication: {name.strip()}")
                st.rerun()
        elif category == "Procedure":
            date = st.text_input("Date", value=pd.Timestamp.today().strftime("%Y-%m-%d"))
            status = st.selectbox("Status", STATUS_OPTIONS, index=0)
            if st.form_submit_button("Add procedure"):
                if not name.strip():
                    st.warning("Procedure name is required.")
                    return
                new_df = st.session_state.care_plan[key].copy()
                new_df.loc[len(new_df)] = {
                    "name": name.strip(),
                    "status": status,
                    "date": date,
                }
                st.session_state.care_plan[key] = new_df
                st.success(f"Added procedure: {name.strip()}")
                st.rerun()
        else:
            status = st.selectbox("Status", STATUS_OPTIONS, index=0)
            notes = st.text_area("Notes", value="")
            if st.form_submit_button("Add treatment"):
                if not name.strip():
                    st.warning("Treatment name is required.")
                    return
                new_df = st.session_state.care_plan[key].copy()
                new_df.loc[len(new_df)] = {
                    "name": name.strip(),
                    "status": status,
                    "notes": notes.strip() or "-",
                }
                st.session_state.care_plan[key] = new_df
                st.success(f"Added treatment: {name.strip()}")
                st.rerun()

    _dialog()


if "care_plan" not in st.session_state:
    st.session_state.care_plan = initialize_care_plan()

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

        .stApp {
            background: transparent;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            color: var(--espresso);
        }

        .hero {
            background: rgba(85, 27, 20, 0.90);
            color: white;
            border-radius: 26px;
            padding: 1.6rem 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.2rem;
        }

        .hero-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .hero-title {
            color: white !important;
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: -0.04em;
        }

        .pill {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.28);
            background: rgba(205, 197, 180, 0.15);
            font-size: 0.74rem;
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
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.15);
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

        .card h3 {
            margin-top: 0;
            margin-bottom: 0.75rem;
        }

        .settings-title {
            color: white !important;
            margin: 0 0 0.8rem;
            font-size: 1.3rem;
        }

        .sidebar-section {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 0.8rem;
            margin-bottom: 0.9rem;
        }

        .sidebar-section h4 {
            color: white !important;
            margin-top: 0;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
            letter-spacing: 0.03em;
        }

        .stSelectbox > div, .stDateInput > div, .stTextInput > div, .stMultiSelect > div,
        .stTextarea > div {
            background: rgba(255,255,255,0.15);
            border-radius: 10px;
            color: white;
        }

        .stSelectbox label, .stTextInput label, .stDateInput label, .stMultiSelect label, .stSlider label,
        .stTextarea label {
            color: rgba(255,255,255,0.92);
        }

        .stDataFrame, .stTable {
            border-radius: 14px;
            overflow: hidden;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--mocha), var(--espresso));
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            width: 100%;
        }

        .stButton > button:hover {
            filter: brightness(1.08);
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

        .status-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.75rem;
            margin-top: 0.8rem;
        }

        .status-pill {
            background: rgba(255,255,255,0.22);
            border: 1px solid rgba(85,27,20,0.08);
            border-radius: 14px;
            padding: 0.75rem;
            text-align: center;
        }

        .status-name {
            display: block;
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .status-value {
            display: block;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--espresso);
            margin-top: 0.3rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.25);
            border: 1px solid rgba(85,27,20,0.08);
            border-radius: 16px;
            padding: 0.8rem;
            box-shadow: 0 10px 18px rgba(85,27,20,0.06);
        }
    </style>
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

with st.sidebar:
    st.markdown('<div class="settings-title">Patient filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("Care plan manager", divider="gray")
    if st.button("Add new medication / procedure / treatment", use_container_width=True):
        add_new_item_dialog()
    if st.button("Reset care plan", use_container_width=True):
        st.session_state.care_plan = initialize_care_plan()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    for key, label in [("medications", "Medications"), ("procedures", "Procedures"), ("treatments", "Treatments")]:
        df = st.session_state.care_plan[key]
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{key}",
            hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    options=STATUS_OPTIONS,
                    required=True,
                )
            },
        )
        st.session_state.care_plan[key] = edited
        st.caption(f"{label}: {len(edited)} items")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class='hero'>
      <div class='hero-row'>
        <div>
          <div class='pill'>Synthetic EHR dashboard</div>
          <h1 class='hero-title'>Care coordination overview</h1>
        </div>
        <div class='pill'>Live review</div>
      </div>
      <div class='kpi-grid'>
        <div class='kpi-card'>
          <div class='kpi-label'>Patients</div>
          <div class='kpi-value'>{total}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>New arrivals</div>
          <div class='kpi-value'>{new}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Historical</div>
          <div class='kpi-value'>{historical_total}</div>
        </div>
        <div class='kpi-card'>
          <div class='kpi-label'>Profiles</div>
          <div class='kpi-value'>{profiles}</div>
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

if new_patients.empty:
    st.warning("No patients labelled NEW.")
    st.stop()

selected_id = st.selectbox("Select active patient", new_patients["subject_id"].tolist(), index=0)
patient = patients[patients["subject_id"] == selected_id].iloc[0]

st.markdown(
    f"""
    <div class='card'>
        <h3>Active patient</h3>
        <div><span class='tiny-tag'>Patient #{selected_id}</span> <span class='tiny-tag'>Age {patient['anchor_age']}</span> <span class='tiny-tag'>Profile {patient['profile_group']}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Current care snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Medications", len(st.session_state.care_plan["medications"]))
c2.metric("Procedures", len(st.session_state.care_plan["procedures"]))
c3.metric("Treatments", len(st.session_state.care_plan["treatments"]))

st.subheader("Treatment status summary by category")
summary_columns = st.columns(3)
for idx, key in enumerate(["medications", "procedures", "treatments"]):
    label = key[:-1].title() if key != "treatments" else "Treatments"
    with summary_columns[idx]:
        summary = summarize_status(st.session_state.care_plan[key])
        st.markdown(f"<div class='card'><h3>{label}</h3><div class='status-summary'>", unsafe_allow_html=True)
        for status in STATUS_OPTIONS:
            st.markdown(
                f"""
                <div class='status-pill'>
                    <span class='status-name'>{status}</span>
                    <span class='status-value'>{summary[status]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

care_tabs = st.tabs(["Medications", "Procedures", "Treatments"])
with care_tabs[0]:
    st.dataframe(st.session_state.care_plan["medications"], use_container_width=True)
with care_tabs[1]:
    st.dataframe(st.session_state.care_plan["procedures"], use_container_width=True)
with care_tabs[2]:
    st.dataframe(st.session_state.care_plan["treatments"], use_container_width=True)

st.divider()
st.header("Query and match engine")


def split_terms(text):
    return [part.strip() for part in text.split(",") if part.strip()]


selected_profiles = profile_filter
query_profile = None if ("Any" in selected_profiles or not selected_profiles) else selected_profiles[0]
query = RetrievalQuery(
    conditions=split_terms(condition_text),
    medications=split_terms(medication_text),
    measurements=split_terms(measurement_text),
    profile_group=query_profile,
    start_date=start_date.isoformat() if start_date else None,
    end_date=end_date.isoformat() if end_date else None,
    medication_basis=basis,
    top_k=top_k,
)

cohort = select_cohort(data, query)
st.write(f"Matching patients: **{len(cohort)}**")
st.dataframe(cohort[["subject_id", "patient_status", "profile_group"]], use_container_width=True)

if not query.conditions and not query.medications and not query.measurements and (query.profile_group is not None):
    validation = validate_cohort_against_pandas(data, query, cohort)
    st.write("Profile-only pandas validation:", "PASS" if validation["validated"] else "FAIL", validation)

features = patient_features(data, selected_id, query)
if not features:
    st.info("No features match these filters. Broaden the query or review the date range.")
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

st.divider()
st.subheader("Encounter detail")
for name in ("edstays", "triage"):
    if name in data:
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        st.dataframe(data[name][data[name]["subject_id"].astype(str) == str(selected_id)], use_container_width=True)

if "allergies" in data:
    st.subheader("Allergy records")
    allergy_rows = data["allergies"][data["allergies"]["subject_id"].astype(str) == str(selected_id)]
    st.dataframe(allergy_rows, use_container_width=True)

st.caption(
    "Similarity is an experimental record-overlap score, not treatment effectiveness or a clinical recommendation. This dashboard is a software prototype and is intended for testing and data review workflows."
)
