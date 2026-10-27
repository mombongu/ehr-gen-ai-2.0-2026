import csv
import random
from pathlib import Path
from datetime import date, timedelta

# Entirely synthetic data. MIMIC-inspired structure, not real MIMIC data.
random.seed(20261027)
OUT = Path("synthetic_mimic_100")
OUT.mkdir(exist_ok=True)

TODAY = date(2025, 6, 1)
NEW_COUNT = 10

# Conditions, ICD-style codes, medications, procedures and common patterns.
PROFILES = {
    "diabetes": {
        "conditions": [
            ("E11.9", "Type 2 diabetes mellitus without complications"),
            ("I10", "Essential hypertension"),
            ("E78.5", "Hyperlipidemia, unspecified"),
        ],
        "medications": ["Metformin", "Insulin glargine", "Atorvastatin"],
        "procedures": ["Hemoglobin A1c testing", "Electrocardiogram"],
        "labs": [("Glucose", "mg/dL", 165, 260),
                 ("Creatinine", "mg/dL", 0.7, 1.5),
                 ("Hemoglobin A1c", "%", 7.0, 11.5)],
    },
    "cardiac": {
        "conditions": [
            ("I50.9", "Heart failure, unspecified"),
            ("I10", "Essential hypertension"),
            ("I25.10", "Atherosclerotic heart disease"),
        ],
        "medications": ["Furosemide", "Lisinopril", "Metoprolol"],
        "procedures": ["Electrocardiogram", "Chest radiograph"],
        "labs": [("Creatinine", "mg/dL", 0.8, 2.0),
                 ("Sodium", "mmol/L", 130, 140),
                 ("Potassium", "mmol/L", 3.2, 5.4)],
    },
    "respiratory": {
        "conditions": [
            ("J18.9", "Pneumonia, unspecified organism"),
            ("J44.1", "COPD with acute exacerbation"),
        ],
        "medications": ["Albuterol", "Prednisone", "Azithromycin"],
        "procedures": ["Chest radiograph", "Oxygen therapy"],
        "labs": [("Oxygen saturation", "%", 86, 98),
                 ("White blood cells", "K/uL", 7, 19),
                 ("Creatinine", "mg/dL", 0.6, 1.4)],
    },
    "infection": {
        "conditions": [
            ("A41.9", "Sepsis, unspecified organism"),
            ("N39.0", "Urinary tract infection, site not specified"),
        ],
        "medications": ["Ceftriaxone", "Vancomycin", "Acetaminophen"],
        "procedures": ["Blood culture", "Intravenous fluid administration"],
        "labs": [("White blood cells", "K/uL", 3, 24),
                 ("Lactate", "mmol/L", 1.0, 5.0),
                 ("Creatinine", "mg/dL", 0.8, 2.4)],
    },
    "renal": {
        "conditions": [
            ("N18.3", "Chronic kidney disease, stage 3"),
            ("I10", "Essential hypertension"),
            ("E87.1", "Hypo-osmolality and hyponatremia"),
        ],
        "medications": ["Furosemide", "Amlodipine", "Lisinopril"],
        "procedures": ["Renal ultrasound", "Electrocardiogram"],
        "labs": [("Creatinine", "mg/dL", 1.3, 3.5),
                 ("Sodium", "mmol/L", 125, 138),
                 ("Potassium", "mmol/L", 3.4, 5.8)],
    },
}

ALLERGIES = [
    ("Penicillin", "Rash"),
    ("Sulfonamide antibiotics", "Hives"),
    ("Latex", "Contact dermatitis"),
    ("Ibuprofen", "Urticaria"),
    ("No known allergies", ""),
]

def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def dt(days_ago):
    return (TODAY - timedelta(days=days_ago)).isoformat()

patients, admissions, diagnoses = [], [], []
prescriptions, procedures, labs, charts = [], [], [], []
edstays, triage, allergies = [], [], []
discharges, radiology = [], []

admission_seq = diagnosis_seq = prescription_seq = 1
procedure_seq = lab_seq = chart_seq = 1
ed_seq = triage_seq = 1
allergy_seq = discharge_seq = radiology_seq = 1

for i in range(1, 101):
    pid = 20000 + i
    is_new = i <= NEW_COUNT
    profile_name = list(PROFILES)[(i - 1) % len(PROFILES)]
    profile = PROFILES[profile_name]

    gender = random.choice(["F", "M"])
    birth_year = random.randint(1940, 2000)
    age = TODAY.year - birth_year
    race = random.choice([
        "White", "Black", "Asian", "Other", "Unknown"
    ])

    patients.append({
        "subject_id": pid,
        "gender": gender,
        "anchor_age": age,
        "anchor_year": TODAY.year,
        "anchor_year_group": "2017 - 2019",
        "dod": "",
        "patient_status": "NEW" if is_new else "HISTORICAL",
        "profile_group": profile_name,
        "race_label": race,
        "synthetic_flag": 1,
    })

    # All patients have a documented allergy status. "Unknown" is
    # distinct from "No known allergies".
    allergy = random.choice(ALLERGIES)
    allergy_text = allergy[0]
    allergy_status = (
        "NO_KNOWN_ALLERGIES" if allergy_text == "No known allergies"
        else "UNKNOWN" if allergy_text == "Unknown"
        else "KNOWN"
    )
    allergies.append({
        "allergy_id": allergy_seq,
        "subject_id": pid,
        "allergy_name": allergy_text,
        "reaction": allergy[1],
        "status": allergy_status,
        "recorded_time": dt(random.randint(30, 700)),
        "synthetic_flag": 1,
    })
    allergy_seq += 1

    # Create one historical encounter for every patient. This provides
    # a prior clinical profile even when the person is a new arrival.
    prior_adm_id = 30000 + pid
    prior_days = random.randint(40, 700)
    prior_start = dt(prior_days)
    prior_end = dt(max(1, prior_days - random.randint(1, 8)))

    admissions.append({
        "subject_id": pid,
        "hadm_id": prior_adm_id,
        "admittime": prior_start + " 08:00:00",
        "dischtime": prior_end + " 14:00:00",
        "deathtime": "",
        "admission_type": random.choice(
            ["EMERGENCY", "URGENT", "ELECTIVE"]
        ),
        "admission_location": "EMERGENCY DEPARTMENT",
        "discharge_location": random.choice(
            ["HOME", "HOME", "SKILLED NURSING FACILITY"]
        ),
        "insurance": random.choice(
            ["Medicare", "Medicaid", "Other"]
        ),
        "language": "English",
        "marital_status": random.choice(
            ["MARRIED", "SINGLE", "WIDOWED", "UNKNOWN"]
        ),
        "race": race,
        "synthetic_flag": 1,
    })

    # Diagnoses: core profile plus occasional overlapping comorbidity.
    conds = list(profile["conditions"])
    if random.random() < 0.35:
        conds.append(("I10", "Essential hypertension"))

    for code, label in dict(conds).items():
        diagnoses.append({
            "subject_id": pid,
            "hadm_id": prior_adm_id,
            "seq_num": 1 if diagnosis_seq else 1,
            "icd_code": code,
            "icd_version": 10,
            "long_title": label,
            "synthetic_flag": 1,
        })
        diagnosis_seq += 1

    meds = list(profile["medications"])
    if random.random() < 0.30:
        meds.append(random.choice([
            "Acetaminophen", "Aspirin", "Omeprazole"
        ]))

    for med in dict.fromkeys(meds):
        prescriptions.append({
            "subject_id": pid,
            "hadm_id": prior_adm_id,
            "pharmacy_id": prescription_seq,
            "starttime": prior_start + " 09:00:00",
            "stoptime": prior_end + " 09:00:00",
            "drug": med,
            "dose_val_rx": random.choice(["5", "10", "20", "500"]),
            "dose_unit_rx": "mg",
            "route": random.choice(["PO", "IV", "INHALATION"]),
            "synthetic_flag": 1,
        })
        prescription_seq += 1

    for proc in profile["procedures"]:
        procedures.append({
            "subject_id": pid,
            "hadm_id": prior_adm_id,
            "seq_num": procedure_seq,
            "chartdate": prior_start,
            "procedure_name": proc,
            "icd_code": "",
            "icd_version": "",
            "synthetic_flag": 1,
        })
        procedure_seq += 1

    for name, unit, low, high in profile["labs"]:
        value = round(random.uniform(low, high), 2)
        labs.append({
            "subject_id": pid,
            "hadm_id": prior_adm_id,
            "itemid": lab_seq,
            "charttime": prior_start + " 10:00:00",
            "label": name,
            "valuenum": value,
            "valueuom": unit,
            "flag": "",
            "synthetic_flag": 1,
        })
        lab_seq += 1

    # A small set of observations with realistic-looking values.
    obs = [
        ("Heart Rate", random.randint(65, 125), "bpm"),
        ("Respiratory Rate", random.randint(12, 30), "breaths/min"),
        ("Temperature", round(random.uniform(36.2, 39.0), 1), "C"),
        ("Oxygen Saturation", random.randint(87, 100), "%"),
        ("Systolic Blood Pressure", random.randint(90, 175), "mmHg"),
    ]
    for label, value, unit in obs:
        charts.append({
            "subject_id": pid,
            "hadm_id": prior_adm_id,
            "stay_id": "",
            "itemid": chart_seq,
            "charttime": prior_start + " 10:15:00",
            "label": label,
            "valuenum": value,
            "valueuom": unit,
            "synthetic_flag": 1,
        })
        chart_seq += 1

    discharge_text = (
        f"SYNTHETIC SUMMARY: Patient evaluated for {profile_name}-related "
        f"conditions. Recorded treatment included "
        f"{', '.join(meds[:2])}. Follow-up and response should be reviewed "
        f"against the associated synthetic laboratory and observation "
        f"records. This summary is generated fictional text, not a real "
        f"clinical note or treatment recommendation."
    )
    discharges.append({
        "note_id": discharge_seq,
        "subject_id": pid,
        "hadm_id": prior_adm_id,
        "charttime": prior_end + " 14:00:00",
        "note_type": "Discharge summary",
        "text": discharge_text,
        "synthetic_flag": 1,
    })
    discharge_seq += 1

    radiology.append({
        "note_id": radiology_seq,
        "subject_id": pid,
        "hadm_id": prior_adm_id,
        "charttime": prior_start + " 11:00:00",
        "report": (
            f"SYNTHETIC RADIOLOGY REPORT: {random.choice(profile['procedures'])}. "
            "Fictional illustrative report; no real image was interpreted."
        ),
        "synthetic_flag": 1,
    })
    radiology_seq += 1

    # Every new patient gets a current ED arrival with no future outcome.
    # Historical patients have completed prior encounters only.
    if is_new:
        current_hadm = 40000 + pid
        arrival = dt(0)

        admissions.append({
            "subject_id": pid,
            "hadm_id": current_hadm,
            "admittime": arrival + " 09:00:00",
            "dischtime": "",
            "deathtime": "",
            "admission_type": "EMERGENCY",
            "admission_location": "EMERGENCY DEPARTMENT",
            "discharge_location": "",
            "insurance": random.choice(["Medicare", "Medicaid", "Other"]),
            "language": "English",
            "marital_status": "UNKNOWN",
            "race": race,
            "synthetic_flag": 1,
        })

        chief = {
            "diabetes": "Weakness and elevated blood glucose",
            "cardiac": "Shortness of breath and leg swelling",
            "respiratory": "Cough and difficulty breathing",
            "infection": "Fever and generalized weakness",
            "renal": "Fatigue and reduced urine output",
        }[profile_name]

        edstays.append({
            "subject_id": pid,
            "stay_id": 50000 + pid,
            "hadm_id": current_hadm,
            "intime": arrival + " 09:00:00",
            "outtime": "",
            "arrival_transport": random.choice(["WALK IN", "EMS"]),
            "disposition": "IN PROGRESS",
            "chief_complaint": chief,
            "patient_status": "NEW",
            "synthetic_flag": 1,
        })

        triage.append({
            "subject_id": pid,
            "stay_id": 50000 + pid,
            "charttime": arrival + " 09:10:00",
            "chiefcomplaint": chief,
            "temperature": round(random.uniform(36.4, 38.8), 1),
            "heartrate": random.randint(75, 125),
            "resprate": random.randint(14, 30),
            "o2sat": random.randint(88, 99),
            "sbp": random.randint(95, 170),
            "dbp": random.randint(55, 100),
            "acuity": random.randint(2, 4),
            "synthetic_flag": 1,
        })

# Save MIMIC-inspired tables.
tables = {
    "patients.csv": (patients, [
        "subject_id", "gender", "anchor_age", "anchor_year",
        "anchor_year_group", "dod", "patient_status", "profile_group",
        "race_label", "synthetic_flag"
    ]),
    "admissions.csv": (admissions, [
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
        "admission_type", "admission_location", "discharge_location",
        "insurance", "language", "marital_status", "race", "synthetic_flag"
    ]),
    "diagnoses_icd.csv": (diagnoses, [
        "subject_id", "hadm_id", "seq_num", "icd_code", "icd_version",
        "long_title", "synthetic_flag"
    ]),
    "prescriptions.csv": (prescriptions, [
        "subject_id", "hadm_id", "pharmacy_id", "starttime", "stoptime",
        "drug", "dose_val_rx", "dose_unit_rx", "route", "synthetic_flag"
    ]),
    "procedures_icd.csv": (procedures, [
        "subject_id", "hadm_id", "seq_num", "chartdate", "procedure_name",
        "icd_code", "icd_version", "synthetic_flag"
    ]),
    "labevents.csv": (labs, [
        "subject_id", "hadm_id", "itemid", "charttime", "label",
        "valuenum", "valueuom", "flag", "synthetic_flag"
    ]),
    "chartevents.csv": (charts, [
        "subject_id", "hadm_id", "stay_id", "itemid", "charttime",
        "label", "valuenum", "valueuom", "synthetic_flag"
    ]),
    "edstays.csv": (edstays, [
        "subject_id", "stay_id", "hadm_id", "intime", "outtime",
        "arrival_transport", "disposition", "chief_complaint",
        "patient_status", "synthetic_flag"
    ]),
    "triage.csv": (triage, [
        "subject_id", "stay_id", "charttime", "chiefcomplaint",
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
        "acuity", "synthetic_flag"
    ]),
    "allergies.csv": (allergies, [
        "allergy_id", "subject_id", "allergy_name", "reaction",
        "status", "recorded_time", "synthetic_flag"
    ]),
    "discharge.csv": (discharges, [
        "note_id", "subject_id", "hadm_id", "charttime", "note_type",
        "text", "synthetic_flag"
    ]),
    "radiology.csv": (radiology, [
        "note_id", "subject_id", "hadm_id", "charttime", "report",
        "synthetic_flag"
    ]),
}

for filename, (rows, fields) in tables.items():
    write_csv(filename, rows, fields)

(OUT / "README.txt").write_text(
    "SYNTHETIC MIMIC-INSPIRED DATASET\n"
    "100 fictional patients: 10 NEW arrivals and 90 HISTORICAL patients.\n"
    "All records are generated, not sourced from real patient records.\n"
    "Tables use MIMIC-inspired names and selected familiar field names;\n"
    "they are NOT a complete, official MIMIC-IV schema or OMOP CDM.\n"
    "Medication and condition names are recognizable clinical terms.\n"
    "ICD-style codes are illustrative and must be validated before clinical use.\n"
    "Synthetic summaries and outcomes are for software testing only.\n"
    "Do not use these records to guide real patient care.\n",
    encoding="utf-8",
)

print(f"Created {len(patients)} synthetic patients in {OUT.resolve()}")
print(f"New arrivals: {sum(p['patient_status'] == 'NEW' for p in patients)}")
print(f"Historical patients: {sum(p['patient_status'] == 'HISTORICAL' for p in patients)}")
print(f"CSV files created: {len(tables)}")
