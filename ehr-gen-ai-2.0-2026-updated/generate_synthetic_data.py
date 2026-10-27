"""Generate a deterministic, fictional MIMIC-inspired dataset for software tests.

This is NOT official MIMIC-IV data or the OMOP CDM. Never use it for clinical care.
Run: python generate_synthetic_data.py
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20261027
OUT = Path(__file__).parent / "synthetic_mimic_100"
TODAY = date(2025, 6, 1)
N_PATIENTS = 100
N_NEW = 10

PROFILES = {
    "diabetes": {
        "conditions": [("E11.9", "Type 2 diabetes mellitus without complications"),
                       ("I10", "Essential hypertension"), ("E78.5", "Hyperlipidemia")],
        "medications": ["Metformin", "Insulin glargine", "Atorvastatin", "Lisinopril"],
        "procedures": ["Hemoglobin A1c testing", "Electrocardiogram"],
        "labs": [("Glucose", "mg/dL", 165, 260), ("Creatinine", "mg/dL", .7, 1.5),
                 ("Hemoglobin A1c", "%", 7, 11.5)],
    },
    "cardiac": {
        "conditions": [("I50.9", "Heart failure, unspecified"), ("I10", "Essential hypertension"),
                       ("I25.10", "Atherosclerotic heart disease")],
        "medications": ["Furosemide", "Lisinopril", "Metoprolol", "Amlodipine"],
        "procedures": ["Electrocardiogram", "Chest radiograph"],
        "labs": [("Creatinine", "mg/dL", .8, 2), ("Sodium", "mmol/L", 130, 140),
                 ("Potassium", "mmol/L", 3.2, 5.4)],
    },
    "respiratory": {
        "conditions": [("J18.9", "Pneumonia, unspecified organism"),
                       ("J44.1", "COPD with acute exacerbation"), ("J45.909", "Asthma, unspecified")],
        "medications": ["Albuterol", "Prednisone", "Azithromycin", "Fluticasone"],
        "procedures": ["Chest radiograph", "Oxygen therapy"],
        "labs": [("Oxygen saturation", "%", 86, 98), ("White blood cells", "K/uL", 7, 19),
                 ("Creatinine", "mg/dL", .6, 1.4)],
    },
    "infection": {
        "conditions": [("A41.9", "Sepsis, unspecified organism"),
                       ("N39.0", "Urinary tract infection"), ("J06.9", "Acute upper respiratory infection")],
        "medications": ["Ceftriaxone", "Vancomycin", "Acetaminophen", "Azithromycin"],
        "procedures": ["Blood culture", "Intravenous fluid administration"],
        "labs": [("White blood cells", "K/uL", 3, 24), ("Lactate", "mmol/L", 1, 5),
                 ("Creatinine", "mg/dL", .8, 2.4)],
    },
    "renal": {
        "conditions": [("N18.3", "Chronic kidney disease, stage 3"),
                       ("I10", "Essential hypertension"), ("E87.1", "Hyponatremia")],
        "medications": ["Furosemide", "Amlodipine", "Lisinopril", "Levothyroxine"],
        "procedures": ["Renal ultrasound", "Electrocardiogram"],
        "labs": [("Creatinine", "mg/dL", 1.3, 3.5), ("Sodium", "mmol/L", 125, 138),
                 ("Potassium", "mmol/L", 3.4, 5.8)],
    },
}

EXTRA_CONDITIONS = [
    ("E78.5", "Hyperlipidemia"),
    ("J30.9", "Allergic rhinitis"),
    ("M54.5", "Low back pain"),
    ("F41.9", "Anxiety disorder"),
    ("F32.9", "Major depressive disorder"),
    ("M06.9", "Rheumatoid arthritis"),
    ("K21.9", "Gastroesophageal reflux disease"),
    ("J02.9", "Acute pharyngitis"),
    ("K52.9", "Noninfective gastroenteritis and colitis"),
    ("L50.9", "Urticaria"),
]

EXTRA_MEDICATIONS = [
    "Acetaminophen",
    "Ibuprofen",
    "Omeprazole",
    "Cetirizine",
    "Losartan",
    "Sertraline",
    "Gabapentin",
    "Levothyroxine",
    "Hydrocortisone",
    "Infliximab",
    "Denosumab",
]

MEDICATION_FREQUENCIES = {
    "acetaminophen": "PRN",
    "ibuprofen": "PRN",
    "albuterol": "PRN",
    "cetirizine": "once daily",
    "furosemide": "once daily",
    "lisinopril": "once daily",
    "metoprolol": "once daily",
    "amlodipine": "once daily",
    "metformin": "twice daily",
    "insulin glargine": "once daily",
    "atorvastatin": "once daily",
    "levothyroxine": "once daily",
    "prednisone": "once daily",
    "azithromycin": "once daily",
    "ceftriaxone": "once daily",
    "vancomycin": "once daily",
    "fluticasone": "once daily",
    "losartan": "once daily",
    "sertraline": "once daily",
    "gabapentin": "twice daily",
    "omeprazole": "once daily",
    "hydrocortisone": "PRN",
    "infliximab": "once a month",
    "denosumab": "once a month",
}

ALLERGIES = [
    ("Penicillin", "Rash", "KNOWN"),
    ("Sulfonamide antibiotics", "Hives", "KNOWN"),
    ("Latex", "Contact dermatitis", "KNOWN"),
    ("Ibuprofen", "Urticaria", "KNOWN"),
    ("", "", "NO_KNOWN_ALLERGIES"),
    ("", "", "UNKNOWN"),
    ("", "", "NOT_DOCUMENTED"),
]


def medication_frequency(med: str) -> str:
    med_key = med.lower().strip()
    if med_key in MEDICATION_FREQUENCIES:
        return MEDICATION_FREQUENCIES[med_key]
    return random.choice(["PRN", "once daily", "twice daily", "once a month"])


def build_patient_conditions(profile: dict, rng: random.Random) -> list[tuple[str, str]]:
    conditions = list(profile["conditions"])
    if rng.random() < 0.75:
        extra = rng.sample(EXTRA_CONDITIONS, k=rng.randint(1, 2))
        conditions.extend(extra)
    if rng.random() < 0.30:
        conditions.append(("I10", "Essential hypertension"))
    return list(dict.fromkeys(conditions))


def build_patient_meds(profile: dict, rng: random.Random) -> list[tuple[str, str]]:
    meds = list(profile["medications"])
    if rng.random() < 0.65:
        extra = rng.sample(EXTRA_MEDICATIONS, k=rng.randint(1, 2))
        meds.extend(extra)
    meds = list(dict.fromkeys(meds))
    return [(med, medication_frequency(med)) for med in meds]


def write_csv(filename, rows):
    if not rows:
        return
    with (OUT / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dt(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def main():
    rng = random.Random(SEED)
    OUT.mkdir(exist_ok=True)
    patients, admissions, diagnoses, prescriptions, administrations = [], [], [], [], []
    procedures, labs, charts, edstays, triage, allergies, discharges, radiology = ([] for _ in range(8))
    ids = {k: 1 for k in ["adm", "diag", "rx", "admin", "proc", "lab", "chart", "ed", "triage", "allergy", "discharge", "radiology"]}

    for i in range(1, N_PATIENTS + 1):
        pid = 20000 + i
        is_new = i <= N_NEW
        profile_name = list(PROFILES)[(i - 1) % len(PROFILES)]
        profile = PROFILES[profile_name]
        age = TODAY.year - rng.randint(1940, 2000)
        race = rng.choice(["White", "Black", "Asian", "Other", "Unknown"])
        patients.append({
            "subject_id": pid, "gender": rng.choice(["F", "M"]), "anchor_age": age,
            "anchor_year": TODAY.year, "anchor_year_group": "2017 - 2019", "dod": "",
            "patient_status": "NEW" if is_new else "HISTORICAL", "profile_group": profile_name,
            "race_label": race, "synthetic_flag": 1,
        })

        allergy_name, reaction, allergy_status = rng.choice(ALLERGIES)
        allergies.append({
            "allergy_id": ids["allergy"], "subject_id": pid, "allergy_name": allergy_name,
            "reaction": reaction, "status": allergy_status, "recorded_time": dt(rng.randint(30, 700)),
            "synthetic_flag": 1,
        })
        ids["allergy"] += 1

        prior_hadm = 30000 + pid
        prior_days = rng.randint(40, 700)
        start, end = dt(prior_days), dt(max(1, prior_days - rng.randint(1, 8)))
        discharge_location = rng.choice(["HOME", "HOME", "SKILLED NURSING FACILITY"])
        outcome = rng.choice(["DISCHARGED_ALIVE", "TRANSFERRED", "LEFT_AGAINST_MEDICAL_ADVICE"])
        if outcome == "TRANSFERRED":
            discharge_location = "SKILLED NURSING FACILITY"
        admissions.append({
            "subject_id": pid, "hadm_id": prior_hadm, "admittime": start + " 08:00:00",
            "dischtime": end + " 14:00:00", "deathtime": "",
            "admission_type": rng.choice(["EMERGENCY", "URGENT", "ELECTIVE"]),
            "admission_location": "EMERGENCY DEPARTMENT", "discharge_location": discharge_location,
            "outcome_status": outcome, "outcome_recorded": 1, "insurance": rng.choice(["Medicare", "Medicaid", "Other"]),
            "language": "English", "marital_status": rng.choice(["MARRIED", "SINGLE", "WIDOWED", "UNKNOWN"]),
            "race": race, "synthetic_flag": 1,
        })

        conditions = build_patient_conditions(profile, rng)
        for code, label in dict(conditions).items():
            diagnoses.append({"subject_id": pid, "hadm_id": prior_hadm, "seq_num": ids["diag"],
                              "icd_code": code, "icd_version": 10, "long_title": label, "synthetic_flag": 1})
            ids["diag"] += 1

        meds_with_freq = build_patient_meds(profile, rng)
        for med, frequency in meds_with_freq:
            rx_id = ids["rx"]
            prescriptions.append({
                "prescription_id": rx_id, "subject_id": pid, "hadm_id": prior_hadm,
                "starttime": start + " 09:00:00", "stoptime": end + " 09:00:00", "drug": med,
                "dose_val_rx": rng.choice(["5", "10", "20", "500"]), "dose_unit_rx": "mg",
                "route": rng.choice(["PO", "IV", "INHALATION"]), "frequency": frequency, "synthetic_flag": 1,
            })
            if rng.random() < .82:
                administrations.append({
                    "administration_id": ids["admin"], "prescription_id": rx_id, "subject_id": pid,
                    "hadm_id": prior_hadm, "charttime": start + " 10:00:00", "drug": med,
                    "dose_administered": rng.choice(["5", "10", "20", "500"]), "dose_unit": "mg",
                    "route": "PO", "administration_status": "GIVEN", "frequency": frequency, "synthetic_flag": 1,
                })
                ids["admin"] += 1
            ids["rx"] += 1

        for proc in profile["procedures"]:
            procedures.append({"subject_id": pid, "hadm_id": prior_hadm, "seq_num": ids["proc"],
                               "chartdate": start, "procedure_name": proc, "icd_code": "",
                               "icd_version": "", "synthetic_flag": 1})
            ids["proc"] += 1
        for name, unit, low, high in profile["labs"]:
            labs.append({"subject_id": pid, "hadm_id": prior_hadm, "itemid": ids["lab"],
                         "charttime": start + " 10:00:00", "label": name,
                         "valuenum": round(rng.uniform(low, high), 2), "valueuom": unit,
                         "flag": "", "synthetic_flag": 1})
            ids["lab"] += 1
        for label, value, unit in [
            ("Heart Rate", rng.randint(65, 125), "bpm"),
            ("Respiratory Rate", rng.randint(12, 30), "breaths/min"),
            ("Temperature", round(rng.uniform(36.2, 39.0), 1), "C"),
            ("Oxygen Saturation", rng.randint(87, 100), "%"),
            ("Systolic Blood Pressure", rng.randint(90, 175), "mmHg"),
        ]:
            charts.append({"subject_id": pid, "hadm_id": prior_hadm, "stay_id": "",
                           "itemid": ids["chart"], "charttime": start + " 10:15:00",
                           "label": label, "valuenum": value, "valueuom": unit, "synthetic_flag": 1})
            ids["chart"] += 1

        discharges.append({"note_id": ids["discharge"], "subject_id": pid, "hadm_id": prior_hadm,
                           "charttime": end + " 14:00:00", "note_type": "Discharge summary",
                           "text": f"SYNTHETIC SUMMARY: {profile_name} profile. Fictional record for software testing only.",
                           "synthetic_flag": 1})
        ids["discharge"] += 1
        radiology.append({"note_id": ids["radiology"], "subject_id": pid, "hadm_id": prior_hadm,
                          "charttime": start + " 11:00:00",
                          "report": f"SYNTHETIC RADIOLOGY REPORT: {rng.choice(profile['procedures'])}. No real image interpreted.",
                          "synthetic_flag": 1})
        ids["radiology"] += 1

        if is_new:
            current_hadm = 40000 + pid
            chief = {
                "diabetes": "Weakness and elevated blood glucose",
                "cardiac": "Shortness of breath and leg swelling",
                "respiratory": "Cough and difficulty breathing",
                "infection": "Fever and generalized weakness",
                "renal": "Fatigue and reduced urine output",
            }[profile_name]
            admissions.append({
                "subject_id": pid, "hadm_id": current_hadm, "admittime": TODAY.isoformat() + " 09:00:00",
                "dischtime": "", "deathtime": "", "admission_type": "EMERGENCY",
                "admission_location": "EMERGENCY DEPARTMENT", "discharge_location": "",
                "outcome_status": "PENDING", "outcome_recorded": 0,
                "insurance": rng.choice(["Medicare", "Medicaid", "Other"]), "language": "English",
                "marital_status": "UNKNOWN", "race": race, "synthetic_flag": 1,
            })
            edstays.append({"subject_id": pid, "stay_id": 50000 + pid, "hadm_id": current_hadm,
                            "intime": TODAY.isoformat() + " 09:00:00", "outtime": "",
                            "arrival_transport": rng.choice(["WALK IN", "EMS"]), "disposition": "IN PROGRESS",
                            "chief_complaint": chief, "patient_status": "NEW", "synthetic_flag": 1})
            triage.append({"subject_id": pid, "stay_id": 50000 + pid,
                           "charttime": TODAY.isoformat() + " 09:10:00", "chiefcomplaint": chief,
                           "temperature": round(rng.uniform(36.4, 38.8), 1), "heartrate": rng.randint(75, 125),
                           "resprate": rng.randint(14, 30), "o2sat": rng.randint(88, 99),
                           "sbp": rng.randint(95, 170), "dbp": rng.randint(55, 100),
                           "acuity": rng.randint(2, 4), "synthetic_flag": 1})

    tables = {
        "patients.csv": patients, "admissions.csv": admissions, "diagnoses_icd.csv": diagnoses,
        "prescriptions.csv": prescriptions, "medication_administrations.csv": administrations,
        "procedures_icd.csv": procedures, "labevents.csv": labs, "chartevents.csv": charts,
        "edstays.csv": edstays, "triage.csv": triage, "allergies.csv": allergies,
        "discharge.csv": discharges, "radiology.csv": radiology,
    }
    for filename, rows in tables.items():
        write_csv(filename, rows)
    (OUT / "README.txt").write_text(
        "SYNTHETIC MIMIC-INSPIRED DATASET\n"
        "100 fictional patients: 10 NEW arrivals and 90 HISTORICAL patients.\n"
        "Not official MIMIC-IV or OMOP CDM; all data are generated for software testing.\n"
        "Medication orders (prescriptions.csv) are separate from administration events.\n"
        "Medication schedules vary realistically with PRN, once daily, twice daily, and once monthly patterns.\n"
        "Allergy status distinguishes KNOWN, NO_KNOWN_ALLERGIES, UNKNOWN, and NOT_DOCUMENTED.\n"
        "Historical outcome_status is explicit; PENDING means no outcome is available yet.\n"
        "Never use these data to guide real patient care.\n", encoding="utf-8")
    print(f"Generated {len(patients)} fictional patients and {len(tables)} CSV tables in {OUT}")


if __name__ == "__main__":
    main()
