import os
import pandas as pd

src = "/blue/cai6734/mombongu/ehr project/raw - from omer!/csv"
out = "/blue/cai6734/mombongu/ehr project/synthea_omop"
os.makedirs(out, exist_ok=True)

KEEP = ["Essential hypertension", "Anemia", "Ischemic heart disease", "Metabolic syndrome X"]

pid = {p: i for i, p in enumerate(pd.read_csv(src + "/patients.csv", usecols=["Id"]).Id)}

c = pd.read_csv(src + "/conditions.csv")
c = c[c.DESCRIPTION.str.contains(r"\(disorder\)", na=False)].copy()
c["condition_concept_name"] = c.DESCRIPTION.str.replace(r"\s*\(disorder\)", "", regex=True)
if KEEP:
    c = c[c.condition_concept_name.isin(KEEP)]
c["person_id"] = c.PATIENT.map(pid)
c["condition_concept_id"] = c.CODE
c["condition_start_date"] = c.START.astype(str).str[:10]
c[["person_id", "condition_concept_name", "condition_concept_id", "condition_start_date"]].to_csv(out + "/condition_occurrence.csv", index=False)

o = pd.read_csv(src + "/observations.csv", low_memory=False)
o = o[o.CATEGORY == "laboratory"].copy()
o["measurement_concept_id"], names = pd.factorize(o.DESCRIPTION)
pd.DataFrame({"concept_id": range(len(names)), "concept_name": names}).to_csv(out + "/measurement_concept_map.csv", index=False)
num = pd.to_numeric(o.VALUE, errors="coerce")
o["person_id"] = o.PATIENT.map(pid)
o["measurement_date"] = o.DATE.astype(str).str[:10]
o["value_as_number"] = num
o["value_as_string"] = o.VALUE.where(num.isna(), "")
o["unit_source_value"] = o.UNITS
o[["person_id", "measurement_concept_id", "measurement_date", "value_as_number", "value_as_string", "unit_source_value"]].to_csv(out + "/measurement.csv", index=False)

p = pd.read_csv(src + "/procedures.csv")
p = pd.DataFrame({"person_id": p.PATIENT.map(pid), "procedure_concept_name": p.DESCRIPTION, "procedure_date": p.START.astype(str).str[:10]})
im = pd.read_csv(src + "/imaging_studies.csv")
im = pd.DataFrame({"person_id": im.PATIENT.map(pid),
                   "procedure_concept_name": im.MODALITY_DESCRIPTION.astype(str) + " of " + im.BODYSITE_DESCRIPTION.astype(str),
                   "procedure_date": im.DATE.astype(str).str[:10]})
pd.concat([p, im]).to_csv(out + "/procedure_occurrence.csv", index=False)
print(os.listdir(out))
