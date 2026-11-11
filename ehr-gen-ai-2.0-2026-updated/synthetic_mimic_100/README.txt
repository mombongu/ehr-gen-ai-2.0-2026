SYNTHETIC MIMIC-INSPIRED DATASET
100 fictional patients: 10 NEW arrivals and 90 HISTORICAL patients.
Not official MIMIC-IV or OMOP CDM; all data are generated for software testing.
Patients 20001 and 20011 form a deterministic known-similar-pair test fixture.
Medication orders (prescriptions.csv) are separate from administration events.
Medication schedules vary realistically with PRN, once daily, twice daily, and once monthly patterns.
Allergy status distinguishes KNOWN, NO_KNOWN_ALLERGIES, UNKNOWN, and NOT_DOCUMENTED.
Historical outcome_status is explicit; PENDING means no outcome is available yet.
hospital_course.csv contains fictional assessments, response checks, complications, final diagnoses, and disposition notes.
Response events are temporal associations only and do not show treatment causality.
Never use these data to guide real patient care.
