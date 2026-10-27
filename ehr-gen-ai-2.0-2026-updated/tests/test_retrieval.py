from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import load_dataset, RetrievalQuery, select_cohort, validate_cohort_against_pandas, patient_features, jaccard_similarity

class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, str(ROOT / "generate_synthetic_data.py")], check=True, capture_output=True, text=True)
        cls.data = load_dataset(ROOT / "synthetic_mimic_100")

    def test_dataset_has_100_patients(self):
        self.assertEqual(len(self.data["patients"]), 100)

    def test_ten_new_patients(self):
        self.assertEqual((self.data["patients"]["patient_status"] == "NEW").sum(), 10)

    def test_ninety_historical_patients(self):
        self.assertEqual((self.data["patients"]["patient_status"] == "HISTORICAL").sum(), 90)

    def test_allergy_status_categories_are_explicit(self):
        statuses = set(self.data["allergies"]["status"])
        self.assertTrue(statuses <= {"KNOWN", "NO_KNOWN_ALLERGIES", "UNKNOWN", "NOT_DOCUMENTED"})
        self.assertIn("UNKNOWN", statuses)
        self.assertIn("NOT_DOCUMENTED", statuses)

    def test_no_allergy_row_is_not_inferred_as_no_known_allergies(self):
        pid = self.data["patients"].iloc[0]["subject_id"]
        features = patient_features(self.data, pid)
        self.assertTrue(any(f.startswith("allergy_status:") or f.startswith("allergy:known:") for f in features))

    def test_orders_and_administrations_are_separate_tables(self):
        self.assertIn("prescriptions", self.data)
        self.assertIn("medication_administrations", self.data)
        self.assertNotEqual(len(self.data["prescriptions"]), len(self.data["medication_administrations"]))

    def test_pending_arrivals_have_no_recorded_outcome(self):
        pending = self.data["admissions"][self.data["admissions"]["outcome_status"] == "PENDING"]
        self.assertTrue((pending["outcome_recorded"] == 0).all())
        self.assertTrue((pending["dischtime"] == "").all())

    def test_historical_outcome_is_not_inferred_from_discharge_location(self):
        hist = self.data["admissions"][self.data["admissions"]["outcome_recorded"] == 1]
        self.assertIn("outcome_status", hist.columns)
        self.assertGreater(hist["outcome_status"].nunique(), 1)

    def test_profile_cohort_filter(self):
        query = RetrievalQuery(profile_group="diabetes")
        cohort = select_cohort(self.data, query)
        self.assertTrue((cohort["profile_group"] == "diabetes").all())
        self.assertEqual(len(cohort), 20)

    def test_profile_cohort_matches_independent_pandas_reference(self):
        query = RetrievalQuery(profile_group="renal")
        cohort = select_cohort(self.data, query)
        result = validate_cohort_against_pandas(self.data, query, cohort)
        self.assertTrue(result["validated"])
        self.assertEqual(result["expected_count"], result["actual_count"])

    def test_condition_query_returns_matching_patients(self):
        query = RetrievalQuery(conditions=["E11.9"])
        cohort = select_cohort(self.data, query)
        self.assertGreater(len(cohort), 0)
        ids = set(cohort["subject_id"].astype(str))
        dx_ids = set(self.data["diagnoses_icd"].loc[self.data["diagnoses_icd"]["icd_code"] == "E11.9", "subject_id"].astype(str))
        self.assertEqual(ids, dx_ids)

    def test_medication_order_filter(self):
        cohort = select_cohort(self.data, RetrievalQuery(medications=["metformin"], medication_basis="orders"))
        self.assertGreater(len(cohort), 0)
        self.assertTrue(self.data["prescriptions"].loc[self.data["prescriptions"]["subject_id"].isin(cohort["subject_id"]), "drug"].str.contains("Metformin", case=False).any())

    def test_administration_filter_uses_administration_table(self):
        cohort = select_cohort(self.data, RetrievalQuery(medications=["metformin"], medication_basis="administrations"))
        self.assertGreater(len(cohort), 0)
        self.assertTrue(self.data["medication_administrations"].loc[self.data["medication_administrations"]["subject_id"].isin(cohort["subject_id"]), "drug"].str.contains("Metformin", case=False).any())

    def test_measurement_filter(self):
        cohort = select_cohort(self.data, RetrievalQuery(measurements=["glucose"]))
        self.assertGreater(len(cohort), 0)
        self.assertTrue(self.data["labevents"].loc[self.data["labevents"]["subject_id"].isin(cohort["subject_id"]), "label"].str.contains("glucose", case=False).any())

    def test_date_window_restricts_medication_cohort(self):
        all_ids = set(select_cohort(self.data, RetrievalQuery(medications=["metformin"]))["subject_id"].astype(str))
        future_ids = set(select_cohort(self.data, RetrievalQuery(medications=["metformin"], start_date="2030-01-01", end_date="2030-12-31"))["subject_id"].astype(str))
        self.assertLessEqual(future_ids, all_ids)
        self.assertEqual(future_ids, set())

    def test_similarity_is_symmetric(self):
        a, b = {"condition:e11.9", "medication:metformin"}, {"condition:e11.9"}
        self.assertEqual(jaccard_similarity(a, b), jaccard_similarity(b, a))

    def test_similarity_bounds(self):
        score = jaccard_similarity({"a"}, {"a", "b"})
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_empty_feature_sets_return_zero(self):
        self.assertEqual(jaccard_similarity(set(), {"a"}), 0.0)

    def test_new_patient_not_returned_as_historical_candidate(self):
        from retrieval import rank_historical_patients
        result = rank_historical_patients(self.data, 20001, RetrievalQuery(top_k=20))
        new_ids = set(self.data["patients"].loc[self.data["patients"]["patient_status"] == "NEW", "subject_id"])
        self.assertTrue(new_ids.isdisjoint(set(result["subject_id"])))

    def test_top_k_is_respected(self):
        from retrieval import rank_historical_patients
        result = rank_historical_patients(self.data, 20001, RetrievalQuery(top_k=7))
        self.assertLessEqual(len(result), 7)

if __name__ == "__main__":
    unittest.main()
