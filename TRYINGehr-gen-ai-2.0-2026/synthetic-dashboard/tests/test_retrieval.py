from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import (
    RetrievalQuery,
    idf_weighted_jaccard,
    load_dataset,
    patient_features,
    rank_historical_patients,
    select_cohort,
    validate_cohort_against_pandas,
    jaccard_similarity,
)

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

    def test_synthetic_hospital_course_includes_response_complication_and_disposition(self):
        course = self.data["hospital_course"]
        self.assertEqual(course["subject_id"].nunique(), 100)
        self.assertEqual(set(course["event_type"]), {
            "Initial assessment", "Response check", "Complication",
            "Final diagnosis", "Disposition and follow-up",
        })
        response = course[course["event_type"] == "Response check"]
        self.assertTrue(response["response_status"].isin({
            "Improved", "Partially improved", "Unchanged", "Worsened", "Not documented",
        }).all())
        self.assertTrue(response["causal_assessment"].str.contains("cannot establish", case=False).all())
        self.assertTrue(response["response_interpretation"].str.contains("not a measured", case=False).all())
        self.assertTrue(self.data["prescriptions"]["rationale"].str.contains("Synthetic").all())
        self.assertTrue(self.data["procedures_icd"]["rationale"].str.contains("Synthetic").all())
        self.assertTrue((course["synthetic_flag"] == 1).all())
        self.assertTrue(course["summary"].str.strip().ne("").all())

    def test_every_historical_patient_has_a_final_diagnosis_linked_to_diagnoses(self):
        historical_ids = set(self.data["patients"].loc[
            self.data["patients"]["patient_status"] == "HISTORICAL", "subject_id"
        ].astype(str))
        finals = self.data["hospital_course"].loc[
            self.data["hospital_course"]["event_type"] == "Final diagnosis"
        ]
        historical_finals = finals[finals["subject_id"].astype(str).isin(historical_ids)]
        final_counts = historical_finals.groupby(
            historical_finals["subject_id"].astype(str)
        ).size()
        self.assertEqual(set(final_counts.index), historical_ids)
        self.assertTrue((final_counts == 1).all())
        diagnosis_keys = set(zip(
            self.data["diagnoses_icd"]["subject_id"].astype(str),
            self.data["diagnoses_icd"]["icd_code"].astype(str),
            self.data["diagnoses_icd"]["long_title"].astype(str),
        ))
        self.assertTrue(all(
            (str(row.subject_id), row.final_diagnosis_code, row.final_diagnosis) in diagnosis_keys
            for row in finals.itertuples()
        ))

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
        self.assertEqual(result["expected_ids"], result["actual_ids"])

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
        result = rank_historical_patients(self.data, 20001, RetrievalQuery(top_k=7))
        self.assertLessEqual(len(result), 7)

    def test_pandas_reference_validates_supported_filter_combinations(self):
        condition_choices = ([], ["E11.9"])
        medication_choices = ([], ["metformin"])
        measurement_choices = ([], ["glucose"])
        profile_choices = (None, "diabetes")
        date_choices = (
            (None, None),
            ("2023-07-01", "2025-06-01"),
            ("2030-01-01", "2030-12-31"),
        )
        for conditions, medications, measurements, profile, dates, basis in product(
            condition_choices,
            medication_choices,
            measurement_choices,
            profile_choices,
            date_choices,
            ("orders", "administrations"),
        ):
            query = RetrievalQuery(
                conditions=conditions,
                medications=medications,
                measurements=measurements,
                profile_group=profile,
                start_date=dates[0],
                end_date=dates[1],
                medication_basis=basis,
            )
            returned = select_cohort(self.data, query)
            result = validate_cohort_against_pandas(self.data, query, returned)
            self.assertTrue(result["validated"], msg=query)

    def test_filtered_features_only_include_requested_evidence_categories(self):
        features = patient_features(
            self.data,
            20001,
            RetrievalQuery(medications=["metformin"]),
        )
        self.assertTrue(features)
        self.assertTrue(all(feature.startswith("medication:") for feature in features))

    def test_empty_filtered_features_stay_empty(self):
        features = patient_features(
            self.data,
            20001,
            RetrievalQuery(
                conditions=["__no_such_condition__"],
                medications=["__no_such_medication__"],
                measurements=["__no_such_measurement__"],
            ),
        )
        self.assertEqual(features, set())

    def test_idf_weighted_jaccard_prefers_rare_shared_features(self):
        rare_shared = idf_weighted_jaccard(
            {"rare", "common"}, {"rare", "other"}, {"rare": 4.0, "common": 1.0, "other": 1.0}
        )
        common_shared = idf_weighted_jaccard(
            {"common", "rare"}, {"common", "other"}, {"rare": 4.0, "common": 1.0, "other": 1.0}
        )
        self.assertGreater(rare_shared, common_shared)
        self.assertEqual(idf_weighted_jaccard(set(), set(), {}), 0.0)

    def test_ranked_results_include_both_similarity_scores(self):
        ranked = rank_historical_patients(
            self.data, 20001, RetrievalQuery(top_k=5, similarity_method="idf")
        )
        self.assertIn("similarity_jaccard", ranked.columns)
        self.assertIn("similarity_idf", ranked.columns)
        self.assertTrue((ranked["similarity"] == ranked["similarity_idf"]).all())

    def test_hand_specified_known_pair_is_top_ranked_for_both_medication_bases(self):
        for basis in ("orders", "administrations"):
            ranked = rank_historical_patients(
                self.data,
                20001,
                RetrievalQuery(medication_basis=basis, top_k=5),
            )
            self.assertEqual(str(ranked.iloc[0]["subject_id"]), "20011")

    def test_twenty_question_evaluator_emits_correctness_for_every_question(self):
        from evaluate import run_evaluation

        report = run_evaluation()
        self.assertEqual(report["summary"]["question_count"], 20)
        self.assertTrue(all(
            isinstance(question["answer_correct"], bool)
            for question in report["questions"]
        ))
        self.assertGreater(
            report["new_patient_diversity"]["conditions"]["mean_distinct_per_patient"],
            0,
        )
        self.assertGreater(
            report["new_patient_diversity"]["medication_orders"]["mean_distinct_per_patient"],
            0,
        )

    def test_add_review_item_dialog_opens(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.button[0].click().run()
        self.assertEqual(len(app.get("dialog")), 1)
        self.assertTrue(any(widget.label == "Item name" for widget in app.text_input))
        self.assertFalse(app.exception)

    def test_current_arrival_and_historical_demographics_are_displayed(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(app.subheader[0].value, "Current arrival")
        rendered_text = "\n".join(str(widget.value) for widget in app.markdown)
        self.assertIn("Weakness and elevated blood glucose", rendered_text)
        metrics = {widget.label: widget.value for widget in app.metric}
        self.assertEqual(metrics["Arrival method"], "EMS")
        self.assertEqual(metrics["Current status"], "In Progress")
        self.assertEqual(metrics["Heart rate"], "97 bpm")
        self.assertEqual(metrics["Blood pressure"], "131/93 mmHg")
        self.assertIn("Approx. age: 80", rendered_text)
        self.assertIn("Sex field: F", rendered_text)
        subheaders = {widget.value for widget in app.subheader}
        self.assertIn("Compare patient records", subheaders)
        self.assertIn("Historical hospital-course timeline", subheaders)
        timeline_text = "\n".join(str(widget.value) for widget in app.markdown)
        self.assertIn("FINAL DIAGNOSIS · HISTORICAL PATIENT", timeline_text)
        timeline_widgets = (
            list(app.caption) + list(app.info) + list(app.success)
            + list(app.warning) + list(app.error)
        )
        timeline_text += "\n" + "\n".join(str(widget.value) for widget in timeline_widgets)
        self.assertIn("Simulated response:", timeline_text)
        self.assertIn("Causal assessment:", timeline_text)
        self.assertIn("Rationale:", timeline_text)
        self.assertNotIn('<div class="timeline', timeline_text)
        self.assertNotIn("timeline-entry", timeline_text)
        self.assertIn("Diagnosis recorded", timeline_text)
        self.assertIn("Event colors distinguish", timeline_text)
        self.assertFalse(app.exception)

if __name__ == "__main__":
    unittest.main()
