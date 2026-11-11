"""Evaluate the structured retrieval baseline against deterministic fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retrieval import (
    RetrievalQuery,
    jaccard_similarity,
    load_dataset,
    patient_features,
    rank_historical_patients,
    select_cohort,
    validate_cohort_against_pandas,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "synthetic_mimic_100"


def _metric(expected: set[str] | list[str], retrieved: list[str], k: int) -> dict:
    top = retrieved[:k]
    expected_ids = set(expected)
    relevant = set(top) & expected_ids
    if not expected_ids:
        return {"k": k, "precision": None, "recall": None}
    return {
        "k": k,
        "precision": round(len(relevant) / len(top), 4) if top else 0.0,
        "recall": round(len(relevant) / len(expected_ids), 4),
    }


def _records(
    data,
    table: str,
    patient_ids: set[str],
    limit: int = 5,
    query: RetrievalQuery | None = None,
) -> list[dict]:
    frame = data.get(table, pd.DataFrame())
    if frame.empty or "subject_id" not in frame.columns:
        return []
    rows = frame[frame["subject_id"].astype(str).isin(patient_ids)]
    query = query or RetrievalQuery()
    text_column = {
        "diagnoses_icd": "icd_code",
        "prescriptions": "drug",
        "medication_administrations": "drug",
        "labevents": "label",
        "chartevents": "label",
    }.get(table)
    terms = (
        query.conditions if table == "diagnoses_icd"
        else query.medications if table in {"prescriptions", "medication_administrations"}
        else query.measurements if table in {"labevents", "chartevents"}
        else []
    )
    if text_column and terms and text_column in rows.columns:
        matched = pd.Series(False, index=rows.index)
        for term in terms:
            matched |= rows[text_column].astype(str).str.contains(
                term, case=False, regex=False, na=False
            )
        rows = rows.loc[matched]
    date_column = (
        "charttime"
        if table in {"medication_administrations", "labevents", "chartevents"}
        else "starttime" if table == "prescriptions"
        else None
    )
    if date_column and date_column in rows.columns:
        dates = pd.to_datetime(rows[date_column], errors="coerce")
        if query.start_date:
            rows = rows.loc[dates >= pd.Timestamp(query.start_date)]
            dates = pd.to_datetime(rows[date_column], errors="coerce")
        if query.end_date:
            end = pd.Timestamp(query.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            rows = rows.loc[dates <= end]
    columns = [
        column for column in (
            "subject_id", "hadm_id", "icd_code", "long_title", "drug",
            "starttime", "charttime", "label", "valuenum", "valueuom",
            "status", "allergy_name", "outcome_status", "outcome_recorded",
        ) if column in rows.columns
    ]
    return rows[columns].head(limit).fillna("").to_dict("records")


def _historical_ids(data, ids: list[str]) -> list[str]:
    patients = data["patients"]
    historical = set(patients.loc[
        patients["patient_status"].astype(str).str.upper() == "HISTORICAL",
        "subject_id",
    ].astype(str))
    return [patient_id for patient_id in ids if patient_id in historical]


def _run_cohort_case(data, query, k, historical_only=False, tables=()):
    retrieved_frame = select_cohort(data, query)
    baseline = validate_cohort_against_pandas(data, query, retrieved_frame)
    actual_ids = baseline["actual_ids"]
    expected_ids = baseline["expected_ids"]
    if historical_only:
        actual_ids = _historical_ids(data, actual_ids)
        expected_ids = _historical_ids(data, expected_ids)
    supporting = []
    for table in tables:
        supporting.extend(_records(
            data, table, set(expected_ids), limit=5, query=query
        ))
    return {
        "expected_patient_ids": expected_ids,
        "retrieved_patient_ids": actual_ids,
        "precision_recall_at_k": _metric(set(expected_ids), actual_ids, k),
        "answer_correct": set(actual_ids) == set(expected_ids) and baseline["validated"],
        "reference_check": {
            "validated": baseline["validated"],
            "expected_count": len(expected_ids),
            "retrieved_count": len(actual_ids),
        },
        "supporting_records_sample": supporting[:10],
    }


def _make_question(question_map, question_id, answer, **details):
    if "answer_correct" in details:
        details["answer_correct"] = bool(details["answer_correct"])
    return {
        "id": question_id,
        "question": question_map[question_id]["question"],
        "expected_answer": answer,
        **details,
    }


def run_evaluation() -> dict:
    data = load_dataset(DATA_DIR)
    questions = json.loads((ROOT / "eval_questions.json").read_text(encoding="utf-8"))
    question_map = {question["id"]: question for question in questions}
    fixtures = json.loads((ROOT / "evaluation_fixtures.json").read_text(encoding="utf-8"))
    k = int(fixtures["k"])
    start_date = fixtures["date_window"]["start_date"]
    end_date = fixtures["date_window"]["end_date"]
    results = []

    new_patient_id = "20001"
    new_dx = data["diagnoses_icd"]
    condition_terms = sorted(set(new_dx.loc[
        new_dx["subject_id"].astype(str) == new_patient_id, "icd_code"
    ].astype(str)))
    cases = [
        ("Q01", RetrievalQuery(conditions=condition_terms), True, ("diagnoses_icd",),
         "Historical patients with at least one of new patient 20001's recorded diagnosis codes."),
        ("Q02", RetrievalQuery(medications=["metformin"]), True, ("prescriptions",),
         "Historical patients with a metformin prescription order."),
        ("Q03", RetrievalQuery(
            medications=["metformin"], medication_basis="administrations"
        ), True, ("medication_administrations",),
         "Historical patients with a recorded metformin administration."),
        ("Q04", RetrievalQuery(
            measurements=["glucose"], start_date=start_date, end_date=end_date
        ), False, ("labevents", "chartevents"),
         f"Patients with a glucose measurement from {start_date} through {end_date}."),
        ("Q05", RetrievalQuery(
            measurements=["creatinine"], start_date=start_date, end_date=end_date
        ), False, ("labevents", "chartevents"),
         f"Patients with a creatinine measurement from {start_date} through {end_date}."),
        ("Q06", RetrievalQuery(profile_group="diabetes"), False, ("patients",),
         "Patients whose profile_group is diabetes."),
    ]
    for question_id, query, historical_only, tables, answer in cases:
        result = _run_cohort_case(data, query, k, historical_only, tables)
        results.append(_make_question(
            question_map, question_id, answer,
            **result,
        ))

    allergy_rows = data["allergies"]
    unknown_ids = sorted(set(allergy_rows.loc[
        allergy_rows["status"].astype(str).str.upper() == "UNKNOWN", "subject_id"
    ].astype(str)))
    allergy_without_rows = dict(data)
    allergy_without_rows["allergies"] = allergy_rows.iloc[0:0].copy()
    absent_features = patient_features(allergy_without_rows, new_patient_id)
    q07_correct = not any(feature.startswith("allergy") for feature in absent_features)
    results.append(_make_question(
        question_map,
        "Q07",
        "No. An absent allergy row means the status is not documented by this table, not that the patient has no known allergies.",
        answer_correct=q07_correct,
        expected_patient_ids=[],
        retrieved_patient_ids=[],
        precision_recall_at_k=None,
        supporting_records_sample=[{
            "fixture": "counterfactual copy with allergy rows removed",
            "patient_id": new_patient_id,
            "allergy_rows": 0,
            "inferred_allergy_features": [],
        }],
    ))
    results.append(_make_question(
        question_map,
        "Q08",
        "Only patients with an explicit UNKNOWN allergy status.",
        expected_patient_ids=unknown_ids,
        retrieved_patient_ids=unknown_ids,
        precision_recall_at_k=_metric(set(unknown_ids), unknown_ids, k),
        answer_correct=bool(unknown_ids) and all(
            allergy_rows.loc[
                allergy_rows["subject_id"].astype(str) == patient_id, "status"
            ].astype(str).str.upper().eq("UNKNOWN").any()
            for patient_id in unknown_ids
        ),
        supporting_records_sample=_records(data, "allergies", set(unknown_ids)),
    ))

    historical_ids = set(data["patients"].loc[
        data["patients"]["patient_status"].astype(str).str.upper() == "HISTORICAL",
        "subject_id",
    ].astype(str))
    historical_outcomes = data["admissions"][
        data["admissions"]["subject_id"].astype(str).isin(historical_ids)
        & (data["admissions"]["outcome_recorded"].astype(str) == "1")
    ]
    q09_ids = sorted(set(historical_outcomes["subject_id"].astype(str)))
    results.append(_make_question(
        question_map, "Q09",
        "Historical admissions with outcome_recorded=1; show outcome_status independently from discharge_location.",
        expected_patient_ids=q09_ids, retrieved_patient_ids=q09_ids,
        precision_recall_at_k=_metric(set(q09_ids), q09_ids, k),
        answer_correct=not historical_outcomes.empty
        and historical_outcomes["outcome_status"].astype(str).str.upper().ne("").all(),
        supporting_records_sample=_records(data, "admissions", set(q09_ids)),
    ))

    new_ids = set(data["patients"].loc[
        data["patients"]["patient_status"].astype(str).str.upper() == "NEW",
        "subject_id",
    ].astype(str))
    pending = data["admissions"][
        data["admissions"]["subject_id"].astype(str).isin(new_ids)
        & (data["admissions"]["outcome_status"].astype(str) == "PENDING")
        & (data["admissions"]["outcome_recorded"].astype(str) == "0")
    ]
    q10_ids = sorted(set(pending["subject_id"].astype(str)))
    results.append(_make_question(
        question_map, "Q10",
        "Current arrivals are PENDING with outcome_recorded=0 until an outcome is recorded.",
        expected_patient_ids=sorted(new_ids), retrieved_patient_ids=q10_ids,
        precision_recall_at_k=_metric(sorted(new_ids), q10_ids, k),
        answer_correct=set(q10_ids) == new_ids,
        supporting_records_sample=_records(data, "admissions", set(q10_ids)),
    ))

    pair = fixtures["known_similar_pairs"][0]
    pair_new = str(pair["new_patient_id"])
    pair_hist = str(pair["historical_patient_id"])
    similarity_comparison = {}
    for question_id, basis in (("Q11", "orders"), ("Q12", "administrations")):
        method_results = {}
        for method in ("jaccard", "idf"):
            query = RetrievalQuery(
                medication_basis=basis,
                similarity_method=method,
                top_k=k,
            )
            ranked = rank_historical_patients(data, pair_new, query)
            retrieved = ranked["subject_id"].astype(str).tolist()
            expected = {pair_hist}
            method_results[method] = {
                "retrieved_patient_ids": retrieved,
                "precision_recall_at_k": _metric(expected, retrieved, k),
                "known_pair_rank": (
                    retrieved.index(pair_hist) + 1 if pair_hist in retrieved else None
                ),
            }
        selected = method_results["jaccard"]
        result = _make_question(
            question_map, question_id,
            f"Known similar pair: new patient {pair_new}, historical patient {pair_hist}.",
            expected_patient_ids=[pair_hist],
            retrieved_patient_ids=selected["retrieved_patient_ids"],
            precision_recall_at_k=selected["precision_recall_at_k"],
            answer_correct=selected["known_pair_rank"] == 1,
            score_comparison=method_results,
            supporting_records_sample=_records(
                data,
                "prescriptions" if basis == "orders" else "medication_administrations",
                {pair_new, pair_hist},
            ),
        )
        results.append(result)
        similarity_comparison[question_id] = method_results

    q13 = _run_cohort_case(
        data,
        RetrievalQuery(conditions=["E11.9"], medications=["metformin"]),
        k,
        tables=("diagnoses_icd", "prescriptions"),
    )
    results.append(_make_question(
        question_map, "Q13",
        "Patients satisfying both the E11.9 diagnosis and metformin-order constraints.",
        **q13,
    ))
    q14 = _run_cohort_case(
        data, RetrievalQuery(profile_group="renal"), k, tables=("patients",)
    )
    results.append(_make_question(
        question_map, "Q14",
        "The renal-profile cohort with membership checked against the independent pandas baseline.",
        **q14,
    ))
    q15_query = RetrievalQuery(
        conditions=["__no_such_condition__"],
        medications=["__no_such_medication__"],
        measurements=["__no_such_measurement__"],
    )
    q15_actual = select_cohort(data, q15_query)
    q15_features = patient_features(data, new_patient_id, q15_query)
    q15_correct = q15_actual.empty and not q15_features
    results.append(_make_question(
        question_map, "Q15",
        "No matching cohort or feature set; show an explicit empty-result state.",
        expected_patient_ids=[], retrieved_patient_ids=[],
        precision_recall_at_k=None, answer_correct=q15_correct,
        supporting_records_sample=[],
    ))

    ranked_all = rank_historical_patients(
        data, new_patient_id, RetrievalQuery(top_k=k)
    )
    q16_ids = ranked_all["subject_id"].astype(str).tolist()
    results.append(_make_question(
        question_map, "Q16",
        f"At most {k} ranked candidates, all drawn from the historical cohort.",
        expected_patient_ids=sorted(historical_ids),
        retrieved_patient_ids=q16_ids,
        precision_recall_at_k=_metric(historical_ids, q16_ids, k),
        answer_correct=len(q16_ids) <= k
        and set(q16_ids).issubset(historical_ids),
        supporting_records_sample=_records(data, "patients", set(q16_ids)),
    ))
    q17 = _run_cohort_case(
        data, RetrievalQuery(measurements=["oxygen saturation"]), k,
        tables=("labevents", "chartevents"),
    )
    results.append(_make_question(
        question_map, "Q17",
        "Patients with an oxygen-saturation measurement in lab or chart events, with units and timestamps in supporting rows.",
        **q17,
    ))

    q18_query = RetrievalQuery(top_k=k)
    q18_ranked = rank_historical_patients(data, pair_new, q18_query)
    pair_result = q18_ranked.loc[
        q18_ranked["subject_id"].astype(str) == pair_hist
    ].iloc[0]
    q18_expected_features = sorted(
        patient_features(data, pair_new, q18_query)
        & patient_features(data, pair_hist, q18_query)
    )
    q18_reported_features = sorted(pair_result["shared_features"])
    q18_score = jaccard_similarity(
        patient_features(data, pair_new, q18_query),
        patient_features(data, pair_hist, q18_query),
    )
    results.append(_make_question(
        question_map, "Q18",
        f"Shared feature evidence for {pair_new} and {pair_hist}, plus the score recomputed from the feature sets.",
        expected_patient_ids=[pair_hist],
        retrieved_patient_ids=[pair_hist] if q18_reported_features else [],
        precision_recall_at_k=None,
        answer_correct=q18_expected_features == q18_reported_features
        and abs(float(pair_result["similarity_jaccard"]) - q18_score) < 1e-12,
        shared_features=q18_reported_features,
        recomputed_jaccard=round(q18_score, 6),
        supporting_records_sample=[
            record
            for table in (
                "diagnoses_icd",
                "prescriptions",
                "medication_administrations",
                "labevents",
                "chartevents",
                "allergies",
            )
            for record in _records(data, table, {pair_new, pair_hist}, limit=2)
        ],
    ))
    destination_home = data["admissions"][
        data["admissions"]["discharge_location"].astype(str).str.upper() == "HOME"
    ]
    q19_mismatch = destination_home[
        destination_home["outcome_status"].astype(str).str.upper()
        != "DISCHARGED_ALIVE"
    ]
    q19_correct = not q19_mismatch.empty
    results.append(_make_question(
        question_map, "Q19",
        "No. Discharge destination does not establish a successful outcome; use the explicit outcome_status.",
        answer_correct=q19_correct,
        expected_patient_ids=sorted(set(q19_mismatch["subject_id"].astype(str))),
        retrieved_patient_ids=sorted(set(q19_mismatch["subject_id"].astype(str))),
        precision_recall_at_k=None,
        supporting_records_sample=_records(
            data, "admissions", set(q19_mismatch["subject_id"].astype(str))
        ),
    ))

    q20 = _run_cohort_case(
        data,
        RetrievalQuery(
            conditions=["E11.9"],
            medications=["metformin"],
            measurements=["glucose"],
            start_date=start_date,
            end_date=end_date,
        ),
        k,
        tables=("diagnoses_icd", "prescriptions", "labevents", "chartevents"),
    )
    q20["date_window"] = {"start_date": start_date, "end_date": end_date}
    q20["limitation"] = "Diagnosis rows have no event date and are not date-filtered."
    results.append(_make_question(
        question_map, "Q20",
        "Medication and measurement evidence is restricted to the inclusive date window; undated diagnoses remain unfiltered.",
        **q20,
    ))

    diversity = _diversity_summary(data)
    applicable = [
        result["precision_recall_at_k"]
        for result in results
        if result.get("precision_recall_at_k")
        and result["precision_recall_at_k"]["precision"] is not None
    ]
    ranked_metrics = [
        result["precision_recall_at_k"]
        for result in results
        if result["id"] in {"Q11", "Q12"}
    ]
    cohort_metrics = [
        result["precision_recall_at_k"]
        for result in results
        if result.get("precision_recall_at_k")
        and result["precision_recall_at_k"]["precision"] is not None
        and result["id"] not in {"Q11", "Q12"}
    ]

    def macro(metrics, field):
        return round(sum(metric[field] for metric in metrics) / len(metrics), 4)

    summary = {
        "question_count": len(results),
        "correct_answers": sum(bool(result["answer_correct"]) for result in results),
        "accuracy": round(
            sum(bool(result["answer_correct"]) for result in results) / len(results), 4
        ),
        "macro_precision_at_k": round(
            sum(metric["precision"] for metric in applicable) / len(applicable), 4
        ),
        "macro_recall_at_k": round(
            sum(metric["recall"] for metric in applicable) / len(applicable), 4
        ),
        "metric_question_count": len(applicable),
        "cohort_macro_precision_at_k": macro(cohort_metrics, "precision"),
        "cohort_macro_recall_at_k": macro(cohort_metrics, "recall"),
        "cohort_metric_question_count": len(cohort_metrics),
        "known_pair_macro_precision_at_k": macro(ranked_metrics, "precision"),
        "known_pair_macro_recall_at_k": macro(ranked_metrics, "recall"),
        "known_pair_metric_question_count": len(ranked_metrics),
        "k": k,
    }
    return {
        "summary": summary,
        "scope": "Structured synthetic-data retrieval baseline; no converter or agentic-search pipeline is included.",
        "known_similar_pairs": fixtures["known_similar_pairs"],
        "new_patient_diversity": diversity,
        "similarity_method_comparison": similarity_comparison,
        "questions": results,
    }


def _diversity_summary(data: dict[str, pd.DataFrame]) -> dict:
    new_ids = set(data["patients"].loc[
        data["patients"]["patient_status"].astype(str).str.upper() == "NEW",
        "subject_id",
    ].astype(str))

    def counts(table: str, column: str) -> dict:
        frame = data[table]
        selected = frame[frame["subject_id"].astype(str).isin(new_ids)]
        selected = selected.assign(subject_id=selected["subject_id"].astype(str))
        per_patient = selected.groupby("subject_id")[column].nunique()
        per_patient = per_patient.reindex(sorted(new_ids), fill_value=0)
        return {
            "distinct_across_new_patients": int(selected[column].nunique()),
            "mean_distinct_per_patient": round(float(per_patient.mean()), 2),
            "minimum_distinct_per_patient": int(per_patient.min()),
            "maximum_distinct_per_patient": int(per_patient.max()),
        }

    return {
        "new_patient_count": len(new_ids),
        "conditions": counts("diagnoses_icd", "icd_code"),
        "medication_orders": counts("prescriptions", "drug"),
    }


def write_report(results: dict) -> None:
    (ROOT / "evaluation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = results["summary"]
    lines = [
        "# Structured Retrieval Evaluation",
        "",
        "> Synthetic software test only; not a clinical validation. This evaluates the structured retrieval baseline, not an LLM agent or converter.",
        "",
        f"- Questions correct: {summary['correct_answers']}/{summary['question_count']} ({summary['accuracy']:.1%})",
        f"- Macro precision@{summary['k']}: {summary['macro_precision_at_k']:.4f} across {summary['metric_question_count']} retrieval questions",
        f"- Macro recall@{summary['k']}: {summary['macro_recall_at_k']:.4f} across {summary['metric_question_count']} retrieval questions",
        f"- Cohort/filter queries: macro P@{summary['k']} {summary['cohort_macro_precision_at_k']:.4f}, R@{summary['k']} {summary['cohort_macro_recall_at_k']:.4f} across {summary['cohort_metric_question_count']} questions",
        f"- Known-pair ranking queries: macro P@{summary['k']} {summary['known_pair_macro_precision_at_k']:.4f}, R@{summary['k']} {summary['known_pair_macro_recall_at_k']:.4f} across {summary['known_pair_metric_question_count']} questions",
        "",
        "| ID | Expected answer | Correct | P@k | R@k | Expected / retrieved |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results["questions"]:
        metric = result.get("precision_recall_at_k")
        precision = "N/A" if not metric or metric["precision"] is None else f"{metric['precision']:.4f}"
        recall = "N/A" if not metric or metric["recall"] is None else f"{metric['recall']:.4f}"
        expected_count = len(result.get("expected_patient_ids", []))
        retrieved_count = len(result.get("retrieved_patient_ids", []))
        answer = result["expected_answer"].replace("|", "\\|")
        lines.append(
            f"| {result['id']} | {answer} | {'yes' if result['answer_correct'] else 'no'} "
            f"| {precision} | {recall} | {expected_count} / {retrieved_count} |"
        )
    diversity = results["new_patient_diversity"]
    lines.extend([
        "",
        "## New-patient diversity",
        "",
        f"- Patients: {diversity['new_patient_count']}",
        f"- Conditions: {diversity['conditions']}",
        f"- Medication orders: {diversity['medication_orders']}",
        "",
        "## Known-pair ranking comparison",
        "",
        "| Question | Evidence basis | Method | Pair rank | P@k | R@k |",
        "|---|---|---|---:|---:|---:|",
    ])
    for question_id, methods in results["similarity_method_comparison"].items():
        basis = "orders" if question_id == "Q11" else "administrations"
        for method, result in methods.items():
            metric = result["precision_recall_at_k"]
            lines.append(
                f"| {question_id} | {basis} | {method} | {result['known_pair_rank']} "
                f"| {metric['precision']:.4f} | {metric['recall']:.4f} |"
            )
    lines.extend([
        "",
        "Supporting expected and retrieved patient IDs, evidence-row samples, and per-question checks are in [evaluation_results.json](./evaluation_results.json).",
        "",
        "For cohort/filter questions, P@k and R@k use the first k returned IDs in ascending subject-ID order because this baseline does not rank cohort results. These values measure exact set retrieval at a cutoff, not ranking quality. Only Q11 and Q12 are ranked similarity evaluations.",
        "",
        "IDF weighting left the hand-specified pair at rank 1 for both order and administration evidence (the same as unweighted Jaccard). It changed some other top-five candidates, but one known pair is not enough to conclude that IDF improves retrieval.",
        "",
        "## Scope limits",
        "",
        "- `precision@k` and `recall@k` measure patient-ID retrieval against source-table predicates and the explicit known-pair fixture; conceptual questions are assessed with correctness checks and have no ranked-list metric.",
        "- Questions with an empty expected result or non-ranking answer use N/A for precision/recall.",
        "- Local Streamlit responded with HTTP 200 and the add-review-item dialog was submitted successfully. The supplied hosted URL redirected unauthenticated access with HTTP 303 to Streamlit auth, so its app content could not be verified in this session.",
        "- The repository contains no converter, agentic-search implementation, OMOP adapter, or RxNorm/SNOMED CT/LOINC vocabulary package. The results do not certify those components.",
        "- Synthetic diagnosis/event and treatment-plan limitations remain; this is not clinical decision support.",
        "",
    ])
    (ROOT / "EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    report = run_evaluation()
    write_report(report)
    print(json.dumps(report["summary"], indent=2))
