"""Generated strict first-round audits, raw source ownership and release guards."""

import csv
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.audit_burden_labels import audit_burden_labels
from src.cohort.burden_io import SUPPLEMENTARY_COLUMNS, TABLE_COLUMNS
from src.cohort.freeze_endpoint import freeze_endpoint
from src.cohort.strict_sources import REVIEW_SOURCES
from src.common.manifest import write_json
from src.common.paths import PipelineError, read_json
from tests.cohort import test_lenient_clinical as clinical
from tests.cohort.test_burden_pipeline import prepare


class StrictPipelineTests(unittest.TestCase):
    def setUp(self):
        fixture = self.fixture = clinical.LenientClinicalTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.capture = redirect_stdout(StringIO())
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)
        self.context, self.store = prepare(fixture)
        self.rules_path = fixture.root / "configs/data/v1_r1/cohort/burden_rules.json"
        cohort = read_json(self.rules_path)
        cohort["review_policy"] = "strict"
        cohort["burden_v2"].update(
            cohort_decisions_path=None, event_decisions_path=None
        )
        write_json(self.rules_path, cohort)
        self.ehr_path = fixture.root / "configs/data/v1_r1/ehr/burden_sources.json"
        ehr = read_json(self.ehr_path)
        ehr["burden_v2"]["review_evidence_sources"] = {}
        write_json(self.ehr_path, ehr)
        for name, extra in SUPPLEMENTARY_COLUMNS.items():
            self.write_table(
                name,
                [*TABLE_COLUMNS[name].split(), *extra.split()],
                fixture.tables[name],
            )
        self.ref = self.store.manifest_ref("step05", "audit")

    def write_table(self, name, fields, rows):
        path = self.fixture.iv / self.fixture.ehr["burden_v2"]["tables"][name]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def review_source(self, name, rows):
        fields = list(
            dict.fromkeys(
                [
                    *REVIEW_SOURCES[name][1].split(),
                    *(field for row in rows for field in row),
                ]
            )
        )
        path = self.fixture.iv / f"review_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        ehr = read_json(self.ehr_path)
        ehr["burden_v2"]["review_evidence_sources"][name] = path.relative_to(
            self.fixture.iv
        ).as_posix()
        write_json(self.ehr_path, ehr)
        return path

    def audit(self):
        return audit_burden_labels(self.context, review_policy="strict")

    def output(self, name):
        manifest = self.store.validate(self.ref)
        return self.context.layout.resolve(self.store.output_ref(manifest, name))

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_first_audit_has_explicit_policy_schema_pending_templates_and_no_baseline(
        self,
    ):
        self.assertEqual(self.audit(), 0)
        manifest = self.store.validate(self.ref)
        self.assertEqual(manifest["details"]["review_policy"], "strict")
        self.assertEqual(manifest["generation"], 1)
        schema = pq.read_schema(self.output("window_72h/event_evidence.parquet"))
        self.assertEqual(schema.field("blocks_event").type, pa.uint8())
        self.assertIn(
            "new_invasive_ventilation_first_event_time_status",
            pq.read_schema(self.output("window_72h/study_event_audit.parquet")).names,
        )
        self.assertEqual(
            pq.read_table(self.output("review_source_records.parquet")).num_rows, 0
        )
        metadata = read_json(self.output("run_metadata.json"))
        self.assertFalse(metadata["audit_ready_for_window_decision"])
        self.assertTrue(metadata["review_application_supported"])
        with self.output("review_templates/events.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            suggestions = list(csv.DictReader(stream))
        self.assertTrue(suggestions)
        self.assertTrue(
            all(
                row["decision"] == "pending"
                and row["event_flag"] == row["first_event_time"] == ""
                for row in suggestions
            )
        )
        self.assertFalse(
            any(
                "baseline" in row["path"] or "48h" in row["path"]
                for row in manifest["outputs"]
            )
        )
        self.error(
            "MISSING_FILE",
            lambda: freeze_endpoint(self.context, review_policy="strict"),
        )

    def test_nonaligned_reviewer_input_is_not_silently_ignored(self):
        cohort = read_json(self.rules_path)
        cohort["burden_v2"]["event_decisions_path"] = (
            "configs/data/v1_r1/cohort/reviews/events.csv"
        )
        write_json(self.rules_path, cohort)
        self.error("MISSING_FILE", self.audit)
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_strict_requires_extra_projection_and_does_not_publish_on_failure(self):
        self.write_table(
            "icustays",
            TABLE_COLUMNS["icustays"].split(),
            self.fixture.tables["icustays"],
        )
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            self.audit()
        self.assertFalse(self.context.layout.resolve(self.ref).exists())

    def test_detail_ownership_uses_all_subject_parent_admissions(self):
        self.review_source(
            "emar_full",
            [
                {
                    "subject_id": "1",
                    "hadm_id": hadm,
                    "emar_id": "ambiguous",
                    "poe_id": "p",
                    "pharmacy_id": "x",
                    "charttime": clinical.dt(1),
                }
                for hadm in ("10", "11")
            ],
        )
        self.review_source(
            "emar_detail_full",
            [
                {
                    "subject_id": "1",
                    "emar_id": "ambiguous",
                    "administration_type": "IV Infusion",
                    "infusion_rate_adjustment": "0",
                }
            ],
        )
        self.audit()
        rows = pq.read_table(self.output("review_source_records.parquet")).to_pylist()
        details = [
            json.loads(row["raw_evidence"])
            for row in rows
            if row["source_file"] == "emar_detail_full"
        ]
        self.assertTrue(details)
        self.assertTrue(all(row["hadm_id"] is None for row in details))
        exported = [row for row in rows if row["source_file"] == "emar_detail_full"]
        self.assertTrue(all(row["hadm_id"] is None for row in exported))
        # Context admits this record for review; it cannot grant ownership.
        self.assertTrue(all(row["subject_id"] == 1 for row in exported))
        known = [row for row in rows if row["source_file"] == "emar_full"]
        self.assertTrue(all(row["hadm_id"] == 10 for row in known))

    def test_nonblank_emar_still_retains_matching_prescriptions(self):
        row = {
            "subject_id": "1",
            "hadm_id": "10",
            "pharmacy_id": "1",
            "drug": "Norepinephrine",
        }
        fields = [
            *TABLE_COLUMNS["prescriptions"].split(),
            *SUPPLEMENTARY_COLUMNS["prescriptions"].split(),
        ]
        self.write_table("prescriptions", fields, [row])
        self.audit()
        rows = pq.read_table(
            self.output("medication_identity_audit.parquet")
        ).to_pylist()
        self.assertTrue(any(row["prescriptions_match_count"] == 1 for row in rows))

    def test_successful_rerun_refreshes_only_current_generation_templates(self):
        self.audit()
        self.audit()
        self.assertEqual(self.store.validate(self.ref)["generation"], 2)
        with self.output("review_templates/events.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            self.assertTrue(
                all(
                    row["source_audit_generation"] == "2"
                    for row in csv.DictReader(stream)
                )
            )

    def test_supplemental_inputevents_are_pending_evidence_not_automatic_positives(
        self,
    ):
        self.write_table(
            "d_items",
            TABLE_COLUMNS["d_items"].split(),
            [
                *self.fixture.tables["d_items"],
                {
                    "itemid": "221906",
                    "label": "Norepinephrine",
                    "linksto": "inputevents",
                },
            ],
        )
        self.review_source(
            "inputevents_full",
            [
                {
                    "subject_id": "1",
                    "hadm_id": "10",
                    "itemid": "221906",
                    "starttime": clinical.dt(3),
                    "endtime": clinical.dt(4),
                    "amount": "1",
                    "rate": "2",
                    "orderid": "order",
                    "linkorderid": "link",
                    "statusdescription": "FinishedRunning",
                }
            ],
        )
        self.audit()
        evidence = pq.read_table(
            self.output("window_72h/event_evidence.parquet")
        ).to_pylist()
        rows = [row for row in evidence if row["source_file"] == "inputevents_full"]
        self.assertTrue(rows)
        self.assertTrue(
            all(row["qualifies"] is None and row["blocks_event"] == 1 for row in rows)
        )
        self.assertTrue(
            all(
                row["exclusion_reason"] == "supplemental_infusion_requires_review"
                for row in rows
            )
        )
        metadata = read_json(self.output("run_metadata.json"))
        source = metadata["review_evidence_sources"]["inputevents_full"]["source"]
        self.assertEqual(source["root"], "mimic_iv")
        self.assertFalse(source["path"].startswith("/"))

    def test_clinical_slashes_survive_source_records_and_diagnostic_csv_publication(
        self,
    ):
        fields = [
            *TABLE_COLUMNS["prescriptions"].split(),
            *SUPPLEMENTARY_COLUMNS["prescriptions"].split(),
        ]
        self.write_table(
            "prescriptions",
            fields,
            [
                {
                    "subject_id": "1",
                    "hadm_id": "10",
                    "pharmacy_id": "1",
                    "drug": "Norepinephrine",
                    "prod_strength": "10 mg /2mL",
                }
            ],
        )
        self.review_source("emar_detail_full", [])
        # Configure the ownership parent before the child is ingested.
        self.review_source(
            "emar_full",
            [
                {
                    "subject_id": "1",
                    "hadm_id": "10",
                    "emar_id": "review",
                    "poe_id": "p",
                    "pharmacy_id": "1",
                    "charttime": clinical.dt(1),
                }
            ],
        )
        self.review_source(
            "emar_detail_full",
            [
                {
                    "subject_id": "1",
                    "emar_id": "review",
                    "administration_type": "IV Infusion",
                    "infusion_rate_adjustment": "0",
                    "product_description": "Synthetic infusion /100mL",
                }
            ],
        )
        from src.common.config import DataContext
        from src.common.paths import Layout

        paths_file = self.fixture.root / "configs/environment/paths.local.json"
        paths = read_json(paths_file)
        paths["roots"]["mimic_note"] = paths["roots"]["mimic_iv"]
        write_json(paths_file, paths)
        self.context = DataContext.load(Layout(self.fixture.root), 1, 1)
        note = "Synthetic clinical narrative /followup and ratio /12.\n" + "x" * 140000
        self.review_source(
            "discharge_notes_full",
            [
                {
                    "subject_id": "1",
                    "hadm_id": "10",
                    "note_id": "synthetic",
                    "charttime": clinical.dt(2),
                    "text": note,
                }
            ],
        )
        self.audit()
        rows = pq.read_table(self.output("review_source_records.parquet")).to_pylist()
        note_rows = [
            json.loads(row["raw_evidence"])
            for row in rows
            if row["source_file"] == "discharge_notes_full"
        ]
        self.assertEqual(note_rows[0]["record"]["text"], note)
        product_rows = [
            json.loads(row["raw_evidence"])
            for row in rows
            if row["source_file"] == "emar_detail_full"
        ]
        self.assertEqual(
            product_rows[0]["record"]["product_description"],
            "Synthetic infusion /100mL",
        )
        from src.cohort.burden_io import record_key

        for row in note_rows + product_rows:
            table = row["source_record_key"].split(":", 1)[0]
            payload = {key: row[key] for key in ("subject_id", "hadm_id", "record")}
            self.assertEqual(row["source_record_key"], record_key(table, payload))

    def fill_event_reviews(self, **changes):
        with self.output("review_templates/events.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = list(reader)
        self.assertTrue(rows)
        for row in rows:
            row.update(
                decision="set_event",
                event_flag="0",
                reviewer="synthetic",
                reviewed_on="2026-11-05",
                rationale="synthetic reviewed source",
            )
        rows[0].update(changes)
        path = self.fixture.root / "configs/data/v1_r1/cohort/reviews/events.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        relative = path.relative_to(self.fixture.root).as_posix()
        cohort = read_json(self.rules_path)
        cohort["burden_v2"]["event_decisions_path"] = relative
        write_json(self.rules_path, cohort)
        return rows, relative

    def test_reviewed_audit_retains_generation_then_freezes_and_builds(self):
        from src.cohort.build_labels import build_burden_labels

        self.audit()
        rows, ref = self.fill_event_reviews()
        self.audit()
        first = read_json(self.output("run_metadata.json"))
        self.assertEqual(first["review_source_generations"], [1])
        self.assertEqual(first["review_evidence_binding_count"], len(rows))
        self.assertTrue(first["audit_ready_for_window_decision"])
        self.audit()
        self.assertEqual(self.store.validate(self.ref)["generation"], 3)
        archive = read_json(self.output("review_sources/index.json"))
        self.assertEqual(list(archive["snapshots"]), ["1"])
        review = {
            "confirmed": True,
            "version": 1,
            "run": 1,
            "audit_generation": 3,
            "review_policy": "strict",
            "horizon_hours": 72,
            "decision_version": "fixture",
            "reviewer": "synthetic",
            "reviewed_on": "2026-11-05",
            "rationale": "synthetic endpoint",
            "rules_version": read_json(self.rules_path)["burden_v2"]["rules_version"],
            "review_records": [ref],
        }
        write_json(
            self.fixture.root
            / "configs/data/v1_r1/cohort/reviews/endpoint_review.json",
            review,
        )
        self.assertEqual(freeze_endpoint(self.context, review_policy="strict"), 0)
        self.assertEqual(build_burden_labels(self.context), 0)
        self.assertEqual(
            self.store.validate(self.store.manifest_ref("step06", "build"))["details"][
                "review_policy"
            ],
            "strict",
        )

    def test_review_stale_generation_rolls_back_without_discarding_previous_evidence(
        self,
    ):
        self.audit()
        before = self.context.layout.resolve(self.ref).read_bytes()
        self.fill_event_reviews(source_audit_generation="9")
        self.error("REVIEW_SOURCE_UNAVAILABLE", self.audit)
        self.assertEqual(self.context.layout.resolve(self.ref).read_bytes(), before)

    def test_review_unowned_record_is_rejected(self):
        self.audit()
        self.fill_event_reviews(source_record_keys='["chartevents:synthetic_missing"]')
        self.error("REVIEW_EVIDENCE_CHANGED", self.audit)

    def test_explicit_cohort_exclusion_keeps_pending_event_inactive(self):
        self.audit()
        rows, _ = self.fill_event_reviews(decision="pending", event_flag="")
        target = rows[0]["sample_id"]
        evidence = pq.read_table(self.output("candidate_evidence.parquet")).to_pylist()
        key = next(
            row["source_record_key"]
            for row in evidence
            if row["sample_id"] == target
            and row["source_record_key"].startswith("admissions:")
        )
        cohort_row = {
            name: rows[0][name]
            for name in (
                "version",
                "run",
                "source_audit_generation",
                "review_policy",
                "sample_id",
                "reviewer",
                "reviewed_on",
                "rationale",
            )
        }
        cohort_row.update(
            decision="exclude",
            reason="synthetic explicit exclusion",
            source_record_keys=json.dumps([key]),
            care_level_at_t0="",
        )
        path = self.fixture.root / "configs/data/v1_r1/cohort/reviews/cohort.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cohort_row))
            writer.writeheader()
            writer.writerow(cohort_row)
        rules = read_json(self.rules_path)
        rules["burden_v2"]["cohort_decisions_path"] = path.relative_to(
            self.fixture.root
        ).as_posix()
        write_json(self.rules_path, rules)
        self.audit()
        metadata = read_json(self.output("run_metadata.json"))
        self.assertEqual(
            metadata["pending_review_decisions"], {"cohort": 0, "event": 0}
        )
        self.assertTrue(
            any(
                row["decision"] == "pending" and row["event_flag"] == ""
                for row in metadata["inactive_event_reviews"]
            )
        )
        wide = pq.read_table(
            self.output("window_72h/study_event_audit.parquet")
        ).to_pylist()
        removed = next(row for row in wide if row["sample_id"] == target)
        self.assertEqual(removed["label_status"], "excluded")
        self.assertIsNone(removed["burden_score"])
        self.assertTrue(metadata["audit_ready_for_window_decision"])

    def test_rule_changes_after_review_source_are_rejected(self):
        self.audit()
        self.fill_event_reviews()
        rules = read_json(self.rules_path)
        rules["burden_v2"]["dictionary_version"] = "different_dictionary"
        write_json(self.rules_path, rules)
        self.error("REVIEW_RULES_CHANGED", self.audit)


if __name__ == "__main__":
    unittest.main()
