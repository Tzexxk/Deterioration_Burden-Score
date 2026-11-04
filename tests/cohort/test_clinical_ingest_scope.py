"""Admission-scope equivalence and bounded-memory clinical CSV ingestion."""

import csv
import json
import unittest
from collections import Counter
from unittest.mock import patch

import duckdb

from src.cohort import burden_io
from src.common.paths import PipelineError
from tests.cohort import test_burden_pipeline as pipeline


class ConnectionProxy:
    def __init__(self, connection, fail=False):
        self.connection = connection
        self.fail = fail
    def __getattr__(self, name):
        return getattr(self.connection, name)
    def execute(self, query, params=None):
        if self.fail and query.startswith("CREATE TABLE chartevents AS"):
            raise duckdb.OutOfMemoryException("synthetic allocation failure")
        return self.connection.execute(query, params) if params is not None else self.connection.execute(query)


class ClinicalIngestScopeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = pipeline.BurdenPipelineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.fixture
        self.csv = self.base.iv / self.base.ehr["burden_v2"]["tables"]["chartevents"]

    def write_rows(self, rows):
        fields = burden_io.TABLE_COLUMNS["chartevents"].split()
        with self.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_scope_preserves_blank_hadm_duplicates_and_raw_identity(self):
        template = self.base.tables["chartevents"][0]
        rows = [{**template, "subject_id": subject, "hadm_id": hadm, "value": value} for subject, hadm, value in [
            ("1", "10", "exact"), ("1", "11", "another_candidate_admission"),
            ("1", "", "blank"), ("1", None, "null"), ("1", " ", "whitespace"),
            ("1", "bad", "invalid_hadm"), ("1", "12", "other_admission"),
            ("bad", "10", "invalid_subject"), (None, "", "null_subject"),
            ("3", "", "other_subject"), (" 1 ", "010", "cast_identity"),
            ("2", "20", "second_subject"), ("1", "10", "duplicate"), ("1", "10", "duplicate")]]
        self.write_rows(rows)
        folder = self.base.root / "scope"
        folder.mkdir()
        matches = [*self.base.matches, {"subject_id": 1, "hadm_id": 11}]
        with burden_io.open_store(folder, "128MB") as connection:
            burden_io.ingest(connection, self.base.iv, self.base.ehr["burden_v2"], matches)
            kept = connection.execute("SELECT value FROM chartevents").fetchall()
        self.assertEqual(Counter(v[0] for v in kept), Counter({"exact": 1, "another_candidate_admission": 1,
                         "blank": 1, "null": 1, "cast_identity": 1, "second_subject": 1, "duplicate": 2}))

    def test_generated_wide_csv_is_filtered_under_128mb(self):
        template = self.base.tables["chartevents"][0]
        def rows():
            yield template
            for index in range(150000):
                yield {**template, "subject_id": str(index + 1000), "hadm_id": str(index + 2000),
                       "value": str(index) + "x" * 512}
        self.write_rows(rows())
        folder = self.base.root / "bounded"
        folder.mkdir()
        with burden_io.open_store(folder, "128MB", max_temp_directory_size="16MB") as connection:
            connection.execute("SET threads=1")
            burden_io.ingest(connection, self.base.iv, self.base.ehr["burden_v2"], self.base.matches)
            self.assertEqual(connection.execute("SELECT count(*) FROM chartevents").fetchone()[0], 1)

    def test_underestimated_csv_never_builds_the_hash_and_settings_are_restored(self):
        folder = self.base.root / "plan"
        folder.mkdir()
        matches = [*self.base.matches, *({"subject_id": i + 1000, "hadm_id": i + 2000} for i in range(800))]
        def contains_csv(node):
            return "READ_CSV" in node["name"] or any(contains_csv(n) for n in node.get("children", []))
        class PlanProxy(ConnectionProxy):
            checked = 0
            def execute(proxy, query, params=None):
                if query.startswith("CREATE TABLE chartevents AS"):
                    roots = json.loads(proxy.connection.execute("EXPLAIN (FORMAT JSON) " + query, params).fetchone()[1])
                    def visit(node):
                        if node["name"].strip() == "HASH_JOIN":
                            self.assertFalse(contains_csv(node["children"][1]))
                            proxy.checked += 1
                        for child in node.get("children", []):
                            visit(child)
                    for node in roots:
                        visit(node)
                return super().execute(query, params)
        with burden_io.open_store(folder, "128MB", max_temp_directory_size="16MB") as connection:
            connection.execute("SET disabled_optimizers='top_n'")
            proxy = PlanProxy(connection)
            burden_io.ingest(proxy, self.base.iv, self.base.ehr["burden_v2"], matches)
            self.assertGreater(proxy.checked, 0)
            self.assertEqual(connection.execute("SELECT current_setting('disabled_optimizers')").fetchone()[0], "top_n")

    def test_spill_failure_is_distinguished_from_memory_exhaustion(self):
        connect = duckdb.connect
        class SpillProxy(ConnectionProxy):
            def execute(proxy, query, params=None):
                if query.startswith("CREATE TABLE chartevents AS"):
                    raise duckdb.OutOfMemoryException("failed to offload data block (4 GiB/4 GiB used)")
                return super().execute(query, params)
        with patch("src.cohort.burden_io.duckdb.connect", side_effect=lambda *a, **k: SpillProxy(connect(*a, **k))):
            with self.assertRaises(PipelineError) as caught:
                self.fixture.audit()
        self.assertEqual(caught.exception.code, "CLINICAL_SPILL_LIMIT_EXCEEDED")
        self.assertIn("chartevents", str(caught.exception))

    def test_out_of_memory_reports_table_and_preserves_prior_audit(self):
        self.fixture.audit()
        before = self.fixture.context.layout.resolve(self.fixture.ref).read_bytes()
        connect = duckdb.connect
        with patch("src.cohort.burden_io.duckdb.connect", side_effect=lambda *a, **k: ConnectionProxy(connect(*a, **k), fail=True)):
            with self.assertRaises(PipelineError) as caught:
                self.fixture.audit()
        self.assertEqual(caught.exception.code, "CLINICAL_INGEST_OUT_OF_MEMORY")
        self.assertIn("chartevents", str(caught.exception))
        self.assertIn("128MB", str(caught.exception))
        self.assertNotIn(str(self.base.root), str(caught.exception))
        self.assertEqual(self.fixture.context.layout.resolve(self.fixture.ref).read_bytes(), before)
        state = json.loads((self.fixture.store.control / "step05.audit.state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.fixture.audit()
        self.assertEqual(self.fixture.store.validate(self.fixture.ref)["generation"], 2)


if __name__ == "__main__":
    unittest.main()
