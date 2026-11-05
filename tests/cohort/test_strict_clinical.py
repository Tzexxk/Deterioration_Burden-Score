"""Generated evidence cases distinguishing lenient and strict policies."""

import copy
import json
import unittest
from datetime import datetime, timedelta

from src.cohort import burden_events, strict_events

modules = {"r1": burden_events, "r2": strict_events}
base = datetime(2120, 1, 1, 12)


def t(h):
    return base + timedelta(hours=h)


study = {
    "sample_id": "synthetic",
    "subject_id": 1,
    "hadm_id": 2,
    "study_id": 3,
    "horizon_hours": 72,
    "t0": base,
    "observation_end": t(72),
}
admission = {
    "subject_id": "1",
    "hadm_id": "2",
    "admittime": str(t(-4)),
    "dischtime": str(t(80)),
}
stay = {
    "subject_id": "1",
    "hadm_id": "2",
    "stay_id": "4",
    "start": t(-3),
    "end": t(75),
    "reason": "",
}
context = {
    "admission": admission,
    "admissions": [admission],
    "stays": [stay],
    "ventilation_episodes": [],
    "ventilation_rejected": [],
    "chartevents": [],
}
confirmed = {"confirmed": True, "reviewer": "synthetic"}
rules = {
    "items": {"220277": confirmed},
    "spo2_units": {"%": confirmed},
    "spo2_threshold": 85,
    "supplementary_audit": {"enabled": True},
}
cases = []


def check(name, operation, expected):
    cases.append((name, operation, expected))


def chart(h, value=80, warning="0", item="220277"):
    return {
        "subject_id": "1",
        "hadm_id": "2",
        "stay_id": "4",
        "itemid": item,
        "charttime": str(t(h)),
        "storetime": str(t(h)),
        "value": str(value),
        "valuenum": str(value),
        "valueuom": "%",
        "warning": warning,
    }


def spo2(rows, m):
    c = copy.deepcopy(context)
    c["chartevents"] = rows
    return [
        (r["qualifies"], r["exclusion_reason"])
        for r in m.hypoxemia_evidence(study, c, rules)
    ]


check(
    "first_low_without_prior_nonlow",
    lambda m, k: spo2([chart(1)], m),
    {"r1": [(1, "")], "r2": [(None, "hypoxemia_onset_not_established")]},
)
check(
    "missing_warning_flag",
    lambda m, k: spo2([chart(0, 95), chart(1, 80, "")], m)[-1],
    {"r1": (1, ""), "r2": (None, "spo2_artifact_review")},
)
check(
    "confirmed_nonlow_then_low",
    lambda m, k: spo2([chart(0, 95), chart(1)], m)[-1],
    {"r1": (1, ""), "r2": (1, "")},
)


def vent(m, segmented=False, device=False):
    c = copy.deepcopy(context)
    if segmented:
        c["ventilation_episodes"] = [
            {
                "subject_id": "1",
                "hadm_id": "2",
                "start": t(-1),
                "end": t(4),
                "reason": "",
                "raw_records": [{"start": t(-1)}, {"start": t(1)}],
            }
        ]
    if device:
        c["chartevents"] = [chart(1, "Nasal cannula", item="10")]
        c["chart_item_labels"] = {"10": "O2 Delivery Device(s)"}
    return [
        (r["qualifies"], r["exclusion_reason"])
        for r in m.ventilation_evidence(study, c)
    ]


check(
    "merged_ventilation_hides_restart",
    lambda m, k: vent(m, segmented=True),
    {
        "r1": [(0, "preexisting_support")],
        "r2": [(None, "segmented_ventilation_onset_requires_review")],
    },
)
check(
    "oxygen_device_context",
    lambda m, k: vent(m, device=True),
    {
        "r1": [(None, "indirect_ventilation_only")],
        "r2": [(0, "oxygen_device_context_only")],
    },
)


def ev(m, key, h, q, reason="", dependency=None):
    raw = {"subject_id": "1", "hadm_id": "2", "source_record_key": key}
    if dependency:
        raw["onset_dependency_keys"] = dependency
    return m.evidence(study, m.COMPONENTS[3], "synthetic", raw, t(h), q, reason)


def override(m, scope, rows, keys):
    decision = {
        "reviewer": "synthetic",
        "rationale": "synthetic",
        "version": "fixture",
        "event_flag": "0",
        "first_event_time": "",
        "source_record_keys": json.dumps(keys),
        "override_scope": scope,
    }
    m.apply_override(study, m.COMPONENTS[3], rows, decision)
    return rows


check(
    "component_review_must_cover_other_positive",
    lambda m, k: override(m, "component", [ev(m, "a", 1, 1), ev(m, "b", 2, 1)], ["a"])[
        1
    ]["qualifies"],
    {"r1": 0, "r2": None},
)
check(
    "excluded_predecessor_invalidates_onset",
    lambda m, k: override(
        m,
        "source_record",
        [ev(m, "a", 1, 0), ev(m, "b", 2, 1, dependency=["a"])],
        ["a"],
    )[1]["qualifies"],
    {"r1": 1, "r2": None},
)


def independent(m, k):
    rows = [ev(m, "a", 1, 1), ev(m, "b", 2, None, "spo2_artifact_review")]
    if k == "r2":
        m.classify_uncertainty(study, rows, True)
    return any(r["qualifies"] is None and (r.get("blocks_event", 1)) for r in rows)


check(
    "later_independent_uncertainty_is_not_blocking",
    independent,
    {"r1": True, "r2": False},
)


def medication(m, k):
    row = {
        "subject_id": "1",
        "hadm_id": "2",
        "pharmacy_id": "5",
        "medication": "named bag",
        "source_record_keys": ["emar:a"],
    }
    sources = {
        "pharmacy": {},
        "prescriptions": {
            ("1", "2", "5"): [
                {
                    "drug": "unmapped component",
                    "duplicate_count": 1,
                    "source_record_key": "prescriptions:a",
                }
            ]
        },
    }
    m.medication_identity(
        row, sources, {"medications": {}, "supplementary_audit": {"enabled": True}}
    )
    return row.get("component_identity_status", "not_checked")


check(
    "nonblank_medication_still_crosschecks_components",
    medication,
    {"r1": "not_checked", "r2": "requires_review"},
)


class StrictClinicalTests(unittest.TestCase):
    def test_published_policy_boundary_cases(self):
        self.assertEqual(len(cases), 9)
        for name, operation, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(
                    {key: operation(module, key) for key, module in modules.items()},
                    expected,
                )

    def test_explicit_dispatch_uses_strict_without_changing_default(self):
        c = copy.deepcopy(context)
        c["chartevents"] = [chart(1)]
        lenient = burden_events.hypoxemia_evidence(study, c, rules)
        strict = burden_events.hypoxemia_evidence(
            study, c, {**rules, "review_policy": "strict"}
        )
        self.assertEqual(lenient[0]["qualifies"], 1)
        self.assertIsNone(strict[0]["qualifies"])
        self.assertNotIn("blocks_event", lenient[0])
        self.assertEqual(strict[0]["blocks_event"], 1)
        self.assertNotIn("blocks_event", burden_events.EVIDENCE_COLUMNS)
        self.assertIn("blocks_event", strict_events.EVIDENCE_COLUMNS)

    def test_historical_only_review_key_requires_exact_payload_and_owner(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from src.cohort.burden_io import record_key
        from src.cohort.strict_reviews import ReviewArchive, raw_source_index
        from src.common.paths import PipelineError

        raw = chart(1)
        key = record_key("chartevents", raw)
        current = raw_source_index([{**raw, "source_record_key": key}])
        candidate = {**study, "dischtime": t(80)}
        snapshot = {
            "inputs": [],
            "input_versions": {},
            "review_evidence_sources": {},
            "configuration": {"cohort": {"burden_v2": {}}},
        }
        row = {
            "source_audit_generation": "1",
            "decision": "exclude_source",
            "source_record_keys": json.dumps([key]),
            "historical_keys": json.dumps([key]),
        }
        with TemporaryDirectory() as temp:
            archive = ReviewArchive(None, Path(temp), {"snapshots": {"1": snapshot}})
            with patch.object(
                archive,
                "source",
                return_value=({candidate["sample_id"]: candidate}, {}),
            ):
                archive.bind(
                    row, candidate, current, {"review_policy": "strict"}, {}, {}, []
                )
                archive.write_historical_evidence()
                self.assertTrue((Path(temp) / "historical_evidence.parquet").is_file())
                self.assertEqual(archive.historical_keys, {key})
                for edit in ({"content": "changed"}, {"hadm_id": None}):
                    with self.assertRaises(PipelineError) as caught:
                        archive.bind(
                            row,
                            candidate,
                            {key: {**current[key], **edit}},
                            {"review_policy": "strict"},
                            {},
                            {},
                            [],
                        )
                    self.assertEqual(caught.exception.code, "REVIEW_EVIDENCE_CHANGED")


if __name__ == "__main__":
    unittest.main()
