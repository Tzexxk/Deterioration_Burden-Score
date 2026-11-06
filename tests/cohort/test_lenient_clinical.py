"""Synthetic clinical checks derived from the published lenient test cases."""

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import pyarrow.parquet as pq

from src.cohort.audit_burden_labels import (
    IDENTITY_POLICY,
    medication_identity_exclusion,
    validate_rules,
)
from src.cohort.build_candidates import (
    build_candidate,
    observation,
    ownership,
    prepare_intervals,
)
from src.cohort.burden_events import (
    aggregate_medications,
    apply_override,
    evaluate,
    hypoxemia_evidence,
    infusion_evidence,
    ventilation_episodes,
    ventilation_evidence,
)
from src.cohort.burden_io import (
    COMPONENTS,
    TABLE_COLUMNS,
    ParquetSink,
    clinical_time,
    score,
)
from src.common.paths import read_json as _json_file

PROJECT = Path(__file__).resolve().parents[2]
BASE = datetime(2120, 1, 1, 10)


def dt(hours):
    return (BASE + timedelta(hours=hours)).isoformat()


def reviewed(**fields):
    return dict(
        confirmed=True,
        reviewer="synthetic_reviewer",
        mapping_reason="synthetic fixture rule",
        **fields,
    )


class LenientClinicalTests(unittest.TestCase):
    def setUp(self) -> None:
        (PROJECT / "outputs/step05_tests").mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            dir=PROJECT / "outputs/step05_tests", prefix="burden_test_"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.iv, self.cxr = (self.root / "iv", self.root / "cxr")
        self.config = self.root / "configs"
        self.config.mkdir()
        self.processed, self.outputs = (self.root / "processed", self.root / "outputs")
        self.tables = {name: [] for name in TABLE_COLUMNS}
        self.paths = dict(
            mimic_iv=str(self.iv),
            mimic_cxr=str(self.cxr),
            processed=str(self.processed),
            outputs=str(self.outputs),
        )
        self.cohort = copy.deepcopy(
            _json_file(PROJECT / "configs/data.template/lenient/cohort/burden_rules.json")
        )
        self.ehr = copy.deepcopy(
            _json_file(PROJECT / "configs/data.template/lenient/ehr/burden_sources.json")
        )
        self.imaging = copy.deepcopy(
            _json_file(PROJECT / "configs/data.template/lenient/imaging/raw_audit.json")
        )
        self.ehr["burden_v2"]["memory_limit"] = "128MB"
        rules = self.cohort["burden_v2"]
        rules["dictionary_handling_reviews"] = {}
        rules["overrides_path"] = None
        rules["candidate_exclusions_path"] = None
        rules["candidate_conflict_policies"] = {}
        rules[IDENTITY_POLICY].update(
            confirmed=False,
            action="exclude",
            reason="synthetic_identity",
            reviewer="synthetic",
            rationale="fixture",
            version="fixture",
        )
        rules["dictionary_version"] = "synthetic_reviewed_v1"
        rules["transfer_event_actions"] = {
            "admit": reviewed(action="location"),
            "transfer": reviewed(action="location"),
            "discharge": reviewed(action="exclude"),
        }
        rules["careunits"] = {
            name: reviewed(care_level=level)
            for name, level in (
                ("Ward", "ward"),
                ("Stepdown", "stepdown_intermediate"),
                ("ICU", "icu_ccu"),
            )
        }
        rules["medications"] = {
            "norepinephrine": reviewed(canonical_drug="norepinephrine"),
            "acetaminophen": reviewed(canonical_drug=None),
        }
        rules["administration_types"] = {
            "IV Infusion": reviewed(action="infusion"),
            "IV Bolus": reviewed(action="exclude"),
        }
        rules["medication_event_actions"] = {
            name: reviewed(action=action)
            for name, action in (
                ("Started", "start"),
                ("Restarted", "restart"),
                ("Continued", "continue"),
                ("Stopped", "stop"),
                ("Given", "given"),
                ("Cancelled", "exclude"),
            )
        }
        rules["procedure_status_actions"] = {
            "Finished": reviewed(action="include"),
            "Cancelled": reviewed(action="exclude"),
        }
        rules["iv_routes"] = {"IV": reviewed(action="iv")}
        rules["spo2_units"] = {"%": reviewed()}
        for value in rules["items"].values():
            value.update(reviewed())
        for source in ("pharmacy", "prescriptions"):
            rules[source + "_identity_lookup"] = {
                "confirmed": True,
                "reviewer": "synthetic",
                "mapping_reason": "fixture",
                "version": "fixture",
            }
        self.tables["d_items"] = [
            dict(itemid=key, label=value["label"], linksto=value["linksto"])
            for key, value in rules["items"].items()
        ]
        self.tables["patients"] = [
            dict(
                subject_id=str(subject),
                gender="F",
                anchor_age="40",
                anchor_year="2120",
                anchor_year_group="2017 - 2019",
                dod="",
            )
            for subject in (1, 2)
        ]
        self.tables["admissions"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                admittime=dt(-10),
                dischtime=dt(70),
                deathtime=dt(60),
                hospital_expire_flag="1",
                discharge_location="DIED",
            ),
            dict(
                subject_id="2",
                hadm_id="20",
                admittime=dt(-10),
                dischtime=dt(20),
                deathtime="",
                hospital_expire_flag="0",
                discharge_location="HOME",
            ),
        ]
        for index, (unit, start, end) in enumerate(
            (
                ("ICU", -9, -8),
                ("Ward", -8, 1),
                ("Stepdown", 1, 2),
                ("ICU", 2, 12),
                ("Ward", 12, 70),
            )
        ):
            self.tables["transfers"].append(
                dict(
                    subject_id="1",
                    hadm_id="10",
                    transfer_id=str(index),
                    eventtype="transfer",
                    careunit=unit,
                    intime=dt(start),
                    outtime=dt(end),
                )
            )
        self.tables["transfers"].append(
            dict(
                subject_id="2",
                hadm_id="20",
                transfer_id="20",
                eventtype="admit",
                careunit="Ward",
                intime=dt(-10),
                outtime=dt(20),
            )
        )
        self.tables["icustays"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id=str(stay),
                intime=dt(start),
                outtime=dt(end),
            )
            for stay, start, end in ((11, -9, -8), (12, 2, 12))
        ]
        self.tables["procedureevents"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id="12",
                itemid="225792",
                starttime=dt(4),
                endtime=dt(6),
                storetime=dt(8),
                orderid="1",
                linkorderid="1",
                statusdescription="Finished",
                continueinnextdept="0",
            )
        ]
        self.tables["chartevents"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id="12",
                itemid="220277",
                charttime=dt(3),
                storetime=dt(4),
                value="85",
                valuenum="85",
                valueuom="%",
                warning="0",
            )
        ]
        self.add_medication("m1", 5, "Started")
        self.add_medication("m2", 14, "Continued")
        self.matches = [
            dict(
                subject_id=subject,
                hadm_id=hadm,
                study_id=study,
                dicom_id=f"d{study}",
                t0=dt(time),
                admittime=dt(-10),
                dischtime=dt(end),
                match_status="unique_match",
            )
            for subject, hadm, study, time, end in (
                (1, 10, 100, 0, 70),
                (1, 10, 101, 13, 70),
                (2, 20, 200, 0, 20),
            )
        ]

    def add_medication(self, key: str, time: float, event: str, **detail) -> None:
        self.tables["emar"].append(
            dict(
                subject_id="1",
                hadm_id="10",
                emar_id=key,
                emar_seq=key[1:],
                pharmacy_id="1",
                charttime=dt(time),
                medication="Norepinephrine",
                event_txt=event,
                scheduletime=dt(time - 1),
                storetime=dt(time + 1),
            )
        )
        value = dict(
            subject_id="1",
            emar_id=key,
            emar_seq=key[1:],
            parent_field_ordinal="",
            administration_type="IV Infusion",
            complete_dose_not_given="0",
            dose_given="1",
            route="IV",
            prior_infusion_rate="",
            infusion_rate="2",
            infusion_complete="0",
            new_iv_bag_hung="0",
            continued_infusion_in_other_location="0",
        )
        value.update(detail)
        self.tables["emar_detail"].append(value)

    def context(self) -> dict:
        admission = self.tables["admissions"][0]
        rules = self.cohort["burden_v2"]
        context = dict(
            admission=admission,
            admissions=self.tables["admissions"],
            source_names=self.ehr["burden_v2"]["tables"],
            transfers=prepare_intervals(
                "transfers",
                self.tables["transfers"][:-1],
                admission,
                self.tables["admissions"],
                rules,
            ),
            stays=prepare_intervals(
                "icustays",
                self.tables["icustays"],
                admission,
                self.tables["admissions"],
                rules,
            ),
            **{
                table: self.tables[table]
                for table in (
                    "procedureevents",
                    "chartevents",
                    "emar",
                    "pharmacy",
                    "prescriptions",
                )
            },
        )
        context["ventilation_episodes"], context["ventilation_rejected"] = (
            ventilation_episodes(context, rules)
        )
        context["medication_events"] = aggregate_medications(
            context, self.tables["emar_detail"], rules
        )
        return context

    def study(self, time=0, hours=72) -> dict:
        candidate = dict(
            sample_id="p1_h10_s100",
            subject_id=1,
            hadm_id=10,
            study_id=100,
            t0=clinical_time(dt(time)),
            dischtime=clinical_time(dt(70)),
            candidate_status="eligible",
            hospital_expire_flag="1",
            exclusion_reason="",
        )
        return observation(candidate, hours)

    def test_onset_boundaries_merged_support_and_restart(self) -> None:
        self.tables["icustays"][1].update(intime=dt(-2), outtime=dt(65))
        self.tables["procedureevents"][0].update(starttime=dt(0), endtime=dt(2))
        self.tables["procedureevents"].append(
            {
                **self.tables["procedureevents"][0],
                "orderid": "2",
                "starttime": dt(2),
                "endtime": dt(4),
            }
        )
        context = self.context()
        self.assertEqual(len(context["ventilation_episodes"]), 1)
        events = ventilation_evidence(self.study(), context)
        self.assertFalse(any((row["qualifies"] == 1 for row in events)))
        self.assertEqual(events[0]["baseline_status"], "active_documented")
        for start, _, expected72 in (
            (1, 1, 1),
            (3, 1, 1),
            (72, 1, 1),
            (49, 0, 1),
            (72, 0, 1),
        ):
            with self.subTest(start=start):
                self.tables["admissions"][0]["dischtime"] = dt(80)
                self.tables["icustays"][1]["outtime"] = dt(80)
                self.tables["procedureevents"] = [
                    {
                        **self.tables["procedureevents"][0],
                        "starttime": dt(start),
                        "endtime": dt(start + 1),
                    }
                ]
                context = self.context()
                study = {**self.study(), "dischtime": clinical_time(dt(80))}
                for horizon, expected in ((72, expected72),):
                    records = ventilation_evidence(observation(study, horizon), context)
                    self.assertEqual(
                        int(any((row["qualifies"] == 1 for row in records))), expected
                    )
        self.tables["emar"], self.tables["emar_detail"] = ([], [])
        self.add_medication("m1", -1, "Started")
        self.add_medication("m2", 1, "Continued", new_iv_bag_hung="1")
        self.add_medication("m3", 2, "Stopped")
        self.add_medication("m4", 3, "Restarted")
        records = infusion_evidence(self.study(), self.context())
        positive = [row for row in records if row["qualifies"] == 1]
        self.assertEqual(
            [row["event_time"] for row in positive], [clinical_time(dt(3))]
        )
        self.assertTrue(
            any((row["exclusion_reason"] == "preexisting_support" for row in records))
        )

    def test_spo2_recovery_artifacts_stale_stay_and_missing_records(self) -> None:
        rules = self.cohort["burden_v2"]
        self.tables["chartevents"] = []
        self.assertEqual(hypoxemia_evidence(self.study(), self.context(), rules), [])
        template = dict(
            subject_id="1",
            hadm_id="10",
            stay_id="11",
            itemid="220277",
            charttime=dt(-8.5),
            storetime=dt(-8),
            value="80",
            valuenum="80",
            valueuom="%",
            warning="0",
        )
        self.tables["chartevents"] = [
            template,
            {**template, "stay_id": "12", "charttime": dt(3)},
        ]
        records = hypoxemia_evidence(self.study(), self.context(), rules)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["qualifies"], 1)
        self.assertEqual(records[0]["baseline_status"], "not_observed")
        self.tables["icustays"][1].update(intime=dt(-1), outtime=dt(12))
        self.tables["chartevents"] = [
            {
                **template,
                "stay_id": "12",
                "charttime": dt(time),
                "valuenum": str(value),
                "value": str(value),
            }
            for time, value in ((-0.5, 80), (1, 82), (2, 95), (3, 85))
        ]
        records = hypoxemia_evidence(self.study(), self.context(), rules)
        self.assertEqual(
            [row["event_time"] for row in records if row["qualifies"] == 1],
            [clinical_time(dt(3))],
        )
        self.tables["chartevents"][-1]["warning"] = "1"
        records = hypoxemia_evidence(self.study(), self.context(), rules)
        self.assertIsNone(records[-1]["qualifies"])

    def test_bolus_cancelled_unknown_onset_and_admission_ownership(self) -> None:
        self.tables["emar"][0]["event_txt"] = "Given"
        records = infusion_evidence(self.study(), self.context())
        self.assertTrue(any((row["qualifies"] is None for row in records)))
        self.tables["emar_detail"][0]["administration_type"] = "IV Bolus"
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 0)
        self.tables["emar_detail"][0]["administration_type"] = "IV Infusion"
        self.tables["emar_detail"][0]["complete_dose_not_given"] = "1"
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 0)
        self.tables["chartevents"][0]["hadm_id"] = ""
        records = hypoxemia_evidence(
            self.study(), self.context(), self.cohort["burden_v2"]
        )
        self.assertEqual(records[0]["qualifies"], 1)
        self.tables["admissions"].append(
            dict(
                subject_id="1",
                hadm_id="11",
                admittime=dt(2),
                dischtime=dt(4),
                deathtime="",
                hospital_expire_flag="0",
                discharge_location="HOME",
            )
        )
        records = hypoxemia_evidence(
            self.study(), self.context(), self.cohort["burden_v2"]
        )
        self.assertIsNone(records[0]["qualifies"])

    def test_excluded_administration_types_preserve_history_and_conflicts(self) -> None:
        rules = self.cohort["burden_v2"]
        excluded_types = (
            "IV Flush",
            "Transdermal Patch",
            "Insulin SC Sliding Scale",
            "Insulin SC Fixed Dose",
            "Epidural",
            "Heparin IV Bolus",
            "Heparin IV Infusion",
            "Insulin Infusion (IV Drip)",
            "TPN",
            "IV Fluid Repletion",
        )
        rules["administration_types"].update(
            {name: reviewed(action="exclude") for name in excluded_types}
        )
        for name in excluded_types:
            with self.subTest(administration_type=name):
                self.tables["emar"], self.tables["emar_detail"] = ([], [])
                self.add_medication("m1", 1, "Started")
                self.add_medication("m2", 2, "Stopped", administration_type=name)
                self.add_medication("m3", 3, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [1, 0, 0])
                self.assertTrue(context["medication_events"][-1]["active_before"])
                self.tables["emar"][1]["medication"] = ""
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [1, 0, 0])
                self.tables["emar"][1]["medication"] = "Norepinephrine"
                self.tables["emar_detail"].append(
                    {
                        **self.tables["emar_detail"][1],
                        "parent_field_ordinal": "1",
                        "administration_type": "IV Infusion",
                    }
                )
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [1, None, 0])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unresolved_administration_type"
                )
                self.assertTrue(context["medication_events"][-1]["active_before"])
        self.tables["emar_detail"].pop()
        self.tables["emar"][1]["medication"] = ""
        self.tables["emar_detail"][1]["subject_id"] = "2"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[1]["qualifies"])
        self.assertEqual(records[1]["exclusion_reason"], "emar_detail_key_mismatch")
        self.tables["emar_detail"][1]["subject_id"] = "1"
        self.tables["emar_detail"][1]["administration_type"] = "IV Infusion"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[1]["qualifies"])
        self.assertEqual(records[1]["exclusion_reason"], "unconfirmed_medication_name")

    def test_review_type_policies_and_blank_detail_aggregation(self) -> None:
        ambiguous_types = (
            "Standard Maintenance Medication",
            "PCA",
            "Sliding Scale",
            "OVR",
            "PACU-only order",
            "",
        )
        rules = self.cohort["burden_v2"]
        rules["administration_types"].update(
            {name: reviewed(action="review") for name in ambiguous_types}
        )
        validate_rules(self.cohort, self.ehr, self.imaging)
        for name in ambiguous_types:
            with self.subTest(administration_type=name):
                self.tables["emar"], self.tables["emar_detail"] = ([], [])
                self.add_medication("m1", 1, "Started")
                self.add_medication("m2", 2, "Stopped", administration_type=name)
                self.add_medication("m3", 3, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [1, None, 0])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unresolved_administration_type"
                )
                self.assertTrue(context["medication_events"][-1]["active_before"])
                self.tables["emar"][1]["medication"] = "Acetaminophen"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [1, 0, 0])
                self.tables["emar"][1]["medication"] = ""
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[1]["qualifies"])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unconfirmed_medication_name"
                )
        self.tables["emar"], self.tables["emar_detail"] = ([], [])
        self.add_medication("m1", 1, "Started", administration_type="")
        self.tables["emar_detail"].append(
            {
                **self.tables["emar_detail"][0],
                "parent_field_ordinal": "1",
                "administration_type": "IV Infusion",
            }
        )
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual([row["qualifies"] for row in records], [1])
        self.tables["emar_detail"][0]["administration_type"] = "PCA"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(records[0]["review_status"], "pending")
        self.tables["emar_detail"][1]["administration_type"] = "IV Bolus"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.tables["emar_detail"][0]["administration_type"] = ""
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual([row["qualifies"] for row in records], [0])

    def test_excluded_preadmission_medications_do_not_reopen_infusion_review(
        self,
    ) -> None:
        for medication, administration_type in (
            ("Acetaminophen", "IV Infusion"),
            ("Norepinephrine", "IV Bolus"),
            ("", "IV Bolus"),
        ):
            with self.subTest(
                medication=medication, administration_type=administration_type
            ):
                self.tables["emar"], self.tables["emar_detail"] = ([], [])
                self.add_medication(
                    "m0", -11, "Stopped", administration_type=administration_type
                )
                self.tables["emar"][0]["medication"] = medication
                self.add_medication("m1", -2, "Started")
                self.add_medication("m2", 1, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual(records[0]["qualifies"], 0)
                self.assertEqual(
                    records[0]["exclusion_reason"], "event_outside_admission"
                )
                self.assertEqual(records[0]["review_status"], "not_required")
                self.assertTrue(context["medication_events"][-1]["active_before"])
                result, _ = evaluate(
                    self.study(), context, self.cohort["burden_v2"], {}
                )
                self.assertEqual(result[COMPONENTS[4]], 0)
                self.assertEqual(result["label_status"], "complete")
                self.tables["emar_detail"][0]["subject_id"] = "2"
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[0]["qualifies"])
                self.assertEqual(
                    records[0]["exclusion_reason"], "emar_detail_key_mismatch"
                )
        self.tables["emar_detail"][0]["subject_id"] = "1"
        self.tables["emar_detail"][0]["administration_type"] = "IV Infusion"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(records[0]["review_status"], "pending")

    def test_review_event_states_preserve_history_and_baseline(self) -> None:
        states = (
            "",
            " in Other Location",
            "Not Confirmed",
            "Delayed ",
            "Infusion Reconciliation Not Done",
            "Documented in O.R. Holding",
        )
        rules = self.cohort["burden_v2"]
        rules["medication_event_actions"].update(
            {state: reviewed(action="review") for state in states}
        )
        validate_rules(self.cohort, self.ehr, self.imaging)
        for state in states:
            with self.subTest(event_state=state):
                self.tables["emar"], self.tables["emar_detail"] = ([], [])
                self.add_medication("m1", -2, "Started")
                self.add_medication("m2", -1, state, infusion_complete="1")
                self.add_medication("m3", 1, "Continued")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [0, None, 0])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unresolved_medication_event"
                )
                self.assertTrue(context["medication_events"][-1]["active_before"])
                self.assertEqual(records[-1]["baseline_status"], "active_documented")
                result, _ = evaluate(self.study(), context, rules, {})
                self.assertIsNone(result[COMPONENTS[4]])
                self.assertIsNone(result["burden_score"])
                self.assertEqual(result["label_status"], "unresolved")
                self.tables["emar"][1]["medication"] = "Acetaminophen"
                context = self.context()
                self.assertEqual(context["medication_events"][1]["action"], "exclude")
                records = infusion_evidence(self.study(), context)
                self.assertEqual([row["qualifies"] for row in records], [0, 0])
                self.assertEqual(records[-1]["baseline_status"], "active_documented")
                self.tables["emar"][1]["medication"] = "Norepinephrine"
                self.tables["emar_detail"][1]["administration_type"] = "IV Bolus"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [0, 0, 0])
                self.tables["emar"][1]["medication"] = ""
                self.tables["emar_detail"][1]["administration_type"] = "IV Infusion"
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[1]["qualifies"])
                self.assertEqual(
                    records[1]["exclusion_reason"], "unconfirmed_medication_name"
                )
                self.tables["emar"], self.tables["emar_detail"] = ([], [])
                self.add_medication("m1", -3, "Started")
                self.add_medication("m2", -2, "Stopped")
                self.add_medication("m3", -1, state)
                self.add_medication("m4", 1, "Restarted")
                context = self.context()
                records = infusion_evidence(self.study(), context)
                self.assertEqual(records[-1]["qualifies"], 1)
                self.assertEqual(records[-1]["baseline_status"], "inactive_documented")

    def test_route_consistency_and_missing_route_policy(self) -> None:
        rules = self.cohort["burden_v2"]
        non_iv_routes = ("PO", "NG", "SC", "IM", "PR")
        rules["iv_routes"].update(
            {route: reviewed(action="non_iv") for route in non_iv_routes}
        )
        rules["iv_routes"][""] = reviewed(action="missing")
        validate_rules(self.cohort, self.ehr, self.imaging)
        for route in non_iv_routes:
            with self.subTest(route=route):
                self.tables["emar"], self.tables["emar_detail"] = ([], [])
                self.add_medication("m1", 1, "Started", route=route)
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[0]["qualifies"])
                self.assertEqual(
                    records[0]["exclusion_reason"], "route_conflict_or_unconfirmed"
                )
                self.tables["emar"][0]["medication"] = "Acetaminophen"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual(records[0]["qualifies"], 0)
                self.tables["emar"][0]["medication"] = "Norepinephrine"
                self.tables["emar_detail"][0]["administration_type"] = "IV Bolus"
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual(records[0]["qualifies"], 0)
        self.tables["emar"], self.tables["emar_detail"] = ([], [])
        self.add_medication("m1", 1, "Started", route="")
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 1)
        self.tables["emar_detail"].append(
            {
                **self.tables["emar_detail"][0],
                "parent_field_ordinal": "1",
                "route": "PO",
            }
        )
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.tables["emar_detail"][1]["route"] = "IV"
        records = infusion_evidence(self.study(), self.context())
        self.assertEqual(records[0]["qualifies"], 1)
        self.tables["emar_detail"] = self.tables["emar_detail"][:1]
        self.tables["emar_detail"][0]["administration_type"] = ""
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(
            records[0]["exclusion_reason"], "unresolved_administration_type"
        )
        self.tables["emar"], self.tables["emar_detail"] = ([], [])
        self.add_medication("m1", -2, "Started", route="")
        self.add_medication("m2", -1, "Stopped", route="PO", infusion_complete="1")
        self.add_medication("m3", 1, "Continued", route="")
        context = self.context()
        records = infusion_evidence(self.study(), context)
        self.assertEqual([row["qualifies"] for row in records], [0, None, 0])
        self.assertTrue(context["medication_events"][-1]["active_before"])
        self.assertEqual(records[-1]["baseline_status"], "active_documented")
        rules["iv_routes"]["PO"]["action"] = "missing"
        with self.assertRaisesRegex(ValueError, "blank route"):
            validate_rules(self.cohort, self.ehr, self.imaging)
        rules["iv_routes"]["PO"]["action"] = "non_iv"
        rules["iv_routes"][""]["action"] = "iv"
        with self.assertRaisesRegex(ValueError, "blank route"):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_partial_administration_and_not_stopped_need_history(self) -> None:
        rules = self.cohort["burden_v2"]
        rules["medication_event_actions"].update(
            {
                "Partial Administered": reviewed(action="given"),
                "Not Stopped": reviewed(action="continue"),
            }
        )
        for state in ("Partial Administered", "Not Stopped"):
            with self.subTest(state=state):
                self.tables["emar"][0]["event_txt"] = "Started"
                self.tables["emar"][1]["event_txt"] = state
                records = infusion_evidence(self.study(), self.context())
                self.assertEqual([row["qualifies"] for row in records], [1, 0])
                records = infusion_evidence(self.study(time=6), self.context())
                self.assertEqual([row["qualifies"] for row in records], [0, 0])
                self.tables["emar"][0]["event_txt"] = state
                self.tables["emar"][1]["event_txt"] = "Stopped"
                records = infusion_evidence(self.study(), self.context())
                self.assertIsNone(records[0]["qualifies"])
                self.assertEqual(records[0]["review_status"], "pending")
        self.tables["emar"][0]["event_txt"] = "Partial Administered"
        self.tables["emar_detail"][0]["complete_dose_not_given"] = "1"
        records = infusion_evidence(self.study(), self.context())
        self.assertIsNone(records[0]["qualifies"])
        self.assertEqual(
            records[0]["exclusion_reason"], "unresolved_partial_administration"
        )

    def test_crossing_intervals_preserve_location_history_and_identity(self) -> None:
        admission = self.tables["admissions"][0]
        rules = self.cohort["burden_v2"]
        crossing = dict(
            subject_id="1",
            hadm_id="10",
            transfer_id="90",
            eventtype="admit",
            careunit="Ward",
            intime=dt(-11),
            outtime=dt(71),
        )
        discharge = dict(
            crossing,
            transfer_id="91",
            eventtype="discharge",
            careunit="UNKNOWN",
            intime=dt(70 + 44 / 3600),
            outtime="",
        )
        values = prepare_intervals(
            "transfers",
            [crossing, discharge],
            admission,
            self.tables["admissions"],
            rules,
        )
        ward, administrative = values
        self.assertEqual(ward["reason"], "")
        self.assertEqual(ward["intime"], crossing["intime"])
        self.assertEqual(ward["outtime"], crossing["outtime"])
        self.assertEqual(ward["start"], clinical_time(dt(-11)))
        self.assertEqual(ward["end"], clinical_time(dt(71)))
        self.assertEqual(ward["admission_overlap_start"], clinical_time(dt(-10)))
        self.assertEqual(ward["admission_overlap_end"], clinical_time(dt(70)))
        self.assertEqual(ward["starts_before_admission"], 1)
        self.assertEqual(ward["ends_after_discharge"], 1)
        self.assertEqual(administrative["reason"], "excluded_nonlocation_record")
        self.assertIsNone(administrative["end"])
        self.assertEqual(administrative["outtime_imputed"], 0)
        self.assertEqual(
            ownership(crossing, admission, self.tables["admissions"], ward["start"]),
            "event_outside_admission",
        )
        for changes, expected in (
            ({"subject_id": "2"}, "subject_mismatch"),
            ({"hadm_id": "20"}, "other_admission"),
            ({"intime": dt(72), "outtime": dt(73)}, "interval_outside_admission"),
            ({"intime": dt(-12), "outtime": dt(-10)}, "interval_outside_admission"),
            ({"intime": dt(2), "outtime": dt(1)}, "invalid_care_interval"),
            ({"intime": dt(1), "outtime": dt(1)}, "invalid_care_interval"),
            (
                {"intime": dt(1), "outtime": "2120-01-01T12:00:00+08:00"},
                "invalid_care_interval",
            ),
        ):
            with self.subTest(changes=changes):
                row = prepare_intervals(
                    "transfers",
                    [{**crossing, **changes}],
                    admission,
                    self.tables["admissions"],
                    rules,
                )[0]
                self.assertEqual(row["reason"], expected)
        row = prepare_intervals(
            "transfers",
            [{**discharge, "subject_id": "2"}],
            admission,
            self.tables["admissions"],
            rules,
        )[0]
        self.assertEqual(row["reason"], "subject_mismatch")
        missing = {**crossing, "hadm_id": "", "intime": dt(0)}
        row = prepare_intervals(
            "transfers", [missing], admission, self.tables["admissions"], rules
        )[0]
        self.assertEqual(row["reason"], "")
        row = prepare_intervals(
            "transfers",
            [missing],
            admission,
            [admission, {**admission, "hadm_id": "11"}],
            rules,
        )[0]
        self.assertEqual(row["reason"], "ambiguous_missing_hadm")
        row = prepare_intervals(
            "transfers",
            [{**missing, "intime": dt(-11)}],
            admission,
            self.tables["admissions"],
            rules,
        )[0]
        self.assertEqual(row["reason"], "other_admission")

    def test_boundary_stays_keep_ventilation_and_hypoxemia(self) -> None:
        self.tables["icustays"][1]["outtime"] = dt(71)
        context = self.context()
        self.assertEqual(context["stays"][1]["reason"], "")
        self.assertEqual(context["ventilation_episodes"][0]["reason"], "")
        result, _ = evaluate(self.study(), context, self.cohort["burden_v2"], {})
        self.assertEqual(result[COMPONENTS[2]], 1)
        self.assertEqual(result[COMPONENTS[3]], 1)
        self.tables["transfers"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id="90",
                eventtype="admit",
                careunit="ICU",
                intime=dt(-11),
                outtime=dt(71),
            ),
            self.tables["transfers"][-1],
        ]
        self.tables["icustays"] = [
            dict(
                subject_id="1",
                hadm_id="10",
                stay_id="12",
                intime=dt(-11),
                outtime=dt(71),
            )
        ]
        context = self.context()
        candidate = build_candidate(
            self.matches[0],
            dict(
                dicom_path="d100.dcm",
                view_position="AP",
                selection_reason="single_frontal",
            ),
            context["admission"],
            self.tables["patients"][0],
            context["transfers"],
            context["stays"],
            self.cohort["burden_v2"],
            True,
        )
        self.assertEqual(candidate["candidate_status"], "excluded")
        self.assertEqual(candidate["exclusion_reason"], "icu_at_t0")

    def test_interval_conflicts_are_scoped_to_current_location_and_events(self) -> None:
        selection = dict(
            dicom_path="d100.dcm", view_position="AP", selection_reason="single_frontal"
        )
        self.tables["transfers"][3]["outtime"] = dt(1)
        self.tables["transfers"].insert(
            0,
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id="91",
                eventtype="transfer",
                careunit="Ward",
                intime=dt(-2),
                outtime=dt(-3),
            ),
        )
        context = self.context()
        candidate = build_candidate(
            self.matches[0],
            selection,
            context["admission"],
            self.tables["patients"][0],
            context["transfers"],
            context["stays"],
            self.cohort["burden_v2"],
            True,
        )
        self.assertEqual(candidate["candidate_status"], "eligible")
        result, records = evaluate(self.study(), context, self.cohort["burden_v2"], {})
        self.assertIsNone(result[COMPONENTS[0]])
        self.assertEqual(result["label_status"], "unresolved")
        self.assertTrue(
            any((row["exclusion_reason"] == "invalid_care_interval" for row in records))
        )
        self.tables["transfers"][0].update(intime=dt(-1), outtime=dt(1))
        self.tables["transfers"][0]["eventtype"] = "unconfirmed"
        context = self.context()
        candidate = build_candidate(
            self.matches[0],
            selection,
            context["admission"],
            self.tables["patients"][0],
            context["transfers"],
            context["stays"],
            self.cohort["burden_v2"],
            True,
        )
        self.assertEqual(candidate["candidate_status"], "needs_review")
        self.assertIn("care_interval_conflict", candidate["exclusion_reason"])

    def test_outtime_imputation_ignores_nonlocation_and_unconfirmed_records(
        self,
    ) -> None:
        admission = self.tables["admissions"][0]
        rows = [
            dict(
                subject_id="1",
                hadm_id="10",
                transfer_id=str(index),
                eventtype=event,
                careunit="Ward",
                intime=dt(start),
                outtime=end,
            )
            for index, (event, start, end) in enumerate(
                (
                    ("admit", -10, ""),
                    ("discharge", -2, ""),
                    ("unconfirmed", -1, dt(1)),
                    ("transfer", 2, dt(70)),
                )
            )
        ]
        values = prepare_intervals(
            "transfers",
            rows,
            admission,
            self.tables["admissions"],
            self.cohort["burden_v2"],
        )
        self.assertEqual(values[0]["end"], clinical_time(dt(2)))
        self.assertEqual(values[0]["outtime_imputed"], 1)
        self.assertEqual(values[0]["imputation_basis"], "next_intime_or_dischtime")
        self.assertIsNone(values[1]["end"])
        self.assertEqual(values[1]["outtime_imputed"], 0)

    def test_pharmacy_identity_requires_one_exact_match_and_confirmed_name(
        self,
    ) -> None:
        original = copy.deepcopy(self.tables["emar"][0])
        order = dict.fromkeys(TABLE_COLUMNS["pharmacy"].split(), "")
        order.update(
            subject_id="1", hadm_id="10", pharmacy_id="1", medication="Norepinephrine"
        )
        cases = (
            ([order], "resolved", "norepinephrine"),
            ([], "no_match", None),
            ([{**order, "subject_id": "2"}], "no_match", None),
            ([{**order, "hadm_id": "20"}], "no_match", None),
            ([{**order, "pharmacy_id": "2"}], "no_match", None),
            ([order, order], "ambiguous_match", None),
            (
                [order, {**order, "medication": "Acetaminophen"}],
                "ambiguous_match",
                None,
            ),
            ([{**order, "medication": ""}], "blank_pharmacy_medication", None),
            ([{**order, "medication": "Unreviewed exact alias"}], "resolved", None),
            ([{**order, "medication": "Acetaminophen"}], "resolved", None),
        )
        for orders, status, canonical in cases:
            with self.subTest(status=status, orders=orders):
                self.tables["emar"] = [{**original, "medication": ""}]
                self.tables["pharmacy"] = orders
                context = self.context()
                event = context["medication_events"][0]
                self.assertEqual(event["medication"], "")
                self.assertEqual(event["pharmacy_lookup_status"], status)
                self.assertEqual(event["canonical_drug"], canonical)
                result, _ = evaluate(
                    self.study(), context, self.cohort["burden_v2"], {}
                )
                expected = (
                    1 if canonical else 0 if event["action"] == "exclude" else None
                )
                self.assertEqual(result[COMPONENTS[4]], expected)
                if status == "ambiguous_match":
                    self.assertEqual(event["pharmacy_match_count"], 2)
        for field in ("subject_id", "hadm_id", "pharmacy_id"):
            self.tables["emar"] = [{**original, "medication": "", field: ""}]
            self.tables["pharmacy"] = [order]
            self.assertEqual(
                self.context()["medication_events"][0]["pharmacy_lookup_status"],
                "missing_lookup_key",
            )
        self.tables["emar"] = [original]
        self.tables["pharmacy"] = [{**order, "medication": "Acetaminophen"}]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["pharmacy_lookup_status"], "not_needed")
        self.assertEqual(event["canonical_drug"], "norepinephrine")
        self.assertFalse(
            any((key.startswith("pharmacy:") for key in event["source_record_keys"]))
        )
        self.tables["emar"] = []
        self.assertEqual(self.context()["medication_events"], [])
        self.tables["emar"] = [{**original, "medication": ""}]
        self.cohort["burden_v2"]["pharmacy_identity_lookup"]["confirmed"] = False
        self.assertEqual(
            self.context()["medication_events"][0]["pharmacy_lookup_status"], "disabled"
        )

    def test_prescriptions_identity_requires_unique_evidence_and_preserves_conflicts(
        self,
    ) -> None:
        original = copy.deepcopy(self.tables["emar"][0])
        order = dict.fromkeys(TABLE_COLUMNS["prescriptions"].split(), "")
        order.update(
            subject_id="1",
            hadm_id="10",
            pharmacy_id="1",
            drug="Norepinephrine",
            drug_type="MAIN",
            route="PO",
            starttime=dt(-100),
            stoptime=dt(-99),
        )
        cases = (
            ([order], "resolved", "norepinephrine"),
            ([], "no_match", None),
            ([{**order, "subject_id": "2"}], "no_match", None),
            ([{**order, "hadm_id": "20"}], "no_match", None),
            ([{**order, "pharmacy_id": "2"}], "no_match", None),
            ([order, order], "ambiguous_match", None),
            (
                [order, {**order, "drug_type": "BASE", "drug": "Acetaminophen"}],
                "ambiguous_match",
                None,
            ),
            ([{**order, "drug": ""}], "blank_prescriptions_drug", None),
            ([{**order, "drug": "Unreviewed alias"}], "resolved", None),
            ([{**order, "drug": "Acetaminophen"}], "resolved", None),
        )
        for orders, status, canonical in cases:
            with self.subTest(status=status, orders=orders):
                self.tables["emar"] = [{**original, "medication": ""}]
                self.tables["prescriptions"] = orders
                context = self.context()
                event = context["medication_events"][0]
                self.assertEqual(event["medication"], "")
                self.assertEqual(event["pharmacy_lookup_status"], "no_match")
                self.assertEqual(event["prescriptions_lookup_status"], status)
                self.assertEqual(event["canonical_drug"], canonical)
                result, _ = evaluate(
                    self.study(), context, self.cohort["burden_v2"], {}
                )
                expected = (
                    1 if canonical else 0 if event["action"] == "exclude" else None
                )
                self.assertEqual(result[COMPONENTS[4]], expected)
                if expected is None:
                    self.assertIsNone(result["burden_score"])
                if status == "ambiguous_match":
                    self.assertEqual(event["prescriptions_match_count"], 2)
        self.tables["prescriptions"] = [order]
        for field in ("subject_id", "hadm_id", "pharmacy_id"):
            self.tables["emar"] = [{**original, "medication": "", field: ""}]
            self.assertEqual(
                self.context()["medication_events"][0]["prescriptions_lookup_status"],
                "missing_lookup_key",
            )
        self.tables["emar"] = [{**original, "medication": ""}]
        pharmacy = dict.fromkeys(TABLE_COLUMNS["pharmacy"].split(), "")
        pharmacy.update(subject_id="1", hadm_id="10", pharmacy_id="1")
        self.tables["pharmacy"] = [pharmacy]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["pharmacy_lookup_status"], "blank_pharmacy_medication")
        self.assertEqual(event["medication_name_source"], "prescriptions")
        self.assertTrue(
            any(
                (
                    key.startswith("prescriptions:")
                    for key in event["source_record_keys"]
                )
            )
        )
        self.tables["pharmacy"] = [pharmacy, pharmacy]
        event = self.context()["medication_events"][0]
        self.assertEqual(
            event["prescriptions_lookup_status"], "blocked_pharmacy_conflict"
        )
        self.assertIsNone(event["canonical_drug"])
        self.tables["pharmacy"] = [{**pharmacy, "medication": "Acetaminophen"}]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["prescriptions_lookup_status"], "not_needed")
        self.assertEqual(event["resolved_medication"], "Acetaminophen")
        self.tables["emar"] = [original]
        event = self.context()["medication_events"][0]
        self.assertEqual(event["medication_name_source"], "emar")
        self.assertEqual(event["prescriptions_lookup_status"], "not_needed")
        self.tables["emar"] = []
        self.assertEqual(self.context()["medication_events"], [])
        self.tables["emar"] = [{**original, "medication": ""}]
        self.tables["pharmacy"] = []
        policy = self.cohort["burden_v2"]["prescriptions_identity_lookup"]
        policy["confirmed"] = False
        self.assertEqual(
            self.context()["medication_events"][0]["prescriptions_lookup_status"],
            "disabled",
        )
        policy["confirmed"] = True
        policy["reviewer"] = ""
        with self.assertRaisesRegex(
            ValueError, "prescriptions identity lookup requires review provenance"
        ):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_unknown_identity_nonadministration_preserves_history_and_conflicts(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        rules[IDENTITY_POLICY]["confirmed"] = True
        for name in ("Not Given", "Not Started"):
            rules["medication_event_actions"][name] = reviewed(action="exclude")
        self.tables["emar"], self.tables["emar_detail"] = ([], [])
        self.add_medication("m1", 1, "Started")
        for index, event in enumerate(("Not Given", "Not Started", "Started"), 2):
            self.add_medication(
                f"m{index}",
                index,
                event,
                complete_dose_not_given="1" if event == "Started" else "0",
            )
            self.tables["emar"][-1].update(medication="", pharmacy_id="")
        self.add_medication("m5", 5, "Continued")
        context = self.context()
        self.assertTrue(
            all((not row["reason"] for row in context["medication_events"]))
        )
        self.assertIsNone(
            medication_identity_exclusion(self.study(), context, rules)[0]
        )
        result, _ = evaluate(self.study(), context, rules, {})
        self.assertEqual(result[COMPONENTS[4]], 1)
        self.assertEqual(
            result[f"{COMPONENTS[4]}_first_event_time"], clinical_time(dt(1))
        )
        self.assertFalse(context["medication_events"][-1]["onset"])
        self.tables["emar_detail"].append(
            {
                **self.tables["emar_detail"][1],
                "parent_field_ordinal": "1",
                "complete_dose_not_given": "1",
            }
        )
        context = self.context()
        self.assertEqual(
            context["medication_events"][1]["reason"], "conflicting_given_status"
        )
        self.assertIsNone(evaluate(self.study(), context, rules, {})[0][COMPONENTS[4]])
        self.assertIsNone(
            medication_identity_exclusion(self.study(), context, rules)[0]
        )
        self.tables["emar"][1]["charttime"] = ""
        self.assertTrue(self.context()["medication_events"][0]["reason"])

    def test_confirmed_nonadministration_review_type_preserves_support_and_conflicts(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        rules["administration_types"].update(
            {
                "Standard Maintenance Medication": reviewed(action="review"),
                "Other Review": reviewed(action="review"),
                "Unconfirmed Review": {**reviewed(action="review"), "confirmed": False},
            }
        )
        for status in ("Not Given", "Not Started"):
            rules["medication_event_actions"][status] = reviewed(action="exclude")
        for start in (-1, 1):
            for medication in ("", "Norepinephrine"):
                with self.subTest(start=start, medication=medication):
                    self.tables["emar"], self.tables["emar_detail"] = ([], [])
                    self.add_medication("m1", start, "Started")
                    for index, status in enumerate(("Not Given", "Not Started"), 2):
                        self.add_medication(
                            f"m{index}",
                            index,
                            status,
                            administration_type="Standard Maintenance Medication",
                            dose_given="0",
                            infusion_complete="1",
                            new_iv_bag_hung="1",
                        )
                        self.tables["emar"][-1].update(
                            medication=medication, pharmacy_id=""
                        )
                    self.add_medication("m4", 4, "Continued")
                    context = self.context()
                    for row in context["medication_events"][1:3]:
                        self.assertFalse(row["reason"])
                        self.assertEqual(row["action"], "exclude")
                        self.assertFalse(row["onset"])
                        self.assertEqual(row["medication"], medication)
                        self.assertTrue(row["source_record_keys"])
                    continuation = context["medication_events"][-1]
                    self.assertTrue(continuation["active_before"])
                    self.assertFalse(continuation["onset"])
                    self.assertEqual(
                        continuation["episode_start_before"], clinical_time(dt(start))
                    )
                    for horizon in (72,):
                        result, evidence = evaluate(
                            self.study(hours=horizon), context, rules, {}
                        )
                        self.assertEqual(result[COMPONENTS[4]], int(start > 0))
                        self.assertEqual(result["label_status"], "complete")
                        self.assertEqual(
                            result["burden_score"], (8) - (2 if start < 0 else 0)
                        )
                        self.assertTrue(
                            all(
                                (
                                    row["qualifies"] == 0
                                    for row in evidence
                                    if row["source_record_key"]
                                    in {
                                        value["source_record_key"]
                                        for value in context["medication_events"][1:3]
                                    }
                                )
                            )
                        )
        for status in ("Not Given", "Not Started"):
            for details, changes, extra, expected in (
                ({"dose_given": "1"}, {}, {}, "conflicting_not_given_dose"),
                ({"dose_given": "unknown"}, {}, {}, "invalid_not_given_dose"),
                ({"dose_given": "-1"}, {}, {}, "invalid_not_given_dose"),
                ({"dose_given": "NaN"}, {}, {}, "invalid_not_given_dose"),
                ({"administration_type": ""}, {}, {}, "unresolved_administration_type"),
                (
                    {"administration_type": "Unconfirmed Review"},
                    {},
                    {},
                    "unresolved_administration_type",
                ),
                (
                    {},
                    {},
                    {"administration_type": "Other Review"},
                    "unresolved_administration_type",
                ),
                (
                    {},
                    {},
                    {"administration_type": "IV Infusion"},
                    "unresolved_administration_type",
                ),
                ({}, {}, {"complete_dose_not_given": "1"}, "conflicting_given_status"),
                ({"route": "PO"}, {}, {}, "route_conflict_or_unconfirmed"),
                (
                    {"prior_infusion_rate": "invalid"},
                    {},
                    {},
                    "invalid_prior_infusion_rate",
                ),
                ({}, {"charttime": ""}, {}, "invalid_medication_time"),
                ({"emar_seq": "wrong"}, {}, {}, "emar_detail_key_mismatch"),
            ):
                with self.subTest(
                    status=status, details=details, changes=changes, extra=extra
                ):
                    self.tables["emar"], self.tables["emar_detail"] = ([], [])
                    base_detail = {
                        "administration_type": "Standard Maintenance Medication",
                        "dose_given": "0",
                        **details,
                    }
                    self.add_medication("m1", 1, status, **base_detail)
                    self.tables["emar"][-1].update(changes)
                    if extra:
                        self.tables["emar_detail"].append(
                            {
                                **self.tables["emar_detail"][0],
                                "parent_field_ordinal": "1",
                                **extra,
                            }
                        )
                    context = self.context()
                    self.assertEqual(
                        context["medication_events"][0]["reason"], expected
                    )
                    result, evidence = evaluate(self.study(), context, rules, {})
                    self.assertIsNone(result[COMPONENTS[4]])
                    self.assertIsNone(result["burden_score"])
                    self.assertTrue(any((row["qualifies"] is None for row in evidence)))
        self.tables["emar"][0].update(charttime=dt(-11))
        self.tables["emar_detail"][0].update(emar_seq="1")
        self.assertEqual(
            self.context()["medication_events"][0]["reason"], "event_outside_admission"
        )

    def test_identity_exclusion_scopes_existing_baseline_and_window_evidence(
        self,
    ) -> None:
        rules = self.cohort["burden_v2"]
        rules[IDENTITY_POLICY]["confirmed"] = True
        original_emar = copy.deepcopy(self.tables["emar"])
        original_details = copy.deepcopy(self.tables["emar_detail"])
        for time, changes, detail, expected in (
            (-1, {}, {}, True),
            (0, {}, {}, True),
            (60, {}, {}, True),
            (70, {}, {}, True),
            (71, {}, {}, False),
            (60, {"hadm_id": "20"}, {}, False),
            (3, {}, {"administration_type": "IV Bolus"}, False),
            (3, {"medication": "Acetaminophen"}, {}, False),
            (3, {"medication": "Unreviewed exact name"}, {}, True),
        ):
            with self.subTest(time=time, changes=changes, detail=detail):
                self.tables["emar"] = copy.deepcopy(original_emar)
                self.tables["emar_detail"] = copy.deepcopy(original_details)
                self.add_medication("m3", time, "Started", **detail)
                self.tables["emar"][-1].update(
                    {"medication": "", "pharmacy_id": "", **changes}
                )
                exclusion, triggers = medication_identity_exclusion(
                    self.study(), self.context(), rules
                )
                self.assertEqual(exclusion is not None, expected)
                if exclusion:
                    self.assertTrue(triggers)
                    self.assertTrue(all((row["qualifies"] is None for row in triggers)))
                    self.assertEqual(
                        exclusion["reviewer"], rules[IDENTITY_POLICY]["reviewer"]
                    )
        for state in ("excluded", "needs_review"):
            self.assertIsNone(
                medication_identity_exclusion(
                    {**self.study(), "candidate_status": state}, self.context(), rules
                )[0]
            )
        rules[IDENTITY_POLICY]["confirmed"] = False
        self.assertIsNone(
            medication_identity_exclusion(self.study(), self.context(), rules)[0]
        )
        rules[IDENTITY_POLICY]["confirmed"] = True
        rules[IDENTITY_POLICY]["reviewer"] = " "
        with self.assertRaisesRegex(ValueError, "review provenance"):
            validate_rules(self.cohort, self.ehr, self.imaging)

    def test_source_record_exclusions_keep_positives_and_future_conflicts(self) -> None:
        template = self.tables["chartevents"][0]
        self.tables["chartevents"] += [
            {**template, "charttime": dt(1.99), "value": "94", "valuenum": "94"},
            {
                **template,
                "charttime": dt(5),
                "value": "90",
                "valuenum": "90",
                "warning": "1",
            },
            {**template, "charttime": dt(7), "itemid": "223872", "value": "BiPAP/CPAP"},
        ]
        rules = self.cohort["burden_v2"]
        context = self.context()
        _, original = evaluate(self.study(), context, rules, {})
        overrides = {}
        for horizon in (72,):
            _, horizon_original = evaluate(
                self.study(hours=horizon), context, rules, {}
            )
            for component in COMPONENTS[2:4]:
                keys = [
                    r["source_record_key"]
                    for r in original
                    if r["component"] == component and r["qualifies"] is None
                ]
                overrides["p1_h10_s100", horizon, component] = dict(
                    event_flag="0",
                    first_event_time="",
                    source_record_keys=json.dumps(keys),
                    reviewer="synthetic_reviewer",
                    rationale="Exclude only reviewed source records.",
                    version="record_review_v1",
                    override_scope="source_record",
                )
            wide, records = evaluate(
                self.study(hours=horizon), context, rules, overrides
            )
            self.assertEqual(wide["burden_score"], 8)
            for component, onset in ((COMPONENTS[2], 4), (COMPONENTS[3], 3)):
                self.assertEqual(wide[component], 1)
                self.assertEqual(
                    wide[f"{component}_first_event_time"], clinical_time(dt(onset))
                )
            originals = {
                (r["component"], r["source_record_key"]): r for r in horizon_original
            }
            for row in records:
                prior = originals.get((row["component"], row["source_record_key"]))
                if prior:
                    self.assertEqual(row["raw_evidence"], prior["raw_evidence"])
                    self.assertEqual(
                        row["automated_qualifies"], prior["automated_qualifies"]
                    )
                    self.assertEqual(
                        row["automated_exclusion_reason"],
                        prior["automated_exclusion_reason"],
                    )
                    if prior["qualifies"] == 1:
                        self.assertEqual(row, {**prior, "horizon_hours": horizon})
        override = overrides["p1_h10_s100", 72, COMPONENTS[3]]
        for invalid in (
            {**override, "override_scope": "unknown"},
            {**override, "event_flag": "1"},
            {**override, "first_event_time": dt(5)},
            {**override, "baseline_status": "active_documented"},
            {**override, "source_record_keys": '["chartevents:missing"]'},
            {
                **override,
                "source_record_keys": json.dumps(
                    [
                        next(
                            (
                                r["source_record_key"]
                                for r in original
                                if r["component"] == COMPONENTS[2]
                            )
                        )
                    ]
                ),
            },
        ):
            with self.assertRaises(ValueError):
                evaluate(
                    self.study(),
                    context,
                    rules,
                    {("p1_h10_s100", 72, COMPONENTS[3]): invalid},
                )
        targeted = [
            copy.deepcopy(r) for r in original if r["component"] == COMPONENTS[3]
        ]
        for field, foreign in (("subject_id", "2"), ("hadm_id", "20")):
            values = copy.deepcopy(targeted)
            row = next((r for r in values if r["qualifies"] is None))
            raw = json.loads(row["raw_evidence"])
            raw[field] = foreign
            row["raw_evidence"] = json.dumps(raw)
            with self.assertRaisesRegex(ValueError, "another patient or admission"):
                apply_override(self.study(), COMPONENTS[3], values, override)
        episode = next(
            (
                r
                for r in original
                if r["component"] == COMPONENTS[2] and r["qualifies"] == 1
            )
        )
        with self.assertRaisesRegex(ValueError, "exact evidence row keys"):
            apply_override(
                self.study(),
                COMPONENTS[2],
                [copy.deepcopy(episode)],
                {**override, "source_record_keys": episode["source_record_keys"]},
            )
        self.tables["chartevents"].append(
            {**template, "charttime": dt(9), "warning": "1"}
        )
        unresolved, evidence_rows = evaluate(
            self.study(), self.context(), rules, overrides
        )
        self.assertIsNone(unresolved[COMPONENTS[3]])
        self.assertIsNone(unresolved["burden_score"])
        self.assertTrue(
            any(
                (
                    r["qualifies"] is None and r["event_time"] == clinical_time(dt(9))
                    for r in evidence_rows
                )
            )
        )
        self.tables["chartevents"].pop()
        self.tables["chartevents"][0].update(value="96", valuenum="96")
        no_hypoxemia, _ = evaluate(self.study(), self.context(), rules, overrides)
        self.assertEqual(no_hypoxemia[COMPONENTS[3]], 0)
        self.assertIsNone(no_hypoxemia[f"{COMPONENTS[3]}_first_event_time"])

    def test_helpers_reject_timezone_and_score_invariants(self) -> None:
        self.assertIsNone(clinical_time("2120-01-01"))
        self.assertIsNone(clinical_time("2120-01-01T10:00:00+08:00"))
        self.assertEqual(score([1] * 6)["burden_score"], 8)
        self.assertEqual(score([1, 1, 1, 1, 0, 0])["burden_score"], 3)
        self.assertIsNone(score([1, 0, None, 0, 0, 0])["burden_score"])
        with self.assertRaises(ValueError):
            score([0, 0, 0, 0, 0, 2])

    def test_score_all_binary_combinations_and_pending_components(self) -> None:
        for flags in product((0, 1), repeat=6):
            result = score(list(flags))
            respiratory = max(flags[2:4])
            expected = max(flags[:2]) + 2 * respiratory + 2 * flags[4] + 3 * flags[5]
            self.assertEqual(result["burden_score"], expected)
            self.assertEqual(
                result["phys_death"], int(bool(respiratory or flags[4] or flags[5]))
            )
            self.assertEqual(result["high_burden"], int(expected >= 3))
        for index in range(6):
            pending: list[int | None] = [0] * 6
            pending[index] = None
            self.assertIsNone(score(pending)["burden_score"])

    def test_closed_parquet_sink_rejects_nonempty_writes(self) -> None:
        path = self.root / "closed.parquet"
        sink = ParquetSink(path, ["sample_id"])
        sink.close()
        with self.assertRaisesRegex(ValueError, "closed Parquet sink"):
            sink.write([{"sample_id": "synthetic"}])
        self.assertEqual(pq.read_table(path).num_rows, 0)
