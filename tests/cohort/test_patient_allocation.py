"""Historical allocation and source-leakage tests on generated cohorts."""
import copy
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa

from src.cohort.burden_io import COMPONENTS
from src.cohort.create_patient_split import (
    SCHEMA,
    aggregate_patients,
    allocate_patients,
    apportion,
    check_assignments,
    check_source_leakage,
    split_reports,
)
from src.common.paths import read_json

PROJECT=Path(__file__).resolve().parents[2]
VERSION="v1_r1"


def approved_settings():
    settings=read_json(PROJECT/"configs/data.template/lenient/cohort/patient_split.json")
    return {**settings,"confirmed":True,"reviewer":"synthetic","reviewed_on":"2026-11-05","rationale":"synthetic approval", "cohort_version":VERSION,"split_version":VERSION}


def synthetic_cohort(n=222, high=8, phys=10):
    schema = pa.schema([('sample_id', pa.string()), ('subject_id', pa.int64()), ('hadm_id', pa.int64()), ('study_id', pa.int64()), *[(c, pa.uint8()) for c in COMPONENTS], ('burden_score', pa.float32()), ('phys_death', pa.uint8()), ('high_burden', pa.uint8()), ('early_discharge', pa.uint8()), ('cohort_version', pa.string())])
    rows = []
    for i in range(n):
        subject, hadm, study = (1000 + i, 2000 + i, 3000 + i)
        flags = [0, 0, int(high <= i < phys), 0, 0, int(i < high)]
        rows.append(dict(sample_id=f'p{subject}_h{hadm}_s{study}', subject_id=subject, hadm_id=hadm, study_id=study, **dict(zip(COMPONENTS, flags)), burden_score=3.0 if i < high else 2.0 if i < phys else 0.0, phys_death=int(i < phys), high_burden=int(i < high), early_discharge=0, cohort_version=VERSION))
    return pa.Table.from_pylist(rows, schema=schema)

class PatientAllocationTests(unittest.TestCase):

    def setUp(self):
        self.settings = approved_settings()

    def test_expected_quotas_order_invariance_and_multiple_admissions(self):
        cohort = synthetic_cohort()
        extra = cohort.to_pylist()[0].copy()
        extra.update(hadm_id=9999, study_id=9999, sample_id='p1000_h9999_s9999')
        cohort = pa.concat_tables([cohort, pa.Table.from_pylist([extra], schema=cohort.schema)])
        assigned, quotas = allocate_patients(cohort, self.settings)
        self.assertEqual(quotas, {'total': [156, 33, 33], 'ever_phys_death': [7, 2, 1], 'ever_high_burden': [6, 1, 1]})
        self.assertTrue(check_assignments(cohort, assigned, self.settings, quotas)['each_study_has_one_split'])
        self.assertEqual(assigned.schema, SCHEMA)
        self.assertEqual(assigned.to_pylist()[0]['study_count'], 2)
        reverse = cohort.take(pa.array(list(reversed(range(cohort.num_rows)))))
        repeated, repeated_quotas = allocate_patients(reverse, self.settings)
        self.assertTrue(assigned.equals(repeated))
        self.assertEqual(quotas, repeated_quotas)

    def test_rounding_ties_bounds_and_impossible_constraints(self):
        self.assertEqual(apportion(10, [70, 15, 15], [1, 1, 1], [156, 33, 33]), [7, 2, 1])
        self.assertEqual(apportion(3, [70, 15, 15], [1, 1, 1], [3, 3, 3]), [1, 1, 1])
        self.assertEqual(apportion(8, [70, 15, 15], [1, 1, 1], [7, 2, 1]), [6, 1, 1])
        for n in range(3, 30):
            counts = apportion(n, [70, 15, 15], [1, 1, 1], [1, n, n])
            self.assertEqual(sum(counts), n)
            self.assertEqual(counts[0], 1)
            self.assertTrue(all((x >= 1 for x in counts)))
        for n, lo, cap in ((2, [1, 1, 1], [2] * 3), (10, [1] * 3, [2] * 3), (3, [1] * 3, [0, 3, 3])):
            with self.assertRaises(ValueError):
                apportion(n, [70, 15, 15], lo, cap)
        with self.assertRaisesRegex(ValueError, 'three high-burden'):
            allocate_patients(synthetic_cohort(n=8, high=2, phys=3), self.settings)

    def test_invalid_cohort_and_assignments_fail(self):
        cohort = synthetic_cohort(n=12, high=3, phys=5)
        assigned, quotas = allocate_patients(cohort, self.settings)
        for mutation in (lambda r: r.append(copy.deepcopy(r[0])), lambda r: r[0].update(sample_id='bad'), lambda r: r[0].update(burden_score=8), lambda r: r[0].update(phys_death=0), lambda r: r[0].update(high_burden=None), lambda r: r[0].update(cohort_version='other'), lambda r: r[0].update(in_hospital_death=2)):
            rows = cohort.to_pylist()
            mutation(rows)
            with self.assertRaises(ValueError):
                aggregate_patients(pa.Table.from_pylist(rows, schema=cohort.schema), VERSION)
        with self.assertRaises(ValueError):
            aggregate_patients(cohort.drop(['phys_death']), VERSION)
        for mutation in (lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])), lambda r: r[0].update(split='invalid'), lambda r: r[0].update(seed=1), lambda r: r[0].update(study_count=99), lambda r: r[0].update(split=None)):
            rows = assigned.to_pylist()
            mutation(rows)
            with self.assertRaises(ValueError):
                check_assignments(cohort, pa.Table.from_pylist(rows, schema=SCHEMA), self.settings, quotas)

    def test_missing_classes_reported_and_shared_source_leakage(self):
        cohort = synthetic_cohort(n=3, high=3, phys=3)
        patients, quotas = allocate_patients(cohort, self.settings)
        summary, checks = split_reports(cohort, patients, check_assignments(cohort, patients, self.settings, quotas))
        self.assertEqual(sum((r['category'] == 'study_score' for r in summary)), 27)
        self.assertTrue(all((not group['high_burden']['discrimination_evaluable'] for group in checks['endpoint_evaluability'].values())))
        repeated = cohort.to_pylist()[0].copy()
        repeated.update(study_id=8000, sample_id='p1000_h2000_s8000')
        cohort = pa.concat_tables([cohort, pa.Table.from_pylist([repeated], schema=cohort.schema)])
        patients, _ = allocate_patients(cohort, self.settings)
        evidence = [{'sample_id': r['sample_id'], 'subject_id': r['subject_id'], 'component': 'in_hospital_death', 'source_record_key': 'same_patient' if r['subject_id'] == 1000 else str(r['subject_id'])} for r in cohort.to_pylist()]
        with duckdb.connect() as connection:
            connection.register('common', cohort)
            connection.register('evidence', pa.Table.from_pylist(evidence))
            check_source_leakage(connection, patients)
            evidence[1]['source_record_key'] = 'same_patient'
            connection.register('evidence', pa.Table.from_pylist(evidence))
            with self.assertRaisesRegex(ValueError, 'crosses patient splits'):
                check_source_leakage(connection, patients)
