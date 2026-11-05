"""Validate clinical ownership, windows, evidence and scores before freezing."""

import pyarrow as pa
import pyarrow.parquet as pq

from src.cohort.audit_burden_labels import WIDE_COLUMNS, query_rows
from src.cohort.build_candidates import CANDIDATE_COLUMNS, OBSERVATION_COLUMNS
from src.cohort.burden_events import EVIDENCE_COLUMNS
from src.cohort.burden_io import COMPONENTS

SCORES = "T R C D burden_score phys_death high_burden".split()


def _reject(connection, condition: str, message: str) -> None:
    if connection.execute(condition).fetchone()[0]:
        raise ValueError(message)


def validate_labels(connection, artifacts: dict, horizon: int, *, review_policy="lenient") -> dict:
    """Reconstruct binary flags and scores from final evidence, independently of raw ingestion."""
    strict = review_policy == "strict"
    wide_columns, evidence_columns = WIDE_COLUMNS, EVIDENCE_COLUMNS
    if strict:
        from src.cohort import strict_events
        wide_columns = list(dict.fromkeys(OBSERVATION_COLUMNS + strict_events.EVENT_COLUMNS))
        evidence_columns = strict_events.EVIDENCE_COLUMNS
    for name, required in (
        ("candidates", CANDIDATE_COLUMNS),
        ("observations", OBSERVATION_COLUMNS),
        ("wide", wide_columns),
        ("evidence", evidence_columns),
    ):
        schema = pq.read_schema(artifacts[name])
        if len(schema.names) != len(set(schema.names)) or set(required) - set(
            schema.names
        ):
            raise ValueError(f"Missing {name} fields")
        if schema.field("sample_id").type != pa.string():
            raise ValueError("Sample keys must be strings")
        for field in schema:
            if pa.types.is_timestamp(field.type) and field.type != pa.timestamp("us"):
                raise ValueError(
                    "Clinical timestamps must be timezone-free microseconds"
                )
        for field in ("subject_id", "hadm_id", "study_id"):
            if schema.field(field).type != pa.int64():
                raise ValueError("Identifiers must be int64")
        connection.read_parquet(str(artifacts[name])).create_view(name)
        if name != "evidence":
            _reject(
                connection,
                f"SELECT count(*) FROM (SELECT sample_id FROM {name} GROUP BY sample_id HAVING count(*)<>1 OR sample_id IS NULL)",
                "Duplicate or null sample key",
            )
    for name, columns in (
        ("observations", CANDIDATE_COLUMNS),
        ("wide", OBSERVATION_COLUMNS),
    ):
        reference = "candidates" if name == "observations" else "observations"
        mismatch = " OR ".join((f"a.{c} IS DISTINCT FROM b.{c}" for c in columns))
        _reject(
            connection,
            f"SELECT count(*) FROM {reference} a FULL JOIN {name} b USING(sample_id) WHERE a.sample_id IS NULL OR b.sample_id IS NULL OR {mismatch}",
            "Candidate, observation and audit rows disagree",
        )
    _reject(
        connection,
        "SELECT count(*) FROM wide WHERE subject_id IS NULL OR hadm_id IS NULL OR study_id IS NULL OR sample_id<>concat('p',subject_id,'_h',hadm_id,'_s',study_id) OR candidate_status IS NULL OR candidate_status NOT IN ('eligible','excluded')",
        "Invalid sample identity or unresolved candidate",
    )
    _reject(
        connection,
        "SELECT count(*) FROM (SELECT subject_id,hadm_id,study_id FROM wide GROUP BY ALL HAVING count(*)<>1)",
        "Duplicate study identifiers",
    )
    _reject(
        connection,
        f"SELECT count(*) FROM wide WHERE horizon_hours IS DISTINCT FROM {horizon}",
        "Mixed observation horizons",
    )
    null_labels = " OR ".join(
        (
            f"{c} IS NOT NULL"
            for c in (
                *COMPONENTS,
                *(f"{c}_first_event_time" for c in COMPONENTS),
                *SCORES,
            )
        )
    )
    _reject(
        connection,
        f"SELECT count(*) FROM wide WHERE candidate_status='excluded' AND (label_status IS DISTINCT FROM 'excluded' OR {null_labels})",
        "Excluded studies must retain null labels",
    )
    _reject(
        connection,
        "SELECT count(*) FROM wide WHERE candidate_status='eligible' AND label_status IS DISTINCT FROM 'complete'",
        "Eligible studies require completed labels",
    )
    connection.execute(
        "CREATE VIEW eligible AS SELECT * FROM wide WHERE candidate_status='eligible'"
    )
    _reject(
        connection,
        f"SELECT count(*) FROM eligible WHERE dicom_id IS NULL OR dicom_id='' OR dicom_path IS NULL OR dicom_path='' OR t0 IS NULL OR admittime IS NULL OR dischtime IS NULL OR t0<admittime OR t0>=dischtime OR horizon_end IS DISTINCT FROM t0+INTERVAL '{horizon} hours' OR observation_end IS DISTINCT FROM least(horizon_end,dischtime) OR observed_hours IS NULL OR NOT isfinite(observed_hours) OR abs(observed_hours-epoch(observation_end-t0)/3600)>0.0001 OR early_discharge IS DISTINCT FROM cast(dischtime<horizon_end AND hospital_expire_flag='0' AND NOT contains(exclusion_reason,'death_time_conflict') AS UTINYINT)",
        "Invalid observation boundaries or discharge status",
    )
    _reject(
        connection,
        "SELECT count(*) FROM evidence e LEFT JOIN wide w USING(sample_id) WHERE w.sample_id IS NULL OR e.subject_id IS DISTINCT FROM w.subject_id OR e.hadm_id IS DISTINCT FROM w.hadm_id OR e.study_id IS DISTINCT FROM w.study_id OR e.horizon_hours IS DISTINCT FROM w.horizon_hours OR e.source_record_key IS NULL OR e.source_record_key=''",
        "Invalid evidence ownership or source key",
    )
    allowed = ",".join((f"'{c}'" for c in COMPONENTS))
    _reject(
        connection,
        f"SELECT count(*) FROM evidence WHERE component IS NULL OR component NOT IN ({allowed}) OR (qualifies IS NOT NULL AND qualifies NOT IN (0,1))",
        "Invalid evidence component or flag",
    )
    if strict:
        _reject(connection,
                "SELECT count(*) FROM evidence WHERE blocks_event IS NULL OR blocks_event NOT IN (0,1) OR (qualifies IS NOT NULL AND blocks_event<>0) OR (qualifies IS NULL AND blocks_event=0 AND (uncertainty_relevance IS NULL OR uncertainty_relevance NOT IN ('independent_confirmed_low_after_recovery','independent_after_confirmed_positive','history_without_window_infusion')))",
                "Invalid strict uncertainty classification")
    unresolved = "(qualifies IS NULL AND blocks_event=1)" if strict else "qualifies IS NULL"
    _reject(
        connection,
        f"SELECT count(*) FROM evidence e JOIN eligible w USING(sample_id) WHERE {unresolved} OR (qualifies=1 AND (event_time IS NULL OR first_event_time IS NULL OR event_time<=t0 OR event_time>observation_end OR first_event_time<=t0 OR first_event_time>observation_end))",
        "Unresolved or out-of-window evidence",
    )
    for c in COMPONENTS:
        _reject(
            connection,
            f"WITH d AS (SELECT sample_id, coalesce(max(qualifies),0) flag, min(first_event_time) FILTER(WHERE qualifies=1) first_time FROM evidence WHERE component='{c}' GROUP BY sample_id) SELECT count(*) FROM eligible w LEFT JOIN d USING(sample_id) WHERE w.{c} IS DISTINCT FROM coalesce(d.flag,0) OR w.{c}_first_event_time IS DISTINCT FROM d.first_time OR w.{c}_event_status IS NULL OR w.{c}_event_status NOT IN ('positive','preexisting_only','no_qualifying_record') OR (w.{c}=1 AND w.{c}_event_status<>'positive') OR (w.{c}=0 AND w.{c}_event_status='positive')",
            f"Evidence does not reproduce {c}",
        )
    if strict:
        for component in COMPONENTS:
            _reject(connection,
                    f"WITH times AS (SELECT sample_id,min(first_event_time) FILTER(WHERE qualifies=1) first_confirmed,min(event_time) FILTER(WHERE qualifies IS NULL) first_uncertain FROM evidence WHERE component='{component}' GROUP BY sample_id) SELECT count(*) FROM eligible w LEFT JOIN times USING(sample_id) WHERE w.{component}_first_event_time_status IS DISTINCT FROM CASE WHEN first_confirmed IS NULL THEN 'no_confirmed_event' WHEN first_uncertain<first_confirmed THEN 'earliest_confirmed_possible_earlier' ELSE 'earliest_confirmed' END",
                    f"Strict onset-time status disagrees with {component} evidence")
    a, b, c, d, e, f = COMPONENTS
    connection.execute(
        f"CREATE VIEW recalculated AS SELECT * EXCLUDE(T,R,C,D,burden_score,phys_death,high_burden), cast(greatest({a},{b}) AS UTINYINT) T, cast(greatest({c},{d}) AS UTINYINT) R, cast({e} AS UTINYINT) C, cast({f} AS UTINYINT) D FROM eligible"
    )
    connection.execute(
        "CREATE VIEW scored AS SELECT *, cast(T+2*R+2*C+3*D AS FLOAT) burden_score, cast(R=1 OR C=1 OR D=1 AS UTINYINT) phys_death, cast(T+2*R+2*C+3*D>=3 AS UTINYINT) high_burden FROM recalculated"
    )
    mismatch = " OR ".join((f"w.{c} IS DISTINCT FROM s.{c}" for c in SCORES))
    _reject(
        connection,
        f"SELECT count(*) FROM eligible w JOIN scored s USING(sample_id) WHERE {mismatch}",
        "Audit scores disagree with recalculated labels",
    )
    for field in (
        *COMPONENTS,
        "T",
        "R",
        "C",
        "D",
        "phys_death",
        "high_burden",
        "early_discharge",
    ):
        if pq.read_schema(artifacts["wide"]).field(field).type != pa.uint8():
            raise ValueError("Binary flags must be uint8")
    for field in ("burden_score", "observed_hours"):
        if pq.read_schema(artifacts["wide"]).field(field).type != pa.float32():
            raise ValueError("Scores and observation hours must be float32")
    counts = query_rows(
        connection,
        "SELECT count(*) eligible_studies, count(DISTINCT hadm_id) eligible_admissions, count(DISTINCT subject_id) eligible_patients, count(*) FILTER(WHERE burden_score>0) nonzero_studies, count(*) FILTER(WHERE phys_death=1) phys_death_studies, count(*) FILTER(WHERE high_burden=1) high_burden_studies, 0 unresolved_studies FROM scored",
    )[0]
    if not counts["eligible_studies"]:
        raise ValueError("No formal labels to publish")
    return counts
