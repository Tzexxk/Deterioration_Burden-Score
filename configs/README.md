# Configuration reference

Use this page to initialize a local configuration and choose the inputs and settings for each stage. See the [command reference](../scripts/README.md) for execution order and output locations.

## Initialize a data run

For a new checkout, run the initializer from the repository root.

```powershell
python -B scripts/utils/initialize_configs.py --template strict --version 1 --run 1
```

| Template | Data and split policy |
| --- | --- |
| `lenient` | Build 72-hour labels from the raw data, use a patient-level 70/15/15 holdout, and prepare 224 × 224 images. |
| `strict` | Apply the expanded evidence review policy, reserve approximately 15% of studies for the patient-disjoint fixed test set, and split the remaining patients into five development folds. Prepare 224 × 224 images. |
| `multiple_size` | Derive a 1024 × 1024 master from an earlier frozen 224 × 224 five-fold run of the same data version. Once the master is accepted, the image variants utility can publish smaller sizes. |

For example, if `v1_r1` is a frozen five-fold release, initialize its master as follows.

```powershell
python -B scripts/utils/initialize_configs.py --template multiple_size --version 1 --run 2 --source-run 1
```

A version identifies the raw data scope. A run identifies a processing release within that version. Run numbers start at 1 and are consecutive. A new version follows the highest existing version and starts at run 1. An existing version can receive another run even if a newer version exists. Skipping a number raises `NONCONSECUTIVE_DATA_RUN`.

The initializer copies the selected data template to `configs/data/v<version>_r<run>/`, fills missing shared configuration, and updates `configs/version_run.json`. It rejects an existing target run with `CONFIG_ALREADY_EXISTS`. Master initialization also writes `configs/utils/v<version>_r<run>/reuse_master_inputs.json`.

Edit `configs/environment/paths.local.json` to point to your data, model weights, and storage roots. The `local_inputs/...` values in `environment.template/paths.local.json` are placeholders. Store source data and weights outside the checkout. Put machine-specific absolute paths in the local root map. Other configurations and published references use project-relative paths or a named root and a path relative to that root.

Data commands use the run selected in `configs/version_run.json`. Model configurations choose a frozen release independently through `data_release`. The public model templates generally select `v1_r1` and require a five-fold release. Set the release and input IDs for your run before executing a model command. For an existing local config tree, copy individual templates for newly needed stages.

## Data configuration

Paths in the following table are relative to the selected directory under `configs/data.template/`. The initializer copies them to the corresponding data run.

| Configuration | What it controls |
| --- | --- |
| `imaging/raw_audit.json` | Chest radiograph indexes, DICOM layout and decoding, worker and batch settings, frontal view types, and acquisition time precedence for Steps 01 to 03. |
| `ehr/raw_audit.json` | Required EHR sources, fields, and laboratory timestamp sampling for Step 01. |
| `cohort/admission_matching.json` | Study sample unit and admissions input for Step 04. The default is `hosp/admissions.csv` under `mimic_iv`. |
| `cohort/burden_rules.json` | Step05 review policy, 72-hour window, clinical evidence dictionaries, processing rules, and review paths. |
| `ehr/burden_sources.json` | Clinical source tables, DuckDB memory and spill limits, and the workspace preflight estimate. |
| `imaging/processing.json` | Step07 image size, percentile clipping, interpolation, padding, preview size, review path, and stratified sampling. |
| `cohort/patient_split.json` | Step09 split method, proportions, seed, and confirmation. |
| `ehr/lab_processing.json` | Step10 sources, reviewer, batch and evidence limits, review paths, and DuckDB settings. |
| `ehr/features.json` | Step11 24-hour lookback, static sources, laboratory aggregations, and resource limits. |
| `export/training_package.json` | Step12 package interface version and workspace preflight estimate. |
| `export/validation.json` | Step13 expected counts, interface version, relocation policy, diagnostic limit, and workspace preflight estimate. |

### Clinical evidence and reviews

For Step05, `--review-policy` must match the configured policy. When omitted, the command reads the policy from the run. A template provides unconfirmed dictionaries. Review the audit inventory and conflicts before recording mappings, supporting evidence, and rationale. Freeze requires confirmed rules and a current audit with all blocking issues resolved.

- With `lenient`, `event_evidence.csv` records decisions about events or individual records. `candidate_exclusions.csv` records study exclusions. Both templates contain headers only. Set an unused review path to `null`.
- With `strict`, start with the review paths unset. After reviewing the current audit, set `cohort_decisions_path` and `event_decisions_path` as needed. Keep the lenient overrides and exclusions paths unset. The audit creates pending review templates under `review_templates/`. Preserve their run, policy, source generation, sample identity, and record keys when completing them.
- Strict cohort decisions support `pending`, `resolve_blocker`, and `exclude`. A decision resolving a care-level conflict at imaging must include `care_level_at_t0`. Strict event decisions support `pending`, `set_event`, and `exclude_source`. A positive component decision needs a time inside the 72-hour window. A negative component decision leaves the time empty. Source exclusion targets a specific record.
- Enter the reviewer, date, and rationale. For a historical decision, record its provenance in `historical_source`. `historical_keys` is for verified keys that the original audit did not emit. The audit reconstructs those keys and checks ownership against the current raw evidence.
- Strict `review_evidence_sources` can include `inputevents_full`, `ingredientevents_full`, `poe_full`, `poe_detail_full`, `pharmacy_full`, `prescriptions_full`, `emar_full`, `emar_detail_full`, and `discharge_notes_full`. Configure only available sources with the required fields. A detail table needs its parent. Supplemental records support review, including leads that require corroboration.

Use record keys from the audit. Once the audit is ready, fill `cohort/reviews/endpoint_review.json` with the current version, run, audit generation, review policy, rule identifiers, reviewer, date, rationale, and actual record review references. Freeze checks these bindings and verifies the evidence and scoring. A new audit generation requires renewed confirmation.

The clinical source templates set `memory_limit` to `2GB`, `max_temp_directory_size` to `4GB`, and `workspace_estimate_bytes` to 8 GiB. The estimate is a disk preflight value. The first two settings limit DuckDB memory and temporary spill separately.

### Image processing and QC

The 224 × 224 templates use valid-pixel percentile clipping from the 1st to the 99th percentile, bilinear image interpolation, nearest-neighbor mask interpolation, and half-up size rounding. The `multiple_size` template selects the 1024 × 1024 master and records the source run. `processing_version` identifies the algorithm, while the data run and manifest generation identify the publication.

`reviews_path` points to a review CSV within the current run, normally `imaging/reviews/image_qc.csv`. A null path means there are no prior decisions to apply. The public CSV contains headers only. Use the generated preview gallery and review table to record QC decisions in the local CSV, then rerun Step07. The generated review report and completed local input are separate files.

### Patient splits and laboratory reviews

Confirm `cohort/patient_split.json` before Step09. The configured method must match `--split-method`. The holdout targets patient counts. The five-fold policy targets study counts for the fixed test set and keeps all studies from each patient in the same role.

Step10 uses `lab_audit_review_v1`. The templates use batches of up to 10 item IDs and configure evidence limits, reviewer, source paths, and resource settings. Review items and units using the generated evidence. Preserve the program-generated `review_key` and `lab_record_key`. The lenient setup uses `ehr/reviews/lab_dictionary.csv` and `ehr/reviews/lab_scope.csv`. The strict setup can place both decisions and `scope_ack` rows in `ehr/reviews/lab_dictionary.csv`.

A decision row must set `keep` to 0 or 1. For a kept item, provide the feature name, canonical unit, conversion multiplier and offset, range basis, and rules for specimen, duplicates, numeric results, and missing units where relevant. For exclusions, give the exclusion reason and leave keep-specific fields empty. Record the reviewer, date, and rationale. JSON rules use the supported schema. A `scope_ack` row uses `scope` as its key and records acknowledgment of the out-of-scope inventory. Review must cover all required units, branches, and promoted items. Review and freeze bind the applicable audit generation.

### EHR features and package validation

`ehr_features_v1` uses 25 reviewed laboratory tests and six aggregations, plus age and time since admission, for 152 numeric and five categorical fields. The laboratory window includes both endpoints of the 24 hours before imaging, and the result must be available at imaging time. Static values use patient, admission, and service information known then. Step11 uses the Step10 laboratory cache. Numeric values use `float32` and `NaN` for missing data. Missing categorical values use `__MISSING__`. Imputation, scaling, and category fitting belong to model training. Rerun Step11 audit before build when the feature configuration changes.

Step12 uses interface version `1.0` and preserves the published image size. Set the workspace estimate to cover the directory, ZIP, and transaction staging. Dataset, cohort, split, dictionary, and feature-set identities follow the data run and its upstream publications.

Before Step13, enter `expected_counts` for studies, admissions, and patients. Enter `expected_split_studies` and `expected_split_patients` for train, validation, and test. For a five-fold release, these configured split counts describe fold 1. The validator checks the remaining folds against the published indices. Use the reviewed cohort, split report, and export metadata as evidence for these counts. The per-split totals must match the overall study and patient totals. Patient counts cannot exceed admission counts, and admission counts cannot exceed study counts.

Validation uses interface version `1.0`, `temporary_directory` relocation, and a `failure_detail_limit` from 1 to 100. The diagnostic limit caps displayed failures. Successful independent validation freezes the full run.

### Copy plans and derived image sizes

The initializer copies `utils.template/` to `configs/utils/` when needed. Use `utils.template/reuse_master_inputs.json` as a reference for manually prepared copy plans. Master initialization creates a run-specific plan targeting `step06/reuse`.

- `source_run` identifies an earlier frozen run within the same version.
- `target.stage` and `target.phase` identify the destination publication.
- Each item uses `source.stage`, `source.phase`, and `source.name` to locate a file registered in a source manifest.
- `target.name` is a destination-relative name. `target.root` selects `processed` or `outputs`.

The utility copies bytes and records the source publication. An old acceptance report remains evidence about its source run. The new run goes through its own validation.

Copy `utils.template/image_variants.json` to `configs/utils/image_variants.json` if needed. Set `batch` to a name of at most 32 lowercase letters, digits, underscores, or hyphens. Fill `sizes` with distinct integer side lengths from 2 to 1023. `workers` controls the number of sizes processed concurrently, from 1 to 8, with a template value of 2.

## Model configuration

Templates under `models.template/{train,assessment,test}/` are copied to `configs/models/`. Training and development assessment use `defaults` and per-experiment `parameters`. An assessment of a trained model uses `train_id` to bind its training result. F1 binds the two source assessments instead. The fields `sensitivity_target` and `specificity_target` set the operating points, both 0.80 in the templates. For iterative EHR models, `plot_training_history` controls the training curve output.

For image training, `image_source` selects either the package images or a registered variant batch and size from the same frozen release. `weights` selects a file relative to the configured `model_weights` root. Image assessment inherits the source and size from training.

For unimodal fixed testing, `data_release` selects the package, `id` identifies the test experiment, and `assessment_id` binds the completed development assessment. Each candidate uses the same local configuration for `refit` and `evaluate`.

| Fusion template under `models.template/` | Key settings |
| --- | --- |
| `assessment/fusion/weighted_late.json` | Bind both source assessments. Set the image weight, operating points, bootstrap settings, and decision margins. |
| `train/fusion/feature_fusion.json` | Bind the EHR and image source assessments. `pca_components` defaults to 8. The training fold fits the scaler and PCA without whitening. CatBoost inherits the source training protocol. |
| `assessment/fusion/feature_fusion.json` | Bind the feature-fusion training ID and set assessment options. |
| `train/fusion/learned_late.json` | Bind both source assessments. `round_policy: training_median` selects the median of best lengths within each training scope, rounding a half round up. For fixed lengths, supply `ehr_rounds` and `image_epochs` in place of `round_policy`. |
| `assessment/fusion/learned_late.json` | Bind the nested training ID. The assessment fits weights and intercepts separately for each outer fold. `fixed_image_weights` controls the matched fixed-weight comparisons. |
| `test/fusion/weighted_late.json` and `test/fusion/learned_late.json` | Set the release, test ID, source assessment ID, and both unimodal refit IDs. The fixed-weight config must match the source assessment's `image_weight`. |

Replace placeholder source IDs such as `selected_image` with completed assessment or refit IDs as appropriate. Configure additional experiment IDs before using them in commands. Use a new experiment ID when changing a completed candidate's settings or source. Fusion comparison takes its explicit source list from command-line arguments.

## Portable references and evidence

Configuration and provenance fields use project-relative paths or named-root references. Raw evidence excerpts such as `raw_evidence` and CSV audit fields such as `header_text` and `last_line_text` preserve the source text. Fields such as `source_file` and `file` still follow the path convention. Keep completed reviews, source excerpts, and training packages in local storage.
