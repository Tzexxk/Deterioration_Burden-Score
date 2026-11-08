# Command reference

Run these commands from the repository root in a prepared Python environment. The research environment used Python 3.11.16. [requirements.txt](../requirements.txt) records the direct dependencies used in that environment. Prepare the matching PyTorch and Torchvision builds for your platform before using the model workflows. [Configuration setup](../configs/README.md) explains the input and review files.

Data commands take `--version` and `--run` matching `configs/version_run.json`. These examples use version 1, run 1 unless they explicitly show a different run. Initialize and configure the matching run before executing its commands. Model commands select a frozen release through their own configuration.

## Utilities

### Initialize a run

```powershell
python -B scripts/utils/initialize_configs.py --template strict --version 1 --run 1
```

Choose `lenient` for a single holdout or `strict` for five development folds and a fixed test set. For an image master, use `multiple_size` and supply `--source-run`. The initializer fills missing shared configuration and refuses to overwrite the target run.

### Copy published data artifacts

```powershell
python -B scripts/utils/reuse_data_artifacts.py --version 1 --run 3 --config configs/utils/v1_r3/reuse_master_inputs.json
```

The plan must be a JSON file under `configs/utils/`. The utility copies registered artifacts from the accepted dependency chain of an earlier frozen run of the same data version. It records the source release, manifest, generation, and actual plan in `copy_plan.json`. Choose a separate destination phase for copied artifacts and use the destination manifest to resolve them. Embedded source references remain historical. The destination run needs its own acceptance.

The utility checks for duplicate names, parent and child path conflicts, and reserved names. A successful replacement before the destination run is frozen increments its generation. After a failed attempt, resolve the issue and retry before consuming the result.

### Build and validate derived image sizes

```powershell
python -B scripts/utils/build_image_variants.py --version 1 --run 3
python -B scripts/utils/build_image_variants.py --version 1 --run 3 --mode validate
```

Set the batch and sizes in `configs/utils/image_variants.json`. Use `--config` to select another JSON file under `configs/utils/`. The selected run must contain a frozen 1024 × 1024 master. The builder derives each size directly from the master with Pillow float32 bilinear interpolation, then checks coverage, shape, dtype, value range, and pixel agreement before publishing the batch.

Outputs go to `data/processed/v<version>_r<run>/image_variants/batch-<batch>/`. The manifest binds the master export and acceptance. Use the master's samples, labels, EHR features, and splits, and resolve `image_relpath` under the selected `<size>x<size>/` image root. Use a new batch name when adding sizes.

## Data preparation

Stage payloads live in `data/processed/v<version>_r<run>/steps/<stage>/<phase>/`. Manifests, summaries, audits, and diagnostics live in the equivalent tree under `outputs/data/`. Read artifact locations from the manifest's `{root, path}` references. Publication and recovery cover both roots, and downstream stages check the successful state and recorded upstream generations. Keep source excerpts, reviews, and patient-level files local.

### Step01 Raw data audit

```powershell
python -B scripts/data/01_audit_data.py --version 1 --run 1
```

The audit streams the chest radiograph indexes, checks DICOM metadata and decoding, summarizes views and time sources, and checks required EHR table fields. The default laboratory time screen uses the first 10,000 records. The audit has no upstream stage dependency. Optional `--imaging` and `--ehr` overrides must be in the current run. Defaults are `imaging/raw_audit.json` and `ehr/raw_audit.json`. Optional radiology notes are checked when the source is configured.

The `step01/audit` output includes `audit_summary.json`, `audit_report.md`, `dicom_readability.csv`, `raw_file_summary.csv`, `cxr_view_distribution.csv`, `dicom_time_summary.csv`, `ehr_time_field_screen.csv`, and `csv_tail_checks.csv`. A `missing_path` result can reflect a local DICOM subset. `unverified` indicates that the reader did not reach a verified logical end of file. Input failures prevent publication and can produce `diagnostics/step01_last_failure.json`.

### Step02 Chest radiograph index

```powershell
python -B scripts/data/02_build_cxr_index.py --version 1 --run 1
```

The builder consumes Step01 and the raw chest radiograph indexes and DICOM files. It accepts a current-run `--imaging` override and supports both configured `dataset/` and standard `files/` layouts. `cxr_index.parquet` contains one row per matched local file, including unreadable files and their status. `cxr_index_summary.json` reports coverage and differences from the audit. Investigate count differences before proceeding.

### Step03 Frontal image selection

```powershell
python -B scripts/data/03_select_frontal_image.py --version 1 --run 1
```

From Step02, select one readable AP or PA image with a reliable acquisition time per study. Break ties by earliest time, largest pixel area, then ascending DICOM ID. Time-zone inconsistencies within a study fail validation. The main outputs are `study_image_selection.parquet` and `study_image_selection_summary.json`. `--imaging` can override the current-run imaging configuration.

### Step04 Admission matching

```powershell
python -B scripts/data/04_match_admissions.py --version 1 --run 1
```

Match selected images against the patient's admission interval `[admittime, dischtime)`. A single definite candidate with no competing uncertain candidate yields `unique_match`. The program records other outcomes as `no_match` or `multiple_match`. Missing or incompatible bounds are handled under the uncertainty rules. An optional `--cohort` override selects the matching configuration.

Outputs include `cxr_admission_matches.parquet`, `cxr_admission_match_failures.parquet`, and `admission_match_summary.csv`. A confirmed truncated final CSV record is skipped and reported. Corruption elsewhere in the CSV fails validation.

### Step05 Audit and freeze 72-hour outcomes

Run one of the following audit commands for the configured policy.

```powershell
python -B scripts/data/05_audit_burden_labels.py --version 1 --run 1 --mode audit --review-policy strict
# For a run initialized with the lenient template:
python -B scripts/data/05_audit_burden_labels.py --version 1 --run 1 --mode audit --review-policy lenient
```

Step05 consumes the selected images and admission matches from Steps 03 and 04. It evaluates candidate eligibility, then builds evidence for six clinical events in `(t0, min(t0 + 72h, dischtime)]`. Audit accepts current-run `--cohort`, `--ehr`, and `--imaging` overrides. Omitting `--review-policy` uses the current configuration. The two policies use different evidence and review rules.

The main tables are `candidate_studies.parquet`, `candidate_evidence.parquet`, and the observation, event audit, and evidence tables under `window_72h/`. The output manifest lists the available inventory, exclusion, conflict, source coverage, and event reports. Review the audit and resolve blocking evidence before confirming the endpoint.

- For `strict`, complete applicable decisions from `review_templates/cohort.csv` and `review_templates/events.csv` and place them in the local run configuration. Set `cohort_decisions_path` and `event_decisions_path`, then rerun the audit. The program verifies record keys and archives reviewed evidence under `review_sources/` by generation. Supplemental sources can provide review evidence. See the [review configuration](../configs/README.md#clinical-evidence-and-reviews).
- For `lenient`, record event evidence and candidate exclusions in the configured local review CSVs as applicable, then rerun the audit.

Once `audit_ready_for_window_decision` is true and review is complete, confirm `cohort/reviews/endpoint_review.json` for that audit generation. Freeze checks the evidence, window, and score. `--decision-review` can select another current-run endpoint review. Freeze inherits the clinical source settings from the audit.

```powershell
python -B scripts/data/05_audit_burden_labels.py --version 1 --run 1 --mode freeze
```

`step05/freeze/endpoint_decision.json` binds the frozen outcome decision to the audit. Step13 performs the final data-run freeze. For `CLINICAL_INGEST_OUT_OF_MEMORY` or `CLINICAL_SPILL_LIMIT_EXCEEDED`, inspect resource settings and the join plan. Large CSV ingestion constrains the hash-build side to the small candidate table and restores the optimizer settings after reading.

### Step06 Build burden labels

```powershell
python -B scripts/data/06_build_burden_labels.py --version 1 --run 1
```

The builder independently checks evidence and scoring from the Step05 freeze and audit snapshots. It publishes `study_burden_labels.parquet` and `event_evidence.parquet` for eligible studies with complete labels. The label table records the data version, run, and source audit and freeze generations. Reports include score and event counts, early events, and early discharge. Changed evidence or rules must first go through a new Step05 audit and freeze.

### Step07 Prepare and review images

```powershell
python -B scripts/data/07_qc_and_prepare_images.py --version 1 --run 1
```

The command consumes the index, selected images, and formal labels from Steps 02, 03, and 06. It checks DICOM identity, decodes pixels, applies display and valid-pixel transforms, and rescales and pads the image. Output is a finite, single-channel float32 NPY in `[0, 1]`. The audit checks exact duplicates and draws a stratified review sample from image metadata.

The default config is `imaging/processing.json`. Optional `--processing` and `--reviews` overrides stay within the current run. Review `image_qc_review.html` and its before and after previews, record decisions in the configured local review CSV, and rerun. The processing and QC tables record NPY paths and decisions. Continue once `unreviewed_studies` is zero, at least one image passes, and `ready_for_cohort_freeze` is true.

### Step08 Freeze the common cohort

```powershell
python -B scripts/data/08_freeze_common_cohort.py --version 1 --run 1
```

The command checks the formal labels and image QC from Steps 06 and 07, then freezes their accepted intersection in `cohort.parquet`. `cohort_audit.parquet` retains upstream candidates and exclusion status. The program rechecks event evidence, QC decisions, NPY identity, and source file state. Images remain in the Step07 cache, referenced through the manifest. The `cohort_freeze.json` and flow reports record counts, exclusions, and provenance.

### Step09 Create patient-level splits

Choose the command matching the configured split policy.

```powershell
python -B scripts/data/09_create_patient_split.py --version 1 --run 1 --split-method five-fold
python -B scripts/data/09_create_patient_split.py --version 1 --run 1 --split-method holdout
```

Confirm the current `cohort/patient_split.json` or pass `--split-config`. The holdout targets patient counts in a 70/15/15 allocation. The five-fold split first targets approximately 15% of studies for the fixed test cohort, then assigns the remaining patients to five validation folds. The default CLI method is `holdout`. Step09 checks complete patient coverage, endpoint feasibility, and isolation of stable event records across groups.

Holdout writes `patient_split.parquet`. Five-fold writes one patient table per fold and `cross_validation_indices.json` with zero-based indices into samples sorted by `sample_id`. `split_checks.json` and `split_summary.csv` describe the allocation and feasibility.

### Step10 Audit and freeze the laboratory dictionary

```powershell
python -B scripts/data/10_audit_laboratory.py --version 1 --run 1 --mode audit
python -B scripts/data/10_audit_laboratory.py --version 1 --run 1 --mode review
python -B scripts/data/10_audit_laboratory.py --version 1 --run 1 --mode freeze
```

Run each mode after checking the preceding result. Audit consumes the common cohort and split, then reads laboratory records and admission history for the included patients. The lookback is `[t0 - 24h, t0]` and records must be available at `t0`. Audit checks patient and admission ownership, timestamps, units, specimen evidence, duplicates, and revisions. Use `--lab-config` only in audit mode.

Use the generated review tables, evidence, and local HTML to fill the configured decision and scope inputs. `--reviews` can select the completed CSVs in review mode. Review checks the configuration and evidence from the current audit. Freeze consumes the published review and produces the laboratory dictionary, including `lab_dictionary.parquet` and `lab_dictionary_freeze.json`. A new audit invalidates the previous review and freeze. See the [laboratory review fields](../configs/README.md#patient-splits-and-laboratory-reviews).

### Step11 Build raw EHR features

```powershell
python -B scripts/data/11_build_ehr_features.py --version 1 --run 1 --mode audit
python -B scripts/data/11_build_ehr_features.py --version 1 --run 1 --mode build
```

Audit consumes the frozen Step10 dictionary and laboratory cache and Step09 split. It reads patient, admission, and service tables for static features. It checks availability, ownership, revisions, feature coverage, and the numeric and categorical contracts. For repeated laboratory records, select the latest available revision before applying the numeric result rule. Optional `--feature-config` applies to audit only.

Inspect `ehr_audit_report.md`, `blocking_conflicts.csv`, `static_conflicts.csv`, `ehr_feature_summary.csv`, and laboratory cleaning and missingness reports before build. Build consumes the completed audit and publishes `ehr_features_raw.parquet` and `feature_schema.json`. Each row has `sample_id`, 152 numeric features, and five categorical features. Labels and model preprocessing are handled in other stages.

### Step12 Export a training package

```powershell
python -B scripts/data/12_export_training_package.py --version 1 --run 1
```

Export assembles the published cohort, split, EHR features, and accepted images into `package/` and `training_package.zip`. It validates the bindings, schema, patient isolation, labels, image bytes, and ZIP members. It preserves the upstream image size. Optional `--export-config` selects a configuration in the current run.

The package includes `samples.parquet`, `ehr_features.parquet`, `feature_schema.json`, `images/`, `summary.csv`, `dataset_metadata.json`, and a loading README. Five-fold packages add `cross_validation_indices.json` and `splits/fold_1` through `fold_5`. After relocation, loading uses the new package root. The export report includes `training_package_export.json` and `image_manifest.parquet`. Exported metadata records the export state. Step13 writes the independent acceptance record separately.

### Step13 Validate and freeze the data run

```powershell
python -B scripts/data/13_validate_training_package.py --version 1 --run 1
```

Fill the expected counts in the current `export/validation.json`. Use `--validation-config` for a current-run override. The validator consumes Step12, checks the dependency chain, and independently verifies sample and split identity, the EHR contract, laboratory and static feature reconstruction, images, metadata and summaries, ZIP relocation, and source stability. The standard package compares NPY bytes with Step07, and the master workflow also reconstructs images from DICOM. Five-fold validation checks that each development sample appears in validation exactly once and that all folds share the fixed test cohort.

See `training_package_acceptance.json`, `cross_file_checks.csv`, `failure_details.csv`, and `validation_report.md` under `step13/validate/`. Successful validation writes `outputs/data/v<version>_r<run>/.control/release.json` and freezes the full run. A failed validation leaves the run unfrozen. For a revised package, initialize the next run. For a changed raw data scope, initialize the next version.

### Derive a 1024 × 1024 master

This example assumes `v1_r2` is already a frozen 224 × 224 five-fold release. Complete the local image QC and validation configuration for the new run before the corresponding stages.

```powershell
python -B scripts/utils/initialize_configs.py --template multiple_size --version 1 --run 3 --source-run 2
python -B scripts/utils/reuse_data_artifacts.py --version 1 --run 3 --config configs/utils/v1_r3/reuse_master_inputs.json
python -B scripts/data/07_qc_and_prepare_images.py --version 1 --run 3
python -B scripts/data/08_freeze_common_cohort.py --version 1 --run 3
python -B scripts/data/12_export_training_package.py --version 1 --run 3
python -B scripts/data/13_validate_training_package.py --version 1 --run 3
```

Step07 rebuilds the master from DICOM and checks image identity, duplicate groups, and QC. Step08 preserves accepted samples and clinical values while updating image fields. Export and validation consume the copied clinical features and five-fold split. Smaller sizes are published separately after master acceptance.


## Model development and fixed testing

Start from a frozen five-fold release. Set `data_release` and select the experiments in the local model configs described in the [configuration reference](../configs/README.md#model-configuration). Training commands take `cv`. Development assessments bind a completed training ID and reload its fold checkpoints. Where an entry point supports `--id`, omitting it runs the configured batch.

Training and development assessment outputs live under `outputs/models/train/<modality>/<model>/<id>/` and `outputs/models/assessment/<modality>/<model>/<id>/`. Assessments save OOF predictions, fold and pooled metrics, and two operating points with confusion matrices. Each development baseline uses the corresponding fold's training label mean. The default operating points target at least 80% sensitivity and 80% specificity. Use checkpoints from trusted sources and outputs.

### EHR development

The 12 EHR models include linear and kernel regressors, tree ensembles, and MLPs. The commands below run CatBoost. For another model in the table, replace `catboost` in both paths with its directory name.

```powershell
python -B scripts/models/train/ehr/catboost/run.py cv
python -B scripts/models/assessment/ehr/catboost/run.py
```

| Model directories | Protocol details |
| --- | --- |
| `catboost` | Uses native categorical inputs. The validation fold selects the best round by RMSE. |
| `ridge`, `elastic_net`, `tweedie` | Fit imputation, scaling, and categorical encoding on the training fold. Ridge and Elastic Net use JSON checkpoints. Tweedie uses a log link. |
| `kernel_ridge`, `svr` | The RBF scale is the inverse of the fold's transformed feature count. Assessment uses the bound training data to reconstruct the required features. |
| `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost` | Use the shared fold preprocessing. XGBoost selects its best round by validation RMSE. |
| `mlp_sklearn`, `mlp_torch` | The scikit-learn candidates use L-BFGS. The PyTorch candidates use deterministic full-batch CPU training and keep the best validation weights. |

Model settings and candidate values are in `configs/models.template/train/ehr/`. The assessment configurations use the corresponding training IDs. Batch indexes record publication status and parameter differences.

### EHR development comparison

```powershell
python -B scripts/models/assessment/ehr/compare.py --id operating_points_80
```

The command discovers completed EHR assessments and verifies common release, samples, folds, baselines, and operating-point targets. It uses 5,000 paired patient bootstrap draws with the development thresholds held fixed. Results go to `outputs/models/assessment/ehr/comparison/<id>/`, with metrics and plots under `threshold_free/` and `threshold_based/`. The manifest records the actual input generations. Use a new comparison ID for a changed candidate set.

### EHR fixed testing

The public configs select XGBoost `x03`, CatBoost `d2_l210`, and Extra Trees `t02`. Set their release and source assessment IDs for the intended run.

```powershell
python -B scripts/models/test/ehr/xgboost/run.py refit
python -B scripts/models/test/ehr/catboost/run.py refit
python -B scripts/models/test/ehr/extra_trees/run.py refit
```

Refit uses the full development cohort. CatBoost and XGBoost use the median of the five best development rounds, while Extra Trees inherits the configured tree count. The assessment supplies the model thresholds. The full development label mean supplies the baseline and its thresholds.

```powershell
python -B scripts/models/test/ehr/xgboost/run.py evaluate
python -B scripts/models/test/ehr/catboost/run.py evaluate
python -B scripts/models/test/ehr/extra_trees/run.py evaluate
python -B scripts/models/test/ehr/compare.py --id operating_points_80
```

Evaluate scores the fixed test cohort from the refit checkpoint and publishes predictions, raw and clipped metrics, frozen operating points, and confusion matrices. The two stages write to `outputs/models/test/ehr/<model>/<id>/{refit,evaluate}/`. Compare discovers the completed EHR test evaluations, checks their common release, and writes the paired patient comparison under `outputs/models/test/ehr/comparison/<id>/`. Test operating-point labels refer to development targets. The results report the sensitivity and specificity observed on test.

### Image development

Set `roots.model_weights` in `configs/environment/paths.local.json`. Select the package images or a validated variant batch and size with `image_source`. Each image load checks the registered dimensions, dtype, and value range. Configure batch size for the image dimensions and available memory.

```powershell
python -B scripts/models/train/image/resnet18/run.py cv --id imagenet1k_frozen
python -B scripts/models/assessment/image/resnet18/run.py --id imagenet1k_frozen
```

The baseline freezes the encoder and normalization statistics and trains a linear head for 15 epochs with augmentation disabled. Each fold retains the checkpoint with the best validation RMSE. The ResNet implementations also support a head warm-up followed by `layer4` fine-tuning. The ResNet50 implementation supports event-component supervision. To run a study-specific optimization or resolution candidate, first define it in the local training and assessment configs.

| Image model directory | Local weight location under `model_weights` |
| --- | --- |
| `resnet18` | `ResNet-18/resnet18-f37072fd.pth` |
| `resnet50` | `ResNet-50/resnet50-0676ba61.pth` for ImageNet-1K. A supported TorchXRayVision variant uses `ResNet-50/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt`. |
| `densenet121` | ImageNet-1K and CheXpert files in `DenseNet-121/`. See the training template for filenames. |
| `swin_b` | `Swin-B/swin_b-68c6b09e.pth` |
| `caformer_s18` | `CAFormer-S18/model.safetensors` |
| `convformer_s18` | `ConvFormer-S18/model.safetensors` |
| `efficientnet_b0` | `EfficientNet-B0/efficientnet_b0_rwightman-7f5810bc.pth` |
| `deit_base_patch16_224` | `DeiT-B16/model.safetensors` |

ImageNet-1K workflows use full-field three-channel input with ImageNet normalization. CheXpert DenseNet and TorchXRayVision ResNet50 use one channel scaled to the configured model's intensity range. The official CheXpert full model file is loaded as a model object. Use only a trusted weight file. For a Windows MKL threading-layer conflict, the study used `$env:MKL_THREADING_LAYER = 'SEQUENTIAL'` in the current terminal.

For example, run the two public DenseNet candidates with these commands. For another backbone, use its model directory and configured experiment IDs.

```powershell
python -B scripts/models/train/image/densenet121/run.py cv
python -B scripts/models/assessment/image/densenet121/run.py
```

### Image development comparison

```powershell
python -B scripts/models/assessment/image/compare.py --id imagenet_backbones_224 `
  --assessment resnet18/imagenet1k_frozen `
  --assessment swin_b/imagenet1k_frozen
```

Repeat `--assessment MODEL/ID` for the selected candidates or omit it to discover all completed image assessments. All candidates must share the frozen release, development samples, folds, and operating-point targets. Image dimensions can differ. The command records the selected generations and uses 5,000 paired patient draws. Results go to `outputs/models/assessment/image/comparison/<id>/`. Use a new ID for a changed selection.

### Image fixed testing

```powershell
python -B scripts/models/test/image/resnet50/run.py refit
python -B scripts/models/test/image/resnet50/run.py evaluate
python -B scripts/models/test/image/compare.py --id fixed_image_80
```

The selected `assessment_id` determines the training configuration and image source. Use `--config` to select a separate local config for each candidate. Refit starts from the original pretrained weights and uses the full development cohort. The fixed epoch budget comes from the median of the five best development epochs. Staged candidates retain their warm-up and unfreezing rules. Evaluate applies development-derived thresholds to the fixed test cohort. The stage outputs follow the EHR fixed-test layout under `outputs/models/test/image/resnet50/<id>/`. Comparison writes to `outputs/models/test/image/comparison/<id>/`.

## Multimodal fusion

The commands below illustrate our fusion study. For a new checkout, configure the source IDs and all referenced experiments before running them. In particular, `imagenet1k_combo_c5_layer4_512` and the additional F1 weights require local configuration. The examples assume the selected sources use the same frozen release. In our study, the two unimodal fixed-test evaluations informed source selection. We interpret the later fusion fixed test as exploratory.

### F1 Fixed score fusion

```powershell
python -B scripts/models/assessment/fusion/weighted_late/run.py --id f1_a020
```

In `configs/models/assessment/fusion/weighted_late.json`, set the source assessments, image weight, and decision margins before reviewing the fusion results. The command pairs existing development OOF scores by `sample_id` and checks the labels, folds, and training means. With image weight `alpha`, it computes `(1-alpha)*EHR + alpha*image`. The matched shrinkage control substitutes the fold training mean for the image score. The image term is the raw scalar burden prediction.

The assessment writes sources, paired predictions, metrics, operating points, and paired RMSE and MAE differences under `outputs/models/assessment/fusion/weighted_late/<id>/`. It records point-estimate and interval checks against the configured margins in `decision_summary.json`. F1 uses published OOF scores directly.

### F2 Image feature and EHR fusion

```powershell
python -B scripts/models/train/fusion/feature_fusion/run.py cv --id f2_pca08
python -B scripts/models/assessment/fusion/feature_fusion/run.py --id f2_pca08
```

Each image fold checkpoint supplies 2,048-dimensional pooled vectors. The training fold fits standardization and PCA without whitening. The projected columns are appended to the original EHR features before fitting CatBoost with the source protocol. The development assessment reloads the published vectors, projection, and CatBoost weights. No fixed-test images enter this workflow.

The training output records checkpoint and source generations, sample ordering, fitted projections, and fold model files. The assessment binds the F2 training result and saves OOF predictions, metrics, operating points, and training history. The image checkpoints used validation-fold early stopping, so the results belong to the historical development protocol.

### F3 Nested learned score fusion

```powershell
python -B scripts/models/train/fusion/learned_late/run.py cv --id f3_nested_median
python -B scripts/models/assessment/fusion/learned_late/run.py --id f3_nested_median
```

For each outer fold, the remaining four folds provide inner OOF predictions for estimating the fusion coefficients. A three-fold base fit can score its two held-out folds for the applicable outer splits. Five four-fold fits supply the outer validation scores. With `training_median`, each three- or four-fold training scope selects its own CatBoost and image training lengths by leaving out one of its eligible folds at a time, taking the median, and refitting on the full scope. Per modality this uses 50 calibration runs and 15 refits. The outer validation fold does not enter its coefficient or round selection.

Training publishes the 15 fit scopes, split membership, histories, models, and predictions in `outputs/models/train/fusion/learned_late/<id>/`. Assessment validates the calibration record, publishes inner and outer predictions, and reports learned coefficients and paired differences. `predicted_score` is F3b, which learns an image weight and intercept. `f3a` learns only the weight. The matched comparison also includes fixed weights, the unimodal base scores, EHR plus an intercept, and the fold mean.

### Fusion development comparison

```powershell
python -B scripts/models/assessment/fusion/compare.py --id fusion_full_oof_v1 `
  --assessment ehr/catboost/d2_l210 `
  --assessment image/resnet50/imagenet1k_combo_c5_layer4_512 `
  --assessment fusion/weighted_late/f1_a020 `
  --assessment fusion/weighted_late/f1_a040 `
  --assessment fusion/weighted_late/f1_a060 `
  --assessment fusion/weighted_late/f1_a080 `
  --assessment fusion/feature_fusion/f2_pca08 `
  --assessment fusion/learned_late/f3_nested_median
```

Supply one EHR, one image, one F2, one F3, and at least one F1 assessment. Under `outputs/models/assessment/fusion/comparison/<id>/`, `historical/` covers the original unimodal scores, F1 and its shrinkage controls, and F2. `nested/` covers F3 and its protocol-matched controls. Both include the fold mean. Outputs include raw and clipped paired error differences, AUROC, AP, operating points and alert counts, fold and subgroup metrics, decision margins, and leave-one-patient-out checks. The bootstrap uses 5,000 paired patient draws and seed 42 by default.

### Fusion fixed testing

The study used F1 with image weight 0.60 and F3b. Prepare their local test configs and complete both matching unimodal fixed tests first.

```powershell
python -B scripts/models/test/fusion/weighted_late/run.py refit
python -B scripts/models/test/fusion/learned_late/run.py refit
python -B scripts/models/test/fusion/weighted_late/run.py evaluate
python -B scripts/models/test/fusion/learned_late/run.py evaluate
```

Each refit binds the two existing full-development unimodal refits. F1 verifies its fixed weight and F3b fits the final coefficients on nested development base scores. Both freeze operating points from development. Evaluate checks sample and label alignment and verifies that the unimodal test scores bind the same refit generations. It then publishes each candidate's predictions and metrics under `outputs/models/test/fusion/<model>/<id>/evaluate/`.

```powershell
python -B scripts/models/test/fusion/compare.py --id fusion_final_test_v1 `
  --evaluation ehr/catboost/d2_l210 `
  --evaluation image/resnet50/imagenet1k_combo_c5_layer4_512 `
  --evaluation fusion/weighted_late/f1_a060 `
  --evaluation fusion/learned_late/f3b_nested_median
```

Compare accepts the explicit four-evaluation list and writes to `outputs/models/test/fusion/comparison/<id>/`. It includes matched F1 shrinkage, F3a, and EHR intercept controls and uses frozen thresholds with a shared set of 5,000 paired patient draws.

## Tests and publication rules

```powershell
python -B -m unittest discover -s tests -v
```

Tests use synthetic inputs and temporary directories. The VS Code test configuration uses the same unittest discovery. On a restricted Windows environment, set `TEMP` and `TMP` to a writable temporary location before running tests.

Stage generations increase on successful publication. Downstream checks cover direct and indirect inputs. After a relevant code, config, or review change, rerun the affected stage and its dependents. File hashes do not detect pending configuration or review edits. Clinical and laboratory record keys retain their content-based identity. Source metadata uses relative or named-root paths. Raw CSV evidence excerpts preserve the source text.
