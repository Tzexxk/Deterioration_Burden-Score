# Burden-Score

We study how chest radiographs and electronic health records (EHR) can help estimate clinical event burden during the 72 hours after imaging. The cohort consists of adult inpatients outside the ICU or CCU at the time of imaging. The research score combines care escalation, respiratory deterioration, circulatory support, and in-hospital death observed within the study window. It is a research outcome.

The data pipeline audits the source files, matches images to admissions, reviews clinical evidence, constructs labels, checks image quality, and builds patient-level splits and EHR features. An independent validation step freezes the exported training package. Each stage records its input publications and their generations so downstream stages can check provenance.

The frozen research release, `v1_r3`, uses five patient-level development folds and a fixed test cohort. It contains a 1024 × 1024 image master from which we derive and validate smaller images with identical samples, labels, and splits. Model workflows cover EHR regression, chest radiograph models, and fixed score fusion, feature fusion, and learned score fusion. Development comparisons use paired patient bootstrap intervals. Fixed-test workflows freeze thresholds from development data. In our fusion study, the unimodal test results informed source selection, so the subsequent fusion test is exploratory.

## Data access and release scope

- MIMIC-IV supplies hospital, ICU, laboratory, and clinical event data.
- MIMIC-CXR supplies chest radiographs and their indexes.
- MIMIC-IV-NOTE is optional. It supplies radiology reports and can provide discharge-note evidence when configured.

We publish the code, tests, and configuration templates. Obtain the required dataset access separately and point the local configuration to data and weights stored outside the checkout. Reproducing the exact research cohort requires the original source subset, effective configuration, and clinical and image review decisions. Raw data, patient-level outputs, completed reviews, and weights stay in local storage.

## Resource planning

The clinical and laboratory audit templates and the EHR feature template set DuckDB's memory limit to 2 GB and temporary spill limit to 4 GB. They use an 8 GiB disk-space preflight estimate. Package export and validation each use a 1 GiB preflight estimate for 224 × 224 packages and 8 GiB for the 1024 × 1024 master. Allow additional space for raw data, caches, archives, staging files, and model weights. GPU memory depends on image size, batch size, and backbone.

In our 682-image release, the 1024 × 1024 master ZIP was about 1.91 GiB, and the NPY files for the first five derived sizes totaled about 3.48 GiB.

## Getting started

- [Configuration setup and reference](configs/README.md)
- [Commands, stage outputs, and tests](scripts/README.md)
