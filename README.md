# IM2Deep
Collisional cross-section prediction for (modified) peptides.

---
## Introduction

IM2Deep is a CCS predictor for (modified) peptides It is able to accurately predct CCS for modified peptides, even if the modification wasn't observed during training.

## Installation
Install with pip:
`pip install im2deep`

## Usage
Basic CLI usage:
```sh
im2deep <path/to/peptide_file.csv>
```
If you want to calibrate your predictions (HIGHLY recommended), please provide a calibration file:
```sh
im2deep <path/to/peptide_file.csv> --calibration_file <path/to/peptide_file_with_CCS.csv>
```
For an overview of all CLI arguments, run `im2deep --help`.

## Input files
TODO

