# IM2Deep
Collisional cross-section prediction for (modified) peptides.

---
## Introduction

IM2Deep is a CCS predictor for (modified) peptides.
It is able to accurately predict CCS for modified peptides, even if the modification wasn't observed during training.

## Installation
Install with pip:
`pip install im2deep`

## Usage
### Basic CLI usage:
```sh
im2deep <path/to/peptide_file.csv>
```
If you want to calibrate your predictions (HIGHLY recommended), please provide a calibration file:
```sh
im2deep <path/to/peptide_file.csv> --calibration_file <path/to/peptide_file_with_CCS.csv>
```
For an overview of all CLI arguments, run `im2deep --help`.

## Input files
Both peptide and calibration files are expected to be comma-separated values (CSV) with the following columns:
  - `seq`: unmodified peptide sequence
  - `modifications`: every modifications should be listed as `location|name`, separated by a pipe character (`|`)
     between the location, the name, and other modifications. `location` is an integer counted starting at 1 for the
     first AA. 0 is reserved for N-terminal modifications, -1 for C-terminal modifications. `name` has to correspond
     to a Unimod (PSI-MS) name.
  - `charge`: peptide precursor charge
  - `CCS`: collisional cross-section (only for calibration file)

For example:

```csv
seq,modifications,charge,CCS
VVDDFADITTPLK,,2,422.9984309464991
GVEVLSLTPSFMDIPEK,12|Oxidation,2,464.6568644356109
SYSGREFDDLSPTEQK,,2,468.9863221739147
SYSQSILLDLTDNR,,2,460.9340710819608
DEELIHLDGK,,2,383.8693416055445
IPQEKCILQTDVK,5|Butyryl|6|Carbamidomethyl,3,516.2079366048176
```

