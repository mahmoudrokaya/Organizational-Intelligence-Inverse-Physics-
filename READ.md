# Data Branch

## Overview

This branch contains the data resources, metadata, configuration files, and documentation required to reproduce the experiments reported in the manuscript:

**"Organizational Intelligence for Real-Time Inverse Physics Problems in Positioning Systems"**

The purpose of this branch is to separate data-related materials from the source-code repository, improving reproducibility and repository organization.

---

## Contents

This branch may contain:

- Dataset acquisition instructions
- Dataset metadata
- Data preprocessing information
- Sample data files
- Experimental configuration files
- Data split information
- Data availability documentation
- Links to external datasets

Large benchmark datasets are not redistributed unless permitted by their respective licenses.

---

## Dataset Source

The experiments are based on propagation-field simulations generated using the PDEBench framework.

**Official PDEBench Repository**

https://github.com/pdebench/PDEBench

Users should obtain the original benchmark datasets directly from the official source.

Please refer to the PDEBench documentation for licensing terms, download procedures, and dataset descriptions.

---

## Data Organization

```text
data/
├── README.md
├── metadata/
├── configs/
├── samples/
└── documentation/
```

Folder contents may vary depending on the experiment.

---

## Data Splits

The experimental protocol uses:

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

To prevent information leakage, parameter-disjoint splitting was employed between training, validation, and testing environments.

---

## Sample Data

Small sample files may be included to verify code execution and demonstrate expected input formats.

These samples are not intended to reproduce the full experimental results.

---

## Reproducing the Experiments

To reproduce the complete experiments:

1. Download the original PDEBench dataset.
2. Place the dataset in the location specified by the configuration files.
3. Follow the instructions provided in the main repository README.
4. Execute the training and evaluation pipelines from the source-code repository.

---

## Data Availability

The original benchmark datasets remain the property of their respective creators and distributors.

This repository does not claim ownership of third-party datasets.

All users are responsible for complying with the licensing conditions of the original data providers.

---

## Citation

If you use this data branch, please cite:

```bibtex
@software{rokaya2026inversephysicsdata,
  author = {Rokaya, Mahmoud},
  title = {Organizational Intelligence for Real-Time Inverse Physics Problems in Positioning Systems: Data Resources},
  year = {2026},
  doi = {10.5281/zenodo.20520585},
  url = {https://github.com/mahmoudrokaya/Organizational-Intelligence-Inverse-Physics-}
}
```

---

## Related Resources

- Main Repository:
  https://github.com/mahmoudrokaya/Organizational-Intelligence-Inverse-Physics-

- DOI:
  https://doi.org/10.5281/zenodo.20520585

---

## Contact

**Mahmoud Rokaya**

College of Computers and Information Technology  
Taif University  
Saudi Arabia