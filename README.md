# Organizational Intelligence for Real-Time Inverse Physics Problems in Positioning Systems

## Overview

This repository contains the source code, experimental framework, evaluation scripts, and reproducibility materials accompanying the manuscript:

**"Organizational Intelligence for Real-Time Inverse Physics Problems in Positioning Systems"**

The study introduces a distributed organizational intelligence framework for solving inverse physics problems under real-time constraints. Instead of relying on centralized deep architectures, the proposed framework employs a population of lightweight Shallow Adaptive Cooperative Units (SACUs) that collectively reconstruct dynamic physical fields through adaptive coordination, localized physics-aware learning, and performance-driven specialization.

The framework is evaluated on propagation-field inverse reconstruction tasks generated from physically governed simulations and compared against centralized learning baselines.

---

## Repository Structure

```text
organizational-intelligence-inverse-physics/
│
├── src/
├── scripts/
├── configs/
├── data/
├── results/
├── docs/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
└── CITATION.cff
```

---

## Scientific Contributions

The repository implements:

- Distributed organizational intelligence for inverse reconstruction
- Shallow Adaptive Cooperative Units (SACUs)
- Physics-aware residual integration
- Adaptive influence redistribution
- Role-conditioned specialization
- Inter-agent communication mechanisms
- Organizational robustness under dynamic perturbations
- Real-time inverse problem solving

---

## Dataset Information

### Data Source

The experiments utilize physically governed propagation-field simulations generated using the PDEBench framework.

PDEBench provides benchmark partial differential equation datasets for scientific machine learning and physics-based modeling.

Official source:

https://github.com/pdebench/PDEBench

The dataset is not redistributed in this repository.

Users should obtain the original dataset directly from the official source.

### Data Splits

The experimental protocol uses:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

Parameter-disjoint splitting is employed to prevent information leakage between training and evaluation environments.

---

## Code Components

| Module | Description |
|----------|-------------|
| train_sacu.py | Training of the proposed organizational framework |
| train_baseline.py | Centralized baseline training |
| evaluate.py | Performance evaluation |
| run_ablations.py | Component contribution analysis |
| stress_test.py | Robustness and perturbation experiments |
| physics_metrics.py | Physics-consistency evaluation |
| compare_eval.py | Comparative analysis |
| utils.py | Utility functions |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/USERNAME/organizational-intelligence-inverse-physics.git

cd organizational-intelligence-inverse-physics
```

### Create Environment

Using Conda:

```bash
conda env create -f environment.yml

conda activate inversephysics
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## Requirements

Typical dependencies include:

- Python 3.12
- TensorFlow
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn

Exact package versions are provided in:

- `requirements.txt`
- `environment.yml`

---

## Training

### Train Proposed SACU Framework

```bash
python src/train_sacu.py
```

### Train Baseline Model

```bash
python src/train_baseline.py
```

---

## Evaluation

Run nominal evaluation:

```bash
python src/evaluate.py
```

Run robustness analysis:

```bash
python src/stress_test.py
```

Run ablation studies:

```bash
python src/run_ablations.py
```

---

## Experimental Outputs

The framework generates:

- Reconstruction accuracy metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Physics residual magnitude
- Latency measurements
- Robustness curves
- Ablation statistics
- Comparative visualizations

Outputs are stored in:

```text
results/
```

---

## Reproducibility

To facilitate reproducibility, the repository additionally provides:

- Hyperparameter configuration
- Random seed information
- Hardware specifications
- Experimental protocol documentation
- Evaluation procedures

See:

```text
docs/hyperparameters.md
docs/random_seeds.md
docs/hardware_and_environment.md
docs/reproducibility_checklist.md
```

---

## Hardware Environment

Experiments reported in the manuscript were conducted using:

- Operating System: Windows 11 (64-bit)
- CPU: Intel Core Ultra 7 265KF
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 5070 (12 GB)

Additional implementation details are provided in:

```text
docs/hardware_and_environment.md
```

---

## Citation

If you use this repository, please cite:

```bibtex
@article{rokaya2026organizational,
  title={Organizational Intelligence for Real-Time Inverse Physics Problems in Positioning Systems},
  author={Rokaya, Mahmoud},
  year={2026}
}
```

The final citation will be updated upon publication.

---

## License

This repository is released under the MIT License.

See:

```text
LICENSE
```

for details.

---

## Contact

**Mahmoud Rokaya**

Department of Information Technology  
College of Computers and Information Technology  
Taif University, Saudi Arabia

Email: mahmoudrokaya@tu.edu.sa
