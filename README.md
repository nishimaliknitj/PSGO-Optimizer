![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Release](https://img.shields.io/badge/Release-v1.0.0-orange)
![Status](https://img.shields.io/badge/Status-Research-red)

# PSGO-Optimizer
# PSGO-Optimizer

## PSGO: A Novel Asymmetric Dual-Agent Swarm Optimization Inspired by Pistol Shrimp–Goby Fish Mutualism for Feature Selection and Engineering Design Optimization

Official implementation of PSGO (Pistol Shrimp–Goby Fish Optimization), a biologically inspired asymmetric dual-agent swarm optimization algorithm designed for global optimization, feature selection, and engineering design applications.

---

## Authors

* Nishi Madan
* Rahul Malik
* Alok Kumar
* Utsav Upadhyaya

Department of Computer Science and Engineering
Galgotias University, India

---

## Abstract

PSGO is a novel asymmetric dual-agent swarm optimization algorithm inspired by the mutualistic relationship between pistol shrimp and goby fish. The proposed optimizer integrates biologically grounded asymmetric cooperation, a graded three-level danger signaling mechanism, and a claw-blast perturbation strategy to balance exploration and exploitation effectively. PSGO demonstrates strong optimization performance on benchmark functions, feature selection tasks, and engineering design optimization problems.

---

## Key Contributions

* Biologically grounded asymmetric dual-agent swarm architecture.
* Novel graded three-level danger signal mechanism.
* Claw-blast perturbation strategy with adaptive decay radius.
* Strong performance on CEC benchmark functions.
* Applications to feature selection and engineering design optimization.

---

## Repository Structure

```text
PSGO-Optimizer/
│
├── src/
├── notebooks/
├── experiments/
├── results/
├── figures/
├── paper/
├── docs/
├── data/
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/nishimaliknitj/PSGO-Optimizer.git

cd PSGO-Optimizer

pip install -r requirements.txt
```

---

## Usage

Run the optimizer:

```bash
python psgo.py
```

Run benchmark experiments:

```bash
python run_cec2017.py
```

Execute complete experiments:

```bash
python run_all_fast.py
```

---

## Benchmark Evaluation

The proposed PSGO algorithm is evaluated on standard numerical optimization benchmarks including:

* CEC 2017 Benchmark Suite
* Feature Selection Datasets
* Engineering Design Optimization Problems

---

## Engineering Design Problems

The repository contains implementations and experiments for engineering optimization tasks reported in the paper.

---

## Reproducibility

All experiments, notebooks, benchmark scripts, and result generation codes required to reproduce the reported findings are included in this repository.

---

## Citation

If you use PSGO in your research, please cite the associated paper.

```bibtex
@article{PSGO2026,
  title={PSGO: A Novel Asymmetric Dual-Agent Swarm Optimization Inspired by Pistol Shrimp--Goby Fish Mutualism for Feature Selection and Engineering Design Optimization},
  author={Madan, Nishi and Malik, Rahul and Kumar, Alok and Upadhyaya, Utsav},
  year={2026}
}
```

---

## Contact

Rahul Malik

Email: [maliknit@gmail.com](mailto:maliknit@gmail.com)
