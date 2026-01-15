# Metric Decision Tree

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> A novel approach for an informed selection of suitable ML metrics for classification and regression tasks

## 📖 Overview

In the area of ML/DL, oftentimes standard evaluation metrics are selected for the training, evaluation, and/or monitoring of ML/DL models.
Mostly, the choice of specific metrics is ad-hoc and not questioned or justified. Therefore, task- or dataset-specific requirements are not sufficiently taken into account.
The choice of ML metrics not only has an impact on the conclusions drawn on the predictive performance of a model, it consequently also affects the actual performance, following the principle: "You can only improve what you measure."

This is a research project with the goal of helping ML/DL practitioners to select metrics appropriate for their tasks or datasets.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-link>

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

*TODO!*



## 🔬 Research Paper

This repository accompanies our research paper submitted to QualITA 2026.
The paper will be referenced here upon publication.


## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=metric_decision_tree --cov-report=html

# Run specific test module
pytest tests/test_classifier.py
```


## 🤝 Contributing

We welcome contributions from the community!
As discussed in the workshop paper, by nature, our evaluation did not cover all metrics and properties you can think of.
Further, it is currently limited to classification and regression tasks.
If a metric or property you consider important is missing, you can contribute to the project as follows.
Also, new features, such as improved tree visualization methods, are welcome.

1. **Fork the repository**
1. **Create a feature branch**: Example: `git checkout -b feature/new-feature`
1. **Make your changes**: For example, update metric-property matrix and re-generate decision trees.
1. **Commit your changes**: `git commit -m 'Add amazing feature'`
1. **Push to the branch**: `git push origin feature/amazing-feature`
1. **Open a Pull Request**


## 📄 Citation

If you use this software in your research, please cite our paper:

TBD

## 📜 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---
