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

The simplest way to use the tool is via command line:
```bash
python main.py --csv_path data/matrix.csv --out_path ./output
```

This will:
1. Load your metric-property matrix from the CSV file
2. Train a decision tree classifier
3. Save the decision tree visualization as `decision_tree.png` in the output directory

#### Command Line Arguments

- `--csv_path` (required): Path to your metric-property matrix CSV file
- `--out_path` (required): Output directory where the decision tree PNG will be saved
- `--max_depth` (optional): Maximum depth of the decision tree (default: 8)
- `--min_samples_leaf` (optional): Minimum number of samples required at a leaf node (default: 4)

## 📊 Input Data Format

The metric-property matrix CSV file should follow this structure:

- **Separator**: Semicolon (`;`)
- **First row (header)**: Column names
  - First column: `metric`
  - Middle columns: Property names (e.g., `property1`, `property2`, ...)
  - Last column: `suitable`
- **Data rows**: 
  - **First column**: `metric` - Name of the metric
  - **Property columns**: Binary values indicating metric properties
    - `1` = property applies to the metric
    - `0` = property does not apply to the metric
  - **Last column**: `suitable` - Binary classification target
    - `1` = metric is suitable for the given scenario
    - `0` = metric is unsuitable for the given scenario
 
### General CSV Structure Example
```csv
metric;property1;property2;property3;property4;...;suitable
metric1;1;0;1;0;...;1
metric2;0;1;0;1;...;0
metric3;1;1;0;0;...;1
metric4;0;0;1;1;...;0
...
```

## 🔧 Module Structure

The code is organized into modular functions:

- **`load_metric_matrix(csv_path)`**: Loads and preprocesses the metric-property matrix
  - Reads CSV with semicolon separator
  - Extracts target variable `suitable` 
  - Removes non-numeric columns (`metric`)
  - Returns: Feature matrix `X` (property values) and target variable `y` (suitability)
  
- **`train_decision_tree(X, y, max_depth, min_samples_leaf)`**: Trains the decision tree classifier
  - Uses entropy criterion for information gain
  - Applies balanced class weights to handle imbalanced data
  - Configurable tree depth and leaf size
  - Returns: Trained classifier object
  
- **`save_decision_tree_png(clf, feature_names, out_path)`**: Exports the decision tree as PNG
  - High-resolution output (800 DPI)
  - Color-coded nodes for better readability
  - Shows property names at decision nodes
  - Displays class distribution at leaf nodes

## 📁 Output

The tool generates a high-resolution PNG file (`decision_tree.png`) showing:
- **Decision nodes**: Property-based splitting criteria (e.g., "multiclass_capable <= 0.5")
- **Leaf nodes**: Final classification (suitable vs. unsuitable)
- **Node information**: Entropy, samples, and class distribution
- **Color-coding**: Visual distinction between suitable (one color) and unsuitable (another color) metrics

The decision tree helps you understand which combination of metric properties leads to suitable or unsuitable metrics for your specific use case.


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
