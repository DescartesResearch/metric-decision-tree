from pathlib import Path
import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def load_metric_matrix(csv_path: Path):
    data = pd.read_csv(csv_path, sep=";")

    y = data["suitable"]
    X = data.drop(columns=["suitable"] + [c for c in data.columns if c == "metric"])
    X = X.select_dtypes(include="number")

    return X, y


def train_decision_tree(X, y, max_depth=8, min_samples_leaf=4):
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=0
    )
    clf.fit(X, y)
    return clf



def save_decision_tree_png(clf, feature_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["unsuitable", "suitable"],
        filled=True
    )
    plt.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a decision tree on a metric-property matrix and export it as PNG."
    )

    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the metric-property matrix CSV file"
    )

    parser.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="Output directory for the generated decision tree PNG"
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
        help="Maximum depth of the decision tree (default: 8)"
    )

    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=4,
        help="Minimum number of samples required to be at a leaf node (default: 4)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_metric_matrix(args.csv_path)
    clf = train_decision_tree(X, y, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)

    out_png = args.out_path / "decision_tree.png"
    save_decision_tree_png(clf, X.columns.tolist(), out_png)

    print(f"Decision tree saved to {out_png}")


if __name__ == "__main__":
    main()
