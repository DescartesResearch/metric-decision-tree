from pathlib import Path
import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def generate_negative_examples(data, save_csv: bool = True, output_dir: Path = Path("output")):

    if isinstance(data, (str, Path)):
        input_path = Path(data)
        df = pd.read_csv(input_path, sep=";")
    else:
        input_path = None
        df = data

    feature_columns = [col for col in df.columns if col not in ['metric', 'suitable']]
    training_data = []
    training_data.extend(df.to_dict('records'))
    metric_groups = df.groupby('metric')

    for metric_name, group in metric_groups:
        positive_tuples = set()
        for _, row in group.iterrows():
            tuple_repr = tuple(int(row[f]) for f in feature_columns)
            positive_tuples.add(tuple_repr)

        for _, positive_row in group.iterrows():
            for feature in feature_columns:
                negative_example = positive_row.to_dict().copy()
                original_value = int(positive_row[feature])
                negative_example[feature] = 1 - original_value
                neg_tuple = tuple(int(negative_example[f]) for f in feature_columns)

                if neg_tuple not in positive_tuples:
                    negative_example['suitable'] = 0
                    training_data.append(negative_example)

    result_df = pd.DataFrame(training_data)

    result_df = result_df.sort_values(
        by=['suitable', 'metric'],
        ascending=[False, True]
    ).reset_index(drop=True)

    if save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path:
            output_filename = f"training_data_{input_path.stem}.csv"
        else:
            output_filename = "training_data.csv"

        output_path = output_dir / output_filename
        result_df.to_csv(output_path, sep=";", index=False)

        return output_path

    return result_df

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

    X, y = load_metric_matrix(generate_negative_examples(args.csv_path))
    clf = train_decision_tree(X, y, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)

    out_png = args.out_path / "decision_tree.png"
    save_decision_tree_png(clf, X.columns.tolist(), out_png)

    print(f"Decision tree saved to {out_png}")


if __name__ == "__main__":
    main()

