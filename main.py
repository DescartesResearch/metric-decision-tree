from pathlib import Path
import pandas as pd
from graphviz import Digraph
import argparse
from typing import Optional
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_training_data(
    data: pd.DataFrame | str | Path,
    save_csv: bool = True,
    output_dir: Path = Path("output"),
    output_filename: Optional[str] = None,
) -> pd.DataFrame | Path:

    if isinstance(data, (str, Path)):
        input_path = Path(data)
        df = pd.read_csv(input_path, sep=";")
    else:
        input_path = None
        df = data

    feature_columns = [col for col in df.columns if col not in ["metric", "suitable"]]
    training_data = []
    training_data.extend(df.to_dict("records"))
    metric_groups = df.groupby("metric")

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
                    negative_example["suitable"] = 0
                    training_data.append(negative_example)

    result_df = pd.DataFrame(training_data)

    result_df = result_df.sort_values(
        by=["suitable", "metric"], ascending=[False, True]
    ).reset_index(drop=True)

    if save_csv and output_filename is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        result_df.to_csv(output_path, sep=";", index=False)

        return output_path

    return result_df


def load_training_data(data: Path | pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    if isinstance(data, Path):
        data = pd.read_csv(data, sep=";")

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
        random_state=0,
    )
    clf.fit(X, y)
    return clf


def generate_base_decision_tree(clf, feature_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["unsuitable", "suitable"],
        filled=True,
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def generate_metric_decision_tree(clf, samples_df, feature_names):
    """
    Generates a pruned Graphviz tree showing where specific samples landed.
    See: https://graphviz.org/doc/info/shapes.html
    """
    tree = clf.tree_
    dot = Digraph(comment="Pruned Sample Path Tree")
    dot.attr(rankdir="TB", size="10")

    # 1. Map samples to leaf IDs
    # leaf_indices is an array where each entry is the leaf node ID for that sample
    leaf_indices = clf.apply(samples_df)

    # 2. Determine which nodes are "active" (on the path of at least one sample)
    active_nodes = set()
    for leaf in leaf_indices:
        curr = leaf
        while curr != -1:  # -1 is the parent of the root
            active_nodes.add(curr)
            # Find parent: we have to search the tree structure
            parent = -1
            for i, (left, right) in enumerate(
                zip(tree.children_left, tree.children_right)
            ):
                if left == curr or right == curr:
                    parent = i
                    break
            curr = parent

    # 3. Recursive function to build the graph
    def build_node(node_id):
        if node_id not in active_nodes:
            return

        # Check if it's a leaf or if we have samples here
        is_leaf = tree.children_left[node_id] == -1

        # Get samples at this node
        samples_at_node = samples_df.index[leaf_indices == node_id].tolist()

        if is_leaf:
            label = f"{', '.join(map(str, samples_at_node))}"
            dot.node(
                str(node_id),
                label,
                shape="box",
                color="black",
                fillcolor="lightgray",
                style="filled,rounded",
            )
        else:
            feature = feature_names[tree.feature[node_id]]
            label = f"{feature}"
            dot.node(str(node_id), label, shape="box")

            # Recurse to children if they are active
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            if left_child in active_nodes:
                dot.edge(str(node_id), str(left_child), label="no")
                build_node(left_child)

            if right_child in active_nodes:
                dot.edge(str(node_id), str(right_child), label="yes")
                build_node(right_child)

    build_node(0)  # Start at root
    return dot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a decision tree on a metric-property matrix and export it as PNG."
    )

    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the metric-property matrix CSV file",
    )

    parser.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="Output directory for the generated decision tree PNG",
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
        help="Maximum depth of the decision tree (default: 8)",
    )

    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=4,
        help="Minimum number of samples required to be at a leaf node (default: 4)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    matrix_df: pd.DataFrame = pd.read_csv(args.csv_path, sep=";")
    X, y = load_training_data(
        generate_training_data(matrix_df, save_csv=True, output_dir=args.out_path, output_filename=f"training_data_{args.csv_path.stem}.csv")
    )
    clf = train_decision_tree(
        X, y, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf
    )

    base_dt_out_pdf = args.out_path / f"base_decision_tree_{args.csv_path.stem}.pdf"
    generate_base_decision_tree(clf, X.columns.tolist(), base_dt_out_pdf)
    logging.info(f"Base Decision Tree saved to {base_dt_out_pdf}")

    viz = generate_metric_decision_tree(
        clf,
        matrix_df.set_index("metric", drop=True).iloc[:, :-1],
        matrix_df.columns.tolist()[1:-1],
    )

    # 4. Save and view
    metric_dt_out_pdf = args.out_path / f"metric_decision_tree_{args.csv_path.stem}.pdf"
    viz.render(
        metric_dt_out_pdf.stem, directory=args.out_path, format="pdf", cleanup=True
    )
    logging.info(f"Metric Decision Tree saved to {metric_dt_out_pdf}")


if __name__ == "__main__":
    main()
