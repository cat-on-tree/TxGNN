import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ndcg_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from txgnn import TxData, TxGNN


TOP_K_LIST = [10, 50, 100, 150, 200]


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate trained graph models on fixed GNN test sets."
    )

    parser.add_argument("--data-folder", type=str, default="./data")
    parser.add_argument("--split", type=str, default="full_graph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help=(
            "Directory containing converted GNN fixed test sets. "
            "If omitted, uses <output-dir>/fixed_test_sets_gnn."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Pipeline output directory that contains fixed_test_sets_gnn/.",
    )

    parser.add_argument(
        "--model-root",
        type=str,
        default="./saved_models",
        help="Directory containing trained model subdirectories.",
    )
    parser.add_argument(
        "--model-paths",
        nargs="*",
        default=None,
        help=(
            "Optional explicit model directories. "
            "If omitted, all subdirectories under --model-root containing config.pkl and model.pt are evaluated."
        ),
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="./result",
        help="Directory to save evaluation results.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used for binary metrics such as accuracy/F1.",
    )

    return parser.parse_args()


def safe_metric(func, y_true, y_score_or_pred, default=np.nan):
    try:
        return func(y_true, y_score_or_pred)
    except Exception:
        return default


def sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def infer_negative_ratio(df):
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    if pos == 0:
        return np.nan
    ratio = neg / pos
    if abs(ratio - round(ratio)) < 1e-9:
        return int(round(ratio))
    return ratio


def infer_strategy_from_name(name):
    """
    Infer negative sampling strategy from filename.

    Important:
    - Check degree_matched_head / degree_matched_tail BEFORE generic "degree".
    - Otherwise both endpoint-specific hard sets collapse into "degree_matched".
    """
    lower = str(name).lower()

    head_patterns = [
        "degree_matched_head",
        "degree-matched-head",
        "degree_head",
        "degree-head",
        "head_degree",
        "head-degree",
    ]
    tail_patterns = [
        "degree_matched_tail",
        "degree-matched-tail",
        "degree_tail",
        "degree-tail",
        "tail_degree",
        "tail-degree",
    ]

    if any(p in lower for p in head_patterns):
        return "degree_matched_head"

    if any(p in lower for p in tail_patterns):
        return "degree_matched_tail"

    if "degree" in lower and "head" in lower:
        return "degree_matched_head"

    if "degree" in lower and "tail" in lower:
        return "degree_matched_tail"

    if "degree" in lower or "hard" in lower:
        return "degree_matched"

    if "random" in lower:
        return "random"

    if "normal" in lower:
        return "normal"

    return "unknown"


def resolve_strategy_from_manifest_or_name(manifest_strategy, file_name):
    """
    Resolve final strategy label.

    Prefer endpoint-specific information from the file name because some older
    manifests may store both degree_matched_head and degree_matched_tail as
    generic "degree_matched".
    """
    inferred = infer_strategy_from_name(file_name)

    if inferred in {"degree_matched_head", "degree_matched_tail"}:
        return inferred

    if manifest_strategy is None:
        return inferred

    if isinstance(manifest_strategy, float) and np.isnan(manifest_strategy):
        return inferred

    strategy = str(manifest_strategy)

    if strategy in {"", "nan", "None", "unknown"}:
        return inferred

    if strategy == "degree_matched" and inferred in {
        "degree_matched_head",
        "degree_matched_tail",
    }:
        return inferred

    return strategy


def infer_seed_from_name(name):
    import re

    lower = str(name).lower()
    patterns = [
        r"seed[_-]?(\d+)",
        r"negative[_-]?sampling[_-]?seed[_-]?(\d+)",
        r"negseed[_-]?(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return np.nan


def load_test_manifest(test_dir):
    test_dir = Path(test_dir)
    manifest_path = test_dir / "manifest.csv"

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)

        rows = []
        for _, row in manifest.iterrows():
            path = Path(row["converted_file_path"])
            if not path.exists():
                path = test_dir / row["converted_file_name"]

            if not path.exists():
                continue

            out = row.to_dict()
            out["resolved_file_path"] = str(path)

            file_name_for_strategy = out.get("converted_file_name", path.name)
            out["negative_sampling_strategy"] = resolve_strategy_from_manifest_or_name(
                out.get("negative_sampling_strategy", np.nan),
                file_name_for_strategy,
            )

            rows.append(out)

        manifest = pd.DataFrame(rows)
        if len(manifest) == 0:
            raise FileNotFoundError(f"No valid test files found from manifest: {manifest_path}")

        return manifest

    files = sorted(test_dir.glob("*.csv"))
    files = [p for p in files if p.name != "manifest.csv"]

    rows = []
    for path in files:
        df = pd.read_csv(path)
        rows.append({
            "source_file_name": path.name,
            "converted_file_name": path.name,
            "converted_file_path": str(path),
            "resolved_file_path": str(path),
            "format": "gnn",
            "graph_group": "graph_present",
            "negative_sampling_strategy": resolve_strategy_from_manifest_or_name(
                np.nan,
                path.name,
            ),
            "negative_ratio": infer_negative_ratio(df),
            "negative_sampling_seed": infer_seed_from_name(path.name),
            "positive_rows": int((df["label"] == 1).sum()),
            "negative_rows": int((df["label"] == 0).sum()),
            "total_rows": int(len(df)),
        })

    if len(rows) == 0:
        raise FileNotFoundError(f"No CSV test files found under: {test_dir}")

    return pd.DataFrame(rows)


def discover_model_paths(model_root, model_paths):
    if model_paths is not None and len(model_paths) > 0:
        paths = [Path(p) for p in model_paths]
    else:
        root = Path(model_root)
        paths = sorted([
            p for p in root.iterdir()
            if p.is_dir() and (p / "config.pkl").exists() and (p / "model.pt").exists()
        ])

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No model directories found. model_root={model_root}, model_paths={model_paths}"
        )

    return paths


def load_txgnn_model(model_path, tx_data, device):
    model = TxGNN(
        data=tx_data,
        weight_bias_track=False,
        proj_name="TxGNN_MiRAGE_Eval",
        exp_name=Path(model_path).name,
        device=device,
    )
    model.load_pretrained(str(model_path))
    model.model.eval()
    return model


def predict_scores_for_test_df(model, df):
    required = {"x_idx", "relation", "y_idx", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for GNN evaluation: {missing}")

    eval_df = df.copy()
    eval_df["x_idx"] = pd.to_numeric(eval_df["x_idx"], errors="raise").astype(int)
    eval_df["y_idx"] = pd.to_numeric(eval_df["y_idx"], errors="raise").astype(int)
    eval_df["relation"] = eval_df["relation"].astype(str)

    unique_relations = eval_df["relation"].unique()
    if len(unique_relations) != 1:
        raise ValueError(
            "This evaluator currently expects one relation per fixed test file. "
            f"Found relations: {unique_relations}"
        )

    relation = unique_relations[0]

    with torch.no_grad():
        pred_dict = model.predict(eval_df[["x_idx", "relation", "y_idx"]])

    key = ("drug", relation, "disease")
    if key not in pred_dict:
        available = list(pred_dict.keys())
        raise KeyError(f"Prediction key {key} not found. Available keys: {available}")

    raw_scores = pred_dict[key].reshape(-1).detach().cpu().numpy()

    if len(raw_scores) != len(eval_df):
        raise ValueError(
            f"Prediction length mismatch: got {len(raw_scores)}, expected {len(eval_df)}"
        )

    prob_scores = sigmoid_np(raw_scores)
    return raw_scores, prob_scores


def topk_metrics(y_true, y_score, k_list=TOP_K_LIST):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    total_pos = int(y_true.sum())
    out = {}

    for k in k_list:
        kk = min(k, len(y_sorted))
        top = y_sorted[:kk]

        hits = int(top.sum())
        precision_at_k = hits / kk if kk > 0 else np.nan
        recall_at_k = hits / total_pos if total_pos > 0 else np.nan

        out[f"top{k}_hits"] = hits
        out[f"top{k}_precision"] = precision_at_k
        out[f"top{k}_recall"] = recall_at_k

        try:
            out[f"ndcg_at_{k}"] = ndcg_score(
                y_true.reshape(1, -1),
                y_score.reshape(1, -1),
                k=kk,
            )
        except Exception:
            out[f"ndcg_at_{k}"] = np.nan

    return out


def mrr_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    order = np.argsort(-y_score)
    ranked_labels = y_true[order]
    pos_positions = np.where(ranked_labels == 1)[0]

    if len(pos_positions) == 0:
        return np.nan

    return 1.0 / float(pos_positions[0] + 1)


def grouped_disease_metrics(df, y_score, k_list=TOP_K_LIST):
    """
    Disease-centric ranking metrics.

    For each disease y_idx:
      - rank candidate drugs by model score;
      - compute recall@K and NDCG@K;
      - average across diseases with at least one positive.
    """
    temp = df.copy()
    temp["score"] = y_score
    temp["label"] = temp["label"].astype(int)

    rows = []

    for disease_idx, g in temp.groupby("y_idx"):
        y_true = g["label"].values.astype(int)
        scores = g["score"].values.astype(float)

        if y_true.sum() == 0:
            continue

        row = {"y_idx": disease_idx}

        order = np.argsort(-scores)
        y_sorted = y_true[order]
        total_pos = int(y_true.sum())

        for k in k_list:
            kk = min(k, len(y_sorted))
            top = y_sorted[:kk]

            row[f"disease_top{k}_recall"] = float(top.sum() / total_pos)
            row[f"disease_top{k}_hits"] = int(top.sum())

            try:
                row[f"disease_ndcg_at_{k}"] = ndcg_score(
                    y_true.reshape(1, -1),
                    scores.reshape(1, -1),
                    k=kk,
                )
            except Exception:
                row[f"disease_ndcg_at_{k}"] = np.nan

        try:
            row["disease_ndcg"] = ndcg_score(
                y_true.reshape(1, -1),
                scores.reshape(1, -1),
            )
        except Exception:
            row["disease_ndcg"] = np.nan

        row["disease_mrr"] = mrr_score(y_true, scores)

        rows.append(row)

    if len(rows) == 0:
        out = {}
        for k in k_list:
            out[f"disease_mean_top{k}_recall"] = np.nan
            out[f"disease_mean_top{k}_hits"] = np.nan
            out[f"disease_mean_ndcg_at_{k}"] = np.nan
        out["disease_mean_ndcg"] = np.nan
        out["disease_mean_mrr"] = np.nan
        out["num_eval_diseases"] = 0
        return out

    disease_df = pd.DataFrame(rows)
    out = {"num_eval_diseases": int(len(disease_df))}

    for col in disease_df.columns:
        if col == "y_idx":
            continue
        out[f"{col.replace('disease_', 'disease_mean_')}"] = float(disease_df[col].mean())

    return out


def compute_metrics(df, y_score, threshold=0.5):
    y_true = df["label"].astype(int).values
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    out = {}

    out["num_rows"] = int(len(df))
    out["num_pos"] = int((y_true == 1).sum())
    out["num_neg"] = int((y_true == 0).sum())
    out["positive_rate"] = float(out["num_pos"] / out["num_rows"]) if out["num_rows"] > 0 else np.nan

    out["auroc"] = safe_metric(roc_auc_score, y_true, y_score)
    out["auprc"] = safe_metric(average_precision_score, y_true, y_score)
    out["average_precision"] = out["auprc"]

    out["accuracy"] = safe_metric(accuracy_score, y_true, y_pred)
    out["precision"] = safe_metric(
        lambda yt, yp: precision_score(yt, yp, zero_division=0),
        y_true,
        y_pred,
    )
    out["recall"] = safe_metric(
        lambda yt, yp: recall_score(yt, yp, zero_division=0),
        y_true,
        y_pred,
    )
    out["f1"] = safe_metric(
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        y_true,
        y_pred,
    )
    out["mcc"] = safe_metric(matthews_corrcoef, y_true, y_pred)

    try:
        out["ndcg"] = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
    except Exception:
        out["ndcg"] = np.nan

    out["mrr"] = mrr_score(y_true, y_score)

    out.update(topk_metrics(y_true, y_score, TOP_K_LIST))
    out.update(grouped_disease_metrics(df, y_score, TOP_K_LIST))

    return out


def metric_summary_columns(df):
    non_metric_cols = {
        "model_name",
        "model_path",
        "test_file",
        "test_path",
        "graph_group",
        "negative_sampling_strategy",
        "negative_ratio",
        "negative_sampling_seed",
        "format",
        "source_file_name",
        "converted_file_name",
        "converted_file_path",
        "resolved_file_path",
    }

    metric_cols = []
    for col in df.columns:
        if col in non_metric_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metric_cols.append(col)

    return metric_cols


def summarize_groups(file_metrics_df):
    group_cols = [
        "model_name",
        "negative_sampling_strategy",
        "negative_ratio",
    ]

    metric_cols = metric_summary_columns(file_metrics_df)

    rows = []

    for keys, g in file_metrics_df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        base["n_files"] = int(len(g))

        for metric in metric_cols:
            vals = pd.to_numeric(g[metric], errors="coerce")
            base[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            base[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan

        rows.append(base)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_txt_summary(summary_df, file_metrics_df, output_path):
    lines = []

    lines.append("GNN fixed test-set evaluation summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Primary reporting convention:")
    lines.append("- AUROC is primarily reported for 1:1 test sets.")
    lines.append("- AUPRC is primarily reported for 1:4 and 1:9 test sets.")
    lines.append("- TopK, NDCG, MRR, and standard classification metrics are also computed.")
    lines.append("")
    lines.append(f"Number of evaluated model/test files: {len(file_metrics_df)}")
    lines.append("")

    for _, row in summary_df.iterrows():
        model = row["model_name"]
        strategy = row["negative_sampling_strategy"]
        ratio = row["negative_ratio"]
        n_files = row["n_files"]

        lines.append("-" * 80)
        lines.append(f"Model: {model}")
        lines.append(f"Negative strategy: {strategy}")
        lines.append(f"Negative ratio: 1:{ratio}")
        lines.append(f"Number of files: {n_files}")

        if ratio == 1:
            lines.append(
                f"AUROC: {row.get('auroc_mean', np.nan):.6f} "
                f"± {row.get('auroc_std', np.nan):.6f}"
            )

        if ratio != 1:
            lines.append(
                f"AUPRC: {row.get('auprc_mean', np.nan):.6f} "
                f"± {row.get('auprc_std', np.nan):.6f}"
            )

        lines.append(
            f"NDCG: {row.get('ndcg_mean', np.nan):.6f} "
            f"± {row.get('ndcg_std', np.nan):.6f}"
        )
        lines.append(
            f"MRR: {row.get('mrr_mean', np.nan):.6f} "
            f"± {row.get('mrr_std', np.nan):.6f}"
        )

        for k in TOP_K_LIST:
            lines.append(
                f"Top{k} recall: "
                f"{row.get(f'top{k}_recall_mean', np.nan):.6f} "
                f"± {row.get(f'top{k}_recall_std', np.nan):.6f}"
            )
            lines.append(
                f"Top{k} precision: "
                f"{row.get(f'top{k}_precision_mean', np.nan):.6f} "
                f"± {row.get(f'top{k}_precision_std', np.nan):.6f}"
            )
            lines.append(
                f"NDCG@{k}: "
                f"{row.get(f'ndcg_at_{k}_mean', np.nan):.6f} "
                f"± {row.get(f'ndcg_at_{k}_std', np.nan):.6f}"
            )

        lines.append(
            f"Accuracy: {row.get('accuracy_mean', np.nan):.6f} "
            f"± {row.get('accuracy_std', np.nan):.6f}"
        )
        lines.append(
            f"F1: {row.get('f1_mean', np.nan):.6f} "
            f"± {row.get('f1_std', np.nan):.6f}"
        )
        lines.append(
            f"MCC: {row.get('mcc_mean', np.nan):.6f} "
            f"± {row.get('mcc_std', np.nan):.6f}"
        )

    lines.append("-" * 80)
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.test_dir is None:
        test_dir = Path(args.output_dir) / "fixed_test_sets_gnn"
    else:
        test_dir = Path(args.test_dir)

    print("Loading fixed GNN test manifest...")
    test_manifest = load_test_manifest(test_dir)
    print(f"Found {len(test_manifest)} fixed GNN test files.")

    print("\nTest files by negative sampling strategy:")
    print(
        test_manifest
        .groupby(["negative_sampling_strategy", "negative_ratio"], dropna=False)
        .size()
        .reset_index(name="n_files")
        .sort_values(["negative_sampling_strategy", "negative_ratio"])
        .to_string(index=False)
    )

    print("\nDiscovering model checkpoints...")
    model_paths = discover_model_paths(args.model_root, args.model_paths)
    print(f"Found {len(model_paths)} model checkpoints:")
    for p in model_paths:
        print(f"- {p}")

    print("Preparing TxData...")
    tx_data = TxData(data_folder_path=args.data_folder)
    tx_data.prepare_split(
        split=args.split,
        seed=args.seed,
        no_kg=False,
    )

    all_metric_rows = []

    for model_path in model_paths:
        model_path = Path(model_path)
        model_name = model_path.name

        print("=" * 80)
        print(f"Evaluating model: {model_name}")
        print(f"Model path: {model_path}")
        print("=" * 80)

        model = load_txgnn_model(model_path, tx_data, args.device)

        for _, test_row in test_manifest.iterrows():
            test_path = Path(test_row["resolved_file_path"])
            print(f"Evaluating test file: {test_path.name}")

            df = pd.read_csv(test_path)

            required_cols = {
                "x_type", "x_id", "relation", "y_type", "y_id",
                "x_idx", "y_idx", "label",
            }
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"{test_path} missing required columns: {missing}")

            df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

            raw_scores, prob_scores = predict_scores_for_test_df(model, df)

            metrics = compute_metrics(df, prob_scores, threshold=args.threshold)

            negative_ratio = test_row.get("negative_ratio", infer_negative_ratio(df))
            strategy = resolve_strategy_from_manifest_or_name(
                test_row.get("negative_sampling_strategy", np.nan),
                test_path.name,
            )
            neg_seed = test_row.get("negative_sampling_seed", infer_seed_from_name(test_path.name))

            metric_row = {
                "model_name": model_name,
                "model_path": str(model_path),
                "test_file": test_path.name,
                "test_path": str(test_path),
                "graph_group": test_row.get("graph_group", "graph_present"),
                "negative_sampling_strategy": strategy,
                "negative_ratio": negative_ratio,
                "negative_sampling_seed": neg_seed,
                "format": test_row.get("format", "gnn"),
                "source_file_name": test_row.get("source_file_name", np.nan),
                "converted_file_name": test_row.get("converted_file_name", test_path.name),
            }

            metric_row.update(metrics)
            all_metric_rows.append(metric_row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    file_metrics_df = pd.DataFrame(all_metric_rows)
    file_metrics_df = file_metrics_df.sort_values(
        [
            "model_name",
            "negative_sampling_strategy",
            "negative_ratio",
            "negative_sampling_seed",
            "test_file",
        ]
    ).reset_index(drop=True)

    summary_df = summarize_groups(file_metrics_df)

    file_metrics_path = result_dir / "gnn_fixed_test_file_metrics.csv"
    summary_csv_path = result_dir / "gnn_fixed_test_group_summary.csv"
    summary_txt_path = result_dir / "gnn_fixed_test_summary.txt"

    file_metrics_df.to_csv(file_metrics_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    write_txt_summary(summary_df, file_metrics_df, summary_txt_path)

    print("=" * 80)
    print("Saved evaluation results:")
    print(f"- Per-file metrics: {file_metrics_path}")
    print(f"- Group summary CSV: {summary_csv_path}")
    print(f"- Group summary TXT: {summary_txt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()