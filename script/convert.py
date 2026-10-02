import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from txgnn import TxData


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PyG/global-index GNN test CSVs to TxGNN local-index evaluation CSVs."
    )

    parser.add_argument(
        "--input-dir",
        default="./data/benchmark/GNN_new",
        help="Directory containing global-index test CSVs: relation,x_index,y_index,label.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/benchmark/GNN_new_txgnn",
        help="Output directory for TxGNN local-index CSVs.",
    )
    parser.add_argument(
        "--nodes-path",
        default="./data/kg/node.csv",
        help=(
            "PrimeKG node table with node_index,node_id,node_type,node_name. "
            "Use the same node-index system that generated x_index/y_index."
        ),
    )
    parser.add_argument(
        "--data-folder",
        default="./data/kg",
        help="TxGNN data folder containing kg.csv / kg_directed.csv.",
    )
    parser.add_argument("--split", default="full_graph")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, fail on any unmapped row or non drug->disease row.",
    )

    return parser.parse_args()


def normalize_id(x):
    """
    Normalize IDs for robust matching between PrimeKG node.csv and TxGNN df.
    Handles:
      - MONDO:0001234 / MONDO_0001234 -> 1234
      - 1234.0 -> 1234
      - DB00123 stays DB00123
    """
    if pd.isna(x):
        return None

    s = str(x).strip()
    if s == "":
        return None

    if s.startswith("MONDO:"):
        s = s.replace("MONDO:", "")
    if s.startswith("MONDO_"):
        s = s.replace("MONDO_", "")

    # Preserve DrugBank IDs
    if s.upper().startswith("DB"):
        return s.upper()

    # Numeric IDs
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def id_aliases(x):
    """
    TxGNN sometimes stores numeric disease IDs as '1234.0' because of convert2str().
    PrimeKG node.csv may store them as '1234'. Generate both aliases.
    """
    s = normalize_id(x)
    out = set()

    if s is None:
        return out

    out.add(s)

    # Add numeric aliases
    try:
        f = float(s)
        out.add(str(f))
        if f.is_integer():
            out.add(str(int(f)))
    except Exception:
        pass

    return out


def build_txgnn_local_maps(tx_df):
    """
    Build:
      local_maps[node_type][node_id_alias] = local_idx
    using TxGNN's prepared df, which contains type-local x_idx/y_idx.
    """
    local_maps = {}

    node_types = set(tx_df["x_type"].astype(str)).union(set(tx_df["y_type"].astype(str)))
    for ntype in node_types:
        local_maps[ntype] = {}

    for _, row in tx_df.iterrows():
        for type_col, id_col, idx_col in [
            ("x_type", "x_id", "x_idx"),
            ("y_type", "y_id", "y_idx"),
        ]:
            ntype = str(row[type_col])
            idx = int(row[idx_col])

            for alias in id_aliases(row[id_col]):
                local_maps[ntype][alias] = idx

    return local_maps


def infer_strategy_from_name(name):
    lower = name.lower()
    if "degree" in lower:
        return "degree_matched"
    if "normal" in lower:
        return "normal"
    if "random" in lower:
        return "random"
    return "unknown"


def infer_ratio_from_name(name, df):
    lower = name.lower()
    m = re.search(r"ratio[_-]?(\d+)", lower)
    if m:
        return int(m.group(1))

    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    if pos == 0:
        return np.nan
    ratio = neg / pos
    if abs(ratio - round(ratio)) < 1e-9:
        return int(round(ratio))
    return ratio


def infer_seed_from_name(name):
    lower = name.lower()
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


def convert_one_file(path, nodes_df, local_maps, tx_train_df, tx_valid_df, strict=False):
    df = pd.read_csv(path)

    required = {"relation", "x_index", "y_index", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["x_index"] = pd.to_numeric(df["x_index"], errors="raise").astype(int)
    df["y_index"] = pd.to_numeric(df["y_index"], errors="raise").astype(int)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["relation"] = df["relation"].astype(str)

    node_index_to_type = dict(zip(nodes_df["node_index"], nodes_df["node_type"].astype(str)))
    node_index_to_id = dict(zip(nodes_df["node_index"], nodes_df["node_id"]))
    if "node_name" in nodes_df.columns:
        node_index_to_name = dict(zip(nodes_df["node_index"], nodes_df["node_name"]))
    else:
        node_index_to_name = dict(zip(nodes_df["node_index"], nodes_df["node_id"]))

    rows = []
    skipped = []

    for i, row in df.iterrows():
        x_global = int(row["x_index"])
        y_global = int(row["y_index"])

        x_type = node_index_to_type.get(x_global)
        y_type = node_index_to_type.get(y_global)
        x_id_raw = node_index_to_id.get(x_global)
        y_id_raw = node_index_to_id.get(y_global)

        if x_type is None or y_type is None:
            skipped.append((i, "missing_global_node", x_global, y_global, x_type, y_type, x_id_raw, y_id_raw))
            continue

        # This benchmark should be drug -> disease.
        if not (x_type == "drug" and y_type == "disease"):
            skipped.append((i, "non_drug_disease", x_global, y_global, x_type, y_type, x_id_raw, y_id_raw))
            if strict:
                continue

        x_idx = None
        x_id = None
        for alias in id_aliases(x_id_raw):
            if alias in local_maps.get(x_type, {}):
                x_idx = local_maps[x_type][alias]
                x_id = alias
                break

        y_idx = None
        y_id = None
        for alias in id_aliases(y_id_raw):
            if alias in local_maps.get(y_type, {}):
                y_idx = local_maps[y_type][alias]
                y_id = alias
                break

        if x_idx is None or y_idx is None:
            skipped.append((i, "missing_txgnn_local_idx", x_global, y_global, x_type, y_type, x_id_raw, y_id_raw))
            continue

        rows.append({
            "x_type": x_type,
            "x_id": x_id,
            "relation": str(row["relation"]),
            "y_type": y_type,
            "y_id": y_id,
            "x_idx": int(x_idx),
            "y_idx": int(y_idx),
            "label": int(row["label"]),
            "x_name": node_index_to_name.get(x_global, x_id_raw),
            "y_name": node_index_to_name.get(y_global, y_id_raw),
            "original_x_index": x_global,
            "original_y_index": y_global,
        })

    out = pd.DataFrame(rows)

    if strict and len(skipped) > 0:
        skipped_df = pd.DataFrame(
            skipped,
            columns=[
                "row_id",
                "reason",
                "x_index",
                "y_index",
                "x_type",
                "y_type",
                "x_id_raw",
                "y_id_raw",
            ],
        )
        raise ValueError(f"{path} has skipped rows under --strict:\n{skipped_df.head(20)}")

    overlap_report = {
        "txgnn_train_overlap_pos": 0,
        "txgnn_train_overlap_neg": 0,
        "txgnn_valid_overlap_pos": 0,
        "txgnn_valid_overlap_neg": 0,
    }

    if len(out) > 0:
        # Report overlap with current TxGNN train/valid positive edges.
        # This does not drop rows; it only warns you.
        def overlap_count(base_df, label_value):
            temp = out[out["label"] == label_value]
            if len(temp) == 0:
                return 0

            merged = temp.merge(
                base_df[
                    ["x_type", "relation", "y_type", "x_idx", "y_idx"]
                ].drop_duplicates(),
                on=["x_type", "relation", "y_type", "x_idx", "y_idx"],
                how="inner",
            )
            return int(len(merged))

        overlap_report["txgnn_train_overlap_pos"] = overlap_count(tx_train_df, 1)
        overlap_report["txgnn_train_overlap_neg"] = overlap_count(tx_train_df, 0)
        overlap_report["txgnn_valid_overlap_pos"] = overlap_count(tx_valid_df, 1)
        overlap_report["txgnn_valid_overlap_neg"] = overlap_count(tx_valid_df, 0)

    skipped_df = None
    if len(skipped) > 0:
        skipped_df = pd.DataFrame(
            skipped,
            columns=[
                "row_id",
                "reason",
                "x_index",
                "y_index",
                "x_type",
                "y_type",
                "x_id_raw",
                "y_id_raw",
            ],
        )

    return out, skipped_df, overlap_report


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Converting global-index GNN tests to TxGNN local-index tests")
    print("=" * 80)
    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Nodes path:   {args.nodes_path}")
    print(f"Data folder:  {args.data_folder}")
    print(f"Split/seed:   {args.split}/{args.seed}")
    print("=" * 80)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    nodes_df = pd.read_csv(args.nodes_path)
    nodes_required = {"node_index", "node_id", "node_type"}
    missing_nodes = nodes_required - set(nodes_df.columns)
    if missing_nodes:
        raise ValueError(f"{args.nodes_path} missing columns: {missing_nodes}")

    nodes_df["node_index"] = pd.to_numeric(nodes_df["node_index"], errors="raise").astype(int)

    print("Preparing TxData...")
    tx_data = TxData(data_folder_path=args.data_folder)
    tx_data.prepare_split(split=args.split, seed=args.seed, no_kg=False)

    tx_df = tx_data.df.copy()
    tx_train_df = tx_data.df_train.copy()
    tx_valid_df = tx_data.df_valid.copy()

    for d in [tx_df, tx_train_df, tx_valid_df]:
        d["x_idx"] = pd.to_numeric(d["x_idx"], errors="raise").astype(int)
        d["y_idx"] = pd.to_numeric(d["y_idx"], errors="raise").astype(int)
        d["x_type"] = d["x_type"].astype(str)
        d["y_type"] = d["y_type"].astype(str)
        d["relation"] = d["relation"].astype(str)

    local_maps = build_txgnn_local_maps(tx_df)

    files = sorted([p for p in input_dir.glob("*.csv") if p.name != "manifest.csv"])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    manifest_rows = []
    skipped_all = []

    for path in files:
        print("-" * 80)
        print(f"Converting: {path.name}")

        converted, skipped_df, overlap_report = convert_one_file(
            path=path,
            nodes_df=nodes_df,
            local_maps=local_maps,
            tx_train_df=tx_train_df,
            tx_valid_df=tx_valid_df,
            strict=args.strict,
        )

        if skipped_df is not None and len(skipped_df) > 0:
            skipped_path = output_dir / f"{path.stem}__skipped.csv"
            skipped_df.to_csv(skipped_path, index=False)
            skipped_all.append(skipped_df.assign(source_file=path.name))
            print(f"WARNING: skipped {len(skipped_df)} rows. Saved: {skipped_path}")

        out_name = path.name.replace("__global.csv", "__gnn.csv")
        if out_name == path.name:
            out_name = path.stem + "__gnn.csv"

        out_path = output_dir / out_name
        converted.to_csv(out_path, index=False)

        pos = int((converted["label"] == 1).sum()) if len(converted) else 0
        neg = int((converted["label"] == 0).sum()) if len(converted) else 0
        ratio = infer_ratio_from_name(path.name, converted) if len(converted) else np.nan
        strategy = infer_strategy_from_name(path.name)
        seed = infer_seed_from_name(path.name)

        print(f"Saved: {out_path}")
        print(f"Rows: total={len(converted)}, pos={pos}, neg={neg}, ratio=1:{ratio}")
        print(f"Strategy={strategy}, seed={seed}")
        print("TxGNN train/valid overlap report:")
        print(overlap_report)

        if overlap_report["txgnn_train_overlap_pos"] > 0 or overlap_report["txgnn_valid_overlap_pos"] > 0:
            print(
                "WARNING: Some positive rows overlap with current TxGNN train/valid edges. "
                "If this benchmark is intended to be strictly unseen for TxGNN, regenerate "
                "the global tests with TxGNN train/valid edges included in the ban set."
            )

        manifest_rows.append({
            "source_file_name": path.name,
            "converted_file_name": out_name,
            "converted_file_path": str(out_path),
            "resolved_file_path": str(out_path),
            "format": "gnn",
            "graph_group": "pyg_style",
            "negative_sampling_strategy": strategy,
            "negative_ratio": ratio,
            "negative_sampling_seed": seed,
            "positive_rows": pos,
            "negative_rows": neg,
            "total_rows": int(len(converted)),
            **overlap_report,
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    if skipped_all:
        skipped_all_df = pd.concat(skipped_all, ignore_index=True)
        skipped_all_path = output_dir / "all_skipped_rows.csv"
        skipped_all_df.to_csv(skipped_all_path, index=False)
        print(f"All skipped rows saved to: {skipped_all_path}")

    print("=" * 80)
    print("Done.")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)
    print(
        manifest[
            [
                "negative_sampling_strategy",
                "negative_ratio",
                "negative_sampling_seed",
                "positive_rows",
                "negative_rows",
                "total_rows",
                "txgnn_train_overlap_pos",
                "txgnn_valid_overlap_pos",
                "converted_file_name",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()