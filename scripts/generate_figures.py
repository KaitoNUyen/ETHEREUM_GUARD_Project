"""
Generate publication-ready static figures (.png) in docs/figures/ from real measured
experiment artifacts (benchmark_comparison.csv, ablation_study.csv, error_cases.csv, best_tgae.pt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def generate_all_figures(
    results_dir: Path | None = None,
    checkpoints_dir: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    results_dir = results_dir or (ROOT_DIR / "artifacts" / "results")
    checkpoints_dir = checkpoints_dir or (ROOT_DIR / "artifacts" / "checkpoints")
    out_dir = out_dir or (ROOT_DIR / "docs" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#1e293b",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#f8fafc",
            "text.color": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "grid.color": "#334155",
            "font.size": 10,
        }
    )

    saved_paths: list[Path] = []

    # 1. Benchmark Comparison Chart (Cumulative Test vs Strict Unseen)
    bench_csv = results_dir / "benchmark_comparison.csv"
    if bench_csv.exists():
        df_b = pd.read_csv(bench_csv).dropna(subset=["pr_auc", "f1_binary"])
        df_b = df_b[df_b["protocol"].isin(["cumulative_test", "strict_unseen_from_train"])]
        if not df_b.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
            for ax, proto, title in zip(
                axes,
                ["cumulative_test", "strict_unseen_from_train"],
                ["Cumulative Test Split (n=211)", "Strict Unseen Holdout (n=23)"],
            ):
                sub = df_b[df_b["protocol"] == proto].copy()
                short_names = [
                    m.replace("Proposed: ", "")
                    .replace("Abl-G: Full Proposed ", "")
                    .replace("BL-1: ", "")
                    .replace("BL-2: ", "")
                    .replace("DL-BL1: ", "")
                    .replace("DL-BL2: ", "")
                    .replace(" (Deep Graph AE)", "\n(Graph AE)")
                    .replace(" (Graph Attention Net)", "\n(GAT)")
                    .replace(" (best_tgae.pt)", "\n(Checkpoint)")
                    .replace(" (Log1p-Seq Fixed)", "\n(Log1p-Fixed)")
                    for m in sub["model"]
                ]
                x = np.arange(len(short_names))
                width = 0.26
                b1 = ax.bar(x - width, sub["pr_auc"], width, label="PR-AUC", color="#38bdf8")
                b2 = ax.bar(x, sub["f1_binary"], width, label="F1-Score", color="#10b981")
                b3 = ax.bar(
                    x + width,
                    1.0 - sub["fpr_at_90_tpr"],
                    width,
                    label="1 - FPR@90%TPR",
                    color="#a855f7",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(short_names, fontsize=9)
                ax.set_ylim(0.0, 1.12)
                ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
                ax.grid(axis="y", linestyle="--", alpha=0.45)
                for bars in (b1, b2, b3):
                    for rect in bars:
                        h = rect.get_height()
                        ax.annotate(
                            f"{h:.3f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                        )
            axes[0].set_ylabel("Score (Higher is Better)")
            axes[1].legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569")
            fig.suptitle(
                "Ethereum Guard — Model Benchmark Comparison on Held-Out Test Graph",
                fontsize=13,
                fontweight="bold",
                y=0.99,
            )
            fig.tight_layout()
            p1 = out_dir / "benchmark_comparison.png"
            fig.savefig(p1, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(p1)

    # 2. 7-Configuration Ablation Study (Abl-A -> Abl-G)
    abl_csv = results_dir / "ablation_study.csv"
    if abl_csv.exists():
        df_a = pd.read_csv(abl_csv)
        if not df_a.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
            cum = df_a[df_a["protocol"] == "cumulative_test"].sort_values("ablation_code")
            uns = df_a[df_a["protocol"] == "strict_unseen_from_train"].sort_values("ablation_code")

            labels = [
                f"{r['ablation_code']}\n({'T' if r['use_tabular'] else '-'}"
                f"{'G' if r['use_graph'] else '-'}"
                f"{'S' if r['use_temporal'] else '-'})"
                for _, r in cum.iterrows()
            ]
            x = np.arange(len(labels))
            w = 0.34

            axes[0].bar(x - w / 2, cum["pr_auc"], w, label="Cumulative PR-AUC", color="#38bdf8")
            axes[0].bar(x + w / 2, uns["pr_auc"], w, label="Strict Unseen PR-AUC", color="#f59e0b")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels)
            axes[0].set_ylim(0.70, 1.03)
            axes[0].set_title("PR-AUC Across All 7 Modality Combinations", fontweight="bold")
            axes[0].grid(axis="y", linestyle="--", alpha=0.45)
            axes[0].legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569")

            axes[1].bar(x - w / 2, cum["f1_binary"], w, label="Cumulative F1", color="#10b981")
            axes[1].bar(x + w / 2, uns["f1_binary"], w, label="Strict Unseen F1", color="#fb7185")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels)
            axes[1].set_ylim(0.82, 1.02)
            axes[1].set_title("F1-Score Across All 7 Modality Combinations", fontweight="bold")
            axes[1].grid(axis="y", linestyle="--", alpha=0.45)
            axes[1].legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569")

            fig.suptitle(
                "7-Configuration Ablation Study (T=Tabular, G=Graph Transformer, S=Temporal GRU-Attn)",
                fontsize=12.5,
                fontweight="bold",
                y=0.99,
            )
            fig.tight_layout()
            p2 = out_dir / "ablation_study.png"
            fig.savefig(p2, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(p2)

    # 3. Modular TG-AE Training Convergence Curve from Checkpoint History
    ckpt_path = checkpoints_dir / "best_tgae.pt"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        history = payload.get("state", {}).get("training_history_", [])
        if history:
            df_h = pd.DataFrame(history)
            fig, ax1 = plt.subplots(figsize=(9.5, 4.8), dpi=180)
            epochs = df_h["epoch"]
            ax1.plot(
                epochs,
                df_h["train_loss"],
                marker="o",
                markersize=4,
                linewidth=2.0,
                color="#38bdf8",
                label="Train Composite Loss",
            )
            ax1.set_xlabel("Training Epoch")
            ax1.set_ylabel("Composite Loss", color="#38bdf8")
            ax1.grid(True, linestyle="--", alpha=0.4)

            if "val_pr_auc" in df_h.columns:
                ax2 = ax1.twinx()
                ax2.plot(
                    epochs,
                    df_h["val_pr_auc"],
                    marker="s",
                    markersize=4,
                    linewidth=2.0,
                    color="#10b981",
                    label="Validation PR-AUC",
                )
                if "val_f1" in df_h.columns:
                    ax2.plot(
                        epochs,
                        df_h["val_f1"],
                        linestyle="--",
                        linewidth=1.8,
                        color="#f59e0b",
                        label="Validation F1",
                    )
                ax2.set_ylabel("Validation Metric", color="#10b981")
                ax2.set_ylim(0.85, 1.01)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc="center right",
                    facecolor="#0f172a",
                    edgecolor="#475569",
                )

            plt.title(
                "Modular TG-AE (best_tgae.pt) — Training Loss & Validation PR-AUC Convergence",
                fontweight="bold",
            )
            fig.tight_layout()
            p3 = out_dir / "tgae_convergence.png"
            fig.savefig(p3, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(p3)

    # 4. Error Analysis Breakdown (False Positives vs False Negatives)
    err_csv = results_dir / "error_cases.csv"
    if err_csv.exists():
        df_e = pd.read_csv(err_csv)
        if not df_e.empty:
            grp = (
                df_e.groupby(["model", "error_type"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["FALSE_POSITIVE", "FALSE_NEGATIVE"], fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=180)
            models_short = [
                m.replace("Abl-G: Full Proposed ", "").replace("BL-1: ", "") for m in grp.index
            ]
            y_pos = np.arange(len(models_short))
            ax.barh(
                y_pos - 0.18,
                grp["FALSE_POSITIVE"],
                height=0.35,
                color="#f59e0b",
                label="False Positives (Whitelisted Hubs)",
            )
            ax.barh(
                y_pos + 0.18,
                grp["FALSE_NEGATIVE"],
                height=0.35,
                color="#fb7185",
                label="False Negatives (Missed Phishing)",
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models_short)
            ax.set_xlabel("Misclassified Wallet Count on Test Split (n=211)")
            ax.set_title(
                "False Positive Reduction: Isolation Forest (54 FP) vs. Modular TG-AE (3-4 FP)",
                fontweight="bold",
            )
            ax.grid(axis="x", linestyle="--", alpha=0.45)
            ax.legend(loc="lower right", facecolor="#0f172a", edgecolor="#475569")
            fig.tight_layout()
            p4 = out_dir / "error_breakdown.png"
            fig.savefig(p4, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(p4)

    # 5. Transaction-Level (Edge-Level) Deep Learning Benchmark
    tx_csv = results_dir / "transaction_benchmark.csv"
    if tx_csv.exists():
        df_tx = pd.read_csv(tx_csv)
        if not df_tx.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=180)
            for ax, proto, title in zip(
                axes,
                ["cumulative_test_txs", "strict_unseen_txs"],
                ["Held-Out Test Transactions (n=25,252)", "Strict Unseen Post-Train Txs (n=10,674)"],
            ):
                sub = df_tx[df_tx["protocol"] == proto].copy()
                if sub.empty:
                    continue
                names = [
                    m.replace("Tx-BL1: ", "")
                    .replace("Tx-DL1: ", "")
                    .replace("Proposed: ", "")
                    .replace(" (TG-AE + Shock)", "\n(TG-AE + Shock)")
                    .replace(" Edge Scorer", "\nEdge")
                    for m in sub["model"]
                ]
                x = np.arange(len(names))
                w = 0.26
                ax.bar(x - w, sub["roc_auc"], w, label="ROC-AUC", color="#38bdf8")
                ax.bar(x, sub["pr_auc"], w, label="PR-AUC (1:100 Imbalance)", color="#10b981")
                ax.bar(x + w, sub["precision_at_50"], w, label="Precision@50", color="#f59e0b")
                ax.set_xticks(x)
                ax.set_xticklabels(names, fontsize=8.5)
                ax.set_ylim(0.0, 1.12)
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.grid(axis="y", linestyle="--", alpha=0.45)
            axes[1].legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569")
            fig.suptitle(
                "Transaction-Level Deep Learning Anomaly Detection (Edge-Level Benchmark)",
                fontsize=13,
                fontweight="bold",
                y=0.99,
            )
            fig.tight_layout()
            p5 = out_dir / "transaction_benchmark.png"
            fig.savefig(p5, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(p5)

    return saved_paths


if __name__ == "__main__":
    generated = generate_all_figures()
    for p in generated:
        print(f"[OK] Generated figure: {p} ({p.stat().st_size:,} bytes)")
