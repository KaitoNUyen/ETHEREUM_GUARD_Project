from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

from src.app.components import (
    SYSTEM_THEME_CSS,
    render_branch_breakdown_bar,
    render_feature_attribution_list,
    render_interactive_graph_widget,
    render_threat_summary_card,
    render_top_navbar,
    render_transaction_summary_card,
)
from src.app.engine import get_engine, is_transaction_hash
from src.schemas import GRAPH_TOPO_COLS, TABULAR_TX_COLS, TEMPORAL_STAT_COLS

st.set_page_config(
    page_title="Ethereum Guard — Deep Learning Transaction & Wallet Anomaly",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(SYSTEM_THEME_CSS, unsafe_allow_html=True)
st.markdown(render_top_navbar(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_engine():
    return get_engine()


PRESET_OPTIONS = [
    "-- Custom Manual Input --",
    "0xfadc2e6f30354a6623b17e78527eecd77659b5ac51a9132ce521d7fdb67a2bb9 (Phishing TxHash · Edge Deep Scorer)",
    "0x5b8d362d45210ab128854f18c317d2781d6b98bfbff6b6f1266a75fe38a5d11a (Normal TxHash · WBTC Interaction)",
    "0x096f226b05777751047d1caa8e06f350fa7c4677 (Known Phishing · 9 Edges)",
    "0x7cd5b32f1c7a3917525efaaf3b8bf286252430b6 (Known Phishing · 7 Edges)",
    "0xe2facbebbe29a94a17590f9a56e721fb7d610795 (Known Phishing · 5 Edges)",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599 (Verified Normal · WBTC · 60 Edges)",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984 (Verified Normal · UNI · 60 Edges)",
]

DEFAULT_ADDRESS = "0x096f226b05777751047d1caa8e06f350fa7c4677"

if "wallet_address_input" not in st.session_state:
    st.session_state["wallet_address_input"] = DEFAULT_ADDRESS


def _on_preset_change() -> None:
    choice = st.session_state.get("preset_selector", "")
    if choice and "--" not in choice:
        st.session_state["wallet_address_input"] = choice.split(" ")[0].strip()


# ==========================================
# Sidebar: Deep Learning Engine Telemetry & Presets
# ==========================================

with st.sidebar:
    st.markdown("### ⚙️ Deep Learning Engine")
    engine = get_cached_engine()

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Engine Status", "ONLINE" if engine.is_loaded else "OFFLINE")
    with col_stat2:
        st.metric("Primary GNN", "TG-AE + EdgeNet")

    model_label_choice = st.radio(
        "Account Detector:",
        [
            "Modular TG-AE (Tri-Modal Deep GNN)",
            "Isolation Forest (Fast Baseline)",
        ],
        index=0,
    )
    selected_model_type = (
        "tgae" if "TG-AE" in model_label_choice else "isolation_forest"
    )

    st.markdown("---")
    st.markdown("### 🎯 Quick Inspect Presets")
    st.caption("Supports BOTH 66-char `tx_hash` & 42-char Wallet `0x...`:")

    st.selectbox(
        "Choose transaction or wallet:",
        PRESET_OPTIONS,
        key="preset_selector",
        on_change=_on_preset_change,
    )

    st.markdown("---")
    st.markdown("### 🧠 Deep Architecture Specs")
    st.markdown(
        """
        - **Transaction Engine:** `DeepTransactionScorer` (`z_u || z_v || Shock`)
        - **Account Engine:** `Modular TG-AE` (`best_tgae.pt`)
          - `TabularMLP` (12 Flow Features)
          - `TransformerConv` (6 Topo + 5D Edge Attr)
          - `GRU + Bahdanau Attention` (`10×8` Windows)
        - **Baselines:** `XGBoost`, `DOMINANT`, `GAT`, `Isolation Forest`
        - **Live Sync:** Etherscan API V2 (`txlist` + `getTransactionByHash`)
        """
    )


def _render_benchmark_section() -> None:
    results_dir = ROOT_DIR / "artifacts" / "results"
    tx_bench_csv = results_dir / "transaction_benchmark.csv"
    bench_csv = results_dir / "benchmark_comparison.csv"
    abl_csv = results_dir / "ablation_study.csv"
    err_csv = results_dir / "error_cases.csv"

    st.markdown("##### ⚡ 1. Transaction-Level Anomaly Detection Benchmark (Edge-Level Deep Learning)")
    if tx_bench_csv.exists():
        tx_df = pd.read_csv(tx_bench_csv)
        st.dataframe(tx_df, width="stretch")

    st.markdown("##### 🏆 2. Account-Level Model Benchmark (ML vs. DOMINANT / GAT vs. Proposed Modular TG-AE)")
    if bench_csv.exists():
        bench_df = pd.read_csv(bench_csv)
        display_cols = [
            c
            for c in (
                "model",
                "protocol",
                "pr_auc",
                "roc_auc",
                "f1_binary",
                "precision",
                "recall",
                "fpr_at_90_tpr",
                "num_eval_nodes",
                "inference_latency_ms_per_wallet",
            )
            if c in bench_df.columns
        ]
        st.dataframe(bench_df[display_cols], width="stretch")

    st.markdown("##### 🧪 3. 7-Configuration Modality Ablation Matrix (`Abl-A` → `Abl-G`)")
    if abl_csv.exists():
        st.dataframe(pd.read_csv(abl_csv), width="stretch")

    st.markdown("##### 🔎 4. Error Analysis Case Studies (`False Positives` / `False Negatives`)")
    if err_csv.exists():
        err_df = pd.read_csv(err_csv)
        st.caption(
            f"Total misclassified cases logged across models: {len(err_df)} "
            f"(FP: {(err_df['error_type'] == 'FALSE_POSITIVE').sum()}, "
            f"FN: {(err_df['error_type'] == 'FALSE_NEGATIVE').sum()})"
        )
        st.dataframe(err_df.head(15), width="stretch")


# ==========================================
# Main Workspace: Unified Transaction & Wallet Scanner
# ==========================================

st.markdown("#### 🔍 Ethereum Transaction (`tx_hash`) & Wallet (`0x...`) Deep Anomaly Scanner")

with st.form("wallet_scan_form", border=False):
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        target_query = st.text_input(
            "Enter 66-char Transaction Hash (0x...) or 42-char Wallet Address (0x...)",
            key="wallet_address_input",
            placeholder="0x... (Transaction Hash 66 chars OR Wallet Address 42 chars)",
            label_visibility="collapsed",
        )
    with btn_col:
        st.form_submit_button("RUN SCAN", width="stretch", type="primary")

if target_query:
    clean_query = target_query.strip().lower()
    api_base_url = os.getenv("ETHEREUM_GUARD_API_URL", "").strip().rstrip("/")

    try:
        # ------------------------------------------------------------------
        # MODE A: DIRECT TRANSACTION-LEVEL (`tx_hash` — 66 characters)
        # ------------------------------------------------------------------
        if is_transaction_hash(clean_query):
            with st.spinner("Extracting sender/receiver TG-AE embeddings and scoring transaction edge..."):
                if api_base_url:
                    import httpx

                    with httpx.Client(timeout=20.0) as http_client:
                        r_tx = http_client.get(f"{api_base_url}/predict/tx/{clean_query}")
                        if r_tx.status_code == 422:
                            raise ValueError(r_tx.json().get("detail", "Invalid transaction hash"))
                        r_tx.raise_for_status()
                        tx_dict = r_tx.json()
                else:
                    tx_obj = engine.predict_transaction(clean_query)
                    tx_dict = tx_obj.__dict__

            st.markdown(render_transaction_summary_card(tx_dict), unsafe_allow_html=True)

            tab_tx_ego, tab_tx_bench = st.tabs(
                [
                    "🕸️ Sender Ego-Network Context",
                    "📊 Dual-Level Benchmark & Ablation",
                ]
            )
            with tab_tx_ego:
                sender_addr = tx_dict.get("from_address", "")
                if sender_addr:
                    ego_obj = engine.get_ego_network(sender_addr, max_edges=50)
                    graph_html = render_interactive_graph_widget(ego_obj.__dict__)
                    if hasattr(st, "iframe"):
                        st.iframe(graph_html, height=510)
                    else:
                        st_components.html(graph_html, height=510, scrolling=False)
            with tab_tx_bench:
                _render_benchmark_section()

        # ------------------------------------------------------------------
        # MODE B: ACCOUNT-LEVEL + PER-TRANSACTION RANKING (`address` — 42 chars)
        # ------------------------------------------------------------------
        else:
            with st.spinner("Running Tri-Modal Deep GNN (Subgraph + 10×8 Sequence) & scoring wallet transactions..."):
                if api_base_url:
                    import httpx

                    with httpx.Client(timeout=20.0) as http_client:
                        r_pred = http_client.get(
                            f"{api_base_url}/predict/{clean_query}",
                            params={"model": selected_model_type},
                        )
                        if r_pred.status_code == 422:
                            raise ValueError(r_pred.json().get("detail", "Invalid Ethereum address"))
                        r_pred.raise_for_status()
                        pred_dict = r_pred.json()

                        r_graph = http_client.get(
                            f"{api_base_url}/graph/{clean_query}",
                            params={"max_edges": 60},
                        )
                        r_graph.raise_for_status()
                        graph_dict = r_graph.json()

                        r_txs = http_client.get(
                            f"{api_base_url}/transactions/{clean_query}",
                            params={"max_txs": 15},
                        )
                        ranked_txs = r_txs.json().get("transactions", []) if r_txs.status_code == 200 else []
                else:
                    pred_obj = engine.predict(clean_query, model_type=selected_model_type)
                    graph_obj = engine.get_ego_network(clean_query, max_edges=60)
                    ranked_txs = engine.score_wallet_transactions(clean_query, max_txs=15)
                    pred_dict = pred_obj.__dict__
                    graph_dict = graph_obj.__dict__

            st.markdown(render_threat_summary_card(pred_dict), unsafe_allow_html=True)
            branch_contribs = (pred_dict.get("metadata") or {}).get("branch_contributions")
            if branch_contribs:
                st.markdown(render_branch_breakdown_bar(branch_contribs), unsafe_allow_html=True)

            tab_forensics, tab_graph, tab_raw, tab_bench = st.tabs(
                [
                    "🔬 XAI & Per-Transaction Anomaly",
                    "🕸️ Ego-Network Topology",
                    "📋 Full 25-Feature Registry",
                    "📊 Benchmark & Error Analysis",
                ]
            )

            with tab_forensics:
                if ranked_txs:
                    st.markdown("##### ⚡ Deep Learning Per-Transaction Anomaly Ranking (`DeepTransactionScorer`)")
                    st.caption(
                        "Each individual transaction of this wallet is scored using Sender/Receiver TG-AE embeddings + Value Shock:"
                    )
                    st.dataframe(ranked_txs, width="stretch")
                st.markdown(
                    render_feature_attribution_list(pred_dict["feature_contributions"]),
                    unsafe_allow_html=True,
                )

            with tab_graph:
                col_net1, col_net2, col_net3, col_net4 = st.columns(4)
                with col_net1:
                    st.metric("Total Observed Volume", f"{graph_dict['total_volume_eth']:.4f} ETH")
                with col_net2:
                    st.metric("Inbound Transactions", graph_dict["in_degree"])
                with col_net3:
                    st.metric("Outbound Transactions", graph_dict["out_degree"])
                with col_net4:
                    st.metric("Unique Counterparties", graph_dict["unique_counterparties"])

                graph_html = render_interactive_graph_widget(graph_dict)
                if hasattr(st, "iframe"):
                    st.iframe(graph_html, height=510)
                else:
                    st_components.html(graph_html, height=510, scrolling=False)

                if graph_dict["edges"]:
                    st.markdown("##### 📜 Recent Direct Transactions")
                    edge_df = [
                        {
                            "Direction": e["direction"],
                            "Counterparty": e["target"] if e["direction"] == "OUT" else e["source"],
                            "Value (ETH)": e["value_eth"],
                            "Timestamp": e["timestamp"],
                            "Tx Hash": (e["tx_hash"][:18] + "...") if e["tx_hash"] else "-",
                        }
                        for e in graph_dict["edges"][:15]
                    ]
                    st.dataframe(edge_df, width="stretch")

            with tab_raw:
                st.markdown("##### 📐 Multi-Modal Feature Vector (25 Raw Unscaled Features)")
                raw_feat_dict = pred_dict["raw_features"]
                col_tab1, col_tab2, col_tab3 = st.columns(3)

                with col_tab1:
                    st.markdown("**12 Tabular Features**")
                    for col in TABULAR_TX_COLS:
                        st.text(f"{col}: {raw_feat_dict.get(col, 0.0)}")

                with col_tab2:
                    st.markdown("**6 Graph Topology Features**")
                    for col in GRAPH_TOPO_COLS:
                        st.text(f"{col}: {raw_feat_dict.get(col, 0.0)}")

                with col_tab3:
                    st.markdown("**7 Temporal Statistics**")
                    for col in TEMPORAL_STAT_COLS:
                        st.text(f"{col}: {raw_feat_dict.get(col, 0.0)}")

            with tab_bench:
                _render_benchmark_section()

    except ValueError as exc:
        st.error(str(exc))
    except KeyError as exc:
        st.warning(f"Target `{clean_query}` was not observed locally and live lookup failed: {exc}")
    except Exception as exc:
        st.error(f"Inference pipeline encountered an unexpected error: {exc}")
