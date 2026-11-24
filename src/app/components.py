"""
Ethereum Guard - Reusable Frontend UI Components & Design System.
Production-ready HTML/CSS rendering engine for risk metrics, feature attribution, and interactive network graphs.
"""

from __future__ import annotations

import html
import json
from typing import Any


def _safe_json_for_script(obj: Any) -> str:
    """Serialize JSON safely for inline <script> blocks (neutralizes </script> breakout)."""
    return json.dumps(obj).replace("</", "<\\/")

# ==========================================
# Core CSS Design System (Fintech Dark Theme)
# ==========================================

SYSTEM_THEME_CSS = """
<style>
/* Reset & Typography */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary: #090d16;
    --bg-card: #0f172a;
    --bg-card-hover: #1e293b;
    --border-color: #1e293b;
    --border-highlight: #334155;
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    
    --accent-blue: #38bdf8;
    --accent-indigo: #6366f1;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
}

.stApp {
    background-color: var(--bg-primary);
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
}

/* Glassmorphism Cards */
.eg-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.4) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
}

.eg-card:hover {
    border-color: var(--border-highlight);
    box-shadow: 0 14px 35px -10px rgba(0, 0, 0, 0.65);
}

/* Threat Badge Styles */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.risk-badge-critical {
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}

.risk-badge-high {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.risk-badge-medium {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
}

.risk-badge-low {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

/* Metric Pulse Value */
.metric-value-huge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.75rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

/* Feature Item Row */
.feature-item-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(51, 65, 85, 0.4);
    margin-bottom: 10px;
    transition: background 0.2s ease;
}

.feature-item-row:hover {
    background: rgba(30, 41, 59, 0.6);
}

.mono-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    background: rgba(0, 0, 0, 0.35);
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--border-highlight);
    border-radius: 3px;
}
</style>
"""


def _clean_html(html: str) -> str:
    """Strip leading indentation and blank lines so Streamlit Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in html.splitlines() if line.strip())


# ==========================================
# Component 1: Top Navigation Bar
# ==========================================

def render_top_navbar() -> str:
    """Header bar with project branding and live telemetry badge."""
    return _clean_html("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 0 24px 0; border-bottom: 1px solid #1e293b; margin-bottom: 28px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%); width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(56, 189, 248, 0.35);">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <path d="m9 12 2 2 4-4"/>
                </svg>
            </div>
            <div>
                <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(to right, #f8fafc, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ETHEREUM GUARD <span style="font-size: 0.75rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 2px 8px; border-radius: 6px; font-weight: 700; border: 1px solid rgba(56, 189, 248, 0.3); margin-left: 6px;">v1.0-PROD</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">
                    Deep Learning Transaction & Account Anomaly Detection over Ethereum Graphs
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 18px;">
            <div style="display: flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); padding: 6px 12px; border-radius: 20px;">
                <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 10px #10b981;"></span>
                <span style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #34d399; font-weight: 600;">ANOMALY ENGINE: ONLINE</span>
            </div>
        </div>
    </div>
    """)


# ==========================================
# Component 2: Threat Summary Assessment Card
# ==========================================

def render_threat_summary_card(pred: dict[str, Any]) -> str:
    """Renders the executive threat level, score gauge, and verification status."""
    score = float(pred["anomaly_score"])
    threshold = float(pred["threshold"])
    risk = html.escape(str(pred["risk_level"]))
    verified_label = pred["verified_label"]
    label_text = html.escape(str(pred["label_text"]))
    latency = html.escape(str(pred["inference_time_ms"]))
    evaluated_at = html.escape(str(pred["evaluated_at"]))
    source = str(pred.get("source", "OFFLINE_SNAPSHOT"))
    metadata = pred.get("metadata", {}) or {}

    badge_class = {
        "CRITICAL": "risk-badge-critical",
        "HIGH": "risk-badge-high",
        "MEDIUM": "risk-badge-medium",
        "LOW": "risk-badge-low",
    }.get(str(pred["risk_level"]), "risk-badge-low")

    score_color = {
        "CRITICAL": "#fb7185",
        "HIGH": "#fbbf24",
        "MEDIUM": "#38bdf8",
        "LOW": "#34d399",
    }.get(str(pred["risk_level"]), "#34d399")

    progress_percent = min(100.0, max(0.0, score * 100.0))
    threshold_percent = threshold * 100.0

    # Source pill badge
    if source == "LIVE_ONCHAIN":
        txs_cnt = int(metadata.get("transactions_analyzed", 0))
        source_badge = f"""
        <span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
            <span style="width: 7px; height: 7px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></span>
            LIVE ON-CHAIN ({txs_cnt} TXs)
        </span>
        """
    elif source == "CACHE_HIT":
        source_badge = """
        <span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
            ⚡ L1 CACHE HIT
        </span>
        """
    else:
        source_badge = """
        <span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
            INDEXED GRAPH (1.79M)
        </span>
        """

    notice_html = ""
    if metadata.get("notice"):
        safe_notice = html.escape(str(metadata["notice"]))
        notice_html = f"""
        <div style="font-size: 0.78rem; color: #94a3b8; background: rgba(15, 23, 42, 0.6); padding: 6px 12px; border-radius: 6px; border-left: 3px solid #38bdf8; margin-top: 12px; font-family: 'JetBrains Mono', monospace;">
            ℹ️ {safe_notice}
        </div>
        """
    if metadata.get("warning"):
        safe_warn = html.escape(str(metadata["warning"]))
        notice_html += f"""
        <div style="font-size: 0.78rem; color: #fbbf24; background: rgba(245, 158, 11, 0.08); padding: 6px 12px; border-radius: 6px; border-left: 3px solid #f59e0b; margin-top: 8px; font-family: 'JetBrains Mono', monospace;">
            ⚠️ {safe_warn}
        </div>
        """

    return _clean_html(f"""
    <div class="eg-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
            <div>
                <div class="metric-label">Account Risk Diagnosis</div>
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <span class="risk-badge {badge_class}">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                            <circle cx="12" cy="12" r="10"/>
                        </svg>
                        {risk} THREAT LEVEL
                    </span>
                    <span style="font-size: 0.85rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                        {label_text}
                    </span>
                    {source_badge}
                </div>
            </div>
            <div style="text-align: right;">
                <div class="metric-label">Inference Latency</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: #38bdf8; font-weight: 700;">
                    {latency} ms
                </div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: center; margin-bottom: 24px;">
            <div>
                <div class="metric-label">Composite Anomaly Score</div>
                <div class="metric-value-huge" style="color: {score_color};">
                    {score:.4f}
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                    Decision Threshold: <code style="color: #cbd5e1;">{threshold:.4f}</code> (Val-Calibrated)
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Normal Baseline (0.00)</span>
                    <span style="color: #fb7185;">Critical Scam (1.00)</span>
                </div>
                <div style="position: relative; height: 12px; background: rgba(51, 65, 85, 0.4); border-radius: 999px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.05);">
                    <div style="width: {progress_percent}%; height: 100%; background: linear-gradient(90deg, #10b981 0%, #f59e0b 60%, #f43f5e 100%); border-radius: 999px; transition: width 0.6s ease;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748b; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
                    <span>Cutoff: τ*={threshold:.2f}</span>
                    <span>Evaluated: {evaluated_at[:19]}</span>
                </div>
            </div>
        </div>
        {notice_html}
    </div>
    """)


# ==========================================
# Component 3: Explainability Breakdown
# ==========================================

def render_feature_attribution_list(contributions: list[dict[str, Any]]) -> str:
    """Renders Top-K feature deviations for audit and forensics."""
    rows_html = ""
    for item in contributions:
        feat = html.escape(str(item["feature"]))
        cat = html.escape(str(item["category"]))
        raw_val = float(item["raw_value"])
        z_score = float(item["z_score"])
        direction = str(item["direction"])

        dir_color = "#fb7185" if direction == "HIGH" else "#38bdf8"
        dir_sign = "+" if z_score >= 0 else ""

        rows_html += f"""
        <div class="feature-item-row">
            <div>
                <div style="font-weight: 600; font-size: 0.92rem; color: #f1f5f9;">
                    {feat}
                </div>
                <div style="font-size: 0.75rem; color: #64748b;">
                    {cat}
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="text-align: right;">
                    <div class="mono-chip" style="color: #cbd5e1;">
                        Raw: {raw_val:,.2f}
                    </div>
                </div>
                <div style="min-width: 90px; text-align: right;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: {dir_color};">
                        {dir_sign}{z_score:.2f} σ
                    </span>
                </div>
            </div>
        </div>
        """

    return _clean_html(f"""
    <div class="eg-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <div class="metric-label">XAI Forensics</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">
                    Top Feature Deviations (vs. Verified Normal Baseline)
                </div>
            </div>
            <span style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.2);">
                Z-Score Deviation
            </span>
        </div>
        <div>
            {rows_html}
        </div>
    </div>
    """)


def render_branch_breakdown_bar(branch_contributions: dict[str, float]) -> str:
    """Renders Modular TG-AE tri-modal error contribution bar (Tabular / Graph / Temporal)."""
    feat_pct = max(0.0, float(branch_contributions.get("feature_pct", 0.0)))
    graph_pct = max(0.0, float(branch_contributions.get("graph_pct", 0.0)))
    temp_pct = max(0.0, float(branch_contributions.get("temporal_pct", 0.0)))
    total = feat_pct + graph_pct + temp_pct
    if total > 0:
        feat_pct, graph_pct, temp_pct = (
            100.0 * feat_pct / total,
            100.0 * graph_pct / total,
            100.0 * temp_pct / total,
        )

    return _clean_html(f"""
    <div class="eg-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <div class="metric-label">Modular TG-AE Tri-Modal Attribution</div>
                <div style="font-size: 1.02rem; font-weight: 700; color: #f8fafc;">
                    Calibrated Reconstruction Error Breakdown (Ẽ_feat · Ẽ_graph · Ẽ_temp)
                </div>
            </div>
            <span style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #34d399; background: rgba(16, 185, 129, 0.12); padding: 3px 10px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.3);">
                TRI-MODAL GNN
            </span>
        </div>
        <div style="display: flex; height: 14px; border-radius: 999px; overflow: hidden; background: rgba(30, 41, 59, 0.7); margin-bottom: 10px;">
            <div style="width: {feat_pct:.2f}%; background: #38bdf8;" title="Tabular Flow"></div>
            <div style="width: {graph_pct:.2f}%; background: #a855f7;" title="Graph Topology"></div>
            <div style="width: {temp_pct:.2f}%; background: #f59e0b;" title="Temporal Sequence"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #38bdf8;">● Tabular Flow: {feat_pct:.1f}%</span>
            <span style="color: #a855f7;">● Graph Structure: {graph_pct:.1f}%</span>
            <span style="color: #f59e0b;">● Temporal Dynamics: {temp_pct:.1f}%</span>
        </div>
    </div>
    """)


def render_transaction_summary_card(tx_pred: dict[str, Any]) -> str:
    """Renders Deep Learning Transaction-Level (`tx_hash`) risk card, sender->receiver flow, and 3-factor attribution."""
    score = float(tx_pred["anomaly_score"])
    threshold = float(tx_pred["threshold"])
    risk = html.escape(str(tx_pred["risk_level"]))
    tx_hash = html.escape(str(tx_pred["tx_hash"]))
    src_addr = html.escape(str(tx_pred["from_address"]))
    dst_addr = html.escape(str(tx_pred.get("to_address") or "Contract Creation"))
    val_eth = float(tx_pred.get("value_eth", 0.0))
    method_lbl = html.escape(str(tx_pred.get("method_label", "Native ETH Transfer")))
    latency = html.escape(str(tx_pred.get("inference_time_ms", 0.0)))
    s_score = float(tx_pred.get("sender_score", 0.0))
    r_score = float(tx_pred.get("receiver_score", 0.0))
    comp = tx_pred.get("component_breakdown") or {}
    s_pct = float(comp.get("sender_risk_pct", 33.3))
    r_pct = float(comp.get("receiver_risk_pct", 33.3))
    d_pct = float(comp.get("tx_dynamics_pct", 33.4))

    badge_class = {
        "CRITICAL": "risk-badge-critical",
        "HIGH": "risk-badge-high",
        "MEDIUM": "risk-badge-medium",
        "LOW": "risk-badge-low",
    }.get(str(tx_pred["risk_level"]), "risk-badge-low")

    score_color = {
        "CRITICAL": "#fb7185",
        "HIGH": "#fbbf24",
        "MEDIUM": "#38bdf8",
        "LOW": "#34d399",
    }.get(str(tx_pred["risk_level"]), "#34d399")

    sig_rows = ""
    for sig in tx_pred.get("forensic_signals", []):
        s_name = html.escape(str(sig.get("signal", "")))
        s_val = html.escape(str(sig.get("value", "")))
        s_stat = html.escape(str(sig.get("status", "")))
        s_det = html.escape(str(sig.get("detail", "")))
        stat_col = "#fb7185" if s_stat in {"ELEVATED", "SPIKE", "WATCH"} else "#34d399"
        sig_rows += f"""
        <div class="feature-item-row">
            <div>
                <div style="font-weight: 600; font-size: 0.9rem; color: #f1f5f9;">{s_name}</div>
                <div style="font-size: 0.75rem; color: #64748b;">{s_det}</div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span class="mono-chip" style="color: #cbd5e1;">{s_val}</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: {stat_col};">{s_stat}</span>
            </div>
        </div>
        """

    return _clean_html(f"""
    <div class="eg-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px;">
            <div>
                <div class="metric-label">Deep Transaction Anomaly Diagnosis (Edge-Level GNN)</div>
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <span class="risk-badge {badge_class}">{risk} TX RISK</span>
                    <span class="mono-chip" style="color: #38bdf8;">{method_lbl}</span>
                    <span class="mono-chip" style="color: #cbd5e1;">{val_eth:,.6f} ETH</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #94a3b8; margin-top: 8px;">
                    TxHash: {tx_hash}
                </div>
            </div>
            <div style="text-align: right;">
                <div class="metric-label">Inference Latency</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; color: #38bdf8; font-weight: 700;">{latency} ms</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1.4fr; gap: 22px; align-items: center; margin-bottom: 20px;">
            <div>
                <div class="metric-label">Transaction Anomaly Score</div>
                <div class="metric-value-huge" style="color: {score_color};">{score:.4f}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                    Edge Decision Threshold: <code style="color: #cbd5e1;">τ*={threshold:.4f}</code> (DeepTransactionScorer)
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.65); padding: 14px; border-radius: 12px; border: 1px solid #1e293b;">
                <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 8px; font-weight: 600;">DIRECTED TRANSACTION FLOW (TG-AE EMBEDDINGS z_u → z_v)</div>
                <div style="display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                    <div>
                        <div style="color: #cbd5e1;">FROM: {src_addr[:12]}...{src_addr[-6:]}</div>
                        <div style="color: #38bdf8; font-size: 0.74rem;">Sender TG-AE: {s_score:.4f}</div>
                    </div>
                    <div style="color: #f59e0b; font-weight: 800;">──►</div>
                    <div style="text-align: right;">
                        <div style="color: #cbd5e1;">TO: {dst_addr[:12]}...{dst_addr[-6:] if len(dst_addr) > 18 else ''}</div>
                        <div style="color: #a855f7; font-size: 0.74rem;">Receiver TG-AE: {r_score:.4f}</div>
                    </div>
                </div>
            </div>
        </div>

        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px; font-weight: 600;">Edge Attribution Breakdown (Sender z_u · Receiver z_v · Behavioral Shock)</div>
            <div style="display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: rgba(30, 41, 59, 0.7); margin-bottom: 8px;">
                <div style="width: {s_pct:.2f}%; background: #38bdf8;"></div>
                <div style="width: {r_pct:.2f}%; background: #a855f7;"></div>
                <div style="width: {d_pct:.2f}%; background: #f59e0b;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;">
                <span style="color: #38bdf8;">● Sender Profile: {s_pct:.1f}%</span>
                <span style="color: #a855f7;">● Receiver Profile: {r_pct:.1f}%</span>
                <span style="color: #f59e0b;">● Tx Shock & Value Dynamics: {d_pct:.1f}%</span>
            </div>
        </div>
        <div>{sig_rows}</div>
    </div>
    """)


# ==========================================
# Component 4: Interactive Ego-Graph Visualizer (D3.js embedded)
# ==========================================

def render_interactive_graph_widget(graph_data: dict[str, Any]) -> str:
    """
    Renders an embedded, self-contained SVG/D3.js force-directed ego network.
    Does not require external python plotting packages, zero-dependency client-side physics.
    """
    nodes_json = _safe_json_for_script(graph_data.get("nodes", []))
    edges_json = _safe_json_for_script(graph_data.get("edges", []))
    center_json = _safe_json_for_script(str(graph_data.get("center_address", "")))

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: 'Plus Jakarta Sans', sans-serif;
                color: #f8fafc;
                overflow: hidden;
            }}
            .eg-graph-wrapper {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.5) 100%);
                border: 1px solid #1e293b;
                border-radius: 16px;
                padding: 16px;
            }}
            .metric-label {{
                color: #94a3b8;
                font-size: 0.78rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 4px;
            }}
        </style>
    </head>
    <body>
    <div class="eg-graph-wrapper">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding: 4px 8px;">
            <div>
                <div class="metric-label">Graph Neighborhood</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">
                    1-Hop Ego Network Topology
                </div>
            </div>
            <div style="display: flex; gap: 12px; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;">
                <span style="color: #ef4444;">● Target Account</span>
                <span style="color: #3b82f6;">● Inbound (Sender)</span>
                <span style="color: #10b981;">● Outbound (Receiver)</span>
            </div>
        </div>

        <div id="ego-graph-container" style="width: 100%; height: 410px; background: rgba(10, 15, 29, 0.95); border-radius: 12px; border: 1px solid rgba(51, 65, 85, 0.5); position: relative; overflow: hidden;">
            <svg id="ego-svg" style="width: 100%; height: 100%;"></svg>
        </div>
    </div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    (function() {{
        const nodes = {nodes_json};
        const edges = {edges_json};
        const centerAddr = {center_json};

        const container = document.getElementById("ego-graph-container");
        const width = container.clientWidth || 750;
        const height = 410;

        // Offline-resilient fallback renderer when D3 CDN is unreachable (air-gapped / defense room)
        if (typeof window.d3 === "undefined") {{
            const svgEl = document.getElementById("ego-svg");
            svgEl.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
            svgEl.innerHTML = "";
            const NS = "http://www.w3.org/2000/svg";
            const gEl = document.createElementNS(NS, "g");
            svgEl.appendChild(gEl);

            const cx = width / 2, cy = height / 2;
            const radius = Math.min(width, height) * 0.34;
            const posMap = {{}};
            const peers = nodes.filter(n => n.id !== centerAddr);
            nodes.forEach((n) => {{
                if (n.id === centerAddr) {{
                    posMap[n.id] = {{ x: cx, y: cy }};
                }}
            }});
            peers.forEach((n, idx) => {{
                const angle = (2 * Math.PI * idx) / Math.max(1, peers.length);
                posMap[n.id] = {{
                    x: cx + radius * Math.cos(angle),
                    y: cy + radius * Math.sin(angle)
                }};
            }});

            edges.forEach((e) => {{
                const s = posMap[e.source] || {{ x: cx, y: cy }};
                const t = posMap[e.target] || {{ x: cx, y: cy }};
                const line = document.createElementNS(NS, "line");
                line.setAttribute("x1", s.x);
                line.setAttribute("y1", s.y);
                line.setAttribute("x2", t.x);
                line.setAttribute("y2", t.y);
                line.setAttribute("stroke", "#475569");
                line.setAttribute("stroke-opacity", "0.65");
                line.setAttribute("stroke-width", String(Math.max(1.5, Math.min(6, (e.value_eth || 0) * 2 + 1))));
                gEl.appendChild(line);
            }});

            nodes.forEach((n) => {{
                const p = posMap[n.id] || {{ x: cx, y: cy }};
                const c = document.createElementNS(NS, "circle");
                c.setAttribute("cx", p.x);
                c.setAttribute("cy", p.y);
                c.setAttribute("r", String(n.size || 16));
                c.setAttribute("fill", n.color || "#38bdf8");
                c.setAttribute("stroke", "#0f172a");
                c.setAttribute("stroke-width", "2.5");
                const title = document.createElementNS(NS, "title");
                title.textContent = n.id;
                c.appendChild(title);
                gEl.appendChild(c);

                const txt = document.createElementNS(NS, "text");
                txt.setAttribute("x", p.x + 18);
                txt.setAttribute("y", p.y + 4);
                txt.setAttribute("font-size", "10px");
                txt.setAttribute("font-family", "monospace");
                txt.setAttribute("fill", "#94a3b8");
                txt.textContent = n.label || "";
                gEl.appendChild(txt);
            }});
            return;
        }}

        const svg = d3.select("#ego-svg")
            .attr("viewBox", [0, 0, width, height]);

        svg.selectAll("*").remove();

        const g = svg.append("g");
        svg.call(d3.zoom()
            .scaleExtent([0.3, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }}));

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(edges).id(d => d.id).distance(90))
            .force("charge", d3.forceManyBody().strength(-240))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(25));

        const link = g.append("g")
            .attr("stroke", "#475569")
            .attr("stroke-opacity", 0.6)
            .selectAll("line")
            .data(edges)
            .join("line")
            .attr("stroke-width", d => Math.max(1.5, Math.min(6, d.value_eth * 2 + 1)));

        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("r", d => d.size || 16)
            .attr("fill", d => d.color || "#38bdf8")
            .attr("stroke", "#0f172a")
            .attr("stroke-width", 2.5)
            .style("cursor", "pointer")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("title")
            .text(d => d.id);

        const label = g.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .text(d => d.label)
            .attr("font-size", "10px")
            .attr("font-family", "JetBrains Mono, monospace")
            .attr("fill", "#94a3b8")
            .attr("dx", 18)
            .attr("dy", 4);

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    }})();
    </script>
    </body>
    </html>
    """
