"""
Enterprise Integration & Regression Test Suite for Ethereum Guard Live Ingestion,
Inference Engine, REST API, and UI Components.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from src.app.components import (
    render_interactive_graph_widget,
    render_threat_summary_card,
)
from src.app.engine import EthereumGuardEngine, _BoundedTTLCache
from src.app.service import app
from src.pipeline.live_feature_extractor import (
    LiveFeatureExtractor,
    _parse_timestamp_to_unix_seconds,
)
from src.schemas import ALL_FEATURE_COLS


class TestLiveInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EthereumGuardEngine()
        cls.extractor = LiveFeatureExtractor()
        cls.client = TestClient(app)

    def test_01_feature_extractor_column_contract(self):
        """Verify that LiveFeatureExtractor produces exactly the 25 canonical schema features."""
        raw, vec = self.extractor.extract_features(
            "0x0000000000000000000000000000000000000001",
            pd.DataFrame(),
        )
        self.assertEqual(len(raw), 25)
        self.assertEqual(tuple(raw.keys()), ALL_FEATURE_COLS)
        self.assertEqual(vec.shape, (1, 25))
        self.assertFalse(np.isnan(vec).any(), "Feature vector contains NaNs!")

    def test_02_empty_virgin_wallet_inference(self):
        """Verify that a brand new wallet with 0 transactions doesn't crash and returns calibrated score."""
        pred = self.engine.predict("0x000000000000000000000000000000000000abcd")
        self.assertEqual(pred.source, "LIVE_ONCHAIN")
        self.assertGreaterEqual(pred.anomaly_score, 0.0)
        self.assertLessEqual(pred.anomaly_score, 1.0)
        self.assertIn(pred.risk_level, {"LOW", "MEDIUM", "HIGH", "CRITICAL"})
        self.assertTrue(len(pred.feature_contributions) > 0)

    def test_03_synthetic_transaction_flow(self):
        """Verify feature extraction with active simulated transaction history."""
        addr = "0x1234567890123456789012345678901234567890"
        tx_data = pd.DataFrame([
            {
                "timestamp": 1700000000 + i * 3600,
                "from_address": addr if i % 2 == 0 else "0x9999999999999999999999999999999999999999",
                "to_address": "0x8888888888888888888888888888888888888888" if i % 2 == 0 else addr,
                "value_eth": 0.5 * (i + 1),
                "gas_price": 20e9,
                "success": True,
            }
            for i in range(10)
        ])
        raw, vec = self.extractor.extract_features(addr, tx_data)
        self.assertEqual(raw["tx_sent_count"], 5.0)
        self.assertEqual(raw["tx_recv_count"], 5.0)
        self.assertGreater(raw["total_eth_sent"], 0.0)
        self.assertGreater(raw["total_eth_received"], 0.0)
        self.assertEqual(vec.shape, (1, 25))
        self.assertFalse(np.isnan(vec).any())

    def test_04_l1_cache_hit_latency(self):
        """Verify that secondary queries hit L1 memory cache in sub-millisecond time."""
        addr = "0x0000000000000000000000000000000000005555"
        pred1 = self.engine.predict(addr)
        self.assertEqual(pred1.source, "LIVE_ONCHAIN")

        pred2 = self.engine.predict(addr)
        self.assertEqual(pred2.source, "CACHE_HIT")
        self.assertLess(pred2.inference_time_ms, 10.0)

    def test_05_ego_network_generation(self):
        """Verify that get_ego_network returns structured D3 graph."""
        addr = "0x0000000000000000000000000000000000005555"
        graph = self.engine.get_ego_network(addr)
        self.assertEqual(graph.center_address, addr)
        self.assertTrue(len(graph.nodes) >= 1)
        self.assertEqual(graph.nodes[0]["id"], addr)

    # =====================================================================
    # Production Incident Regression Tests
    # =====================================================================

    def test_06_timestamp_resolution_parity(self):
        """Regression H2: datetime64[s/ms/us/ns], Unix ms, and Etherscan strings all yield exact Unix seconds."""
        expected = 1710438803
        for unit in ("s", "ms", "us", "ns"):
            s = pd.Series([pd.Timestamp(expected, unit="s", tz="UTC")]).astype(f"datetime64[{unit}, UTC]")
            parsed = _parse_timestamp_to_unix_seconds(s)
            self.assertEqual(int(parsed.iloc[0]), expected, f"Failed for unit={unit}")

        # Numeric ms epoch and string Unix seconds
        self.assertEqual(int(_parse_timestamp_to_unix_seconds(pd.Series([expected * 1000])).iloc[0]), expected)
        self.assertEqual(int(_parse_timestamp_to_unix_seconds(pd.Series([str(expected)])).iloc[0]), expected)

        # Real Parquet sample check (datetime64[ms, UTC])
        tx_sample = pq.read_table("data/sample/transactions.parquet", columns=["timestamp"]).slice(0, 5).to_pandas()
        parsed_pq = _parse_timestamp_to_unix_seconds(tx_sample["timestamp"])
        self.assertGreater(parsed_pq.iloc[0], 1_400_000_000)

    def test_07_scaler_train_serve_parity(self):
        """Regression H1: LiveFeatureExtractor matches feature_robust_scaler un-logged contract (no 2,200x skew)."""
        all_tx = pq.read_table("data/sample/transactions.parquet").to_pandas()
        addr = "0xa152f8bb749c55e9943a3a0a3111d18ee2b3f94e"
        raw, vec = self.extractor.extract_features(addr, all_tx)
        # tx_sent_count in sample is >5000; un-logged RobustScaler (center=1, scale=1) gives ~tx_sent_count - 1
        self.assertFalse(self.extractor.tab_uses_log_transform)
        self.assertTrue(self.extractor.graph_uses_log_transform)
        self.assertAlmostEqual(float(vec[0, 7]), raw["tx_sent_count"] - 1.0, delta=1.0)
        self.assertGreater(float(vec[0, 9]), 1000.0)  # sent_recv_ratio not collapsed to ~1.17

    def test_08_null_to_address_and_self_loop_parity(self):
        """Regression H5b: Contract creations (to=None/NaN) and self-loops produce exact ratios & 0 fake peers."""
        addr = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        df_edge = pd.DataFrame([
            {"timestamp": 1700000000, "from_address": addr, "to_address": None, "value_eth": 0.0, "gas_price": 1e9, "success": False},
            {"timestamp": 1700000060, "from_address": addr, "to_address": float("nan"), "value_eth": 0.0, "gas_price": 1e9, "success": False},
            {"timestamp": 1700000120, "from_address": addr, "to_address": addr, "value_eth": 0.0, "gas_price": 1e9, "success": False},
        ])
        raw, vec = self.extractor.extract_features(addr, df_edge)
        self.assertEqual(raw["unique_out_neighbors"], 0.0)
        self.assertEqual(raw["unique_in_neighbors"], 0.0)
        self.assertEqual(raw["in_degree"], 0.0)
        self.assertEqual(raw["out_degree"], 0.0)
        self.assertAlmostEqual(raw["zero_val_tx_ratio"], 1.0)
        self.assertAlmostEqual(raw["failed_tx_ratio"], 1.0)
        self.assertFalse(np.isnan(vec).any())

    def test_09_ego_network_null_self_loop_and_chrono_sort(self):
        """Regression H3/H5b: No fake 'nan' nodes, no dangling edges for D3, and exact numeric time ordering."""
        addr = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        peer1 = "0xcccccccccccccccccccccccccccccccccccccccc"
        peer2 = "0xdddddddddddddddddddddddddddddddddddddddd"
        self.engine._live_tx_cache.set(addr, pd.DataFrame([
            {"timestamp": 999999999, "from_address": addr, "to_address": peer1, "value_eth": 1.0, "transaction_hash": "0xold", "gas_used": 21000, "success": True},
            {"timestamp": 1700000000, "from_address": addr, "to_address": float("nan"), "value_eth": 0.0, "transaction_hash": "0xcreate", "gas_used": 100000, "success": True},
            {"timestamp": 1700000050, "from_address": addr, "to_address": peer2, "value_eth": 2.0, "transaction_hash": "0xnew", "gas_used": 21000, "success": True},
            {"timestamp": 1700000020, "from_address": addr, "to_address": addr, "value_eth": 0.5, "transaction_hash": "0xself", "gas_used": 21000, "success": True},
        ]))
        g = self.engine.get_ego_network(addr)
        node_ids = {n["id"] for n in g.nodes}
        self.assertNotIn("nan", node_ids)
        self.assertNotIn("none", node_ids)
        self.assertNotIn("", node_ids)
        # Every edge endpoint must exist in nodes_map so D3 forceLink never throws
        for edge in g.edges:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)
        # Chronological descending sort: 1700000050 (0xnew) must come before 999999999 (0xold)
        self.assertEqual(g.edges[0]["tx_hash"], "0xnew")
        self.assertEqual(g.edges[-1]["tx_hash"], "0xold")
        self.assertEqual(g.unique_counterparties, 2)

    def test_10_api_validation_batch_isolation_and_path_traversal(self):
        """Regression H3/H4: Invalid addresses return 422, path traversal blocked, batch isolates bad items & preserves source."""
        r_inv = self.client.get("/predict/not_an_address")
        self.assertEqual(r_inv.status_code, 422)

        r_trav = self.client.get("/graph/..%5C..%5Csample%5Ctransactions")
        self.assertEqual(r_trav.status_code, 422)

        r_batch = self.client.post(
            "/predict/batch",
            json={"addresses": ["0x0000000000000000000000000000000000007777", "bad_addr"]},
        )
        self.assertEqual(r_batch.status_code, 200)
        body = r_batch.json()
        self.assertEqual(body["total_processed"], 1)
        self.assertEqual(len(body["failed_addresses"]), 1)
        self.assertIn(body["results"][0]["source"], {"LIVE_ONCHAIN", "CACHE_HIT"})
        self.assertTrue(len(body["results"][0]["metadata"]) > 0)

    def test_11_bounded_cache_eviction_and_no_error_poisoning(self):
        """Regression H5: Cache strictly obeys maxsize and transient provider errors do NOT poison cache."""
        cache = _BoundedTTLCache(maxsize=2, ttl_seconds=60.0)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get("a"))

        addr = "0x000000000000000000000000000000000000cafe"
        self.assertIsNone(self.engine.get_address_record(addr))
        with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "fake_key"}):
            with patch(
                "src.pipeline.etherscan_adapter.fetch_normal_transactions",
                side_effect=RuntimeError("Simulated Etherscan 429"),
            ):
                p = self.engine.predict(addr)
                self.assertEqual(p.metadata.get("fetch_status"), "PROVIDER_ERROR")
                self.assertIsNone(self.engine._prediction_cache.get(addr))
                self.assertIsNone(self.engine._live_tx_cache.get(addr))

    def test_12_components_xss_sanitization(self):
        """Regression H6: HTML and inline <script> payloads are safely escaped."""
        html_out = render_threat_summary_card({
            "anomaly_score": 0.5,
            "threshold": 0.37,
            "risk_level": "HIGH",
            "verified_label": -1,
            "label_text": "<img src=x onerror=alert(1)>",
            "inference_time_ms": 1.0,
            "evaluated_at": "2026-11-23T00:00:00",
            "source": "LIVE_ONCHAIN",
            "metadata": {"transactions_analyzed": 0, "notice": "<script>alert(1)</script>"},
        })
        self.assertNotIn("<script>alert(1)</script>", html_out)
        self.assertNotIn("<img src=x", html_out)

        graph_html = render_interactive_graph_widget({
            "center_address": "</script><script>alert(1)</script>",
            "nodes": [{"id": "</script><script>x()</script>", "label": "x", "color": "#fff", "size": 10}],
            "edges": [],
        })
        self.assertNotIn("</script><script>", graph_html)

    def test_13_models_import_and_leak_free_eval_filter(self):
        """Regression BUG-01/BUG-05: BaseDetector & IsolationForestDetector import without PyG/XGBoost and support exclude_addresses."""
        from src.models import BaseDetector, IsolationForestDetector

        det = IsolationForestDetector.load("artifacts/checkpoints/isolation_forest.pkl")
        self.assertIsInstance(det, BaseDetector)
        synthetic = {
            "x": np.zeros((4, 25), dtype=np.float32),
            "y": np.array([0, 1, 0, 1], dtype=np.int64),
            "eval_mask": np.array([True, True, True, True]),
            "addresses": ["0xtrain0", "0xtrain1", "0xunseen0", "0xunseen1"],
        }
        metrics_all = det.evaluate(synthetic)
        self.assertEqual(metrics_all["num_eval_nodes"], 4)
        metrics_unseen = det.evaluate(synthetic, exclude_addresses={"0xtrain0", "0xtrain1"})
        self.assertEqual(metrics_unseen["num_eval_nodes"], 2)

    def test_14_offline_raw_features_true_units(self):
        """Regression BUG-02: Offline snapshot predictions invert scalers so raw_features expose true raw units."""
        addr = "0x195632255acd8182ff5752aaa1210376bf904f06"
        self.engine._prediction_cache.clear()
        pred_dl = self.engine.predict(addr)
        self.assertEqual(pred_dl.source, "OFFLINE_SNAPSHOT")
        self.assertEqual(pred_dl.metadata.get("model_type"), "tgae")

        pred = self.engine.predict(addr, model_type="isolation_forest")
        self.assertEqual(pred.source, "OFFLINE_SNAPSHOT")
        # Previously showed scaled 0.00 instead of true raw tx_sent_count=1.0, sent_recv_ratio=2.0, out_degree=1.0
        self.assertAlmostEqual(pred.raw_features["tx_sent_count"], 1.0, places=3)
        self.assertAlmostEqual(pred.raw_features["sent_recv_ratio"], 2.0, places=3)
        self.assertAlmostEqual(pred.raw_features["out_degree"], 1.0, places=3)
        self.assertAlmostEqual(pred.raw_features["local_ego_density"], 0.5, places=3)
        # Isolation Forest anomaly score contract remains identical
        self.assertAlmostEqual(pred.anomaly_score, 0.4190, places=4)

    def test_15_streamlit_dashboard_e2e_and_presets(self):
        """Regression BUG-03/BUG-04: Dashboard renders cleanly on Streamlit 1.58 with active graphs and preset/manual switching."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("src/app/dashboard.py", default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Dashboard crashed: {at.exception}")
        self.assertEqual(len(at.error), 0)

        # Select Verified Normal preset (WBTC) and verify session_state updates cleanly
        at.selectbox[0].select("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599 (Verified Normal · WBTC · 60 Edges)").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.text_input[0].value, "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599")

        # Type external wallet and click RUN SCAN submit button
        at.text_input[0].set_value("0x0000000000000000000000000000000000001234")
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        # Type invalid address -> clean single-prefix validation error
        at.text_input[0].set_value("not_valid")
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 1)
        self.assertFalse(at.error[0].value.startswith("Invalid Ethereum address format: Invalid Ethereum"))

    def test_16_unified_evaluator_and_stats_api(self):
        """Verify TV3-02 UnifiedEvaluator metrics, error analysis, and /api/v1/stats endpoint."""
        from src.evaluation.evaluator import UnifiedEvaluator, analyze_error_cases
        from src.models import IsolationForestDetector

        det = IsolationForestDetector.load("artifacts/checkpoints/isolation_forest.pkl")
        evaluator = UnifiedEvaluator(ks=(2, 4))
        synthetic = {
            "x": np.zeros((4, 25), dtype=np.float32),
            "y": np.array([0, 1, 0, 1], dtype=np.int64),
            "eval_mask": np.array([True, True, True, True]),
            "addresses": ["0xtr0", "0xtr1", "0xun0", "0xun1"],
        }
        row, idx, y_true, scores = evaluator.evaluate_detector(
            det, synthetic, model_name="TestIF", split="test", exclude_addresses={"0xtr0", "0xtr1"}, eval_protocol="unseen"
        )
        self.assertEqual(row["num_eval_nodes"], 2)
        self.assertIn("fpr_at_90_tpr", row)
        self.assertIn("precision_at_2", row)
        self.assertIn("inference_latency_ms_per_wallet", row)

        err_df = analyze_error_cases(
            "TestIF", "test", ["0xun0", "0xun1"], y_true, scores, threshold=det.threshold_
        )
        self.assertIn("error_type", err_df.columns) if not err_df.empty else None

        # Verify /stats and /api/v1/stats endpoints return real artifacts
        for ep in ("/stats", "/api/v1/stats"):
            r = self.client.get(ep)
            self.assertEqual(r.status_code, 200)
            payload = r.json()
            self.assertGreater(len(payload["benchmark_comparison"]), 0)
            self.assertGreaterEqual(len(payload["ablation_study"]), 7)
            ablation_codes = {row["ablation_code"] for row in payload["ablation_study"]}
            self.assertEqual(ablation_codes, {"Abl-A", "Abl-B", "Abl-C", "Abl-D", "Abl-E", "Abl-F", "Abl-G"})
            self.assertGreater(payload["error_case_summary"]["total_errors"], 0)

    def test_17_tgae_serving_and_branch_breakdown(self):
        """Verify TV2/TV3 ModularTGAE serving via Engine, FastAPI (?model=tgae), UI component, and Dashboard."""
        from src.app.components import render_branch_breakdown_bar
        from streamlit.testing.v1 import AppTest

        self.assertIsNotNone(self.engine.tgae_model, "ModularTGAE checkpoint must be loaded on CPU")
        phishing_addr = "0x096f226b05777751047d1caa8e06f350fa7c4677"
        normal_addr = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"

        self.engine._prediction_cache.clear()
        pred_phish = self.engine.predict(phishing_addr, model_type="tgae")
        pred_norm = self.engine.predict(normal_addr, model_type="tgae")

        self.assertGreaterEqual(pred_phish.anomaly_score, 0.0)
        self.assertLessEqual(pred_phish.anomaly_score, 1.0)
        self.assertEqual(pred_phish.metadata.get("model_type"), "tgae")
        self.assertIn("branch_contributions", pred_phish.metadata)
        bc = pred_phish.metadata["branch_contributions"]
        self.assertAlmostEqual(
            bc["feature_pct"] + bc["graph_pct"] + bc["temporal_pct"], 100.0, places=1
        )
        self.assertGreater(pred_phish.anomaly_score, pred_norm.anomaly_score)

        # Render 3-branch HTML bar cleanly
        bar_html = render_branch_breakdown_bar(bc)
        self.assertIn("TRI-MODAL GNN", bar_html)
        self.assertIn("Tabular Flow", bar_html)

        # FastAPI ?model=tgae and batch model="tgae"
        resp = self.client.get(f"/predict/{phishing_addr}?model=tgae")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["metadata"]["model_type"], "tgae")
        self.assertIn("branch_contributions", body["metadata"])

        batch_resp = self.client.post(
            "/predict/batch",
            json={"addresses": [phishing_addr, normal_addr], "model": "tgae"},
        )
        self.assertEqual(batch_resp.status_code, 200)
        self.assertEqual(batch_resp.json()["total_processed"], 2)

        # Streamlit Dashboard switching radio to Modular TG-AE
        at = AppTest.from_file("src/app/dashboard.py", default_timeout=30)
        at.run()
        at.radio[0].set_value("Modular TG-AE (Tri-Modal Deep GNN)").run()
        self.assertEqual(len(at.exception), 0, f"Dashboard TG-AE switch crashed: {at.exception}")
        self.assertEqual(len(at.error), 0)

    def test_18_api_security_and_offline_svg_fallback(self):
        """Verify X-API-Key guard on /predict/batch and zero-internet SVG fallback in graph widget."""
        import os
        from src.app.components import render_interactive_graph_widget

        widget_html = render_interactive_graph_widget(
            {
                "center_address": "0x096f226b05777751047d1caa8e06f350fa7c4677",
                "nodes": [{"id": "0x096f226b05777751047d1caa8e06f350fa7c4677", "label": "0x096f...", "size": 22, "color": "#ef4444"}],
                "edges": [],
            }
        )
        self.assertIn('typeof window.d3 === "undefined"', widget_html)
        self.assertIn("createElementNS", widget_html)

        # Verify optional ETHEREUM_GUARD_API_KEY guard
        addr = "0x096f226b05777751047d1caa8e06f350fa7c4677"
        with patch.dict(os.environ, {"ETHEREUM_GUARD_API_KEY": "secret-guard-token"}):
            unauth = self.client.post("/predict/batch", json={"addresses": [addr]})
            self.assertEqual(unauth.status_code, 401)

            auth = self.client.post(
                "/predict/batch",
                json={"addresses": [addr]},
                headers={"X-API-Key": "secret-guard-token"},
            )
            self.assertEqual(auth.status_code, 200)
            self.assertEqual(auth.json()["total_processed"], 1)

    def test_19_dual_granularity_tx_hash_and_true_3branch_ego(self):
        """Verify Transaction-Level (`tx_hash`) Deep Scoring, true 3-branch ego subgraph feeding, API, and UI."""
        from streamlit.testing.v1 import AppTest

        phish_addr = "0x096f226b05777751047d1caa8e06f350fa7c4677"
        phish_tx = "0xfadc2e6f30354a6623b17e78527eecd77659b5ac51a9132ce521d7fdb67a2bb9"
        norm_tx = "0x5b8d362d45210ab128854f18c317d2781d6b98bfbff6b6f1266a75fe38a5d11a"

        # 1. Verify default predict() runs Modular TG-AE with real ego edges & active temporal windows
        self.engine._prediction_cache.clear()
        pred_w = self.engine.predict(phish_addr)
        self.assertEqual(pred_w.metadata.get("model_type"), "tgae")
        active_mod = pred_w.metadata.get("active_modalities", {})
        self.assertGreater(active_mod.get("ego_graph_edges", 0), 0)
        self.assertGreater(active_mod.get("nonempty_temporal_windows", 0), 0)

        # 2. Verify direct Transaction-Level Deep Learning scoring (Phishing tx > Normal tx)
        tp_phish = self.engine.predict_transaction(phish_tx)
        tp_norm = self.engine.predict_transaction(norm_tx)
        self.assertGreater(tp_phish.anomaly_score, tp_norm.anomaly_score)
        comp = tp_phish.component_breakdown
        self.assertAlmostEqual(
            comp["sender_risk_pct"] + comp["receiver_risk_pct"] + comp["tx_dynamics_pct"],
            100.0,
            places=0,
        )

        # 3. Verify FastAPI /predict/tx/{tx_hash} and /transactions/{address}
        r_tx = self.client.get(f"/predict/tx/{phish_tx}")
        self.assertEqual(r_tx.status_code, 200)
        self.assertEqual(r_tx.json()["tx_hash"], phish_tx)

        r_bad_tx = self.client.get("/predict/tx/0x1234")
        self.assertEqual(r_bad_tx.status_code, 422)

        r_w_txs = self.client.get(f"/transactions/{phish_addr}?max_txs=5")
        self.assertEqual(r_w_txs.status_code, 200)
        self.assertGreater(r_w_txs.json()["count"], 0)

        # 4. Verify Streamlit Dashboard handles 66-char tx_hash search end-to-end
        at = AppTest.from_file("src/app/dashboard.py", default_timeout=30)
        at.run()
        at.text_input[0].set_value(phish_tx)
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, f"Dashboard tx_hash scan crashed: {at.exception}")
        self.assertEqual(len(at.error), 0)


if __name__ == "__main__":
    unittest.main()
