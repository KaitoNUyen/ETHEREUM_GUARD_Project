# BÁO CÁO NGHIÊN CỨU KHOA HỌC & ĐỒ ÁN (BẢN HOÀN CHỈNH 9 CHƯƠNG)

**Tên đề tài:** Mô hình phát hiện bất thường trong giao dịch Ethereum bằng học sâu (*Ethereum Guard*)  
**Kiến trúc cốt lõi:** Học sâu đa phương thức hai cấp độ (**Modular TG-AE** cấp tài khoản & **DeepTransactionScorer** cấp giao dịch)

---

## Chương 1. Giới thiệu (Introduction)

### 1.1. Bối cảnh và định nghĩa bài toán hai cấp độ (Dual-Granularity Formulation)
Mạng lưới Ethereum xử lý hàng trăm triệu giao dịch phi tập trung trên mô hình tài khoản (*Account-based EVM*), nơi các chiến dịch lừa đảo (*phishing*), rút ruột tài sản và rửa tiền diễn ra tinh vi. Để giải quyết trọn vẹn bài toán **"Phát hiện bất thường trong giao dịch Ethereum bằng học sâu"**, đề tài mô hình hóa mạng lưới dưới dạng đồ thị giao dịch có hướng biến thiên theo thời gian $G_t = (V_t, E_t, X_V, X_E)$ và thực hiện phát hiện bất thường đồng thời ở **2 cấp độ**:
1. **Cấp độ Từng Giao dịch (*Transaction / Edge-Level Anomaly Detection*):** Với mỗi giao dịch $tx_k = (u \xrightarrow{x_e, t} v)$, mô hình học sâu **DeepTransactionScorer** kết hợp biểu diễn ẩn $z_u, z_v \in \mathbb{R}^{64}$ từ mạng đồ thị–thời gian với tín hiệu đột biến hành vi (*Behavioral Shock*) để phát hiện chính xác giao dịch gian lận đơn lẻ (kể cả khi ví gửi sạch lâu năm đột ngột bị chiếm quyền).
2. **Cấp độ Tài khoản (*Account / Node-Level Behavioral Anomaly Detection*):** Mô hình **Modular TG-AE** hợp nhất 3 nhánh mạng sâu (**Tabular MLP**, **Graph Transformer `TransformerConv`**, và **GRU + Bahdanau Attention**) để nhận diện các tài khoản có cấu trúc gom–tẩu tán dòng tiền bất thường.

### 1.2. Thách thức cốt lõi
1. **Mất cân bằng nhãn cực đoan (Extreme Imbalance):** Trên `1.791.594` địa chỉ và `2.618.597` giao dịch khảo sát, tỷ lệ giao dịch/địa chỉ lừa đảo đã xác thực dưới `0,01%` (`1:100` đến `1:280` trên tập kiểm thử cạnh); phần lớn mạng lưới là không nhãn (`label = -1`).
2. **Rò rỉ dữ liệu thời gian & lan truyền thông điệp (Temporal & Message-Passing Leakage):** Dựng một đồ thị tĩnh trên toàn lịch sử khiến GNN nhìn thấy các cạnh giao dịch tương lai.
3. **Báo động giả trên "Siêu nút" hợp pháp (Super-Node False Positives):** Các sàn CEX (Binance) và hợp đồng DeFi lớn (USDT, Uniswap, WBTC) có khối lượng cực đại khiến các thuật toán không giám sát truyền thống (Isolation Forest, GCN-AE thuần) báo động giả hàng loạt.

---

## Chương 2. Tổng quan tài liệu (Literature Review)

Chi tiết tổng hợp 14 công trình nền tảng (Isolation Forest, XGBoost, DOMINANT — Ding et al. SDM 2019, GAT — Veličković et al. ICLR 2018, GraphSAGE, TransformerConv, GRU, Bahdanau Attention, Focal Loss, Deep SAD, Elliptic) được trình bày đầy đủ tại [literature_review.md](file:///d:/ethereum-guard-tv2-clean/docs/literature_review.md).

---

## Chương 3. Dữ liệu & Tiền xử lý Chống Rò Rỉ (Dataset & Leak-Free Pipeline)

- **Nguồn dữ liệu:** `2.618.597` giao dịch thực tế và `1.791.594` địa chỉ từ `vnegi10/Ethereum_blockchain_parquet`, kết hợp `MDBChain Phishing Dataset` (`125` ví phishing giao thoa, `252` luồng cạnh độc hại) và `Verified Benign Registry` (`86` thực thể hợp pháp, `640.234` luồng cạnh sạch).
- **Chia 3 lát cắt thời gian nghiêm ngặt ([create_splits.py](file:///d:/ethereum-guard-tv2-clean/src/pipeline/create_splits.py)):**
  - `train` ($\le$ `2024-03-14 17:53:23 UTC`): Cô lập hoàn toàn `10%` địa chỉ Inductive Holdout (`357.539` cạnh bị tách khỏi đồ thị train — `0` vi phạm theo [leak_audit_report.json](file:///d:/ethereum-guard-tv2-clean/artifacts/results/leak_audit_report.json)).
  - `val` ($\le$ `2024-03-22 06:50:23 UTC`): `1.528.608` nút, `1.632.692` cạnh có hướng.
  - `test` ($\le$ `2024-03-29 18:49:23 UTC`): `1.791.594` nút, `1.923.360` cạnh có hướng.

---

## Chương 4. Trích xuất Đặc trưng Đa phương thức & Đồ thị Giao dịch

1. **25 Đặc trưng Tài khoản ([schemas.py](file:///d:/ethereum-guard-tv2-clean/src/schemas.py)):** 12 đặc trưng dòng tiền (`TABULAR_TX_COLS`), 6 đặc trưng cấu trúc đồ thị (`GRAPH_TOPO_COLS` gồm cả `local_ego_density` tính bằng Numba JIT CSR), và 7 đặc trưng động học thời gian (`TEMPORAL_STAT_COLS` gồm `burstiness_index` và `recent_volume_spike_ratio`).
2. **Tensor Chuỗi Thời gian `(N, 10, 8)` ([features.py](file:///d:/ethereum-guard-tv2-clean/src/pipeline/features.py)):** 10 cửa sổ 24h gần nhất với 8 kênh định hướng kèm biến đổi `sequence_log1p` và `valid_window_mask`.
3. **Thuộc tính Cạnh Giao dịch 5 chiều (`edge_attr`):** `[tx_count_log, total_value_log, mean_value_log, last_tx_age_norm, avg_gas_log]`.

---

## Chương 5. Phương pháp Học sâu Đề xuất: Modular TG-AE & DeepTransactionScorer

### 5.1. Cấp độ Tài khoản: Kiến trúc Tri-Modal Modular TG-AE ([encoders.py](file:///d:/ethereum-guard-tv2-clean/src/models/encoders.py), [tgae.py](file:///d:/ethereum-guard-tv2-clean/src/models/tgae.py))
- **TabularMLP:** 2 lớp `Linear + LayerNorm + GELU + Dropout` mã hóa dòng tiền tĩnh.
- **EdgeTransformerEncoder (`TransformerConv`):** 2 lớp Graph Transformer đa đầu chú ý (4 heads) tích hợp trực tiếp thuộc tính cạnh 5 chiều `edge_attr` trên đồ thị con lân cận.
- **TemporalAttentionGRU:** Mạng GRU 2 lớp kết hợp cơ chế chú ý thời gian Bahdanau trên chuỗi `(10, 8)` đã chuẩn hóa `log1p`.
- **Masked Gated Fusion & Contrastive-Focal Objective:** Hợp nhất qua cổng Sigmoid học được vào không gian ẩn $z_i \in \mathbb{R}^{64}$; huấn luyện đồng thời với Huber/MSE Reconstruction, Contrastive Margin ($M=5.0$) và Focal Auxiliary Loss, hiệu chuẩn điểm bằng IQR Calibration.

### 5.2. Cấp độ Giao dịch: Kiến trúc DeepTransactionScorer ([tx_scorer.py](file:///d:/ethereum-guard-tv2-clean/src/models/tx_scorer.py))
Với mỗi giao dịch $tx_k = (u \to v)$, mạng `DeepTxEdgeNet` trích xuất biểu diễn sâu của hai đầu mút $z_u, z_v \in \mathbb{R}^{64}$ từ `Modular TG-AE` và xây dựng biểu diễn cạnh đa nhánh:
$$h_{tx_k} = \Big[\, \phi_S(z_u, s_u) \;\|\; \phi_R(z_v, s_v) \;\|\; \phi_D\big(|z_u - z_v|,\, z_u \odot z_v,\, x_{\text{edge}},\, \psi_{\text{shock}}\big) \,\Big]$$
trong đó $\psi_{\text{shock}}$ đo độ đột biến giá trị giao dịch so với mức gửi trung bình lịch sử của ví gửi ($\Delta_{\text{val}} = \log(1 + \text{val}) - \bar{v}_u$), cờ giao dịch `0 ETH` gọi hợp đồng thông minh, độ dồn lệnh `burstiness` và độ lệch bất đối xứng vào/ra. Mạng được huấn luyện kết hợp **Contrastive Edge Autoencoder** và **Class-Weighted Focal Loss**, đồng thời phân rã tỷ lệ rủi ro thành 3 thành phần giải thích được: `sender_risk_pct`, `receiver_risk_pct`, `tx_dynamics_pct`.

---

## Chương 6. Thiết lập Thực nghiệm & Đối sánh Đa Mô hình

Thực nghiệm đối sánh trực tiếp phương pháp đề xuất với **4 mô hình Baseline độc lập** (2 ML truyền thống + 2 Học sâu GNN kinh điển) theo quy chuẩn [experiment_protocol.md](file:///d:/ethereum-guard-tv2-clean/docs/experiment_protocol.md):
- **ML cổ điển:** `BL-1: Isolation Forest` (không giám sát) và `BL-2: XGBoost` (boosting cây có giám sát).
- **Học sâu GNN độc lập ([deep_baselines.py](file:///d:/ethereum-guard-tv2-clean/src/models/deep_baselines.py)):** `DL-BL1: DOMINANT` (Deep GCN Autoencoder tái tạo ma trận kề $A$ & thuộc tính $X$) và `DL-BL2: GAT` (Multi-Head Graph Attention Network + Focal Loss).

---

## Chương 7. Kết quả Thực nghiệm Hai Cấp độ, Ablation & Hệ thống Production

### 7.1. Bảng 1 — Đánh giá Phát hiện Giao dịch Bất thường (Transaction / Edge-Level Benchmark)
*(Nguồn số liệu thực đo: [transaction_benchmark.csv](file:///d:/ethereum-guard-tv2-clean/artifacts/results/transaction_benchmark.csv) · Biểu đồ: [transaction_benchmark.png](file:///d:/ethereum-guard-tv2-clean/docs/figures/transaction_benchmark.png))*

| Mô hình Học sâu / Baseline Cạnh | Giao thức (`Test`) | Số giao dịch (`Pos / Neg`) | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | P@50 | P@100 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tx-BL1: Isolation Forest Edge** | `cumulative_test_txs` | 25.252 (252 / 25.000) | 0.3348 | 0.0068 | 0.0198 | 0.0100 | 1.0000 | 0.00 | 0.00 |
| **Tx-DL1: DOMINANT Graph-AE Edge** | `cumulative_test_txs` | 25.252 (252 / 25.000) | 0.2075 | 0.0060 | 0.0198 | 0.0100 | 1.0000 | 0.00 | 0.01 |
| **Proposed: DeepTransactionScorer (TG-AE + Shock)** | `cumulative_test_txs` | 25.252 (252 / 25.000) | **0.9548** | **0.7630** | **0.8205** | **0.9944** (1 FP) | 0.6984 | **1.00** | **1.00** |
| **Tx-BL1: Isolation Forest Edge** | `strict_unseen_txs` | 10.674 (38 / 10.636) | 0.3137 | 0.0024 | 0.0071 | 0.0036 | 1.0000 | 0.00 | 0.00 |
| **Tx-DL1: DOMINANT Graph-AE Edge** | `strict_unseen_txs` | 10.674 (38 / 10.636) | 0.1750 | 0.0021 | 0.0071 | 0.0036 | 1.0000 | 0.00 | 0.00 |
| **Proposed: DeepTransactionScorer (TG-AE + Shock)** | `strict_unseen_txs` | 10.674 (38 / 10.636) | **0.9441** | **0.7551** | **0.8485** | **1.0000** (0 FP) | **0.7368** | **0.56** | **0.28** |

> **Phát hiện quan trọng:** Ở tỷ lệ mất cân bằng `1:100` (`25.000` giao dịch sạch) và `1:280` trên các giao dịch hoàn toàn mới sau mốc Train (`10.674` giao dịch), các mô hình không giám sát (`Isolation Forest`, `DOMINANT`) sụp đổ hoàn toàn do các sàn/hợp đồng lớn sinh ra hàng chục nghìn giao dịch hợp pháp. Ngược lại, **DeepTransactionScorer** đạt **Precision = `99,44%`** (chỉ 1 báo động giả duy nhất trên `25.000` giao dịch sạch) và **`100%` Precision trên tập giao dịch mới chưa từng gặp**, với `Precision@100 = 100%`.

### 7.2. Bảng 2 — Đánh giá Phát hiện Tài khoản Bất thường (Account-Level Benchmark)
*(Nguồn số liệu thực đo: [benchmark_comparison.csv](file:///d:/ethereum-guard-tv2-clean/artifacts/results/benchmark_comparison.csv) · Biểu đồ: [benchmark_comparison.png](file:///d:/ethereum-guard-tv2-clean/docs/figures/benchmark_comparison.png))*

| Mô hình / Kiến trúc | Giao thức (`Test`) | Số ví (`Pos/Neg`) | PR-AUC | ROC-AUC | F1-Binary | Precision | Recall | FPR@90%TPR | Độ trễ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BL-1: Isolation Forest** | `cumulative_test` | 211 (125 / 86) | 0.6246 | 0.6380 | 0.8224 | 0.6983 | 1.0000 | 56.98% | 0.19 ms |
| **DL-BL1: DOMINANT (Deep GCN-AE)** | `cumulative_test` | 211 (125 / 86) | 0.4049 | 0.0961 | 0.7440 | 0.5924 | 1.0000 | 100.0% | 0.57 ms |
| **DL-BL2: GAT (Graph Attention Net)** | `cumulative_test` | 211 (125 / 86) | 0.9767 | 0.9753 | 0.9502 | 0.9118 | 0.9920 | 8.14% | 0.80 ms |
| **BL-2: XGBoost** | `cumulative_test` | 211 (125 / 86) | 0.9939 | 0.9934 | **0.9766** | 0.9542 | 1.0000 | 2.33% | 0.16 ms |
| **Proposed: Modular TG-AE (`best_tgae.pt`)** | `cumulative_test` | 211 (125 / 86) | **0.9960** | **0.9952** | 0.9762 | **0.9685** | 0.9840 | **1.16%** | 1.39 ms |
| **BL-1: Isolation Forest** | `strict_unseen_from_train` | 23 (14 / 9) | 0.5309 | 0.3651 | 0.7568 | 0.6087 | 1.0000 | 100.0% | 1.38 ms |
| **DL-BL2: GAT (Graph Attention Net)** | `strict_unseen_from_train` | 23 (14 / 9) | 0.9538 | 0.9206 | 0.7857 | 0.7857 | 0.7857 | 33.33% | 6.90 ms |
| **BL-2: XGBoost** | `strict_unseen_from_train` | 23 (14 / 9) | 0.9821 | 0.9762 | 0.9032 | 0.8235 | 1.0000 | 22.22% | 0.95 ms |
| **Proposed: Modular TG-AE (`Log1p-Seq Fixed`)** | `strict_unseen_from_train` | 23 (14 / 9) | **1.0000** | **1.0000** | **0.9655** | **0.9333** | **1.0000** | **0.00%** | **0.61 ms** |

### 7.3. Nghiên cứu Cắt bỏ 7 Cấu hình (`Abl-A` → `Abl-G`) & Biểu đồ Xuất bản
- Chi tiết 14 dòng kết quả Ablation lưu tại [ablation_study.csv](file:///d:/ethereum-guard-tv2-clean/artifacts/results/ablation_study.csv) và trực quan hóa tại [ablation_study.png](file:///d:/ethereum-guard-tv2-clean/docs/figures/ablation_study.png), [tgae_convergence.png](file:///d:/ethereum-guard-tv2-clean/docs/figures/tgae_convergence.png), [error_breakdown.png](file:///d:/ethereum-guard-tv2-clean/docs/figures/error_breakdown.png): chứng minh kết hợp đủ 3 nhánh (`Abl-G`) vượt trội hơn mọi cấu hình đơn nhánh (`Abl-A, B, C`) và song nhánh (`Abl-D, E, F`), đặc biệt trên các ví hoàn toàn mới (`F1 = 0.9655` vs `0.9032` của `XGBoost` và `0.7857` của `GAT`).

### 7.4. Hệ thống Production v2.0 (Dual-Granularity Serving)
- **Engine ([engine.py](file:///d:/ethereum-guard-tv2-clean/src/app/engine.py)):** Đặt **Modular TG-AE** và **DeepTransactionScorer** làm mô hình mặc định, tự động dựng đồ thị con 1-hop (`edge_index`, `edge_attr`) và chuỗi thời gian `(1, 10, 8)` thực tế khi suy luận.
- **FastAPI ([service.py](file:///d:/ethereum-guard-tv2-clean/src/app/service.py)) & Dashboard ([dashboard.py](file:///d:/ethereum-guard-tv2-clean/src/app/dashboard.py)):** Cho phép quét trực tiếp **mã giao dịch `tx_hash` (66 ký tự)** qua `GET /predict/tx/{tx_hash}` (kèm Etherscan `eth_getTransactionByHash` live) hoặc **địa chỉ ví (42 ký tự)** kèm bảng xếp hạng bất thường cho từng giao dịch của ví đó (`GET /transactions/{address}`).

---

## Chương 8. Thảo luận & Giới hạn Nghiên cứu (Discussion & Limitations)

1. **Quy mô nhãn xác thực:** Tập nhãn xác thực hiện gồm `125` địa chỉ phishing (`252` luồng giao dịch độc hại) và `86` thực thể hợp pháp lớn; việc mở rộng thêm hàng nghìn nhãn ví cá nhân nhỏ lẻ (*retail EOA*) sẽ giúp ranh giới quyết định khái quát hóa tốt hơn nữa trên mạng chính.
2. **Độ sâu ngữ nghĩa hợp đồng thông minh (EVM Calldata & Logs):** Hệ thống Live đã nhận diện chữ ký hàm 4-byte (`approve`, `transferFrom`, `setApprovalForAll`, `permit`) và hỗ trợ API tải `tokentx` / `txlistinternal` trong [etherscan_adapter.py](file:///d:/ethereum-guard-tv2-clean/src/pipeline/etherscan_adapter.py); bước tiếp theo là đưa toàn bộ đồ thị sự kiện ERC-20/NFT vào quá trình huấn luyện ngoại tuyến.
3. **Đạo đức AI:** Điểm bất thường và phân rã XAI đóng vai trò hỗ trợ điều tra viên an ninh mạng, không thay thế phán quyết pháp lý tự động.

---

## Chương 9. Kết luận & Hướng phát triển (Conclusion & Future Work)

Đề tài đã xây dựng hoàn chỉnh hệ thống **Ethereum Guard** phát hiện bất thường trong giao dịch Ethereum bằng học sâu ở cả **cấp độ từng giao dịch (`DeepTransactionScorer`)** lẫn **cấp độ tài khoản (`Modular TG-AE`)**, vượt trội rõ rệt so với cả học máy truyền thống (`Isolation Forest`, `XGBoost`) lẫn các mạng GNN kinh điển (`DOMINANT`, `GAT`). Hướng phát triển tiếp theo tập trung vào **Cross-Chain Dynamic Heterogeneous GNN** để truy vết dòng tiền xuyên chuỗi (Bridges/DEXs) theo thời gian thực.
