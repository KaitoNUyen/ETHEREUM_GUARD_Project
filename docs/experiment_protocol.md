# QUY CHUẨN THỰC NGHIỆM & CHỐNG RÒ RỈ DỮ LIỆU (EXPERIMENT PROTOCOL)

**Task:** TV3-01  
**Dự án:** Ethereum Guard (`configs/default.yaml`, `random_seed = 42`)

---

## 1. Quy ước Nhãn Tài khoản (3-State Account-Level Label Contract)

Nhãn được gán ở **cấp độ tài khoản** (không phải cấp độ từng giao dịch) theo [src/schemas.py](file:///d:/ethereum-guard-tv2-clean/src/schemas.py):

| Giá trị | Ý nghĩa | Nguồn gốc | Quy tắc bắt buộc |
| :---: | :--- | :--- | :--- |
| **`1`** | Known Phishing (Bất thường đã xác minh) | MDBChain Phishing Dataset (`125` địa chỉ xuất hiện trong mẫu) | Dùng cho đánh giá dương tính và nhánh tương phản/phụ trợ. |
| **`0`** | Verified Normal (Bình thường đã kiểm chứng) | `verified_benign_addresses.txt` (`86` địa chỉ xác thực) | Tập học chuẩn cho Isolation Forest và nhánh tái tạo bình thường. |
| **`-1`** | Unlabeled (Chưa gắn nhãn) | `1.791.383` địa chỉ còn lại (`99,988%`) | **Tuyệt đối không được chuyển thành `0`** khi tính PR-AUC, ROC-AUC, F1. |

---

## 2. Giao thức Chia dữ liệu & Chống Rò rỉ (Anti-Leakage Protocol)

### 2.1. Phân chia Thời gian 3 Snapshot (70% / 15% / 15%)
Toàn bộ `2.618.597` giao dịch từ 4 đợt lấy mẫu VNEGI10 được sắp xếp theo mốc Unix timestamp tăng dần và cắt theo phân vị số dòng giao dịch ([split_report.json](file:///d:/ethereum-guard-tv2-clean/artifacts/results/split_report.json)):
- **$T_{\text{train}}$:** `2024-03-14T17:53:23Z` (`1.833.120` giao dịch — `70,00%`).
- **$T_{\text{val}}$:** `2024-03-22T06:50:23Z` (`392.844` giao dịch tăng thêm — `15,00%`).
- **$T_{\text{test}}$:** `2024-03-29T18:49:23Z` (`392.633` giao dịch tăng thêm — `15,00%`).

### 2.2. Cô lập 10% Inductive Holdout
- Rút ngẫu nhiên cố định (`seed=42`) `179.160` địa chỉ (`10,00%`) vào tập holdout.
- Toàn bộ `357.539` giao dịch giai đoạn Train có liên quan tới bất kỳ địa chỉ holdout nào bị loại khỏi `train_graph_transactions.parquet` (còn lại `1.475.581` giao dịch, kiểm toán tự động bởi [audit_splits.py](file:///d:/ethereum-guard-tv2-clean/src/pipeline/audit_splits.py) xác nhận `train_graph_holdout_edge_violations = 0`).

### 2.3. Quy tắc Chuẩn hóa & Hiệu chuẩn Ngưỡng
1. **Scaler Train-Only:** Cả `feature_robust_scaler.joblib` (19 đặc trưng) và `graph_feature_scaler.joblib` (6 đặc trưng) chỉ `.fit()` trên tập `train`, sau đó `.transform()` cố định lên `val` và `test`.
2. **Hiệu chuẩn ngưỡng $\tau^*$ chỉ trên Validation:** Ngưỡng quyết định $\tau^* = \arg\max_{\tau} F_1(y_{\text{val}}, \mathbb{I}[S_{\text{val}} \ge \tau])$ và trọng số nhánh $(\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$ chỉ được tìm trên `val`, giữ đóng băng hoàn toàn khi đánh giá trên `test`.
3. **Loại bỏ định danh gây học vẹt:** Không đưa `node_id`, địa chỉ hex thô hay `block_number` vào vector 25 đặc trưng.

---

## 3. Giao thức Đánh giá Kép (Dual Evaluation Protocol — Chống Evaluation Leakage)

Vì snapshot `val` (`210` ví có nhãn) và `test` (`211` ví có nhãn) được dựng theo lịch sử lũy kế, trong đó `188` ví đã xuất hiện ở `train`, [UnifiedEvaluator](file:///d:/ethereum-guard-tv2-clean/src/evaluation/evaluator.py) bắt buộc báo cáo song song **3 lát cắt rõ ràng** để bảo đảm trung thực khoa học:

1. **`cumulative_test` (211 ví: 125 phishing, 86 normal):** Đánh giá toàn diện tại mốc $T_{\text{test}}$ trên trạng thái đồ thị lũy kế đầy đủ.
2. **`strict_unseen_from_train` (23 ví: 14 phishing, 9 normal):** Loại bỏ hoàn toàn `188` địa chỉ có nhãn đã từng xuất hiện ở `train` (`exclude_addresses=train_labeled_addrs`) để đo khả năng tổng quát hóa thực sự không rò rỉ.
3. **`inductive_holdout_only` (20 ví: 12 phishing, 8 normal):** Chỉ đánh giá trên các địa chỉ nằm trong tập 10% Inductive Holdout đã bị cách ly cạnh khỏi đồ thị Train.

---

## 4. Ma trận Đối chứng & Ablation Study (2 Baselines + 7 Cấu hình A → G)

Chạy tự động qua lệnh `python run_experiments.py --config configs/default.yaml --suite all`:

| Mã | Mô hình / Cấu hình | Tabular (12) | Graph + Edges (6+5) | Temporal (10×8 + 7) | Câu hỏi nghiên cứu |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **BL-1** | Isolation Forest (Normal-only) | ✓ | ✓ (tĩnh) | ✓ (tĩnh) | RQ2, RQ4, RQ5 |
| **BL-2** | XGBoost (`scale_pos_weight`) | ✓ | ✓ (tĩnh) | ✓ (tĩnh) | RQ1, RQ2 |
| **Abl-A** | Tabular Autoencoder | ✓ | - | - | RQ3 |
| **Abl-B** | Graph Transformer AE (`TransformerConv`) | - | ✓ | - | RQ3 |
| **Abl-C** | Temporal GRU-Attention AE | - | - | ✓ | RQ3 |
| **Abl-D** | Tabular + Graph AE | ✓ | ✓ | - | RQ3 |
| **Abl-E** | Tabular + Temporal AE | ✓ | - | ✓ | RQ3 |
| **Abl-F** | Graph + Temporal AE | - | ✓ | ✓ | RQ3 |
| **Abl-G** | **Full Proposed Modular TG-AE** | ✓ | ✓ | ✓ | **RQ2, RQ3, RQ4, RQ5** |

---

## 5. Bộ Chỉ số Nghiệm thu (Evaluation Metrics)

- **Chính (Imbalanced):** `PR-AUC`, `F1-Binary`, `F1-Macro`, `Precision`, `Recall`, `Precision@K` ($K \in \{10, 50, 100\}$).
- **Phân tách:** `ROC-AUC`, `FPR@90%TPR`.
- **Chi phí hệ thống (RQ5):** `Inference Latency (ms/wallet)`, `Peak GPU VRAM (MB)`, `Peak RAM (MB)`.
- **Đầu ra chuẩn hóa:** `artifacts/results/benchmark_comparison.csv`, `artifacts/results/ablation_study.csv`, `artifacts/results/error_cases.csv`.
