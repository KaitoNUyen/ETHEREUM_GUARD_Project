# TỔNG QUAN TÀI LIỆU & KHOẢNG TRỐNG NGHIÊN CỨU (LITERATURE REVIEW)

**Task:** TV3-01  
**Đề tài:** *Deep Learning-Based Anomaly Detection Using Graph Structure and Temporal Behavioral Analysis in Ethereum Transaction Networks (Ethereum Guard)*

---

## 1. Phân loại các hướng tiếp cận hiện nay

Nghiên cứu phát hiện bất thường và gian lận (phishing, scam, rửa tiền) trên mạng lưới giao dịch Ethereum có thể chia thành 4 nhóm chính:

1. **Nhóm 1 — Dữ liệu bảng & Học máy truyền thống (Tabular ML & Anomaly Detection):** Trích xuất đặc trưng thống kê cấp tài khoản (tổng ETH gửi/nhận, tần suất, gas, tỷ lệ giao dịch lỗi) rồi huấn luyện Isolation Forest, Random Forest, XGBoost hoặc LightGBM.
2. **Nhóm 2 — Nhúng đồ thị & Mạng nơ-ron đồ thị tĩnh (Static Graph Embedding & GNN):** Biểu diễn mạng Ethereum thành đồ thị có hướng có trọng số (đỉnh = ví, cạnh = giao dịch), áp dụng DeepWalk, Node2Vec, Trans2Vec, GCN, GraphSAGE hoặc Graph Transformer (`TransformerConv`).
3. **Nhóm 3 — Học biểu diễn chuỗi thời gian & Đồ thị động (Temporal & Dynamic Graph Learning):** Mô hình hóa nhịp độ giao dịch theo chuỗi thời gian (GRU/LSTM + Attention, Burstiness, Temporal Graph Networks) để bắt các mẫu đột biến (burst, peel chain, xả tiền sau thời gian ngủ đông).
4. **Nhóm 4 — Phát hiện bất thường bán giám sát & Autoencoder trên đồ thị (Deep Anomaly Detection):** Tối thiểu hóa sai số tái tạo trên nhóm tài khoản bình thường kết hợp hàm mất mát tương phản (Margin/Contrastive/Deviation Loss) để tận dụng số ít nhãn bất thường đã biết mà không học vẹt.

---

## 2. Tổng hợp 14 công trình nghiên cứu nền tảng & liên quan

| STT | Công trình (Tác giả, Năm, Hội nghị/Tạp chí) | Hướng tiếp cận | Ưu điểm chính | Hạn chế khi áp dụng lên Ethereum thực tế |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Liu et al. (2008)**, *Isolation Forest*, IEEE ICDM | Unsupervised Tabular | Nhanh ($O(n \log n)$), không cần nhãn gian lận, tách tốt điểm ngoại lai toàn cục. | Bỏ qua cấu trúc đồ thị lân cận và chuỗi thời gian; dễ báo động giả (FP) với ví sàn CEX / hợp đồng lưu lượng lớn. |
| **2** | **Chen & Guestrin (2016)**, *XGBoost*, ACM SIGKDD | Supervised Tabular | Độ chính xác rất cao trên đặc trưng dạng bảng, hỗ trợ TreeSHAP giải thích. | Phụ thuộc mạnh vào nhãn có giám sát; dễ suy giảm trên dạng tấn công mới (unseen scams) hoặc nhãn cực thưa. |
| **3** | **Farrugia et al. (2020)**, *Detection of illicit accounts over the Ethereum blockchain*, Expert Systems with Applications | Tabular XGBoost (Ethereum) | Chứng minh hiệu quả bộ đặc trưng tài khoản Ethereum (ETH sent/recv, inter-tx time, ERC-20). | Sử dụng dữ liệu tổng hợp tĩnh (pre-aggregated), không có thông tin cạnh đồ thị và chuỗi hành vi theo thời gian. |
| **4** | **Wu et al. (2020)**, *Who Are the Phishers? Phishing Scam Detection on Ethereum via Network Embedding (Trans2Vec)*, IEEE TSMC | Biased Random Walk (XBlock) | Kết hợp giá trị giao dịch và dấu thời gian vào xác suất bước đi ngẫu nhiên trên đồ thị. | Transductive — phải chạy lại embedding khi xuất hiện địa chỉ ví mới (không hỗ trợ inductive live lookup). |
| **5** | **Chen et al. (2020)**, *Phishing Scams Detection in Ethereum Transaction Network*, ACM TOIT | Graph + LightGBM (DElight) | Kết hợp đặc trưng đồ thị 1-hop/2-hop với phân loại cây tăng cường. | Xây dựng đồ thị tĩnh trên toàn bộ lịch sử dễ phát sinh rò rỉ thông tin tương lai (temporal leakage). |
| **6** | **Hamilton et al. (2017)**, *Inductive Representation Learning on Large Graphs (GraphSAGE)*, NeurIPS | Inductive GNN + Neighbor Sampling | Tổng quát hóa được cho đỉnh mới (inductive) và mở rộng tới hàng triệu đỉnh qua lấy mẫu lân cận. | Bản gốc chưa tận dụng đầy đủ đặc trưng cạnh (`edge_attr`) và động lực chuỗi thời gian. |
| **7** | **Shi et al. (2021)**, *Masked Label Prediction: Unified Message Passing Model (`TransformerConv`)*, IJCAI | Graph Transformer | Cơ chế đa đầu (Multi-Head Attention) tích hợp trực tiếp thuộc tính cạnh (`edge_dim`) vào lan truyền thông điệp. | Chi phí tính toán cao hơn GCN/SAGE nếu không kết hợp Neighbor Sampling trên đồ thị lớn. |
| **8** | **Cho et al. (2014)**, *Learning Phrase Representations using RNN Encoder–Decoder (GRU)*, EMNLP | Recurrent Sequence | Gọn nhẹ hơn LSTM, nắm bắt tốt phụ thuộc tuần tự ngắn-trung hạn với ít tham số. | Không phân biệt trọng số các cửa sổ đột biến nếu thiếu tầng Attention. |
| **9** | **Bahdanau et al. (2015)**, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR | Additive Attention | Cho phép mô hình tập trung trọng số vào các cửa sổ thời gian có biến động bất thường nhất. | Cần kết hợp với đặc trưng tĩnh và cấu trúc đồ thị để giảm nhiễu trên chuỗi thưa. |
| **10** | **Goh & Barabási (2008)**, *Burstiness and memory in complex systems*, Europhysics Letters | Temporal Statistics | Định nghĩa chỉ số bùng nổ $B = \frac{\sigma - \mu}{\sigma + \mu} \in [-1, 1]$ mô tả tính giật cục của chuỗi sự kiện. | Đơn lẻ chưa đủ phân biệt bot hợp pháp (CEX/MEV) với ví phishing xả tiền nhanh. |
| **11** | **Lin et al. (2017)**, *Focal Loss for Dense Object Detection*, IEEE ICCV | Imbalanced Learning | Giảm trọng số mẫu dễ phân loại, tập trung gradient vào mẫu thiểu số khó. | Chỉ giải quyết phần mất cân bằng lớp ở nhánh phân loại có giám sát. |
| **12** | **Ruff et al. (2020)**, *Deep Semi-Supervised Anomaly Detection (Deep SAD)*, ICLR | Semi-Supervised Anomaly Detection | Chứng minh lý thuyết việc kết hợp dữ liệu bình thường + một lượng nhỏ nhãn bất thường đẩy biên phân tách tốt hơn unsupervised thuần. | Thiết kế cho ảnh/bảng đơn mô thức, chưa tích hợp đồ thị có hướng và chuỗi thời gian. |
| **13** | **Ding et al. (2019)**, *Deep Anomaly Detection on Attributed Networks (DOMINANT)*, SDM | Graph Autoencoder | Tái tạo đồng thời ma trận thuộc tính đỉnh và ma trận kề cấu trúc đồ thị. | Tái tạo toàn bộ ma trận kề $O(N^2)$ không khả thi với đồ thị $1{,}79\text{ triệu}$ đỉnh Ethereum. |
| **14** | **Weber et al. (2019)**, *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks*, KDD Workshop (Elliptic) | Temporal Graph Benchmark | Chỉ ra tầm quan trọng của chia tách theo thời gian (temporal split) và vấn đề nhãn cực thưa trên blockchain. | Tập trung vào Bitcoin UTXO (cấp giao dịch), khác mô hình Account-based của Ethereum. |

---

## 3. Khoảng trống Nghiên cứu (Research Gap) & Đóng góp của Ethereum Guard

Từ bảng đối chiếu 14 công trình trên, có **4 khoảng trống kỹ thuật rõ rệt** mà **Ethereum Guard** trực tiếp giải quyết:

1. **Rò rỉ thời gian và rò rỉ nhãn ngầm (Temporal & Label Leakage):** Nhiều nghiên cứu trước dựng 1 đồ thị tĩnh duy nhất rồi chia `train/test` ngẫu nhiên hoặc coi toàn bộ ví không nằm trong danh sách phishing là "ví sạch" (`label = 0`). **Ethereum Guard** áp dụng kiến trúc **3 Snapshot lũy kế theo mốc thời gian** ($T_{\text{train}} < T_{\text{val}} < T_{\text{test}}$), cô lập 10% nút holdout khỏi đồ thị train (`0` vi phạm cạnh) và giữ nghiêm quy ước 3 nhãn (`1`, `0`, `-1`).
2. **Điểm mù đơn mô thức (Single-Modality Blind Spot):** Mô hình bảng bỏ lỡ quan hệ đối tác; GNN tĩnh bỏ lỡ nhịp độ burst theo thời gian; chuỗi thời gian bỏ lỡ ngữ cảnh mạng lưới. **Modular TG-AE** hợp nhất 3 nhánh độc lập (`TabularMLP` + `TransformerConv` với `NeighborLoader` + `TemporalAttentionGRU`) thông qua **Masked Gated Fusion** cho phép đánh giá cô lập 7 cấu hình Ablation (A → G).
3. **Khả năng mở rộng trên đồ thị triệu đỉnh & ngân sách 1 GPU:** Thay vì tái tạo ma trận kề $O(N^2)$ như DOMINANT, TG-AE tái tạo đặc trưng hình học 1-hop cục bộ (`x_topo`) và trung bình lân cận ($\bar{x}_{\mathcal{N}(i)}^{\text{topo}}$) kết hợp `NeighborLoader [15, 10]` + `FP16`, giữ VRAM thấp và tương thích với đặc trưng cục bộ khi suy luận ví mới.
4. **Khoảng cách từ nghiên cứu sang sản phẩm có giải thích (XAI + Live Serving):** Hệ thống gắn trực tiếp quy trình offline với **Live Etherscan API V2 Adapter**, cơ chế **Z-Score / IQR Attribution** và đồ thị **Ego-Network D3 1-hop**, hỗ trợ nhà phân tích giải thích nguyên nhân thay vì đưa ra hộp đen.
