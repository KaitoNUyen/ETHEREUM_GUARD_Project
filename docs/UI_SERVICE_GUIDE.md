# Hướng dẫn Khởi chạy Hệ thống UI & API Ethereum Guard (Release v1.0)

Hệ thống phục vụ suy luận phát hiện gian lận và bảng điều khiển tương tác (Task **TV3-03** & **TV3-06** theo Master Plan) đã được xây dựng hoàn chỉnh và module hóa ở cấp độ doanh nghiệp:

---

## 1. Kiến trúc Thành phần (Production Modular Architecture)

1. **Inference & Explainability Engine (`src/app/engine.py`):**
   - Đóng gói logic suy luận `IsolationForest` (200 cây, 25 đặc trưng đa mô thức) kèm cơ chế **Hybrid Live On-Chain Fallback** (tự động cào Etherscan V2 khi ví không có trong snapshot offline).
   - Chuẩn hóa Z-Score và khoảng tứ phân vị (IQR) để xếp hạng các đặc trưng đóng góp lớn nhất (Top-8 Feature Contributions), tự động đảo ngược RobustScaler để hiển thị đúng giá trị thô thực tế.
   - Truy vấn đồ thị con 1-hop (Ego-Network) trực tiếp từ cache live hoặc tập dữ liệu Parquet với cơ chế `pyarrow.dataset` lọc dữ liệu tức thì (< 50ms).

2. **RESTful API Backend (`src/app/service.py`):**
   - Xây dựng trên nền **FastAPI** và **Pydantic v2** với chuẩn dữ liệu nghiêm ngặt.
   - Endpoint:
     - `GET /health`: Trạng thái máy chủ, phiên bản và ngưỡng quyết định $\tau^*$.
     - `GET /predict/{address}`: Đánh giá rủi ro và trả về Top-8 đặc trưng bất thường.
     - `GET /graph/{address}`: Trích xuất danh sách nút và cạnh mạng lưới lân cận.
     - `POST /predict/batch`: Quét rủi ro hàng loạt (tối đa 50 địa chỉ đồng thời).
   - Tài liệu OpenAPI Swagger tương tác sẵn sàng tại: `http://localhost:8000/docs`.

3. **Thư viện UI Component Tái sử dụng (`src/app/components.py`):**
   - Thiết kế theo ngôn ngữ Fintech Dark Glassmorphism (font chữ Plus Jakarta Sans & JetBrains Mono).
   - Các widget:
     - `render_top_navbar()`: Thanh điều hướng thương hiệu và trạng thái engine.
     - `render_threat_summary_card()`: Thẻ chẩn đoán rủi ro (CRITICAL, HIGH, MEDIUM, LOW) kèm thanh đo điểm số so với ngưỡng $\tau^*$.
     - `render_feature_attribution_list()`: Danh sách bóc tách độ lệch Z-Score của các đặc trưng.
     - `render_interactive_graph_widget()`: Đồ thị tương tác mạng lưới giao dịch Ego-Network bằng **D3.js Force Simulation** nhúng trực tiếp, cho phép phóng to, thu nhỏ và kéo thả nút mạng.

4. **Giao diện Web Console (`src/app/dashboard.py`):**
   - Ứng dụng **Streamlit** trực quan, tích hợp thanh tìm kiếm ví Ethereum, danh mục ca kiểm thử mẫu (presets) và 3 tab phân tích chuyên sâu.

---

## 2. Hướng dẫn Khởi chạy Độc lập

### Khởi động REST API Server:
```powershell
uvicorn src.app.service:app --host 127.0.0.1 --port 8000 --reload
```
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: `curl http://localhost:8000/health`

### Khởi động Web Dashboard UI:
```powershell
streamlit run src/app/dashboard.py --server.port 8501
```
- Giao diện người dùng: [http://localhost:8501](http://localhost:8501)

---

## 3. Các địa chỉ ví mẫu để kiểm thử thực tế (có sẵn Ego-Network)

- **Ví lừa đảo đã xác thực (Known Phishing — HIGH Risk):**
  - `0x096f226b05777751047d1caa8e06f350fa7c4677` (9 cạnh giao dịch)
  - `0x7cd5b32f1c7a3917525efaaf3b8bf286252430b6` (7 nút / 7 cạnh)
  - `0xe2facbebbe29a94a17590f9a56e721fb7d610795` (5 nút / 5 cạnh)
- **Ví hợp pháp đã xác thực (Verified Normal — dưới ngưỡng cảnh báo τ*):**
  - `0x2260fac5e5542a773aa44fbcfedf7c193bc2c599` (WBTC · 16 nút / 60 cạnh)
  - `0x1f9840a85d5af5bf1d1762f925bdaddc4201f984` (UNI · 24 nút / 60 cạnh)
