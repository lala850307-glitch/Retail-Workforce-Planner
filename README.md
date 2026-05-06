# Retail Workforce Planner — 零售智能排班系統

整合 PostgreSQL 門市營運資料與 LightGBM 機器學習模型，預測各時段最佳正職／兼職人力配置，並提供勞動效率稽核與 Q2 排班規劃功能。

---

## 功能特色

- **智能排班預測**：依早班／午班／晚班時段，預測最佳正職（FT）與兼職（PT）人數
- **勞動效率稽核**：以歷史實際資料回測排班績效，找出人力超配或不足的時段
- **Q2 排班規劃引擎**：結合去年同期 AOV 與客流量，自動生成 2026 Q2 排班計畫
- **混合預測機制**：規則式先驗知識（時段 Buffer）＋ ML 模型雙軌驗證

---

## 技術架構

| 層級 | 技術 |
|---|---|
| 前端 | HTML / Jinja2 Templates |
| 後端 | FastAPI + Uvicorn |
| 預測模型 | LightGBM（`retail_staffing_model_v1.pkl`）|
| 資料庫 | PostgreSQL on GCP Cloud SQL（schema：`store_optimization`）|
| 資料處理 | Pandas、NumPy、SQLAlchemy |

---

## 系統架構

```
Retail-Workforce-Planner/
├── main.py                          # FastAPI 主程式
├── .env                             # 資料庫連線設定
├── api/
│   ├── audit_service.py             # 勞動效率稽核模組
│   ├── forecast_service.py          # 排班預測模組
│   └── retail.js                    # 前端互動邏輯
├── scripts/
│   ├── model_training_pipeline.py   # 核心運算與模型訓練管線
│   ├── q2_forecast_engine.py        # Q2 排班規劃引擎
│   └── data_simulator_2026.py       # 2026 模擬資料產生器
├── data_scripts/
│   ├── gen_master_data.py           # 主資料表產生
│   ├── gen_2025_ops.py              # 2025 營運資料模擬
│   └── gen_2026_ops.py              # 2026 營運資料模擬
├── models/
│   ├── retail_staffing_model_v1.pkl # 訓練完成的排班預測模型
│   └── feature_columns.pkl          # 模型特徵欄位清單
├── templates/
│   ├── index.html                   # 主畫面
│   ├── smart_staffing_forecast.html # 排班預測頁面
│   ├── Labor_Efficiency_Audit.html  # 勞動稽核頁面
│   └── system_architecture_diagram.html
├── stylesheet/
│   └── styles.css
└── output_data/
    └── 2026_Q2_Hybrid_Forecast.csv  # Q2 規劃輸出結果
```

---

## 排班規則設計

| 時段 | 時間 | Buffer | 偏好人力類型 | 流量權重 |
|------|------|--------|-------------|---------|
| 早班 | 09:00–12:00 | 1.00 | 兼職（PT）| 20% |
| 午班 | 13:00–16:00 | 1.15 | 混合 | 35% |
| 晚班 | 16:00–22:00 | 1.30 | 正職（FT）| 45% |

---

## 快速開始

**環境需求：** Python 3.9+、GCP Cloud SQL（PostgreSQL）

### 安裝

```bash
git clone <your-repo-url>
cd Retail-Workforce-Planner/Retail-Workforce-Planner
pip install -r models/requirements.txt
```

### 設定資料庫連線

資料庫部署於 **GCP Cloud SQL**，在專案根目錄的 `.env` 填入連線資訊：

```env
POSTGRESQL_URL=postgresql://user:password@<gcp-cloud-sql-ip>:5432/dbname
```

> 若使用 GCP Cloud SQL Auth Proxy 連線，請先啟動 Proxy 再執行服務。

### 啟動服務

```bash
uvicorn main:app --reload
```

服務啟動後開啟瀏覽器：`http://127.0.0.1:8000`

---

## API 端點

| 端點 | 說明 |
|------|------|
| `GET /` | 主畫面 |
| `POST /audit/...` | 勞動效率稽核 |
| `POST /forecast/...` | 智能排班預測 |

完整互動式文件：`http://127.0.0.1:8000/docs`

---

## 重新訓練模型

```bash
python scripts/model_training_pipeline.py
```

模型輸出至 `models/`，包含 `retail_staffing_model_v1.pkl` 與 `feature_columns.pkl`。
