import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# =============================================================================
# 常數與規則定義 (這些是模型的先驗知識)
# =============================================================================

TIME_SLOTS = {
    "morning":   {"start": 9,  "end": 12, "buffer": 1.00, "preference": "pt",    "weight_pct": 0.20},
    "afternoon": {"start": 13, "end": 16, "buffer": 1.15, "preference": "mixed", "weight_pct": 0.35},
    "evening":   {"start": 16, "end": 22, "buffer": 1.30, "preference": "ft",    "weight_pct": 0.45},
}

STAFF_WEIGHTS = {"ft": 1.0, "pt": 0.5}

ALLOCATION_RULES = {
    "min_weight": 1.0,
    "base_ft": 2,
    "ft_only_threshold": 2.0,
}

# =============================================================================
# 1. 資料抓取
# =============================================================================

def fetch_data(table: str, start: str, end: str, store_id: int = None):
    load_dotenv()
    db_url = os.getenv("POSTGRESQL_URL")
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        conn.execute(text("SET search_path TO store_optimization, public"))
    
    query = f"""
    SELECT 
        record_timestamp,
        EXTRACT(HOUR FROM record_timestamp)::int AS hour,
        EXTRACT(DOW FROM record_timestamp)::int AS day_of_week,
        customer_traffic,
        transaction_count, 
        sales_amount,
        ft_count,
        pt_count,
        is_peak_month,
        store_id
    FROM store_optimization.{table}
    WHERE record_timestamp >= '{start}' 
      AND record_timestamp < ('{end}'::date + INTERVAL '1 day')
    """
    if store_id:
        query += f" AND store_id = {store_id}"
    
    query += " ORDER BY record_timestamp ASC"
    print(f"--- 執行 SQL 查詢中: {table} ---")
    return pd.read_sql(query, engine)

# =============================================================================
# 2. 資料清洗 (保留所有 ML 特徵欄位)
# =============================================================================

def validate_input(df):
    if df.empty: return df
    data = df.copy()
    data["actual_weight"] = (data["ft_count"] * STAFF_WEIGHTS["ft"] + data["pt_count"] * STAFF_WEIGHTS["pt"])
    
    def get_slot_info(h):
        for name, cfg in TIME_SLOTS.items():
            if cfg["start"] <= h < cfg["end"]: 
                return name, cfg["buffer"]
        return "other", 1.0
    
    slot_info = data["hour"].apply(get_slot_info)
    data["slot_label"] = slot_info.apply(lambda x: x[0])
    data["time_buffer"] = slot_info.apply(lambda x: x[1])
    
    return data[data["customer_traffic"] >= 5].reset_index(drop=True)

# =============================================================================
# 3. 產能審計核心邏輯 (這將生成 ML 的 Target Label)
# =============================================================================

def _allocate_staff(w):
    """將權重轉換為排班建議人數"""
    w = max(ALLOCATION_RULES["min_weight"], float(w))
    if w <= 1.0: return 1, 1 
    if w <= ALLOCATION_RULES["ft_only_threshold"]: return 2, 0
    ft = ALLOCATION_RULES["base_ft"]
    pt = int(np.ceil((w - 2.0) / STAFF_WEIGHTS["pt"]))
    return ft, pt

def compute_ideal_staffing_v2(df, handle_capacity=2.5):
    """
    💡 硬核產能審計：
    1. 理想需求 (y) = 實際成交 / 2.5 * 時段係數
    2. 保留客流量 (X) 欄位供未來特徵工程使用
    """
    data = df.copy()
    
    # 計算基於成交的硬核需求 (這是標竿，不為低轉單率找藉口)
    data["ideal_weight_demand"] = (data["transaction_count"] / handle_capacity) * data["time_buffer"]
    
    # 產能保底 1.0 (開門基本成本)
    data["ideal_weight_demand"] = data["ideal_weight_demand"].clip(lower=1.0)
    
    # 生成模擬排班建議
    alloc = data["ideal_weight_demand"].apply(_allocate_staff)
    data["sim_ft"], data["sim_pt"] = alloc.apply(lambda x: x[0]), alloc.apply(lambda x: x[1])
    data["sim_weight"] = data["sim_ft"] * STAFF_WEIGHTS["ft"] + data["sim_pt"] * STAFF_WEIGHTS["pt"]
    
    return data

# =============================================================================
# 4. 財務回測與利用率 (核心審計指標)
# =============================================================================

def run_financial_backtest_audit(data, wage=230):
    df = data.copy()
    
    # 薪資損失 = 實際 - 建議
    df["cost_saving"] = ((df["actual_weight"] - df["sim_weight"]) * wage).clip(lower=0)
    
    # 權重差異
    df["weight_diff"] = df["actual_weight"] - df["sim_weight"]
    
    # 實際產值：每人每小時成交幾筆
    df["prod_per_person"] = df["transaction_count"] / df["actual_weight"].replace(0, np.nan)
    
    # 利用率 (Hardcore)：理想需求 / 實際排班
    df["utilization_rate"] = (df["ideal_weight_demand"] / df["actual_weight"].replace(0, np.nan)).clip(0, 1.2)
    df["efficiency_score"] = (df["utilization_rate"] * 100).round(1)
    
    return df

# =============================================================================
# 5. 進階特徵工程 (Machine Learning Ready - 這是您要保留的寶貴部分)
# =============================================================================

def build_advanced_features(data: pd.DataFrame, rolling_window: int = 4):
    if data.empty: return data, None
    feat = data.copy().sort_values("record_timestamp").reset_index(drop=True)

    # --- 💡 [新增：商業價值審計特徵] ---
    # 1. 計算該門市的基準 AOV (全期平均)
    base_aov = feat["sales_amount"].sum() / (feat["transaction_count"].sum() + 1e-6)
    
    # 2. 小時客單價強度 (該小時 AOV / 基準 AOV)
    # 越高代表該時段進來的客人越「貴」，值得花更多人力服務
    hourly_aov = feat["sales_amount"] / (feat["transaction_count"] + 1e-6)
    feat["aov_intensity"] = (hourly_aov / base_aov).clip(0, 5) # 限制在 5 倍以內防止極端值
    
    # 3. 價值加權人流 (讓 AI 發現：1 個大戶等於 10 個散客)
    feat["value_weighted_traffic"] = feat["customer_traffic"] * feat["aov_intensity"]

    # --- [原有特徵處理] ---
    feat["traffic_log"] = np.log1p(feat["customer_traffic"])
    feat["actual_cr"] = np.where(feat["customer_traffic"] > 0, feat["transaction_count"] / feat["customer_traffic"], 0)
    feat["actual_cr_lag1"] = feat["actual_cr"].shift(1).fillna(feat["actual_cr"].mean())
    feat["ft_ratio"] = np.where(feat["actual_weight"] > 0, feat["ft_count"] / feat["actual_weight"], 0)
    
    for col in ["customer_traffic", "actual_cr", "aov_intensity"]:
        feat[f"{col}_rolling_mean"] = feat[col].rolling(window=rolling_window, min_periods=1).mean()

    # 4. 交叉特徵：強化 AI 對「時段+價值」的敏感度
    feat["traffic_intensity"] = feat["customer_traffic"] * feat["time_buffer"]

    # 💡 更新特徵清單，加入價值維度
    feature_cols = [
        "store_id","hour", "day_of_week", "customer_traffic", "traffic_log", 
        "actual_cr_lag1", "ft_ratio", "traffic_intensity",
        "customer_traffic_rolling_mean",
        "aov_intensity", "value_weighted_traffic" # 👈 新加入的商業審計特徵
    ]
    return feat[feature_cols].fillna(0), feat["ideal_weight_demand"]
def generate_audit_scenarios(df):
    """
    1. 自動計算歷史 AOV (不自訂)
    2. 先算成交率 (避免 KeyError)
    3. 生成隨機業績目標 (模擬達標/不達標情境)
    """
    data = df.copy()
    
    # --- [關鍵：動態 AOV 手法] ---
    # 用總業績除以總成交數，得到這間店最真實的標竿，AI 訓練才不會偏離現實
    # 加上 1e-6 是為了防止分母為 0 導致程式崩潰
    historical_aov = data["sales_amount"].sum() / (data["transaction_count"].sum() + 1e-6)
    
    # 💡 修正順序：先算好成交率，後面的 groupby 才能找到它
    data["actual_cr"] = np.where(data["customer_traffic"] > 0, 
                                 data["transaction_count"] / data["customer_traffic"], 0)
    
    # 核心：隨機生成目標倍率 (模擬偶爾挑戰成功，偶爾失敗)
    random_factors = np.random.normal(1.2, 0.2, size=len(data))
    data["revenue_target"] = data["sales_amount"] * random_factors
    
    # 1. 管理期望：根據「真實歷史 AOV」算出的理論人力
    data["target_weight"] = (data["revenue_target"] / historical_aov) / 2.5
    
    # 2. 歷史上限：該時段過去最好的表現 (90% 分位數)
    slot_cap = data.groupby("slot_label")["transaction_count"].transform(lambda x: x.quantile(0.9))
    data["historical_limit_weight"] = slot_cap / 2.5
    
    # 3. 現實支撐：根據該時段平均成交率算出的人力 (真相指標)
    avg_cr = data.groupby("slot_label")["actual_cr"].transform("mean")
    data["traffic_support_weight"] = (data["customer_traffic"] * avg_cr) / 2.5
    
    # 標記：是否達標 (這在 ML 訓練中是非常強的特徵)
    data["is_target_met"] = (data["sales_amount"] >= data["revenue_target"]).astype(int)
    
    return data

# =============================================================================
# 6. 報告生成
# =============================================================================

def generate_detailed_report(data):
    summary = {
        "總計浪費時數(權重)": f"{data[data['weight_diff'] > 0]['weight_diff'].sum():,.1f} 小時",
        "預計可省下薪資": f"${data['cost_saving'].sum():,.0f}",
        "平均人效 (成交/小時)": round(data["prod_per_person"].mean(), 2),
        "低利用率時數佔比 (<60%)": f"{(data['utilization_rate'] < 0.6).mean()*100:.1f}%",
        "平均效率評分 (硬核)": round(data["efficiency_score"].mean(), 1)
    }
    
    hourly = data.groupby("hour").agg(
        實際排班權重=("actual_weight", "mean"),
        理想需求權重=("ideal_weight_demand", "mean"),
        權重差異=("weight_diff", "mean"),
        產能利用率=("efficiency_score", "mean")
    ).round(2)
    
    return summary, hourly

# =============================================================================
# 🚀 主程式執行
# =============================================================================

if __name__ == "__main__":
    
    # Step 1: 數據獲取 (從資料庫抓取 2025 全年資料)
    df_raw = fetch_data("store_performance_data", "2025-01-01", "2025-12-31",)
    
    if not df_raw.empty:
        # Step 2: 基礎資料清洗
        df_clean = validate_input(df_raw)
        
        # --- 🚀 [核心改動：特徵工程與情境模擬] ---
        # 1. 隨機生成「挑戰型」業績目標 (模擬偶爾達標、偶爾未達標的情境)
        # 2. 同時計算三種權重：管理期望 (Target)、歷史上限 (Limit)、人流支撐 (Support)
        # 注意：我們使用 1500 作為平均客單價 (AOV)
        df_audit = generate_audit_scenarios(df_clean,)
        
        # Step 3: 執行事實審計 (計算這段時間真正「理想」的人力 y)
        # 這裡的 2.5 依然作為你對「成交事實」的標竿尺
        df_ideal = compute_ideal_staffing_v2(df_audit, handle_capacity=2.5)
        
        # Step 4: 執行財務與利用率分析 (計算浪費與利用率)
        final_df = run_financial_backtest_audit(df_ideal, wage=230)
        
        # Step 5: 產出報告
        sum_rep, hour_rep = generate_detailed_report(final_df)
        
        print("\n--- 🏆 [業績可行性審計] 門市效率報告 ---")
        for k, v in sum_rep.items(): print(f"{k}: {v}")
        
        
        print("\n--- 📈 效率分析矩陣 (事實利用率) ---")
        # 這裡會多出 target_weight 等欄位讓你觀察差異
        print(hour_rep)
        
        # --- 🔬 [Step 6 執行結果] ---
        X, y_label = build_advanced_features(final_df)
        print(f"\n[ML Ready] 特徵數量: {X.shape[1]}, 樣本數: {X.shape[0]}")

        # --- 🤖 [Step 7: 隨機森林模型訓練] ---
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        # 1. 拆分訓練集與測試集 (80% 訓練, 20% 驗證)
        X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.2, random_state=42)

        # 2. 初始化隨機森林
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        # 3. 執行訓練
        rf_model.fit(X_train, y_train)

        # 4. 評估模型表現
        y_pred = rf_model.predict(X_test)
        print(f"\n--- 📊 AI 模型審計結果 ---")
        print(f"模型解釋力 (R2 Score): {r2_score(y_test, y_pred):.2f}")
        print(f"平均預測誤差 (MAE): {mean_absolute_error(y_test, y_pred):.2f} 權重單位")

        # 5. [核心] 特徵重要性分析：看 AI 到底聽誰的？
        importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\n--- 🔍 AI 決策依據排行 (Top 5) ---")
        print(importances.head(5))

        # --- [Step 6 訓練完後加入這段] ---
        import joblib

        # 儲存模型檔案
        joblib.dump(rf_model, 'retail_staffing_model_v1.pkl')
        # 儲存特徵名稱清單 (這很重要，預測時欄位順序不能錯)
        joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
        
        print("\n✅ 模型已打包完成：retail_staffing_model_v1.pkl")