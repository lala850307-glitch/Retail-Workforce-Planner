import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import calendar

# =============================================================================
# 1. 搬運你原本的所有定義 (必須包含在同一個檔案內)
# =============================================================================

STAFF_WEIGHTS = {"ft": 1.0, "pt": 0.5}
TIME_SLOTS = {
    "morning":   {"start": 9,  "end": 12, "buffer": 1.00},
    "afternoon": {"start": 13, "end": 16, "buffer": 1.15},
    "evening":   {"start": 16, "end": 22, "buffer": 1.30},
}

def fetch_data(table, start, end, store_id=None):
    load_dotenv()
    engine = create_engine(os.getenv("POSTGRESQL_URL"))
    query = f"""
    SELECT record_timestamp, EXTRACT(HOUR FROM record_timestamp)::int AS hour,
           customer_traffic, transaction_count, sales_amount, ft_count, pt_count
    FROM store_optimization.{table}
    WHERE record_timestamp >= '{start}' AND record_timestamp < ('{end}'::date + INTERVAL '1 day')
    """
    if store_id: query += f" AND store_id = {store_id}"
    return pd.read_sql(query, engine)

def validate_input(df):
    data = df.copy()
    data["actual_weight"] = (data["ft_count"] * 1.0 + data["pt_count"] * 0.5)
    def get_slot_info(h):
        for name, cfg in TIME_SLOTS.items():
            if cfg["start"] <= h < cfg["end"]: return cfg["buffer"]
        return 1.0
    data["time_buffer"] = data["hour"].apply(get_slot_info)
    return data

def compute_ideal_staffing_v2(df, handle_capacity=2.5):
    data = df.copy()
    data["ideal_weight_demand"] = (data["transaction_count"] / handle_capacity) * data["time_buffer"]
    data["ideal_weight_demand"] = data["ideal_weight_demand"].clip(lower=1.0)
    # 簡化 sim_weight 計算供審計使用
    data["sim_weight"] = np.ceil(data["ideal_weight_demand"] * 2) / 2
    return data

def run_financial_backtest_audit(data, wage=230):
    df = data.copy()
    df["weight_diff"] = df["actual_weight"] - df["sim_weight"]
    df["cost_saving"] = (df["weight_diff"] * wage).clip(lower=0)
    df["efficiency_score"] = (df["ideal_weight_demand"] / df["actual_weight"].replace(0, np.nan) * 100).clip(0, 100)
    return df

# =============================================================================
# 2. 導出邏輯 (這是這份檔案的新任務)
# =============================================================================

def run_and_export_audit(store_id, start_date, end_date):
    load_dotenv()
    engine = create_engine(os.getenv("POSTGRESQL_URL"))
    
    print(f"--- 🚀 開始執行 {store_id} 號店審計程序 ---")
    
    # 執行運算鏈
    df_raw = fetch_data("store_performance_data", start_date, end_date, store_id)
    if df_raw.empty: return print("❌ 沒資料")
    
    df_clean = validate_input(df_raw)
    df_ideal = compute_ideal_staffing_v2(df_clean)
    final_df = run_financial_backtest_audit(df_ideal)

    # A. 製作小時矩陣 (小時平均)
    hour_rep = final_df.groupby("hour").agg(
        actual_staff_weight=("actual_weight", "mean"),
        ideal_staff_weight=("ideal_weight_demand", "mean"),
        weight_diff=("weight_diff", "mean"),
        utilization_rate=("efficiency_score", "mean")
    ).reset_index()
    hour_rep['store_id'] = store_id
    
    # B. 製作每日總結 (算錢用)
    daily_audit = final_df.copy()
    daily_audit['audit_date'] = daily_audit['record_timestamp'].dt.date
    daily_summary = daily_audit.groupby('audit_date').agg(
        daily_sales=("sales_amount", "sum"),
        daily_waste_salary=("cost_saving", "sum"),
        avg_utilization=("efficiency_score", "mean")
    ).reset_index()
    daily_summary['store_id'] = store_id

    # 寫入資料庫
    print("--- 💾 正在寫入資料庫 ---")
    with engine.begin() as conn:
        hour_rep.to_sql('audit_hourly_matrix', conn, schema='store_optimization', if_exists='replace', index=False)
        daily_summary.to_sql('audit_daily_summary', conn, schema='store_optimization', if_exists='replace', index=False)

    print(f"✅ 審計匯出成功！去打開 Power BI 吧。")

if __name__ == "__main__":
    run_and_export_audit(store_id=1, start_date="2025-01-01", end_date="2025-12-31")