import pandas as pd
import joblib
import os
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
# 確保這裡包含 compute_ideal_staffing_v2，如果你的原本檔案沒有，我下面會補定義
from model_training_pipeline import build_advanced_features, generate_audit_scenarios, validate_input

load_dotenv()

# --- 💡 補上缺失的產能審計函式 ---
def compute_ideal_staffing_v2(df, handle_capacity=2.5):
    data = df.copy()
    # 取得時段緩衝係數 (若 validate_input 有跑，應該會有 time_buffer)
    buffer = data.get("time_buffer", 1.0)
    data["ideal_weight_demand"] = (data["transaction_count"] / handle_capacity) * buffer
    data["ideal_weight_demand"] = data["ideal_weight_demand"].clip(lower=1.0)
    return data

def run_audit():
    # 1. 載入模型與工具 (路徑正確)
    model_path = '/Users/laylatang8537/Documents/.vscode/門市人力配置/模型/retail_staffing_model_v1.pkl'
    feature_path = '/Users/laylatang8537/Documents/.vscode/門市人力配置/模型/feature_columns.pkl'
    
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)
    
    engine = create_engine(os.getenv("POSTGRESQL_URL"))
    
    # 2. 從資料庫讀取數據 (必須先抓資料！)
    query = "SELECT * FROM store_optimization.store_performance_data_2026_sim WHERE record_timestamp >= '2026-01-01'"
    df_q1 = pd.read_sql(query, engine)
    
    print(f"--- 數據抓取測試：抓到 {len(df_q1)} 筆資料 ---")

    if df_q1.empty:
        print("❌ 找不到 2026 Q1 資料，請確認是否已執行資料生成腳本！")
        return

    # 3. 手動修正時間特徵 (資料庫欄位補位)
    df_q1['record_timestamp'] = pd.to_datetime(df_q1['record_timestamp'])
    df_q1['hour'] = df_q1['record_timestamp'].dt.hour
    df_q1['day_of_week'] = df_q1['record_timestamp'].dt.dayofweek
    df_q1['is_peak_month'] = df_q1['record_timestamp'].dt.month.apply(lambda x: 1 if x in [1, 5, 9, 10, 11] else 0)

    # 4. 執行審計數據流
    df_clean = validate_input(df_q1)
    df_audit = generate_audit_scenarios(df_clean)
    
    # 💡 補齊 build_advanced_features 吵著要的 'ideal_weight_demand'
    df_processed = compute_ideal_staffing_v2(df_audit, handle_capacity=2.5)

    # 5. 特徵工程
    X_q1, _ = build_advanced_features(df_processed)

    # 6. AI 預測理想權重
    # 確保只餵入模型當初學過的 feature_cols
    df_processed['ai_recommended_weight'] = model.predict(X_q1[feature_cols])
    
    # 7. 計算浪費 (實際權重 - AI 建議權重)
    df_processed['actual_weight'] = df_processed['ft_count'] * 1.0 + df_processed['pt_count'] * 0.5
    df_processed['wage_waste'] = (df_processed['actual_weight'] - df_processed['ai_recommended_weight']) * 230
    df_processed['wage_waste'] = df_processed['wage_waste'].clip(lower=0)
    
    # 8. 印出審計報告
    print("\n" + "="*40)
    print("🔍 2026 Q1 門市薪資浪費審計報告")
    print("="*40)
    print(f"💰 Q1 總計浪費薪資: ${df_processed['wage_waste'].sum():,.0f}")
    
    # 月份分析
    print("\n📅 各月份浪費金額分布:")
    monthly_waste = df_processed.groupby(df_processed['record_timestamp'].dt.month)['wage_waste'].sum()
    print(monthly_waste)

if __name__ == "__main__":
    print("🚀 審計任務開始執行...")
    run_audit()
    print("🏁 任務執行完畢。")