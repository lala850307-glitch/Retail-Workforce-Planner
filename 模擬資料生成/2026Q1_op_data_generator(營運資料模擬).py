import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. 載入連線資訊
load_dotenv()
DB_URL = os.getenv("POSTGRESQL_URL")
engine = create_engine(DB_URL)

# 2. 審計參數設定 (2026 Q1)
START_DATE = datetime(2026, 1, 1)
DAYS_IN_Q1 = 90  # 1月到3月底大約 90 天
AVG_TICKET_SIZE = 1800 
SERVICE_CAPACITY = 2.5 # 產能標竿

def save_to_db(records):
    if not records: return
    # 這裡存入 store_performance_data，模型會自動讀取 record_timestamp 來判斷旺季
    df = pd.DataFrame(records, columns=[
        'store_id', 'record_timestamp', 'sales_amount', 'transaction_count', 
        'customer_traffic', 'ft_count', 'pt_count', 'actual_ot_hours', 
        'pt_cost', 'ot_cost' 
    ])
    try:
        df.to_sql('store_performance_data_2026_sim', engine, schema='store_optimization', 
                  if_exists='append', index=False, chunksize=5000)
        print(f"--- [Audit Ready] 已寫入 {len(df)} 筆 2026 Q1 審計數據 ---")
    except Exception as e:
        print(f"!!! [Error] 寫入失敗：{e}")

def generate_2026_q1_audit_data():
    all_records = []
    # 假設針對 Store 1 進行回測審計
    s_id = 1 
    print(f"開始生成 Store {s_id} 2026 Q1 模擬數據...")

    for d_idx in range(DAYS_IN_Q1):
        for hour in range(24):
            ts = START_DATE + timedelta(days=d_idx, hours=hour)
            month, day_of_week = ts.month, ts.weekday()
            
            # --- 💡 核心審計邏輯 ---
            is_peak = (month == 1) # 1月過年是旺季
            is_weekend = day_of_week >= 5
            
            # 1. 模擬人流 (1月旺季 vs 2/3月淡季)
            traffic_base = 25 if is_peak else 12
            if is_weekend: traffic_base *= 1.5
            
            # 營業時間限制 (09-22)
            if 9 <= hour <= 22:
                time_factor = 1.2 if (11 <= hour <= 14 or 18 <= hour <= 21) else 0.8
                customer_traffic = int(traffic_base * time_factor * np.random.uniform(0.8, 1.2))
            else:
                customer_traffic = 0

            # 2. 模擬「僵化排班」 (這是你要抓的妖)
            # 店長習慣下午 (13-17) 固定排 4FT + 2PT，不管淡旺季
            if 9 <= hour <= 22:
                if 13 <= hour <= 17:
                    ft_count, pt_count = 4, 2 # 💡 這裡在 2, 3 月會產生極大浪費
                elif 18 <= hour <= 21:
                    ft_count, pt_count = 3, 1 # 尖峰正常排班
                else:
                    ft_count, pt_count = 2, 0 # 基本排班
            else:
                ft_count, pt_count = 0, 0

            # 3. 成交與業績 (1月 AOV 較高)
            labor_score = (ft_count * 1.0) + (pt_count * 0.5)
            # 產能限制：如果人排太多，成交筆數也不會超過人流
            potential_trans = customer_traffic * 0.15 # 假設 15% 成交率
            transaction_count = int(min(potential_trans, labor_score * SERVICE_CAPACITY))
            
            current_aov = AVG_TICKET_SIZE * 1.3 if is_peak else AVG_TICKET_SIZE
            sales_amount = transaction_count * current_aov
            
            # 成本計算 (190 為 PT 時薪)
            pt_cost = pt_count * 190
            ot_hours = 1.0 if (is_peak and hour >= 18) else 0.0 # 僅旺季晚上模擬加班
            ot_cost = ot_hours * (250 * 1.5)

            if customer_traffic > 0 or ft_count > 0:
                all_records.append([
                    s_id, ts, sales_amount, transaction_count, customer_traffic,
                    ft_count, pt_count, ot_hours, pt_cost, ot_cost
                ])

    save_to_db(all_records)

if __name__ == "__main__":
    # 清除舊的 2026 Q1 數據 (選用，避免重複寫入)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM store_optimization.store_performance_data_2026_sim  WHERE record_timestamp >= '2026-01-01'"))
        conn.commit()
    
    generate_2026_q1_audit_data()