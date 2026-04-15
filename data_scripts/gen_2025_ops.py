import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. 載入環境變數與連線
load_dotenv()
DB_URL = os.getenv("POSTGRESQL_URL")
if not DB_URL:
    raise ValueError("找不到 DB_URL，請檢查 .env 檔案！")
engine = create_engine(DB_URL)

# 2. 模擬參數設定
START_DATE = datetime(2025, 1, 1)
HOURS_IN_YEAR = 365 * 24
BATCH_SIZE = 10 

# 核心事實參數 (配合新的人流定義)
MAX_CR_POTENTIAL = 0.28  # 成交率上限設為 28%
AVG_TICKET_SIZE = 1800   # 修改客單價為你之前提到的 1800
PT_HOURLY_RATE = 190
FT_BASE_HOURLY_FOR_OT = 250 
SERVICE_CAPACITY = 2.5   # 修正產能：1 個權重每小時合理處理 4.5 單

def save_to_db(records):
    if not records:
        return
    
    # 🔥 關鍵：從清單中刪除 'is_peak_month'
    df = pd.DataFrame(records, columns=[
        'store_id', 'record_timestamp', 'sales_amount', 'transaction_count', 
        'customer_traffic', 'ft_count', 'pt_count', 'actual_ot_hours', 
        'pt_cost', 'ot_cost' 
    ])
    
    try:
        df.to_sql('store_performance_data', engine, schema='store_optimization', 
                  if_exists='append', index=False, chunksize=5000)
        print(f"--- [Success] 已寫入 {len(df)} 筆數據，由資料庫自動計算旺季標記 ---")
    except Exception as e:
        print(f"!!! [Error] 資料寫入失敗：{e}")

def generate_abuse_fact_data():
    all_records = []
    try:
        stores = pd.read_sql("SELECT store_id FROM store_optimization.stores", engine)
        store_ids = stores['store_id'].tolist()
        print(f"開始處理 {len(store_ids)} 間門市的現實化模擬...")
    except Exception as e:
        print(f"讀取失敗：{e}"); return

    for idx, s_id in enumerate(store_ids):
        print(f"[{idx+1}/{len(store_ids)}] 正在生成 Store {s_id}...")
        
        for h_idx in range(HOURS_IN_YEAR):
            ts = START_DATE + timedelta(hours=h_idx)
            month, day_of_week, hour = ts.month, ts.weekday(), ts.hour
            is_peak_month = month in [1, 5, 9, 10, 11]
            is_weekend = day_of_week >= 5
            
            # --- [🔥 核心邏輯修正：符合用戶定義的現實限制] ---
            
            # 1. 決定人流上限 (Traffic Cap)
            if is_peak_month:
                traffic_cap = 30 if is_weekend else 20
            else:
                traffic_cap = 20 if is_weekend else 10
            
            # 2. 時段權重因子 (尖峰 11-14, 18-21 點)
            if (11 <= hour <= 14) or (18 <= hour <= 21):
                time_factor = 1.2
            elif hour < 9 or hour > 22:
                time_factor = 0.2 # 關店或深夜時段
            else:
                time_factor = 0.8

            # 3. 生成隨機人流 (上限 * 時段權重 * 隨機擾動)
            customer_traffic = int(traffic_cap * time_factor * np.random.uniform(0.7, 1.1))
            customer_traffic = max(2, customer_traffic) # 保底 2 人
            
            # --- [人力排班模擬：維持原邏輯，但這會顯得過剩] ---
            ft_count, ot_hours = 0, 0.0
            if 9 <= hour <= 20:
                ft_count += 1
                if hour >= 18: ot_hours += 1.0
            if 12 <= hour <= 21:
                ft_count += 1
            if 13 <= hour <= 21:
                ft_count += 1
            
            pt_count = np.random.choice([1, 2]) if 11 <= hour <= 21 else 0
            
            # --- [產能與成交邏輯] ---
            labor_score = (ft_count * 1.0) + (pt_count * 0.5)
            # 理想成交筆數 (20%-28% 隨機成交率)
            ideal_transactions = customer_traffic * np.random.uniform(0.20, MAX_CR_POTENTIAL)
            # 產能限制
            service_limit = labor_score * SERVICE_CAPACITY
            transaction_count = int(min(ideal_transactions, service_limit))
            
            # 業績與成本
            sales_amount = transaction_count * np.random.uniform(1500, 2100)
            pt_cost = pt_count * PT_HOURLY_RATE
            ot_cost = ot_hours * (FT_BASE_HOURLY_FOR_OT * 1.5)

            all_records.append([
                s_id, ts, sales_amount, transaction_count, customer_traffic,
                ft_count, pt_count, ot_hours, pt_cost, ot_cost
                ])
            
        if (idx + 1) % BATCH_SIZE == 0:
            save_to_db(all_records)
            all_records = []

    if all_records:
        save_to_db(all_records)

if __name__ == "__main__":
    print(f"任務開始：{datetime.now()}")
    generate_abuse_fact_data()
    print(f"任務結束：{datetime.now()}")