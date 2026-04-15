import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# 1. 資料庫連線設定 (請替換為你的資訊)
load_dotenv()  # 從 .env 檔案載入環境變數
DB_URL = os.getenv("POSTGRESQL_URL")  # 從環境變數讀取資料庫連線字串
engine = create_engine(DB_URL)

# 2. 生成 140 間門市資料
stores_df = pd.DataFrame({
    'store_id': range(1, 141),
    'store_name': [f'Store_{i}' for i in range(1, 141)],
    'region': np.random.choice(['North', 'Central', 'South'], 140),
    'store_type': np.random.choice(['Mall', 'Street'], 140)
})

# 3. 生成 員工資料 (每店 3正 4工)
staff_data = []
for s_id in range(1, 141):
    for _ in range(3): staff_data.append([s_id, 'FT', 32000]) # 正職
    for _ in range(4): staff_data.append([s_id, 'PT', 0])     # 工讀時薪制
staff_df = pd.DataFrame(staff_data, columns=['store_id', 'role', 'monthly_base_salary'])

# 4. 寫入基本表 (Schema 名稱務必對齊)
stores_df.to_sql('stores', engine, schema='store_optimization', if_exists='append', index=False)
staff_df.to_sql('staff', engine, schema='store_optimization', if_exists='append', index=False)

print("--- 門市與員工資料已寫入成功 ---")