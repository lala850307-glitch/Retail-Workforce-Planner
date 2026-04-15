import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import joblib

# ======================
# 初始化
# ======================
load_dotenv()
engine = create_engine(os.getenv("POSTGRESQL_URL"))

model = joblib.load('/Users/laylatang8537/Documents/.vscode/門市人力配置/模型/retail_staffing_model_v1.pkl')
feature_cols = joblib.load('/Users/laylatang8537/Documents/.vscode/門市人力配置/模型/feature_columns.pkl')

# ======================
# 參數設定
# ======================
TARGET_STORE_ID = 2
GROWTH_RATE = 0.05
DAILY_MIN_BASE = 1.5

HOURLY_CAPACITY = 2.5
SLOT_DURATION = 4
SLOT_CAPACITY_PER_PERSON = HOURLY_CAPACITY * SLOT_DURATION

TIME_BUFFERS = {
    "morning": 1.00,
    "afternoon": 1.15,
    "evening": 1.30
}

# ======================
# 工具函式
# ======================

def get_period(h):
    if 9 <= h < 13: return "morning"
    if 13 <= h < 17: return "afternoon"
    if 17 <= h <= 21: return "evening"
    return "other"


def load_last_year_data():
    query = f"""
    SELECT 
        record_timestamp::date as sale_date,
        EXTRACT(HOUR FROM record_timestamp)::int AS hour,
        SUM(transaction_count) as trans,
        SUM(sales_amount) as sales,
        SUM(customer_traffic) as customer_traffic
    FROM store_optimization.store_performance_data
    WHERE record_timestamp >= '2025-03-30'
      AND record_timestamp <= '2025-07-05'
      AND store_id = {TARGET_STORE_ID}
    GROUP BY 1, 2
    """
    df = pd.read_sql(query, engine)
    df['period'] = df['hour'].apply(get_period)
    return df


def build_ai_features(day, p, p_trans, day_raw):
    """
    建立 AI feature（修正版）
    """

    p_ly_data = day_raw[day_raw['period'] == p]

    # --- 動態 conversion rate ---
    if not p_ly_data.empty and 'customer_traffic' in p_ly_data.columns:
        total_trans = p_ly_data['trans'].sum()
        total_traffic = p_ly_data['customer_traffic'].sum()

        if total_traffic > 0:
            conversion_rate = total_trans / total_traffic
        else:
            conversion_rate = 0.12
    else:
        conversion_rate = 0.12

    # --- 推估人流 ---
    customer_traffic = p_trans / max(conversion_rate, 0.01)
    traffic_log = np.log1p(customer_traffic)

    # --- 建立 feature ---
    feature_dict = {}

    for col in feature_cols:
        if col == "hour":
            feature_dict[col] = 11 if p == "morning" else (15 if p == "afternoon" else 19)

        elif col == "day_of_week":
            feature_dict[col] = day.weekday()

        elif col == "customer_traffic":
            feature_dict[col] = customer_traffic

        elif col == "traffic_log":
            feature_dict[col] = traffic_log

        else:
            feature_dict[col] = 0  # 若模型不能吃 NaN，這裡維持 0（較安全）

    return pd.DataFrame([feature_dict])


def rule_based_staff(p_trans, period):
    base_demand = p_trans / SLOT_CAPACITY_PER_PERSON
    return (base_demand * TIME_BUFFERS.get(period, 1.0)) + 0.5


def hybrid_decision(logic_val, ai_val):
    """
    改良融合邏輯（避免 AI 只會加人）
    """
    if ai_val > logic_val * 1.2:
        return ai_val
    elif ai_val < logic_val * 0.8:
        return logic_val
    else:
        return (logic_val + ai_val) / 2


# ======================
# 主流程
# ======================

def get_q2_hybrid_forecast_report():

    ly_data = load_last_year_data()

    q2_days = pd.date_range(start='2026-04-01', end='2026-06-30', freq='D')
    final_data = []

    for day in q2_days:

        ly_target_date = (day - pd.Timedelta(weeks=52)).date()
        day_raw = ly_data[ly_data['sale_date'] == ly_target_date]

        if day_raw.empty:
            continue

        p_res = {}
        day_total_work_hours = 0
        day_est_sales = 0
        day_est_trans = 0

        for p in ["morning", "afternoon", "evening"]:

            # --- 去年交易 ---
            p_ly_trans = day_raw[day_raw['period'] == p]['trans'].sum()

            # --- 成長預估 ---
            p_trans = p_ly_trans * (1 + GROWTH_RATE)

            # --- Rule-based ---
            logic_staff = rule_based_staff(p_trans, p)

            # --- AI ---
            features = build_ai_features(day, p, p_trans, day_raw)
            ai_staff = model.predict(features[feature_cols])[0]

            # --- Hybrid ---
            final_staff = hybrid_decision(logic_staff, ai_staff)

            # --- rounding ---
            final_staff = round(max(1.5, final_staff) * 2) / 2

            p_res[p] = final_staff

            day_total_work_hours += (final_staff * SLOT_DURATION)
            day_est_trans += p_trans

            day_est_sales += (
                day_raw[day_raw['period'] == p]['sales'].sum() * (1 + GROWTH_RATE)
            )

        daily_total = round(day_total_work_hours / 8, 2)
        gap = round(max(0.0, daily_total - DAILY_MIN_BASE), 2)

        final_data.append({
            "預估日期": day.strftime('%Y-%m-%d'),
            "星期": day.strftime('%a'),
            "去年同期日期": ly_target_date.strftime('%Y-%m-%d'),
            "去年同期總成交": int(day_raw['trans'].sum()),
            "預計成交筆數": int(day_est_trans),
            "預估銷售": int(day_est_sales),
            "早上建議": p_res['morning'],
            "中午建議": p_res['afternoon'],
            "晚上建議": p_res['evening'],
            "全天建議人力": daily_total,
            "人力缺口": gap
        })

    return pd.DataFrame(final_data)


# ======================
# 執行
# ======================

if __name__ == "__main__":
    df = get_q2_hybrid_forecast_report()
    df.to_csv('2026_Q2_Hybrid_Forecast.csv', index=False, encoding='utf-8-sig')
    print("✅ 報表產出完成")