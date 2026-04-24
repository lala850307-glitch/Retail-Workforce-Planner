import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import joblib
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import calendar
from fastapi import APIRouter, Form, Request

# ======================
# 1. 初始化與路徑設定
# ======================
load_dotenv()
router = APIRouter()
engine = create_engine(os.getenv("POSTGRESQL_URL"))

# 請確保路徑與你的電腦一致
MODEL_PATH = "/Users/laylatang8537/Documents/vscold/Retail-Workforce-Planner/Retail-Workforce-Planner/models/retail_staffing_model_v1.pkl"
FEAT_PATH = "/Users/laylatang8537/Documents/vscold/Retail-Workforce-Planner/Retail-Workforce-Planner/models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEAT_PATH)

# 常規參數 (可依實際需求調整)
DEFAULT_CAPACITY = 5.0 
DEFAULT_BUFFERS = {"morning": 1.0, "afternoon": 1.15, "evening": 1.30}

# ======================
# 2. 核心運算引擎 (含活動模式判定)
# ======================
def run_planning_engine(store_id, month, month_goal, input_aov):
    # 一次抓取全月資料 (優化效能)
    query = f"""
        SELECT record_timestamp::date as sale_date, 
               transaction_count as trans, 
               sales_amount as sales, 
               customer_traffic, 
               EXTRACT(HOUR FROM record_timestamp)::int AS hour 
        FROM store_optimization.store_performance_data 
        WHERE store_id = {store_id} 
          AND EXTRACT(MONTH FROM record_timestamp) = {month} 
          AND EXTRACT(YEAR FROM record_timestamp) = 2025
    """
    ly_df = pd.read_sql(query, engine)
    
    if ly_df.empty:
        return None, "No Data"

    # 💡 活動模式自動判定：當 AOV 顯著偏高時，啟動「精緻服務模式」
    # 若 AOV >= 4000，將產能從 10 降為 5 (模擬週年慶高壓服務)
    is_campaign = input_aov >= 4000
    current_capacity = 5.0 if is_campaign else DEFAULT_CAPACITY
    current_buffers = {"morning": 1.2, "afternoon": 1.5, "evening": 1.8} if is_campaign else DEFAULT_BUFFERS
    mode_text = "🔥 活動高壓模式 (產能校正)" if is_campaign else "✅ 常規營運模式"

    def get_p(h):
        if 9 <= h < 13: return "morning"
        elif 13 <= h < 17: return "afternoon"
        elif 17 <= h <= 21: return "evening"
        return "other"
    
    ly_df['period'] = ly_df['hour'].apply(get_p)

    # 去年 AOV 基準
    ly_aov_base = ly_df['sales'].sum() / max(ly_df['trans'].sum(), 1)

    # 業績權重分配
    ly_daily_sales = ly_df.groupby('sale_date')['sales'].sum()
    _, last_day = calendar.monthrange(2026, month)
    days = pd.date_range(f"2026-{month:02d}-01", f"2026-{month:02d}-{last_day}")
    weights = {d: ly_daily_sales.get((d - pd.Timedelta(weeks=52)).date(), ly_daily_sales.mean()) for d in days}
    total_w = sum(weights.values())

    results = []
    for d in days:
        ly_date = (d - pd.Timedelta(weeks=52)).date()
        day_raw = ly_df[ly_df['sale_date'] == ly_date]
        
        # 業績與筆數分配
        day_target_sales = month_goal * (weights[d] / total_w)
        day_target_trans = day_target_sales / input_aov
        
        day_h = 0
        p_res = {}
        for p in ["morning", "afternoon", "evening"]:
            p_ly = day_raw[day_raw['period'] == p]
            p_ratio = p_ly['sales'].sum() / day_raw['sales'].sum() if not day_raw.empty else 0.33
            p_trans = day_target_trans * p_ratio
            
            # 精確轉換率推算
            conv = p_ly['trans'].sum() / p_ly['customer_traffic'].sum() if (not p_ly.empty and p_ly['customer_traffic'].sum() > 0) else 0.12
            traffic = p_trans / max(conv, 0.01)
            
            logic = (p_trans / current_capacity) * current_buffers.get(p, 1.0)
            
            feat = {col: 0 for col in feature_cols}
            feat.update({
                "hour": 15, "day_of_week": d.weekday(), 
                "customer_traffic": traffic, "traffic_log": np.log1p(traffic)
            })
            
            try:
                ai = model.predict(pd.DataFrame([feat])[feature_cols])[0]
            except:
                ai = logic
            
            # 活動模式下，決策邏輯轉向保守(高標)，確保人力充足
            final = max(logic, ai) if is_campaign else (logic + ai) / 2
            final = round(max(1.5, final) * 2) / 2
            p_res[p] = final
            day_h += (final * 4)

        results.append({
            "date": d.strftime("%m-%d"),
            "weekday": d.strftime("%a"),
            "sales": int(day_target_sales),
            "trans": int(day_target_trans),
            "m": p_res["morning"],
            "a": p_res["afternoon"],
            "e": p_res["evening"],
            "staff": round(day_h / 8, 2)
        })
    return results, mode_text

# ======================
# 3. 輔助功能
# ======================
def get_db_ly_aov(store_id, month):
    query = f"SELECT SUM(sales_amount) as s, SUM(transaction_count) as t FROM store_optimization.store_performance_data WHERE store_id={store_id} AND EXTRACT(MONTH FROM record_timestamp)={month} AND EXTRACT(YEAR FROM record_timestamp)=2025"
    res = pd.read_sql(query, engine)
    if not res.empty and res['t'][0] > 0:
        return int(res['s'][0] / res['t'][0])
    return 1800

def get_month_opts(selected=4):
    return "".join([f"<option value='{i}' {'selected' if i==selected else ''}>{i}月</option>" for i in range(1, 13)])

def get_store_opts(selected=2):
    stores = [
        (1, "忠孝門市 #1"), (2, "忠孝Sogo #2"),
        (3, "復興Sogo #3"), (4, "敦化Sogo #4"), (5, "天母Sogo #5")
    ]
    return "".join([f"<option value='{v}' {'selected' if int(v)==int(selected) else ''}>{n}</option>" for v, n in stores])

# ======================
# 4. HTML 模板
# ======================
LAYOUT = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <style>
        :root {{ --primary: #2c3e50; --accent: #3498db; --bg: #f4f7f6; --border: #e0e0e0; --danger: #e74c3c; }}
        body {{ font-family: 'PingFang TC', sans-serif; background: var(--bg); color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .nav-bar {{ display: flex; gap: 15px; align-items: flex-end; margin-bottom: 25px; padding: 20px; border: 1px solid var(--border); border-radius: 8px; background: #fff; }}
        .info-box {{ background: #eef7ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid var(--accent); }}
        .campaign-box {{ background: #fff5f5; border-left: 5px solid var(--danger); padding: 15px; border-radius: 8px; margin-bottom: 20px; color: var(--danger); font-weight: bold; }}
        .form-group {{ display: flex; flex-direction: column; gap: 5px; }}
        label {{ font-size: 11px; font-weight: bold; color: #7f8c8d; }}
        select, input {{ padding: 10px; border: 1px solid var(--border); border-radius: 4px; width: 160px; outline: none; }}
        button {{ background: var(--accent); color: white; border: none; padding: 10px 30px; border-radius: 4px; cursor: pointer; font-weight: bold; height: 42px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #f8f9fa; color: var(--primary); padding: 15px; border-bottom: 2px solid var(--border); }}
        td {{ padding: 12px; border-bottom: 1px solid var(--border); text-align: center; }}
        .weekend {{ color: var(--danger); font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>系統 </h2>
        <form method="post" action="./run" class="nav-bar">
            <div class="form-group"><label>店別</label>
                <select name="store_id" id="store_id" onchange="updateLYAOV()">
                    {store_options}
                </select>
            </div>
            <div class="form-group"><label>月份</label><select name="month" id="month" onchange="updateLYAOV()">{month_options}</select></div>
            <div class="form-group"><label>月業績目標</label><input type="number" name="goal" value="{goal}"></div>
            <div class="form-group"><label>設定今年 AOV</label><input type="number" name="target_aov" id="target_aov" value="{target_aov}" placeholder="請參考下方數據"></div>
            <button type="submit">執行規劃</button>
        </form>
        <div id="aov_display_area">{initial_info}</div>
        {content}
    </div>
    <script>
        async function updateLYAOV() {{
            const sid = document.getElementById('store_id').value;
            const m = document.getElementById('month').value;
            const displayArea = document.getElementById('aov_display_area');
            displayArea.innerHTML = '<div class="info-box">⏳ 正在查詢去年數據...</div>';
            try {{
                const res = await fetch(`./get_ly_aov?store_id=${{sid}}&month=${{m}}`);
                const data = await res.json();
                displayArea.innerHTML = `<div class="info-box">💡 去年同期平均 AOV 為 <span style="color:#27ae60; font-weight:bold;">$${{data.aov.toLocaleString()}}</span></div>`;
                document.getElementById('target_aov').placeholder = `去年平均: ${{data.aov}}`;
            }} catch (e) {{ displayArea.innerHTML = '<div class="info-box" style="color:red;">⚠️ 無法取得去年數據</div>'; }}
        }}
        window.onload = updateLYAOV;
    </script>
</body>
</html>
"""

# ======================
# 5. API 路由
# ======================
@router.get("/", response_class=HTMLResponse)
def home():
    ly_aov = get_db_ly_aov(2, 4)
    info = f"<div class='info-box'>💡 去年同期參考 AOV 為 <span style='color:#27ae60; font-weight:bold;'>${ly_aov:,}</span></div>"
    return LAYOUT.format(
        store_options=get_store_opts(2),
        month_options=get_month_opts(4), 
        goal=3000000,
        target_aov="",
        initial_info=info, 
        content=""
    )

@router.get("/get_ly_aov")
async def api_aov(store_id: int, month: int):
    val = get_db_ly_aov(store_id, month)
    return JSONResponse({"aov": val})

@router.post("/run", response_class=HTMLResponse)
def run(store_id: int = Form(...), month: int = Form(...), goal: int = Form(...), target_aov: str = Form(...)):
    ly_aov = get_db_ly_aov(store_id, month)
    final_aov = int(target_aov) if target_aov and target_aov.strip() else ly_aov
    
    data, mode_text = run_planning_engine(store_id, month, goal, final_aov)
    
    mode_style = "campaign-box" if "活動" in mode_text else "info-box"
    header_info = f"<div class='{mode_style}'>模式：{mode_text} | 本次設定 AOV: ${final_aov:,} (去年同期: ${ly_aov:,})</div>"
    
    rows = ""
    for r in data:
        wk_cls = "weekend" if r["weekday"] in ["Sat", "Sun"] else ""
        rows += f"<tr><td>{r['date']}</td><td class='{wk_cls}'>{r['weekday']}</td><td>${r['sales']:,}</td><td>{r['trans']}</td><td>{r['m']}</td><td>{r['a']}</td><td>{r['e']}</td><td style='font-weight:bold;'>{r['staff']}人</td></tr>"
    
    table = f"<table><thead><tr><th>日期</th><th>星期</th><th>預計業績</th><th>預計成交</th><th>早</th><th>中</th><th>晚</th><th>全天建議</th></tr></thead><tbody>{rows}</tbody></table>"
    
    return LAYOUT.format(
        store_options=get_store_opts(store_id),
        month_options=get_month_opts(month), 
        goal=goal,
        target_aov=target_aov,
        initial_info=header_info, 
        content=table
    )

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    test_app = FastAPI()
    test_app.include_router(router)
    print("🚀 正在以單獨模式啟動 Forecast Service (http://127.0.0.1:8000)...")
    uvicorn.run(test_app, host="127.0.0.1", port=8000)
