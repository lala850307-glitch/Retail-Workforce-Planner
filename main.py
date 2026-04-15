import uvicorn
import os
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from api.forecast_service import get_db_ly_aov, get_month_opts
from fastapi import Form
from api.forecast_service import run_planning_engine, get_db_ly_aov, get_month_opts

# 1. 確定專案根目錄與環境初始化
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 匯入自定義腳本（請確保 scripts 資料夾在根目錄）
try:
    from scripts.model_training_pipeline import (
        validate_input, 
        compute_ideal_staffing_v2, 
        generate_audit_scenarios, 
        build_advanced_features
    )
    print("✅ [核心腳本] Pipeline 載入成功")
except ImportError as e:
    print(f"❌ [核心腳本] 載入失敗: {e}")

# 2. FastAPI 實例與 Metadata
tags_metadata = [
    {"name": "Audit", "description": "人力效率審計與業績平衡分析"},
    {"name": "Forecast", "description": "AI 門市智慧排班與未來業績預估"},
    {"name": "頁面導覽", "description": "HTML 頁面渲染入口"}
]

app = FastAPI(
    title="Retail AI Intelligence Portal",
    openapi_tags=tags_metadata
)

# 3. 靜態檔案掛載 (解決 CSS 抓不到的核心區塊)
# 這裡對齊妳 HTML 裡的 url_for('stylesheet', path='styles.css')
STATIC_CSS_DIR = BASE_DIR / "stylesheet"
if STATIC_CSS_DIR.exists():
    app.mount("/stylesheet", StaticFiles(directory=str(STATIC_CSS_DIR)), name="stylesheet")
    print(f"📂 靜態資源掛載成功: {STATIC_CSS_DIR}")
else:
    print(f"⚠️ 找不到 stylesheet 資料夾: {STATIC_CSS_DIR}")

# 4. 模板與路由載入
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 嘗試載入子模組路由
try:
    from api.audit_service import router as audit_router, engine, model, feature_cols
    app.include_router(audit_router, prefix="/audit", tags=["Audit"])
    print("✅ [模組 A] Audit 服務載入成功")
except Exception as e:
    print(f"❌ [模組 A] Audit 載入失敗: {e}")
    engine = model = feature_cols = None

try:
    from api.forecast_service import router as forecast_router, run_planning_engine, get_db_ly_aov, get_month_opts
    app.include_router(forecast_router, prefix="/forecast", tags=["Forecast"])
    print("✅ [模組 B] Forecast 服務載入成功")
except Exception as e:
    print(f"❌ [模組 B] Forecast 載入失敗: {e}")

# --- 核心運算函數 (Audit 實績抓取) ---
async def get_real_benchmark():
    """抓取 2026 模擬資料並執行 AI 審計計算"""
    if engine is None: return None
    try:
        query = "SELECT * FROM store_optimization.store_performance_data_2026_sim WHERE record_timestamp >= '2026-01-01'"
        df = pd.read_sql(query, engine)
        if df.empty: return None

        df['record_timestamp'] = pd.to_datetime(df['record_timestamp'])
        df['hour'] = df['record_timestamp'].dt.hour
        df['day_of_week'] = df['record_timestamp'].dt.dayofweek
        df_clean = validate_input(df)
        df_audit = generate_audit_scenarios(df_clean)
        df_processed = compute_ideal_staffing_v2(df_audit, handle_capacity=2.5)
        
        X, _ = build_advanced_features(df_processed)
        if model is not None and feature_cols is not None:
            valid_features = [f for f in feature_cols if f in X.columns]
            df_processed['ai_recommended_weight'] = model.predict(X[valid_features])
        else:
            df_processed['ai_recommended_weight'] = df_processed['ideal_weight_demand']

        df_processed['wage_waste'] = (df_processed['actual_weight'] - df_processed['ai_recommended_weight']).clip(lower=0) * 230
        
        return {
            "cr": round((df['transaction_count'].sum() / (df['customer_traffic'].sum() + 1e-6)) * 100, 2),
            "aov": round(df['sales_amount'].sum() / (df['transaction_count'].sum() + 1e-6), 0),
            "ft": round(df['ft_count'].mean(), 1),
            "pt": round(df['pt_count'].mean(), 1),
            "waste_cost": round(df_processed['wage_waste'].sum(), 0),
            "orig_total_revenue": df['sales_amount'].sum()
        }
    except Exception as e:
        print(f"運算錯誤: {e}")
        return None

# 5. 頁面導覽路由
@app.get("/", tags=["頁面導覽"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/system_architecture", tags=["頁面導覽"])
async def get_architecture_page(request: Request):
    """渲染系統架構圖頁面"""
    return templates.TemplateResponse("system_architecture_diagram.html", {"request": request})

@app.get("/Labor_Efficiency_Audit", tags=["頁面導覽"])
async def get_audit_page(request: Request):
    return templates.TemplateResponse("labor_efficiency_audit.html", {"request": request})

@app.get("/smart_staffing_forecast", tags=["頁面導覽"])
async def get_forecast_page(request: Request):
    # 預設參數 (Store 2, April)
    store_id, month = 2, 4
    ly_aov = get_db_ly_aov(store_id, month) if 'get_db_ly_aov' in globals() else 0
    return templates.TemplateResponse(
        "smart_staffing_forecast.html", 
        {
            "request": request,
            "results": None,
            "store_id": store_id,
            "month": month,
            "ly_aov": ly_aov,
            "month_options": get_month_opts(month) if 'get_month_opts' in globals() else ""
        }
    )

# 6. API 實作
# --- 5. API 邏輯路由 ---

@app.get("/api/init_benchmark", tags=["Audit"])
async def init_benchmark():
    data = await get_real_benchmark()
    if not data:
        return JSONResponse(status_code=404, content={"error": "資料庫內無 2026 Q1 數據"})
    
    # 強制檢查前端需要的欄位，避免 JS 報錯
    required_keys = ["cr", "aov", "ft", "pt", "waste_cost", "waste_hours"]
    for key in required_keys:
        if key not in data:
            data[key] = 0  # 給予預設值防止前端 toLocaleString() 崩潰
            
    return data

@app.post("/api/simulate_decision", tags=["Audit"])
async def simulate_decision(data: dict):
    benchmark = await get_real_benchmark()
    if not benchmark:
        return JSONResponse(status_code=404, content={"error": "無法取得基準數據"})
    
    # 執行模擬計算
    try:
        cr_multiplier = data['target_cr'] / (benchmark['cr'] + 1e-6)
        aov_multiplier = data['target_aov'] / (benchmark['aov'] + 1e-6)
        rev_multiplier = cr_multiplier * aov_multiplier
        estimated_rev_growth = benchmark['orig_total_revenue'] * (rev_multiplier - 1)
        
        # 薪資成本計算 (以季為單位)
        ft_diff = (benchmark['ft'] - data['new_ft']) * 45000 * 3
        pt_diff = (benchmark['pt'] - data['new_pt']) * 32384 * 3
        cost_saved = ft_diff + pt_diff

        insights = []
        if data['target_cr'] > benchmark['cr']:
            insights.append(f"<b>成交率提升</b>：預計帶來 ${round(estimated_rev_growth, 0):,} 的業績增量。")
        if cost_saved > 0:
            insights.append(f"<b>成本優化</b>：人力調整預計節省 ${round(cost_saved, 0):,} 薪資支出。")
        
        if not insights: 
            insights.append("<b>基準對照</b>：目前設定與 Q1 實績一致。")

        return {
            "rev_growth": round(estimated_rev_growth, 0),
            "cost_saved": round(cost_saved, 0),
            "total_benefit": round(estimated_rev_growth + cost_saved, 0),
            "insights": "<br>".join(insights)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"模擬運算出錯: {str(e)}"})

@app.post("/forecast/run", tags=["Forecast"])
async def execute_forecast(
    request: Request,
    store_id: int = Form(...), 
    month: int = Form(...), 
    goal: int = Form(...), 
    target_aov: str = Form("")
):
    """處理表單提交，並將預測結果重新渲染到網頁上"""
    
    # 1. 取得 AOV 設定
    ly_aov = get_db_ly_aov(store_id, month)
    final_aov = int(target_aov) if target_aov.strip() else ly_aov
    results, mode_text = run_planning_engine(store_id, month, goal, final_aov)
    
    # 3. 透過 Jinja2 將乾淨的「資料」送到前端，讓 HTML 自己去畫表格！
    return templates.TemplateResponse(
        "smart_staffing_forecast.html", 
        {
            "request": request, 
            "results": results, 
            "mode_text": mode_text,
            "ly_aov": ly_aov,
            "final_aov": final_aov,
            "store_id": store_id,
            "month": month,
            "goal": goal,
            "month_options": get_month_opts(month)
        }
    )

@app.get("/pages/forecast", tags=["頁面導覽"])
async def get_forecast_page(request: Request):
    # 1. 補齊原本遺漏的初始化運算
    store_id, month = 2, 4
    ly_aov = get_db_ly_aov(store_id, month)
    
    # 2. 傳遞完整的 Context 給模板
    return templates.TemplateResponse(
        request=request, 
        name="smart_staffing_forecast.html", 
        context={
            "results": None, 
            "store_id": store_id, 
            "month": month, 
            "goal": 3000000,
            "ly_aov": ly_aov, # 💡 補上這個
            "month_options": get_month_opts(month) # 💡 補上這個，並確保 HTML 裡用 | safe
        }
    )

# 6. 頁面導覽路由 (渲染入口)
@app.get("/", tags=["頁面導覽"])
async def index(request: Request):
    """主首頁入口"""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/Labor_Efficiency_Audit", tags=["頁面導覽"])
async def get_audit_page(request: Request):
    """人力審計與業績平衡頁面"""
    # 對應妳的 Labor Efficiency Audit.html
    return templates.TemplateResponse(request=request, name="Labor Efficiency Audit.html")

@app.get("/smart_staffing_forecast", tags=["頁面導覽"])
async def get_forecast_page(request: Request):
    """智慧排班預測頁面"""
    # 這裡可以根據妳原本的 get_forecast_page 邏輯加入 AOV 查詢
    return templates.TemplateResponse(request=request, name="smart_staffing_forecast.html", 
    context={
        "results": None, "store_id": 2, "month": 4, "goal": 3000000
        })

# 7. 錯誤處理
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"message": "找不到路徑或 HTML 檔案", "path": str(request.url.path)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)