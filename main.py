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

# 這裡假設妳的 scripts 與相關模型變數已在外部定義或匯入
# 若 scripts 在 api 內部匯入，請確保路徑正確
from scripts.model_training_pipeline import (
    validate_input, 
    compute_ideal_staffing_v2, 
    generate_audit_scenarios, 
    build_advanced_features
)

# 2. 標籤元數據 (讓 Swagger UI 更專業)
tags_metadata = [
    {
        "name": "Audit",
        "description": "人力效率審計與業績平衡分析，包含實績比對與 AI 建議模型。",
    },
    {
        "name": "Forecast",
        "description": "AI 門市智慧排班與未來業績預估引擎。",
    },
    {
        "name": "頁面導覽",
        "description": "提供前端 HTML 頁面的渲染入口。",
    }
]

app = FastAPI(
    title="Retail AI Intelligence Portal - Integrated Hub",
    openapi_tags=tags_metadata
)

# 3. 掛載靜態檔案與檢查
static_path = BASE_DIR / "scripts"
if static_path.exists():
    app.mount("/scripts", StaticFiles(directory=str(static_path)), name="static")
    print(f"📂 靜態資源掛載成功: {static_path}")
else:
    print(f"⚠️ 找不到 static 資料夾: {static_path}")

# 設定模板路徑
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 4. 強化的模組載入機制
# 確保即使子模組出錯，主程式依然能啟動並回報錯誤
audit_router = forecast_router = None

try:
    from api.audit_service import router as audit_router
    app.include_router(audit_router, prefix="/audit", tags=["Audit"])
    print("✅ [模組 A] 人力審計服務 (Audit) 載入成功")
except Exception as e:
    print(f"❌ [模組 A] 載入失敗: {e}")

try:
    from api.forecast_service import router as forecast_router
    app.include_router(forecast_router, prefix="/forecast", tags=["Forecast"])
    print("✅ [模組 B] 智慧排班服務 (Forecast) 載入成功")
except Exception as e:
    print(f"❌ [模組 B] 載入失敗: {e}")

# 這裡假設 engine, model, feature_cols 已在外部或 api 內部定義
# 為確保 main.py 邏輯完整，這裡需引用妳的資料庫連線與模型
from api.audit_service import engine, model, feature_cols

# --- 核心運算函數 ---
async def get_real_benchmark():
    """抓取 2026 模擬資料並執行 AI 審計計算"""
    if engine is None: return None
    
    query = "SELECT * FROM store_optimization.store_performance_data_2026_sim WHERE record_timestamp >= '2026-01-01'"
    df = pd.read_sql(query, engine)
    
    if df.empty: return None

    df['record_timestamp'] = pd.to_datetime(df['record_timestamp'])
    df['hour'] = df['record_timestamp'].dt.hour
    df['day_of_week'] = df['record_timestamp'].dt.dayofweek
    
    # 執行清洗與審計流程
    df_clean = validate_input(df)
    df_audit = generate_audit_scenarios(df_clean)
    df_processed = compute_ideal_staffing_v2(df_audit, handle_capacity=2.5)
    
    X, _ = build_advanced_features(df_processed)
    
    if model is not None and feature_cols is not None:
        valid_features = [f for f in feature_cols if f in X.columns]
        df_processed['ai_recommended_weight'] = model.predict(X[valid_features])
    else:
        df_processed['ai_recommended_weight'] = df_processed['ideal_weight_demand']

    df_processed['wage_waste'] = (df_processed['actual_weight'] - df_processed['ai_recommended_weight']) * 230
    df_processed['wage_waste'] = df_processed['wage_waste'].clip(lower=0)
    
    return {
        "cr": round((df['transaction_count'].sum() / (df['customer_traffic'].sum() + 1e-6)) * 100, 2),
        "aov": round(df['sales_amount'].sum() / (df['transaction_count'].sum() + 1e-6), 0),
        "ft": round(df['ft_count'].mean(), 1),
        "pt": round(df['pt_count'].mean(), 1),
        "waste_cost": round(df_processed['wage_waste'].sum(), 0),
        "waste_hours": round((df_processed['actual_weight'] - df_processed['ai_recommended_weight']).clip(lower=0).sum(), 1),
        "orig_total_revenue": df['sales_amount'].sum()
    }

# 5. API 邏輯路由
@app.get("/api/init_benchmark", tags=["Audit"])
async def init_benchmark():
    data = await get_real_benchmark()
    if not data:
        return JSONResponse(status_code=404, content={"error": "資料庫內無 2026 Q1 數據"})
    return data

@app.post("/api/simulate_decision", tags=["Audit"])
async def simulate_decision(data: dict):
    benchmark = await get_real_benchmark()
    if not benchmark:
        return JSONResponse(status_code=404, content={"error": "無法取得基準數據"})
    
    cr_multiplier = data['target_cr'] / (benchmark['cr'] + 1e-6)
    aov_multiplier = data['target_aov'] / (benchmark['aov'] + 1e-6)
    rev_multiplier = cr_multiplier * aov_multiplier
    estimated_rev_growth = benchmark['orig_total_revenue'] * (rev_multiplier - 1)
    
    ft_diff = (benchmark['ft'] - data['new_ft']) * 45000 * 3
    pt_diff = (benchmark['pt'] - data['new_pt']) * 32384 * 3
    cost_saved = ft_diff + pt_diff

    insights = []
    # (此處省略妳原本的 Insights 判斷邏輯，請完整保留在原處)
    # ... 原本的 insights 邏輯 ...
    if not insights: insights.append("<b>基準對照</b>：目前設定與 Q1 實績一致。")

    return {
        "rev_growth": round(estimated_rev_growth, 0),
        "cost_saved": round(cost_saved, 0),
        "total_benefit": round(estimated_rev_growth + cost_saved, 0),
        "insights": "<br><br>".join(insights)
    }
@app.post("/forecast/run", tags=["頁面導覽"])
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
    final_aov = int(target_aov) if target_aov and target_aov.strip() else ly_aov
    
    # 2. 執行核心排班引擎 (取得 List 資料與模式文字)
    results, mode_text = run_planning_engine(store_id, month, goal, final_aov)
    
    # 3. 透過 Jinja2 將乾淨的「資料」送到前端，讓 HTML 自己去畫表格！
    return templates.TemplateResponse(
        request=request, 
        name="smart_staffing_forecast.html", 
        context={
            "results": results,          # 將 List 傳入，前端的 {% for r in results %} 會自動生成表格
            "mode_text": mode_text,      # 傳入活動模式文字
            "ly_aov": ly_aov,
            "final_aov": final_aov,
            "store_id": store_id,
            "month": month,
            "goal": goal,
            "month_options": get_month_opts(month) # 確保下拉選單維持使用者的選擇
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

@app.get("/pages/audit", tags=["頁面導覽"])
async def get_audit_page(request: Request):
    """人力審計與業績平衡頁面"""
    # 對應妳的 Labor Efficiency Audit.html
    return templates.TemplateResponse(request=request, name="Labor Efficiency Audit.html")

@app.get("/pages/forecast", tags=["頁面導覽"])
async def get_forecast_page(request: Request):
    """智慧排班預測頁面"""
    # 這裡可以根據妳原本的 get_forecast_page 邏輯加入 AOV 查詢
    return templates.TemplateResponse(request=request, name="smart_staffing_forecast.html", 
    context={
        "results": None, "store_id": 2, "month": 4, "goal": 3000000
        })

# 7. 自定義 404 錯誤處理
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "message": "路徑或 HTML 檔案找不到，請確認 templates 資料夾內容。",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": "伺服器發生未預期錯誤。",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )

# 8. 執行主機
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)