import uvicorn
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

# Forecast
from api.forecast_service import run_planning_engine, get_db_ly_aov, get_month_opts

# Audit
from scripts.model_training_pipeline import (
    validate_input, 
    compute_ideal_staffing_v2, 
    generate_audit_scenarios, 
    build_advanced_features
)

# ======================
# 初始化
# ======================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Retail AI Intelligence Portal - Integrated Hub")

# ======================
# Static
# ======================
static_path = BASE_DIR / "scripts"
if static_path.exists():
    app.mount("/scripts", StaticFiles(directory=str(static_path)), name="static")

# ======================
# Templates（關鍵）
# ======================
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ======================
# Router（保留）
# ======================
try:
    from api.audit_service import router as audit_router
    app.include_router(audit_router, prefix="/audit", tags=["Audit"])
    print("✅ Audit Router OK")
except Exception as e:
    print(f"❌ Audit Router error: {e}")

try:
    from api.forecast_service import router as forecast_router
    app.include_router(forecast_router, prefix="/forecast", tags=["Forecast"])
    print("✅ Forecast Router OK")
except Exception as e:
    print(f"❌ Forecast Router error: {e}")

# DB / Model
from api.audit_service import engine, model, feature_cols

# ======================
# Audit 核心
# ======================
async def get_real_benchmark():
    if engine is None:
        return None

    query = """
    SELECT * 
    FROM store_optimization.store_performance_data_2026_sim 
    WHERE record_timestamp >= '2026-01-01'
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        return None

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

    # ⭐ 補回完整分析（你之前少的）
    df_processed['wage_waste'] = (
        df_processed['actual_weight'] - df_processed['ai_recommended_weight']
    ) * 230
    df_processed['wage_waste'] = df_processed['wage_waste'].clip(lower=0)

    return {
        "cr": round((df['transaction_count'].sum() / (df['customer_traffic'].sum() + 1e-6)) * 100, 2),
        "aov": round(df['sales_amount'].sum() / (df['transaction_count'].sum() + 1e-6), 0),
        "ft": round(df['ft_count'].mean(), 1),
        "pt": round(df['pt_count'].mean(), 1),
        "waste_cost": round(df_processed['wage_waste'].sum(), 0),
        "waste_hours": round(
            (df_processed['actual_weight'] - df_processed['ai_recommended_weight'])
            .clip(lower=0).sum(), 1
        ),
        "orig_total_revenue": df['sales_amount'].sum()
    }

# ======================
# Audit API
# ======================
@app.get("/api/init_benchmark")
async def init_benchmark():
    data = await get_real_benchmark()
    if not data:
        return JSONResponse(status_code=404, content={"error": "無資料"})
    return data

# ======================
# Forecast 頁面（唯一一個）
# ======================
@app.get("/pages/forecast", response_class=HTMLResponse)
async def forecast_page(request: Request):
    store_id, month = 2, 4
    ly_aov = get_db_ly_aov(store_id, month)

    return templates.TemplateResponse(
        "smart_staffing_forecast.html",
        {
            "request": request,
            "results": None,
            "store_id": store_id,
            "month": month,
            "goal": 3000000,
            "ly_aov": ly_aov,
            "month_options": get_month_opts(month)
        }
    )

# ======================
# Forecast 執行（可輸出）
# ======================
@app.post("/forecast/run", response_class=HTMLResponse)
async def execute_forecast(
    request: Request,
    store_id: int = Form(...),
    month: int = Form(...),
    goal: int = Form(...),
    target_aov: str = Form("")
):
    ly_aov = get_db_ly_aov(store_id, month)
    final_aov = int(target_aov) if target_aov.strip() else ly_aov

    results, mode_text = run_planning_engine(store_id, month, goal, final_aov)

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

# ======================
# 其他頁面（保留）
# ======================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/pages/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    return templates.TemplateResponse("Labor Efficiency Audit.html", {"request": request})

# ======================
# 404
# ======================
@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    return JSONResponse(
        status_code=404,
        content={"message": "路徑或 HTML 檔案找不到，請確認 templates 資料夾內容。"}
    )

# ======================
# 啟動
# ======================
if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)