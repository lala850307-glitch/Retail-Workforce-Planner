import uvicorn
import os
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

# 1. 確定專案根目錄與環境初始化
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 匯入自定義腳本
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

# 2. FastAPI 實例
app = FastAPI(title="Retail AI Intelligence Portal")

# 3. 靜態檔案掛載
STATIC_CSS_DIR = BASE_DIR / "stylesheet"
if STATIC_CSS_DIR.exists():
    app.mount("/stylesheet", StaticFiles(directory=str(STATIC_CSS_DIR)), name="stylesheet")

# 4. 模板與路由載入
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 載入子模組邏輯
try:
    from api.audit_service import router as audit_router, engine, model, feature_cols
    app.include_router(audit_router, prefix="/audit", tags=["Audit"])
except Exception as e:
    print(f"❌ [模組 A] Audit 載入失敗: {e}")
    engine = model = feature_cols = None

try:
    from api.forecast_service import router as forecast_router, run_planning_engine, get_db_ly_aov, get_month_opts
    app.include_router(forecast_router, prefix="/forecast", tags=["Forecast"])
except Exception as e:
    print(f"❌ [模組 B] Forecast 載入失敗: {e}")

# --- 核心運算函數 ---
async def get_real_benchmark():
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
            "waste_hours": round((df_processed['actual_weight'] - df_processed['ai_recommended_weight']).clip(lower=0).sum(), 1),
            "orig_total_revenue": df['sales_amount'].sum()
        }
    except Exception as e:
        print(f"運算錯誤: {e}")
        return None

# 5. 頁面導覽路由 (對齊關鍵字參數)
@app.get("/", tags=["頁面導覽"])
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/system_architecture", tags=["頁面導覽"])
async def get_architecture_page(request: Request):
    return templates.TemplateResponse(request=request, name="system_architecture_diagram.html")

@app.get("/Labor_Efficiency_Audit", tags=["頁面導覽"])
async def get_audit_page(request: Request):
    return templates.TemplateResponse(request=request, name="Labor_Efficiency_Audit.html")

@app.get("/smart_staffing_forecast", tags=["頁面導覽"])
async def get_forecast_page(request: Request):
    store_id, month = 2, 4
    ly_aov = get_db_ly_aov(store_id, month) if 'get_db_ly_aov' in globals() else 1800
    return templates.TemplateResponse(
        request=request, 
        name="smart_staffing_forecast.html", 
        context={
            "results": None, 
            "store_id": store_id, 
            "month": month, 
            "goal": 3000000,
            "ly_aov": ly_aov,
            "month_options": get_month_opts(month) if 'get_month_opts' in globals() else ""
        }
    )

# 6. API 實作
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
    
    try:
        # 1. 基礎業績與成本計算
        cr_multiplier = data['target_cr'] / (benchmark['cr'] + 1e-6)
        aov_multiplier = data['target_aov'] / (benchmark['aov'] + 1e-6)
        rev_multiplier = cr_multiplier * aov_multiplier
        estimated_rev_growth = benchmark['orig_total_revenue'] * (rev_multiplier - 1)
        
        ft_diff = (benchmark['ft'] - data['new_ft']) * 45000 * 3
        pt_diff = (benchmark['pt'] - data['new_pt']) * 32384 * 3
        cost_saved = ft_diff + pt_diff

        # 2. 豐富的 AI 診斷判斷邏輯 (從 audit_service 同步)
        insights = []
        
        # 客單價 (AOV) 槓桿診斷
        if data['target_aov'] != benchmark['aov']:
            aov_diff = data['target_aov'] - benchmark['aov']
            if aov_diff > 0:
                insights.append(f"<b>客單價槓桿</b>：將 AOV 提升至 ${data['target_aov']:,.0f}，顯示高品質服務的產出效率。")
            else:
                insights.append(f"<b>低價策略預警</b>：客單價低於基準值，需仰賴極高成交量維持業績。")

        # 成交率 (CR) 服務力診斷 - 改為不論是否變動都進行基本分析或移除強制警告
        if data['target_cr'] > benchmark['cr']:
            insights.append(f"<b>服務力提升</b>：設定較高成交率標竿，預示門市需更精準的導購流程。")
        elif data['target_cr'] < benchmark['cr']:
            insights.append(f"<b>流失風險偵測</b>：成交率設定過低，這可能與人力縮減過度有關。")

        # 人力配置權衡 (Trade-off)
        total_new_weight = data['new_ft'] * 1.0 + data['new_pt'] * 0.5
        total_old_weight = benchmark['ft'] * 1.0 + benchmark['pt'] * 0.5
        
        if total_new_weight < total_old_weight:
            insights.append(f"<b>精簡風險評估</b>：縮減人力配置（省下 ${cost_saved:,.0f}）時請注意服務品質穩定。")
        elif total_new_weight > total_old_weight:
            insights.append(f"<b>人力投資決策</b>：增加人力配置，需帶動成交率或 AOV 提升才有經濟效益。")

        # 綜合診斷結論
        if estimated_rev_growth > 0 and cost_saved > 0:
            insights.append(f"<b>最優解路徑</b>：目前配置顯示您試圖在低勞動力下達成高產出，是極佳的營運目標。")
        elif estimated_rev_growth < 0 and cost_saved > 0:
            insights.append(f"<b>盲目節流警告</b>：雖然省下成本，但總收益下滑，建議重新評估策略。")

        if not insights:
            insights.append("<b>基準對照</b>：目前設定與 Q1 實績一致。")

        return {
            "rev_growth": round(estimated_rev_growth, 0),
            "cost_saved": round(cost_saved, 0),
            "total_benefit": round(estimated_rev_growth + cost_saved, 0),
            "insights": "<br><br>".join(insights)
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
    ly_aov = get_db_ly_aov(store_id, month)
    try:
        final_aov = int(target_aov) if target_aov.strip() else ly_aov
    except ValueError:
        final_aov = ly_aov
    results, mode_text = run_planning_engine(store_id, month, goal, final_aov)
    return templates.TemplateResponse(
        request=request,
        name="smart_staffing_forecast.html", 
        context={
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
