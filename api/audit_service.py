import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys
import joblib
import pandas as pd

# 💡 [救命關鍵]：強制將根目錄加入 Python 搜尋路徑
# 這段程式碼會抓到 audit_service.py 的上一層，也就是專案根目錄
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from scripts.model_training_pipeline import (
    validate_input, 
    compute_ideal_staffing_v2, 
    run_financial_backtest_audit,
    generate_audit_scenarios,
    build_advanced_features
)

router = APIRouter()
load_dotenv()
engine = create_engine(os.getenv("POSTGRESQL_URL"))

# --- [路徑確保] ---
# 請確保這裡的路徑與妳剛才儲存 .pkl 的地方完全一致
MODEL_PATH = '/Users/laylatang8537/Documents/vscold/Retail-Workforce-Planner/Retail-Workforce-Planner/models/retail_staffing_model_v1.pkl'
FEAT_PATH = '/Users/laylatang8537/Documents/vscold/Retail-Workforce-Planner/Retail-Workforce-Planner/models/feature_columns.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH):
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    print(f"✅ 成功載入模型，特徵數量: {len(feature_cols)}")
else:
    model = None
    feature_cols = None
    print("❌ 找不到模型檔案，請先執行『清洗與模型生成.py』")

async def get_real_benchmark():
    # 抓取 2026 模擬資料 (這部分妳原本就寫得很好)
    query = "SELECT * FROM store_optimization.store_performance_data_2026_sim WHERE record_timestamp >= '2026-01-01'"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return None

    # 基礎時間特徵處理
    df['record_timestamp'] = pd.to_datetime(df['record_timestamp'])
    df['hour'] = df['record_timestamp'].dt.hour
    df['day_of_week'] = df['record_timestamp'].dt.dayofweek
    df['is_peak_month'] = df['record_timestamp'].dt.month.apply(lambda x: 1 if x in [1, 5, 9, 10, 11] else 0)

    # 執行清洗與審計流程
    df_clean = validate_input(df)
    df_audit = generate_audit_scenarios(df_clean)
    df_processed = compute_ideal_staffing_v2(df_audit, handle_capacity=2.5)
    
    # 產出 AI 特徵矩陣 (這裡會產出 11 個素顏特徵)
    X, _ = build_advanced_features(df_processed)
    
    # --- [核心修正區] ---
    if model is not None and feature_cols is not None:
        # 為了防止 KeyError，我們只取模型『真正有紀錄』且 X 矩陣『確實存在』的欄位
        valid_features = [f for f in feature_cols if f in X.columns]
        
        # 預測：將 66 萬筆資料丟進去產出建議
        df_processed['ai_recommended_weight'] = model.predict(X[valid_features])
    else:
        # 如果模型載入失敗，退而求其次使用公式計算的理想需求
        df_processed['ai_recommended_weight'] = df_processed['ideal_weight_demand']

    # 計算薪資浪費 (以 230 元計算)
    df_processed['wage_waste'] = (df_processed['actual_weight'] - df_processed['ai_recommended_weight']) * 230
    df_processed['wage_waste'] = df_processed['wage_waste'].clip(lower=0)
    
    # 彙整指標
    waste_cost = df_processed['wage_waste'].sum()
    waste_hours = (df_processed['actual_weight'] - df_processed['ai_recommended_weight']).clip(lower=0).sum()

    # 計算 2026 Q1 實績 (供前端顯示)
    real_cr = (df['transaction_count'].sum() / (df['customer_traffic'].sum() + 1e-6)) * 100
    real_aov = df['sales_amount'].sum() / (df['transaction_count'].sum() + 1e-6)
    real_ft = df['ft_count'].mean()
    real_pt = df['pt_count'].mean()

    return {
        "cr": round(real_cr, 2),
        "aov": round(real_aov, 0),
        "ft": round(real_ft, 1),
        "pt": round(real_pt, 1),
        "waste_cost": round(waste_cost, 0),
        "waste_hours": round(waste_hours, 1),
        "orig_total_revenue": df['sales_amount'].sum()
    }

@router.get("/api/init_benchmark")
async def init_benchmark():
    data = await get_real_benchmark()
    if not data:
        return {"error": "資料庫內無 2026 Q1 數據"}
    return data

@router.post("/api/simulate_decision")
async def simulate_decision(data: dict):
    benchmark = await get_real_benchmark()
    if not benchmark:
        return {"error": "無法取得基準數據"}
    
    cr_multiplier = data['target_cr'] / (benchmark['cr'] + 1e-6)
    aov_multiplier = data['target_aov'] / (benchmark['aov'] + 1e-6)
    rev_multiplier = cr_multiplier * aov_multiplier
    estimated_rev_growth = benchmark['orig_total_revenue'] * (rev_multiplier - 1)
    
    ft_diff = (benchmark['ft'] - data['new_ft']) * 45000 * 3
    pt_diff = (benchmark['pt'] - data['new_pt']) * 32384 * 3
    cost_saved = ft_diff + pt_diff

    insights = []
    
    # 1. 客單價 (AOV) 的槓桿效應診斷
    if data['target_aov'] != benchmark['aov']:
        aov_diff = data['target_aov'] - benchmark['aov']
        if aov_diff > 0:
            insights.append(f"<b>客單價槓桿</b>：將 AOV 提升至 ${data['target_aov']:,.0f}，代表即使在來客數不變、甚至略微下滑的情況下，透過提升「單次消費價值」，能最快速度拉抬整體業績，這顯示了高品質服務的產出效率。")
        else:
            insights.append(f"<b>低價策略預警</b>：客單價低於基準值，這意味著您必須仰賴極高的「成交量」才能維持業績，這會大幅增加第一線員工的勞動強度，若人力未同步增加，將面臨服務崩潰。")

    # 2. 成交率 (CR) 的工作環節診斷
    if data['target_cr'] != benchmark['cr']:
        cr_diff = data['target_cr'] - benchmark['cr']
        if cr_diff > 0:
            insights.append(f"服務力與轉換率掛鉤：設定較高的成交率標竿，預示著門市必須具備更精準的攔截與導購流程。請注意，若您在下方同時縮減了人力，卻期待成交率大幅上升，這在現實中極難達成；成交率的提升應建立在『排班優化』讓員工能在對的時間服務對的人。")
        else:
            insights.append(f"<b>流發風險偵測</b>：成交率設定過低，這通常與「店員導購積極度」或「排班時段錯置」有關。請思考是否因為人力縮減過度，導致店員無法即時回應顧客，造成業績白白流失。")

    # 3. 人力配置與營業額的權衡 (Trade-off)
    total_new_weight = data['new_ft'] * 1.0 + data['new_pt'] * 0.5
    total_old_weight = benchmark['ft'] * 1.0 + benchmark['pt'] * 0.5
    
    if total_new_weight < total_old_weight:
        insights.append(f"<b>精簡風險評估</b>：您縮減了人力配置（從 {total_old_weight} 降至 {total_new_weight} 權重）。雖然成本節省了 ${cost_saved:,.0f}，但請監控成交率是否受影響——若成交率因此下降 1% 以上，損失的業績可能遠高於省下的薪資。")
    elif total_new_weight > total_old_weight:
        insights.append(f"<b>人力投資決策</b>：您選擇增加人力配置。這必須帶動成交率或 AOV 同步提升才有經濟效益，否則將轉化為新的薪資浪費。")

    # 4. 綜合路徑診斷
    if estimated_rev_growth > 0 and cost_saved > 0:
        insights.append(f"<b>最優解路徑</b>：目前配置顯示您試圖在「低勞動力投入」下達成「高產出（高AOV/CR）」。這依賴極強的員工銷售能力與自動化流程，是門市營運的終極目標。")
    elif estimated_rev_growth < 0 and cost_saved > 0:
        insights.append(f"<b>盲目節流警告</b>：雖然省下了成本，但總收益是下滑的。這代表您的縮減策略可能殺傷了業績根基，建議調整策略方向。")

    if not insights:
        insights.append("<b>基準對照</b>：目前設定與 Q1 實績一致，可作為後續優化的起始點。")

    return {
        "rev_growth": round(estimated_rev_growth, 0),
        "cost_saved": round(cost_saved, 0),
        "total_benefit": round(estimated_rev_growth + cost_saved, 0),
        "insights": "<br><br>".join(insights)
    }

@router.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <title>門市人力配置與業績平衡儀表板</title>
        <style>
            body { font-family: 'PingFang TC', 'Microsoft JhengHei', sans-serif; background: #f0f2f5; margin: 0; padding: 20px; }
            .container { max-width: 1100px; margin: auto; }
            .card { background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; }
            .header { display: flex; align-items: center; margin-bottom: 25px; border-bottom: 3px solid #1a2a6c; padding-bottom: 15px; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .benchmark-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #eef0f2; }
            .bench-item { text-align: center; border-right: 1px solid #dee2e6; }
            .bench-item:last-child { border: none; }
            .bench-item span { font-size: 13px; color: #7f8c8d; display: block; margin-bottom: 8px; }
            .bench-item b { font-size: 22px; color: #2c3e50; }
            .input-group { margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f1f1f1; }
            .input-group label { font-weight: bold; color: #34495e; }
            input { width: 120px; padding: 10px; border: 2px solid #ecf0f1; border-radius: 8px; text-align: center; font-size: 16px; background: #fff; transition: border 0.3s; }
            .btn-exec { width: 100%; padding: 18px; background: #27ae60; color: white; border: none; border-radius: 10px; cursor: pointer; font-size: 20px; font-weight: bold; transition: all 0.3s; margin-top: 15px; }
            .btn-exec:hover { background: #2ecc71; transform: translateY(-2px); }
            .result-panel { background: #2c3e50; color: white; display: flex; flex-direction: column; }
            .res-val { font-size: 34px; font-weight: bold; margin: 12px 0; }
            .res-label { font-size: 14px; color: #bdc3c7; }
            .insight-box { background: rgba(255,255,255,0.08); border-radius: 10px; padding: 20px; margin-top: 25px; font-size: 15px; line-height: 1.8; color: #ecf0f1; border-left: 5px solid #f1c40f; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>門市人力配置與業績平衡儀表板</h1></div>
            <div class="card">
                <h3 style="margin-top:0; color: #1a2a6c;">2026 Q1 原始現況與診斷</h3>
                <div class="benchmark-grid">
                    <div class="bench-item"><span>原始成交率</span><b id="oCR">--</b></div>
                    <div class="bench-item"><span>原始客單價</span><b id="oAOV">--</b></div>
                    <div class="bench-item"><span>平均 FT 人數</span><b id="oFT">--</b></div>
                    <div class="bench-item"><span>平均 PT 人數</span><b id="oPT">--</b></div>
                </div>
                <div class="grid">
                    <div style="text-align: center; background: #fff5f5; padding: 15px; border-radius: 10px; border: 1px solid #ffdada;">
                        <span style="color: #c0392b; font-weight:bold;">預計季度薪資浪費</span>
                        <div class="res-val" id="oCost" style="color: #e74c3c;">$ --</div>
                    </div>
                    <div style="text-align: center; background: #fffaf0; padding: 15px; border-radius: 10px; border: 1px solid #ffe8cc;">
                        <span style="color: #d35400; font-weight:bold;">總計低效權重時數</span>
                        <div class="res-val" id="oHours" style="color: #f39c12;">-- 小時</div>
                    </div>
                </div>
            </div>
            <div class="grid">
                <div class="card">
                    <h3 style="margin-top:0; color: #27ae60;">經營策略調整</h3>
                    <div class="input-group"><label>目標成交率 (CR %)</label><input type="number" id="inCR" step="0.1"></div>
                    <div class="input-group"><label>目標客單價 (AOV $)</label><input type="number" id="inAOV"></div>
                    <div class="input-group"><label>調整後平均 FT (位)</label><input type="number" id="inFT" step="0.1"></div>
                    <div class="input-group"><label>調整後平均 PT (位)</label><input type="number" id="inPT" step="0.1"></div>
                    <button class="btn-exec" onclick="runSim()">執行決策模擬</button>
                </div>
                <div class="card result-panel">
                    <div style="padding: 10px; text-align: center;">
                        <span class="res-label">業績預計成長 (季度)</span>
                        <div class="res-val" id="resRev" style="color: #2ecc71;">+$ 0</div>
                        <span class="res-label">人力成本節省 (季度)</span>
                        <div class="res-val" id="resCost" style="color: #3498db;">+$ 0</div>
                        <hr style="border-color: #444; margin: 20px 0;">
                        <span style="font-size: 18px; color: #f1c40f; font-weight:bold;">預計總經濟價值</span>
                        <div class="res-val" id="resTotal" style="font-size: 52px; color: #f1c40f;">$ 0</div>
                    </div>
                    <div id="insightSection" style="display:none; padding: 0 15px 15px 15px;">
                        <div class="insight-box">
                            <div style="font-weight: bold; margin-bottom: 12px; color: #f1c40f; font-size: 17px;">🤖 AI 策略診斷報告：</div>
                            <div id="resInsight"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            async function init() {
                try {
                    const r = await fetch('/api/init_benchmark');
                    const d = await r.json();
                    if (d.error) { alert(d.error); return; }
                    document.getElementById('oCR').innerText = d.cr + ' %';
                    document.getElementById('oAOV').innerText = '$' + d.aov.toLocaleString();
                    document.getElementById('oFT').innerText = d.ft;
                    document.getElementById('oPT').innerText = d.pt;
                    document.getElementById('oCost').innerText = '$' + d.waste_cost.toLocaleString();
                    document.getElementById('oHours').innerText = d.waste_hours.toLocaleString() + ' 小時';
                    document.getElementById('inCR').value = d.cr;
                    document.getElementById('inAOV').value = d.aov;
                    document.getElementById('inFT').value = d.ft;
                    document.getElementById('inPT').value = d.pt;
                } catch (e) { console.error("初始化失敗", e); }
            }
            async function runSim() {
                const body = {
                    target_cr: parseFloat(document.getElementById('inCR').value),
                    target_aov: parseFloat(document.getElementById('inAOV').value),
                    new_ft: parseFloat(document.getElementById('inFT').value),
                    new_pt: parseFloat(document.getElementById('inPT').value)
                };
                try {
                    const r = await fetch('/api/simulate_decision', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(body)
                    });
                    const res = await r.json();
                    document.getElementById('resRev').innerText = (res.rev_growth >= 0 ? '+$ ' : '-$ ') + Math.abs(res.rev_growth).toLocaleString();
                    document.getElementById('resCost').innerText = (res.cost_saved >= 0 ? '+$ ' : '-$ ') + Math.abs(res.cost_saved).toLocaleString();
                    document.getElementById('resTotal').innerText = (res.total_benefit >= 0 ? '$ ' : '-$ ') + Math.abs(res.total_benefit).toLocaleString();
                    document.getElementById('resInsight').innerHTML = res.insights;
                    document.getElementById('insightSection').style.display = 'block';
                } catch (e) { console.error("模擬執行失敗", e); }
            }
            init();
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    # 💡 建立一個臨時的 App
    test_app = FastAPI()
    
    # 💡 把零件 (router) 裝上去
    test_app.include_router(router)
    
    # 💡 執行這個臨時 App
    print("🚀 正在以單獨模式啟動 Audit Service...")
    uvicorn.run(test_app, host="127.0.0.1", port=8000)
