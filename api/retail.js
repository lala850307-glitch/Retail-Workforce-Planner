flowchart TD

subgraph group_offline["Offline pipeline"]
  node_store_people_gen["Store & people<br/>synthetic data generator"]
  node_op2025_gen["2025 ops<br/>synthetic op generator"]
  node_op2026q1_gen["2026 Q1 ops<br/>synthetic op generator"]
  node_clean_train["Clean & train<br/>batch prep<br/>[清洗與模型生成.py]"]
  node_forecast_engine["Forecast engine<br/>forecast workflow"]
  node_audit_report["Audit report<br/>audit batch"]
  node_export_db["Export audit<br/>report export"]
  node_forecast_csv["Hybrid forecast<br/>forecast output"]
end

subgraph group_artifact["Artifact store"]
  node_model_bundle[("Staffing model<br/>serialized model")]
  node_feature_cols["Feature cols<br/>feature metadata"]
end

subgraph group_api["API service"]
  node_api_main["API bootstrap<br/>service entrypoint<br/>[main.py]"]
  node_audit_api["Audit API<br/>http endpoint"]
  node_predict_api["Predict API<br/>http endpoint"]
end

node_store_people_gen -->|"master data"| node_clean_train
node_op2025_gen -->|"ops data"| node_clean_train
node_op2026q1_gen -->|"ops data"| node_clean_train
node_clean_train -->|"train"| node_model_bundle
node_clean_train -->|"schema"| node_feature_cols
node_clean_train -->|"features"| node_forecast_engine
node_forecast_engine -->|"write"| node_forecast_csv
node_forecast_csv -->|"input"| node_audit_report
node_audit_report -->|"export"| node_export_db
node_model_bundle -->|"load model"| node_api_main
node_feature_cols -->|"load schema"| node_api_main
node_api_main -->|"route"| node_audit_api
node_api_main -->|"route"| node_predict_api
node_model_bundle -->|"infer"| node_predict_api
node_feature_cols -->|"validate"| node_predict_api
node_model_bundle -->|"score"| node_audit_api
node_forecast_csv -->|"report data"| node_audit_api

click node_store_people_gen "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/模擬資料生成/data_generator(店與人資料模擬生成).py"
click node_op2025_gen "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/模擬資料生成/2025_op_data_generator(營運資料模擬).py"
click node_op2026q1_gen "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/模擬資料生成/2026Q1_op_data_generator(營運資料模擬).py"
click node_clean_train "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/清洗模型準備/清洗與模型生成.py"
click node_forecast_engine "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/清洗模型準備/q2_forecast_engine.py"
click node_audit_report "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/清洗模型準備/run_audit_report_2026Q1模型預測結果.py"
click node_export_db "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/清洗模型準備/export_audit_to_db.py"
click node_forecast_csv "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/清洗模型準備/2026_Q2_Hybrid_Forecast.csv"
click node_model_bundle "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/模型生成/retail_staffing_model_v1.pkl"
click node_feature_cols "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/模型生成/feature_columns.pkl"
click node_api_main "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/api/main.py"
click node_audit_api "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/api/2026Q1_(人力效率審計 API).py"
click node_predict_api "https://github.com/lala850307-glitch/retail-workforce-planner/blob/main/api/LaborEfficiencyAuditAPI(智慧排班預測 API).py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_store_people_gen,node_op2025_gen,node_op2026q1_gen,node_clean_train,node_forecast_engine,node_audit_report,node_export_db,node_forecast_csv toneBlue
class node_model_bundle,node_feature_cols toneAmber
class node_api_main,node_audit_api,node_predict_api toneMint