import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import json

# 設定 Matplotlib 支援中文顯示 (若在 Mac 系統請將 'Microsoft JhengHei' 改為 'Arial Unicode MS')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def train_xgboost_and_evaluate(X, y):
    #region agent log
    try:
        with open("/Users/ning/Desktop/Archangel/.cursor/debug.log", "a") as f:
            f.write(json.dumps({
                "id": "log_train_xgboost_entry",
                "timestamp": __import__("time").time(),
                "location": "train_xgboost_and_evaluate.py:train_xgboost_and_evaluate",
                "message": "function_entry",
                "data": {"n_samples": int(getattr(X, "shape", [0])[0]), "n_features": int(getattr(X, "shape", [0, 0])[1]) if len(getattr(X, "shape", [])) > 1 else 0},
                "runId": "pre-fix",
                "hypothesisId": "HX1"
            }) + "\n")
    except Exception:
        pass
    #endregion agent log
    print("啟動 XGBoost 防詐預測模型訓練管線...\n")
    
    # 1. 切分訓練集與測試集 (80% 訓練, 20% 測試，並使用分層抽樣)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"訓練集維度: {X_train.shape}, 測試集維度: {X_test.shape}")

    # 2. 建立與訓練 XGBoost 分類器 (對應 8.5 節)
    # 設定 eval_metric 避免版本警告，並使用預設決策樹深度
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # 3. 模型預測與業務指標評估 (對應 9.3.8 節)
    y_pred = xgb_model.predict(X_test)
    
    print("\n[模型評估報告 - Classification Report]:")
    print("--------------------------------------------------")
    # 產出包含 Precision (精準率) 與 Recall (召回率) 的報告
    print(classification_report(y_test, y_pred, target_names=['正常(0)', '詐騙(1)']))
    print("--------------------------------------------------")
    
    return xgb_model, X_train.columns

def plot_feature_importance(model, feature_names):
    # 1. 萃取特徵重要性 (對應 8.6.7 節 模型的可解釋性)
    importances = model.feature_importances_
    
    # 將特徵與重要性打包為 DataFrame 並降冪排序，取前 10 大關鍵特徵
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    # 2. 繪製高質感 Seaborn 長條圖
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    
    plt.title('XGBoost 防詐模型 - Top 10 核心特徵重要性 (Feature Importance)', fontsize=16, fontweight='bold')
    plt.xlabel('重要性權重 (相對值)', fontsize=12)
    plt.ylabel('數據特徵', fontsize=12)
    plt.tight_layout()
    
    # 儲存圖片供簡報或 POC 展示使用
    plt.savefig('xgboost_feature_importance.png', dpi=300)
    print("\n特徵重要性圖表已成功儲存為 'xgboost_feature_importance.png'！")
    plt.show()

# 延續階段一的產出執行測試 (假設您已經取得了 X, y)
if __name__ == "__main__":
    # 這裡的 X, y 來自您前一個步驟的 clean_and_prepare_data() 產出
    # xgb_model, feature_names = train_xgboost_and_evaluate(X, y)
    # plot_feature_importance(xgb_model, feature_names)
    print("請將此程式碼接續在階段一之後執行，即可看到完整的模型訓練與圖表輸出。")