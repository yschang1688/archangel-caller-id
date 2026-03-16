# 防詐 DRE 面試：Python 實戰 POC 開發藍圖

本指南對應《全民瘋 AI 系列》教材，展現處理不平衡數據（Imbalanced Data）與模型解釋化（XAI）的專業即戰力。

---

## 階段一：資料清洗與特徵萃取（數據工程力）

**對應章節：** 第 2 章 發現資料的秘密  
**核心套件：** `pandas`, `numpy`, `scipy`

### 1.1 離群值處理 (2.3)

**技術要點：**
- 使用 `scipy.stats` 計算 `Report_Count` 的 Z-score
- 或使用 `pandas` 實作 IQR 法
- 針對回報次數異常高（如機器人灌水）的樣本進行 **Clipping（極端值平滑化）**，而非直接刪除
- 保留潛在的攻擊訊號

**實作建議：**
```python
from scipy import stats
import numpy as np

# Z-score 方法
z_scores = np.abs(stats.zscore(df['Report_Count']))
df_clean = df[(z_scores < 3)]  # 保留 3 個標準差內的資料

# 或使用 Clipping
df['Report_Count'] = df['Report_Count'].clip(
    lower=df['Report_Count'].quantile(0.01),
    upper=df['Report_Count'].quantile(0.99)
)
```

### 1.2 類別資料編碼 (2.4.2)

**技術要點：**
- 使用 **Category Encoders** 套件（需額外安裝：`pip install category_encoders`）
- 針對 `Tags`（詐騙標籤）採用 **Target Encoding（目標編碼）**
- 將類別轉化為該標籤下的平均詐騙機率
- 比 One-Hot 更能處理高維度特徵並提升模型效能

**實作建議：**
```python
import category_encoders as ce

# Target Encoding
encoder = ce.TargetEncoder(cols=['Tags', 'Victim_Demographic'])
df_encoded = encoder.fit_transform(df, df['Is_Fraud'])
```

### 1.3 特徵縮放 (2.5)

**技術要點：**
- 調用 `sklearn.preprocessing.RobustScaler` 處理 `Transaction_Amount`
- 相較於 `StandardScaler`，`RobustScaler` 對離群值更具魯棒性
- 確保金額特徵在模型中權重正確

**實作建議：**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[['Transaction_Amount']] = scaler.fit_transform(df[['Transaction_Amount']])
```

---

## 階段二：建立防詐預測核心模型（算法深度與解釋力）

**對應章節：** 第 8 章 整體學習 與 第 7 章 決策樹  
**核心套件：** `xgboost`, `shap`, `sklearn`

### 2.1 極限梯度提升 XGBoost (8.5)

**技術要點：**
- 實作 `XGBClassifier` 並設定 `scale_pos_weight` 參數
- 這是處理防詐數據中「正負樣本極度不均」的關鍵技術
- 能強制模型更關注少數的詐騙樣本

**實作建議：**
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 計算正負樣本比例
pos_weight = (y == 0).sum() / (y == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=pos_weight,  # 關鍵參數
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

### 2.2 SHAP 值解釋化 (8.6.7)

**技術要點：**
- 捨棄傳統的 Feature Importance
- 改用 `shap` 套件產出 **Summary Plot**

**面試關鍵論述：**
> 「我不僅知道模型預測它是詐騙，還能透過 SHAP 說明是因為該門號在 1 小時內有 50 次撥打記錄（特徵 A）且發話地不在台灣（特徵 B），這才是防詐業務真正需要的可解釋 AI。」

**實作建議：**
```python
import shap

# 建立 SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 繪製 Summary Plot
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)  # 詳細分佈圖
```

### 2.3 Baseline 建立 (4.3)

**技術要點：**
- 使用 `LogisticRegression` 建立基準
- 若 XGBoost 提升幅度不大，則討論是否需回頭強化特徵工程
- 展現科學嚴謹性

**實作建議：**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Baseline 模型
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_f1 = f1_score(y_test, lr_pred)

# XGBoost 模型
xgb_pred = xgb_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_pred)

print(f"Baseline (LogisticRegression) F1: {lr_f1:.4f}")
print(f"XGBoost F1: {xgb_f1:.4f}")
print(f"提升幅度: {(xgb_f1 - lr_f1) / lr_f1 * 100:.2f}%")
```

---

## 階段三：探索未知詐騙犯罪集團（非監督式洞察）

**對應章節：** 第 3 章 非監督式學習  
**核心套件：** `scikit-learn`, `matplotlib`

### 3.1 DBSCAN 或 K-means (3.3)

**技術要點：**
- 針對未標記（Unlabeled）的通訊行為數據進行分群
- **DBSCAN** 能識別出高密度的「詐騙集群」並自動過濾雜訊
- 展現主動防護（Proactive Defense）的思維

**實作建議：**
```python
from sklearn.cluster import DBSCAN
import numpy as np

# DBSCAN 分群
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# 分析各群組的詐騙比例
for cluster_id in np.unique(clusters):
    if cluster_id != -1:  # 排除雜訊點
        cluster_mask = clusters == cluster_id
        fraud_rate = y[cluster_mask].mean()
        print(f"Cluster {cluster_id}: 詐騙比例 = {fraud_rate:.2%}")
```

### 3.2 t-SNE 視覺化 (3.5)

**技術要點：**
- 將高維特徵投影至 2D 平面
- 視覺化呈現正常用戶與詐騙集團的空間分佈差異
- 作為面試簡報的強力視覺證據

**實作建議：**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# t-SNE 降維
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 視覺化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Is_Fraud')
plt.title('t-SNE 視覺化：正常用戶 vs 詐騙集團分佈')
plt.xlabel('t-SNE 維度 1')
plt.ylabel('t-SNE 維度 2')
plt.savefig('tsne_visualization.png', dpi=300)
plt.show()
```

---

## 階段四：嚴謹驗證與指標優化（科學家素養）

**對應章節：** 第 9 章 交叉驗證和錯誤修正  
**核心套件：** `sklearn.metrics`, `imbalanced-learn`

### 4.1 分層 K-Fold 驗證 (9.2.2)

**技術要點：**
- 使用 `StratifiedKFold` 確保每一折的詐騙比例一致
- 避免驗證偏差

**實作建議：**
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring='f1')
print(f"5-Fold CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### 4.2 Precision-Recall Curve (9.3.8)

**技術要點：**
- 實作 `classification_report`
- 在防詐場景中，強調 **F1-Score** 或 **AUPRC（Precision-Recall 曲線下面積）** 優於 Accuracy
- 說明如何根據業務需求調校門檻（Threshold）以平衡誤殺率與漏抓率

**實作建議：**
```python
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt

# 取得預測機率
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'XGBoost (AUPRC = {auprc:.4f})')
plt.xlabel('Recall (召回率)')
plt.ylabel('Precision (精準率)')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('pr_curve.png', dpi=300)
plt.show()

# Classification Report
print(classification_report(y_test, xgb_model.predict(X_test), 
                          target_names=['正常(0)', '詐騙(1)']))
```

### 4.3 處理不平衡數據

**技術要點：**
- 使用 `imbalanced-learn` 套件進行過採樣或欠採樣
- 可嘗試 SMOTE、ADASYN 等方法

**實作建議：**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# SMOTE 過採樣
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 在平衡後的資料上訓練
xgb_model_balanced = xgb.XGBClassifier(random_state=42)
xgb_model_balanced.fit(X_resampled, y_resampled)
```

---

## 階段五：模型落地與 API 化（產品化思維）

**對應章節：** 第 10 章 模型落地實踐  
**核心套件：** `FastAPI`, `uvicorn`, `joblib`

### 5.1 FastAPI 服務化 (10.4)

**技術要點：**
- 撰寫一個 `/predict` 接口
- 輸入 JSON 格式的用戶特徵
- 回傳詐騙機率

**實作建議：**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="防詐預測 API")

# 載入模型與預處理器
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

class FraudPredictionRequest(BaseModel):
    report_count: float
    transaction_amount: float
    # ... 其他特徵欄位

@app.post("/predict")
async def predict_fraud(request: FraudPredictionRequest):
    # 轉換為 numpy array
    features = np.array([[request.report_count, request.transaction_amount, ...]])
    
    # 預處理
    features_scaled = scaler.transform(features)
    
    # 預測
    fraud_probability = model.predict_proba(features_scaled)[0][1]
    
    return {
        "fraud_probability": float(fraud_probability),
        "is_fraud": fraud_probability > 0.5
    }

@app.get("/")
async def root():
    return {"message": "防詐預測 API 運行中"}
```

### 5.2 模型序列化

**技術要點：**
- 使用 `joblib` 儲存 Pipeline
- 包含預處理與模型本身
- 確保訓練與推論環境的資料一致性（Data Consistency）
- 展現對生產環境（Production）的理解

**實作建議：**
```python
from sklearn.pipeline import Pipeline
import joblib

# 建立完整的 Pipeline（包含預處理 + 模型）
full_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', xgb.XGBClassifier(...))
])

# 訓練
full_pipeline.fit(X_train, y_train)

# 儲存整個 Pipeline
joblib.dump(full_pipeline, 'fraud_detection_pipeline.pkl')

# 載入與使用
loaded_pipeline = joblib.load('fraud_detection_pipeline.pkl')
predictions = loaded_pipeline.predict(X_new)
```

### 5.3 啟動 API 服務

```bash
# 啟動 FastAPI 服務
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 測試 API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"report_count": 10, "transaction_amount": 5000, ...}'
```

---

## 環境設定

所有套件已安裝在 `fraudml` conda 環境中：

- ✅ xgboost (3.2.0)
- ✅ shap (0.51.0)
- ✅ scikit-learn (1.8.0)
- ✅ scipy (1.17.1)
- ✅ pandas (3.0.1)
- ✅ numpy (2.2.6)
- ✅ imbalanced-learn (0.14.1)
- ✅ FastAPI (0.135.1)
- ✅ uvicorn (0.41.0)
- ✅ joblib (1.5.3)
- ✅ matplotlib, seaborn

**啟動環境：**
```bash
conda activate fraudml
```

---

## 面試準備重點

1. **技術深度**：能解釋每個步驟的理論依據與實作細節
2. **業務理解**：說明為何在防詐場景中 F1-Score 比 Accuracy 重要
3. **可解釋性**：使用 SHAP 展示模型決策過程
4. **產品化思維**：展示 API 化與模型序列化的實作
5. **科學嚴謹性**：使用交叉驗證、Baseline 比較等方法

---

**祝面試順利！** 🚀
