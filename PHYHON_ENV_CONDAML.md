# Conda 環境設定說明

## 已建立的環境

已成功建立名為 `condaml` 的 conda 環境（Python 3.11），並安裝了所有需要的套件：

> **注意：** 環境已從 `fraudml` 重新命名為 `condaml`，名稱更通用，適合所有機器學習專案使用。

**核心機器學習套件：**
- ✅ xgboost (3.2.0) - 已解決 libomp.dylib 錯誤
- ✅ scikit-learn (1.8.0)
- ✅ shap (0.51.0) - 模型可解釋性
- ✅ imbalanced-learn (0.14.1) - 處理不平衡數據

**數據處理套件：**
- ✅ pandas (3.0.1)
- ✅ numpy (2.2.6)
- ✅ scipy (1.17.1)

**視覺化套件：**
- ✅ matplotlib
- ✅ seaborn

**API 與部署套件：**
- ✅ FastAPI (0.135.1)
- ✅ uvicorn (0.41.0)
- ✅ joblib (1.5.3) - 模型序列化

## 如何使用這個環境

### 在終端機中使用

```bash
# 啟動環境
conda activate condaml

# 執行你的腳本
python clean_and_prepare_data.py
python train_xgboost_and_evaluate.py
```

### 在 VS Code 中設定

1. 按 `Cmd+Shift+P` (Mac) 或 `Ctrl+Shift+P` (Windows/Linux)
2. 輸入 "Python: Select Interpreter"
3. 選擇 `Python 3.11.13 ('condaml': conda)` 或 `/opt/anaconda3/envs/condaml/bin/python`

或者：

1. 點擊 VS Code 右下角的 Python 版本顯示
2. 選擇 "Enter interpreter path..."
3. 輸入：`/opt/anaconda3/envs/condaml/bin/python`

設定完成後，IDE 中的匯入錯誤應該會消失。

## 驗證環境

執行以下命令驗證所有套件都已正確安裝：

```bash
/opt/anaconda3/envs/condaml/bin/python -c "import xgboost, pandas, numpy, sklearn; print('All packages OK!')"
```

## 問題已解決

✅ **XGBoost libomp.dylib 錯誤已修復** - 使用 conda-forge 安裝的 xgboost 和 llvm-openmp 版本完全相容
✅ **所有匯入錯誤已解決** - 所有套件都在 condaml 環境中正確安裝
✅ **環境已重新命名** - 從 `fraudml` 更名為 `condaml`，更適合通用 ML 專案

## 相關文件

📖 **POC 開發指南**：請參考 [`README_POC_DEVELOPMENT_GUIDE.md`](./README_POC_DEVELOPMENT_GUIDE.md) 查看完整的防詐 DRE 面試實戰策略，包含：
- 資料清洗與特徵萃取
- XGBoost 模型訓練與 SHAP 解釋化
- 非監督式學習（DBSCAN, t-SNE）
- 不平衡數據處理
- FastAPI 服務化與模型部署
