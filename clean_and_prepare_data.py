import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import sys
import os

# 導入訓練與評估模組
try:
    from train_xgboost_and_evaluate import train_xgboost_and_evaluate, plot_feature_importance
except ImportError as e:
    print(f"警告：無法導入訓練模組 - {e}")
    train_xgboost_and_evaluate = None
    plot_feature_importance = None
def clean_and_prepare_data(file_path):
    #region agent log
    try:
        with open("/Users/ning/Desktop/Archangel/.cursor/debug.log", "a") as f:
            f.write(json.dumps({
                "id": "log_clean_and_prepare_entry",
                "timestamp": __import__("time").time(),
                "location": "clean_and_prepare_data.py:clean_and_prepare_data",
                "message": "function_entry",
                "data": {"file_path": str(file_path)},
                "runId": "pre-fix",
                "hypothesisId": "H1"
            }) + "\n")
    except Exception:
        pass
    #endregion agent log
    print("啟動防詐數據集清洗與特徵工程管線...")
    
    # 1. 載入資料集 (對應 2.2.4)
    df = pd.read_csv(file_path)
    print(f"原始資料筆數: {len(df)}")

    # 2. 離群值與雜訊處理 (對應 2.3)
    # 邏輯：剔除機器人測試標籤，且排除極端回報次數 (Report_Count >= 5000)
    df_clean = df[~df['Tags'].str.contains("機器人測試", na=False)].copy()
    df_clean = df_clean[df_clean['Report_Count'] < 5000]
    print(f"清洗後資料筆數: {len(df_clean)} (移除 {len(df) - len(df_clean)} 筆極端值雜訊)")

    # 3. 捨棄無預測價值的欄位
    # Incident_ID 與 Phone_Number 屬於流水號與唯一值，不具備機器學習泛化特徵
    df_clean = df_clean.drop(columns=['Incident_ID', 'Phone_Number', 'Report_Time'])

    # 4. 類別資料處理：One-Hot Encoding (對應 2.4.2)
    # 將 Victim_Demographic 與 Tags 展開為 Dummy Variables
    df_encoded = pd.get_dummies(df_clean, columns=['Victim_Demographic', 'Tags'], drop_first=False)
    
    # 5. 數據標準化 (對應 2.5.2)
    # 針對連續數值特徵進行 StandardScaler
    scaler = StandardScaler()
    numeric_features = ['Report_Count', 'Transaction_Amount']
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
    
    # 分離特徵矩陣 (X) 與 目標變數 (y)
    y = df_encoded['Is_Fraud']
    X = df_encoded.drop(columns=['Is_Fraud'])
    
    print(f"特徵工程完成！共萃取出 {X.shape[1]} 個特徵維度。")
    return X, y, scaler

# 執行測試與完整管線
if __name__ == "__main__":
    # 設定資料檔案路徑
    data_file = "fraud_1000_dataset.csv"
    
    # 檢查檔案是否存在
    if not os.path.exists(data_file):
        print(f"錯誤：找不到資料檔案 '{data_file}'")
        print(f"請確認檔案路徑是否正確，當前工作目錄：{os.getcwd()}")
        sys.exit(1)
    
    try:
        # 階段一：資料清洗與特徵工程
        print("=" * 60)
        print("階段一：資料清洗與特徵工程")
        print("=" * 60)
        X, y, trained_scaler = clean_and_prepare_data(data_file)
        
        # 顯示資料統計資訊
        print("\n[資料統計摘要]")
        print(f"  特徵矩陣維度: {X.shape[0]} 筆 × {X.shape[1]} 個特徵")
        print(f"  目標變數分布:")
        print(f"    - 正常案例 (0): {sum(y == 0)} 筆 ({sum(y == 0)/len(y)*100:.2f}%)")
        print(f"    - 詐騙案例 (1): {sum(y == 1)} 筆 ({sum(y == 1)/len(y)*100:.2f}%)")
        
        # 顯示特徵矩陣預覽
        print("\n[特徵矩陣 X 前 3 筆預覽]:")
        print(X.head(3))
        
        # 階段二：模型訓練與評估（如果模組可用）
        if train_xgboost_and_evaluate is not None and plot_feature_importance is not None:
            print("\n" + "=" * 60)
            print("階段二：XGBoost 模型訓練與評估")
            print("=" * 60)
            xgb_model, feature_names = train_xgboost_and_evaluate(X, y)
            plot_feature_importance(xgb_model, feature_names)
            print("\n✓ 完整管線執行完成！")
        else:
            print("\n⚠ 跳過模型訓練階段（訓練模組未導入）")
            print("  如需執行完整管線，請確認 train_xgboost_and_evaluate.py 存在且可導入")
            
    except FileNotFoundError as e:
        print(f"錯誤：找不到指定的檔案 - {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"錯誤：資料檔案缺少必要的欄位 - {e}")
        print("請確認資料檔案格式是否正確")
        sys.exit(1)
    except Exception as e:
        print(f"錯誤：執行過程中發生未預期的錯誤 - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

  
