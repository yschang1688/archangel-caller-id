import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 設定隨機種子以確保實驗可重現性
np.random.seed(42)

# 1. 基礎參數與資料規模設定
TOTAL_SAMPLES = 100000
FRAUD_RATIO = 0.10
FRAUD_SIZE = int(TOTAL_SAMPLES * FRAUD_RATIO)  # 10,000
NORMAL_SIZE = TOTAL_SAMPLES - FRAUD_SIZE       # 90,000

# 2. 生成正常雜訊樣本 (90%)
def generate_normal_data(size):
    return pd.DataFrame({
        'Report_Time': [datetime.now() - timedelta(minutes=np.random.randint(0, 43200)) for _ in range(size)],
        'Phone_Number': [f"+886-9{np.random.randint(10, 89):02d}-{np.random.randint(100, 999):03d}-{np.random.randint(100, 999):03d}" for _ in range(size)],
        'Report_Count': np.random.poisson(lam=1.5, size=size), # 常態回報次數偏低
        'Financial_Loss': np.random.choice([0, np.random.randint(100, 5000)], size=size, p=[0.95, 0.05]),
        'Age_Group': np.random.choice(['18-25', '26-40', '41-60', '70+'], size=size, p=[0.2, 0.4, 0.3, 0.1]),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master/PhD'], size=size, p=[0.3, 0.6, 0.1]),
        'Cluster_ID': ['C_NORMAL' for _ in range(size)],
        'Label': 0
    })

# 3. 生成異常詐騙樣本 (10%) - 包含四大威脅區塊
def generate_fraud_data(size):
    # 區塊 A: GNN 拓撲特徵群聚 (40% 詐騙樣本)
    gnn_size = int(size * 0.4)
    gnn_data = pd.DataFrame({
        'Report_Time': [datetime.now() - timedelta(minutes=np.random.randint(0, 10080)) for _ in range(gnn_size)],
        'Phone_Number': [f"+886-903-88{np.random.randint(0, 9)}-{np.random.randint(100, 999):03d}" for _ in range(gnn_size)],
        'Report_Count': np.random.randint(10, 50, size=gnn_size),
        'Financial_Loss': np.random.randint(10000, 50000, size=gnn_size),
        'Age_Group': np.random.choice(['18-25', '26-40', '41-60', '70+'], size=gnn_size),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master/PhD'], size=gnn_size),
        'Cluster_ID': 'C_88X_RING',
        'Label': 1
    })
    
    # 區塊 B: 特徵交叉驗證 (高財損 + 70+長者/碩博士) (40% 詐騙樣本)
    demo_size = int(size * 0.4)
    demo_data = pd.DataFrame({
        'Report_Time': [datetime.now() - timedelta(minutes=np.random.randint(0, 43200)) for _ in range(demo_size)],
        'Phone_Number': [f"+886-9{np.random.randint(10, 99):02d}-{np.random.randint(100, 999):03d}-{np.random.randint(100, 999):03d}" for _ in range(demo_size)],
        'Report_Count': np.random.randint(5, 20, size=demo_size),
        'Financial_Loss': np.random.randint(500000, 5000000, size=demo_size), # 極高財損
        'Age_Group': np.random.choice(['70+', '41-60'], size=demo_size, p=[0.8, 0.2]),
        'Education': np.random.choice(['Master/PhD', 'Bachelor'], size=demo_size, p=[0.8, 0.2]),
        'Cluster_ID': 'C_DEMO_TARGET',
        'Label': 1
    })

    # 區塊 C: Data Poisoning 機器人攻擊 (20% 詐騙樣本)
    poison_size = size - gnn_size - demo_size
    poison_data = pd.DataFrame({
        'Report_Time': [datetime.now() - timedelta(seconds=np.random.randint(0, 3600)) for _ in range(poison_size)], # 密集回報
        'Phone_Number': ['+886-999-999-999' for _ in range(poison_size)],
        'Report_Count': np.random.randint(9999, 50000, size=poison_size), # 異常飆高
        'Financial_Loss': 0,
        'Age_Group': 'UNKNOWN',
        'Education': 'UNKNOWN',
        'Cluster_ID': 'C_POISON_BOT',
        'Label': 1
    })

    return pd.concat([gnn_data, demo_data, poison_data], ignore_index=True)

# 4. 合併、打亂與輸出
df_normal = generate_normal_data(NORMAL_SIZE)
df_fraud = generate_fraud_data(FRAUD_SIZE)
df_final = pd.concat([df_normal, df_fraud], ignore_index=True)

# 打亂資料集順序
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# 導出 CSV
output_filename = "fraud_100000_dataset.csv"
df_final.to_csv(output_filename, index=False)
print(f"✅ 成功生成 {TOTAL_SAMPLES} 筆測試資料，已儲存至 {output_filename}")
print(f"數據分佈：正常樣本 {NORMAL_SIZE} 筆 (Label=0)，異常樣本 {FRAUD_SIZE} 筆 (Label=1)")