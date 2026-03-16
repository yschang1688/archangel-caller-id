import pandas as pd
import json

def main(args):
    # 模擬接收上一個節點傳來的檔案路徑或數據
    # 在 Dify/n8n 中，通常會透過 args['file_path'] 或直接讀取 webhook 數據
    file_path = "fraud_1000_dataset.csv" 
    
    # 1. 載入數據集
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"讀取檔案失敗: {str(e)}"}

    # 2. 資料清洗：過濾機器人灌水與極端值 (Data Skew Mitigation)
    # 條件：排除 Tags 包含「機器人測試」且 Report_Count 小於 5000 的合理數據
    clean_df = df[~df['Tags'].str.contains("機器人測試", na=False)]
    clean_df = clean_df[clean_df['Report_Count'] < 5000]

    # 3. 數據萃取：針對詐騙行為進行統計分析
    fraud_df = clean_df[clean_df['Is_Fraud'] == 1]
    
    # 計算 Top 3 詐騙標籤
    top_3_tags = fraud_df['Tags'].value_counts().head(3).to_dict()
    
    # 計算平均詐騙金額
    avg_fraud_amount = fraud_df['Transaction_Amount'].mean()
    
    # 統計受害者輪廓分布
    demographic_dist = fraud_df['Victim_Demographic'].value_counts().to_dict()

    # 4. 結構化輸出 (餵給下一個 LLM Agent 的 Payload)
    output_payload = {
        "status": "success",
        "data_quality": {
            "original_records": len(df),
            "cleaned_records": len(clean_df),
            "removed_outliers": len(df) - len(clean_df)
        },
        "fraud_insights": {
            "total_fraud_cases": len(fraud_df),
            "average_transaction_amount_twd": round(avg_fraud_amount, 2),
            "top_3_scam_tags": top_3_tags,
            "victim_demographics": demographic_dist
        }
    }
    
    # 轉換為 JSON 字串返回
    return json.dumps(output_payload, ensure_ascii=False, indent=2)

# 單機測試執行
if __name__ == "__main__":
    result = main({})
    print(result)