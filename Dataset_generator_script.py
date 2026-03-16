import csv
import random
import uuid
from datetime import datetime, timedelta

# 定義基礎設定
TOTAL_RECORDS = 1000
FRAUD_RATIO = 0.7
FRAUD_COUNT = int(TOTAL_RECORDS * FRAUD_RATIO)
NORMAL_COUNT = TOTAL_RECORDS - FRAUD_COUNT

# 定義受害者輪廓
DEMOGRAPHICS = ["70+ 長者", "50-69 中高齡", "上班族", "大專院校", "學生", "碩博士", "高中職以下"]

# 模擬時間生成函數
def generate_time(base_time, minutes_offset):
    return (base_time + timedelta(minutes=minutes_offset)).strftime("%Y-%m-%d %H:%M:%S")

def generate_dataset():
    data = []
    base_time = datetime(2026, 3, 12, 9, 0, 0)
    
    # ==========================================
    # 區塊一：高頻度網購詐騙與機器人行為模擬 (Fraud)
    # 特徵：高 Report_Count, 低 Transaction_Amount
    # ==========================================
    for _ in range(150):
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-901-{random.randint(100, 999):03d}-{random.randint(1, 50):03d}",
            "Tags": random.choice(["網路購物詐騙,解除分期付款", "網路購物詐騙,假賣場", "網路購物詐騙,ATM操作", "幽靈包裹,超商取貨"]),
            "Report_Count": int(random.paretovariate(1.5) * 100) + 50, # 模擬高頻長尾分佈
            "Transaction_Amount": random.randint(1000, 45000),
            "Victim_Demographic": random.choice(["上班族", "大專院校", "學生", "高中職以下"]),
            "Is_Fraud": 1
        })

    # ==========================================
    # 區塊二：假冒公署與高財損複合型威脅 (Fraud)
    # 特徵：極低 Report_Count, 極高 Transaction_Amount, 長者居多
    # ==========================================
    for _ in range(150):
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-902-{random.randint(100, 200):03d}-{random.randint(100, 999):03d}",
            "Tags": random.choice(["假冒郵局,普發現金冒領", "假員警,偵查不公開", "假扣押財產,面交黃金白銀", "假檢警,監管帳戶"]),
            "Report_Count": random.randint(1, 3),
            "Transaction_Amount": random.randint(500000, 8000000),
            "Victim_Demographic": random.choices(["70+ 長者", "50-69 中高齡", "碩博士"], weights=[0.6, 0.3, 0.1])[0],
            "Is_Fraud": 1
        })

    # ==========================================
    # 區塊三：假投資與圖神經網路 (GNN) 拓撲關聯 (Fraud)
    # 特徵：同網段號碼群聚, 高財損, 碩博士受害比例高
    # ==========================================
    for _ in range(250):
        cluster_id = random.randint(880, 889) # 製造 GNN 拓撲節點群聚
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-903-{cluster_id:03d}-{random.randint(1, 99):03d}",
            "Tags": random.choice(["假投資,虛擬貨幣,高報酬", "假投資,內線交易,VIP通道", "假投資,資金盤", "假投資,保證獲利"]),
            "Report_Count": random.randint(5, 20),
            "Transaction_Amount": random.randint(100000, 5000000),
            "Victim_Demographic": random.choices(["碩博士", "上班族", "50-69 中高齡", "大專院校"], weights=[0.4, 0.3, 0.2, 0.1])[0],
            "Is_Fraud": 1
        })

    # ==========================================
    # 區塊四：社交工程、新興 AI 威脅 (Fraud)
    # 特徵：低財損起手式, 結合 Threads/AI 濾鏡
    # ==========================================
    for _ in range(150):
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-904-{random.randint(100, 999):03d}-{random.randint(1, 99):03d}",
            "Tags": random.choice(["網路交友,要求儲值,點數卡", "網路交友,AI濾鏡假冒", "Threads廣告,假兼職", "殺豬盤,平台無法出金"]),
            "Report_Count": random.randint(2, 30),
            "Transaction_Amount": random.randint(0, 150000),
            "Victim_Demographic": random.choice(["大專院校", "上班族", "學生", "碩博士"]),
            "Is_Fraud": 1
        })

    # ==========================================
    # 正常交易糾紛、錯誤通報或系統雜訊 (Normal - Is_Fraud = 0)
    # 特徵：真實世界的雜訊，考驗 NLP 與 Agent 的誤判率 (False Positive)
    # ==========================================
    for _ in range(NORMAL_COUNT):
        is_bot = random.random() < 0.05
        if is_bot:
            # 惡意機器人干擾模式 (防禦阻斷攻擊模擬)
            data.append({
                "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
                "Report_Time": generate_time(base_time, random.randint(1, 10)),
                "Phone_Number": "+886-999-999-999",
                "Tags": "機器人測試,惡意灌水",
                "Report_Count": random.randint(8000, 15000),
                "Transaction_Amount": 0,
                "Victim_Demographic": "未知",
                "Is_Fraud": 0
            })
        else:
            data.append({
                "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
                "Report_Time": generate_time(base_time, random.randint(1, 1440)),
                "Phone_Number": f"+886-905-{random.randint(100, 999):03d}-{random.randint(100, 999):03d}",
                "Tags": random.choice(["交易糾紛,商品瑕疵", "一般客訴,服務態度", "債務不履行,退款延遲", "司法文書送達,正常公務", "正常投資諮詢,理財規劃"]),
                "Report_Count": random.randint(1, 2),
                "Transaction_Amount": random.choice([0, 500, 3000, 15000]),
                "Victim_Demographic": random.choice(DEMOGRAPHICS),
                "Is_Fraud": 0
            })

    # 打亂數據順序以模擬真實串流
    random.shuffle(data)

    # 寫入 CSV 檔案
    keys = ["Incident_ID", "Report_Time", "Phone_Number", "Tags", "Report_Count", "Transaction_Amount", "Victim_Demographic", "Is_Fraud"]
    with open("fraud_1000_dataset.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"成功生成測試數據：共 {TOTAL_RECORDS} 筆資料 (詐騙 {FRAUD_COUNT} 筆, 正常 {NORMAL_COUNT} 筆)")
    print("檔案已儲存為: fraud_1000_dataset.csv")

if __name__ == "__main__":
    generate_dataset()