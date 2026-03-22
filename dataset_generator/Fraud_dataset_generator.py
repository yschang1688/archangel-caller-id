import csv
import random
import uuid
import argparse
from datetime import datetime, timedelta

DEMOGRAPHICS = ["70+ 長者", "50-69 中高齡", "上班族", "大專院校", "學生", "碩博士", "高中職以下"]

def generate_time(base_time, minutes_offset):
    return (base_time + timedelta(minutes=minutes_offset)).strftime("%Y-%m-%d %H:%M:%S")

def generate_dataset(total_records: int, fraud_ratio: float, output_file: str):
    fraud_count = int(total_records * fraud_ratio)
    normal_count = total_records - fraud_count

    data = []
    base_time = datetime(2026, 3, 12, 9, 0, 0)

    # 固定各 Fraud 區塊數量，可以依需要再改成比例式
    block1_n = 1500
    block2_n = 1500
    block3_n = 2500
    block4_n = 1500
    fixed_fraud_blocks = block1_n + block2_n + block3_n + block4_n

    if fraud_count != fixed_fraud_blocks:
        print(f"[警告] 目前 fraud_count={fraud_count}，但四個 Fraud 區塊固定總和={fixed_fraud_blocks}，"
              f"會以區塊設定為主，實際詐騙筆數={fixed_fraud_blocks}")
        fraud_count = fixed_fraud_blocks
        normal_count = total_records - fraud_count

    # 區塊一
    for _ in range(block1_n):
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-901-{random.randint(100, 999):03d}-{random.randint(1, 50):03d}",
            "Tags": random.choice([
                "網路購物詐騙,解除分期付款",
                "網路購物詐騙,假賣場",
                "網路購物詐騙,ATM操作",
                "幽靈包裹,超商取貨"
            ]),
            "Report_Count": int(random.paretovariate(1.5) * 100) + 50,
            "Transaction_Amount": random.randint(1000, 45000),
            "Victim_Demographic": random.choice(["上班族", "大專院校", "學生", "高中職以下"]),
            "Is_Fraud": 1
        })

    # 區塊二
    for _ in range(block2_n):
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-902-{random.randint(100, 200):03d}-{random.randint(100, 999):03d}",
            "Tags": random.choice([
                "假冒郵局,普發現金冒領",
                "假員警,偵查不公開",
                "假扣押財產,面交黃金白銀",
                "假檢警,監管帳戶"
            ]),
            "Report_Count": random.randint(1, 3),
            "Transaction_Amount": random.randint(500000, 8000000),
            "Victim_Demographic": random.choices(
                ["70+ 長者", "50-69 中高齡", "碩博士"],
                weights=[0.6, 0.3, 0.1]
            )[0],
            "Is_Fraud": 1
        })

    # 區塊三
    for _ in range(block3_n):
        cluster_id = random.randint(880, 889)
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-903-{cluster_id:03d}-{random.randint(1, 99):03d}",
            "Tags": random.choice([
                "假投資,虛擬貨幣,高報酬",
                "假投資,內線交易,VIP通道",
                "假投資,資金盤",
                "假投資,保證獲利"
            ]),
            "Report_Count": random.randint(5, 20),
            "Transaction_Amount": random.randint(100000, 5000000),
            "Victim_Demographic": random.choices(
                ["碩博士", "上班族", "50-69 中高齡", "大專院校"],
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0],
            "Is_Fraud": 1
        })

    # 區塊四
    for _ in range(block4_n):
        data.append({
            "Incident_ID": f"UID-{str(uuid.uuid4())[:8]}",
            "Report_Time": generate_time(base_time, random.randint(1, 1440)),
            "Phone_Number": f"+886-904-{random.randint(100, 999):03d}-{random.randint(1, 99):03d}",
            "Tags": random.choice([
                "網路交友,要求儲值,點數卡",
                "網路交友,AI濾鏡假冒",
                "Threads廣告,假兼職",
                "殺豬盤,平台無法出金"
            ]),
            "Report_Count": random.randint(2, 30),
            "Transaction_Amount": random.randint(0, 150000),
            "Victim_Demographic": random.choice(["大專院校", "上班族", "學生", "碩博士"]),
            "Is_Fraud": 1
        })

    # 正常樣本
    for _ in range(normal_count):
        is_bot = random.random() < 0.05
        if is_bot:
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
                "Tags": random.choice([
                    "交易糾紛,商品瑕疵",
                    "一般客訴,服務態度",
                    "債務不履行,退款延遲",
                    "司法文書送達,正常公務",
                    "正常投資諮詢,理財規劃"
                ]),
                "Report_Count": random.randint(1, 2),
                "Transaction_Amount": random.choice([0, 500, 3000, 15000]),
                "Victim_Demographic": random.choice(DEMOGRAPHICS),
                "Is_Fraud": 0
            })

    random.shuffle(data)

    keys = [
        "Incident_ID",
        "Report_Time",
        "Phone_Number",
        "Tags",
        "Report_Count",
        "Transaction_Amount",
        "Victim_Demographic",
        "Is_Fraud",
    ]
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"成功生成測試數據：共 {total_records} 筆資料 (詐騙 {fraud_count} 筆, 正常 {normal_count} 筆)")
    print(f"檔案已儲存為: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="詐騙與正常樣本 Dataset 產生器")
    parser.add_argument("--total", type=int, default=10000, help="總筆數，預設 1000")
    parser.add_argument("--fraud-ratio", type=float, default=0.7, help="詐騙比例 0~1，預設 0.7")
    parser.add_argument("--output", type=str, default="fraud_1000_dataset.csv", help="輸出 CSV 檔名")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.total, args.fraud_ratio, args.output)