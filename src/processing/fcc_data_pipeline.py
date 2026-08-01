"""
fcc_data_pipeline.py — FCC Consumer Complaints 特徵工程管線
============================================================
從 FCC 消費者投訴原始資料中，以「電話號碼」為單位聚合出 20 個特徵，
用於 SVM 二元分類器判斷來電是否為垃圾電話（spam）。

設計原則：
  - 推論延遲 < 0.5 秒：所有特徵皆為 per-number 預聚合值，推論時直接查表
  - 20 維特徵涵蓋 5 大面向：時間模式 / 通話類型 / 投訴內容 / 號碼特徵 / 地理行為
  - 效能與精確度權衡：犧牲少量精確度換取即時回應能力

特徵分類：
  ┌───────────────────────────┬──────────────────────────────────────┐
  │ 面向                       │ 對應偵測對象                          │
  ├───────────────────────────┼──────────────────────────────────────┤
  │ 時間模式 (5)               │ 上班族 / 學生 / 家庭成員              │
  │ 通話類型 (5)               │ 推銷 / 機器人 / 騷擾電話              │
  │ 投訴內容 (4)               │ 色情 / 詐騙 / 未授權收費              │
  │ 號碼特徵 (3)               │ VoIP / 免費號碼 / 行動電話            │
  │ 地理行為 (3)               │ 跨州撥打 / 高頻 / 多號碼              │
  └───────────────────────────┴──────────────────────────────────────┘

資料來源：FCC Consumer Complaints RAW Data
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FCC_DATASET_2017 = os.path.join(_PROJECT_ROOT, "FCC_Consumer_Complaints_RAW_Data_2017.csv")

# ── 免費號碼區號 ──
TOLL_FREE_PREFIXES = {"800", "888", "877", "866", "855", "844", "833"}

# ── 垃圾電話相關 Issue 類別 ──
SPAM_ISSUES = {
    "Telemarketing (including do not call and spoofing)",
    "Robocalls",
    "Unwanted Calls",
    "Junk Faxes",
    "Cramming (unauthorized charges on your phone bill)",
    "Slamming (change of your carrier without permission)",
    "Indecency",
}

# ── 20 個特徵名稱（固定順序）──
FEATURE_NAMES = [
    # 時間模式 (5) — 偵測對象：上班族/學生/家庭成員
    "business_hour_ratio",      # 上班時間 (09-17) 比例 → 瞄準上班族
    "evening_ratio",            # 晚間 (18-22) 比例 → 瞄準家庭成員
    "night_ratio",              # 深夜 (22-06) 比例 → 騷擾/色情電話
    "school_hour_ratio",        # 學校時間 (08-15 weekday) 比例 → 瞄準學生
    "weekend_ratio",            # 週末比例

    # 通話類型 (5) — 偵測：推銷/機器人/騷擾
    "robocall_flag",            # 有 Robocall 投訴
    "telemarketing_flag",       # 有電話推銷投訴
    "prerecorded_ratio",        # 預錄語音比例
    "abandoned_call_ratio",     # 掛斷/未接比例
    "live_voice_ratio",         # 真人語音比例

    # 投訴內容 (4) — 偵測：色情/詐騙/騷擾
    "indecency_flag",           # 色情/不雅內容
    "cramming_flag",            # 未授權收費 (詐騙)
    "issue_diversity",          # 投訴問題多樣性
    "complaint_count_log",      # 被投訴次數 (log1p 壓縮)

    # 號碼特徵 (3) — 技術判斷
    "is_toll_free",             # 是否免費號碼
    "is_voip",                  # VoIP 比例
    "is_wireless",              # 行動電話比例

    # 地理行為 (3) — 大範圍撥打偵測
    "unique_states_norm",       # 跨州投訴數（正規化）
    "complaint_velocity",       # 投訴增長速率 (投訴數/天數)
    "multi_caller_flag",        # 同廣告商多號碼關聯
]


# ─────────────────────────────────────────────────────────────────────────────
# 時間解析工具
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hour(time_str: str) -> Optional[int]:
    """
    解析 FCC 資料中各種格式的時間字串，回傳 24 小時制的小時數。

    支援格式：'1:00 pm', '8:08 PM', '2:30 p.m.', '14:30', '7:00 am' 等。

    參數：
        time_str: 原始時間字串

    回傳：
        int (0-23) 或 None（解析失敗時）
    """
    if not isinstance(time_str, str) or not time_str.strip():
        return None

    time_str = time_str.strip().lower().replace(".", "")

    # 匹配 HH:MM am/pm 格式
    match = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?", time_str)
    if match:
        hour = int(match.group(1))
        period = match.group(3)
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23:
            return hour

    return None


def _extract_area_code(phone_number: str) -> Optional[str]:
    """
    從電話號碼中萃取區號 (前三碼)。

    參數：
        phone_number: 例如 '866-410-0458'

    回傳：
        str 或 None
    """
    if not isinstance(phone_number, str):
        return None
    digits = re.sub(r"\D", "", phone_number)
    if len(digits) >= 3:
        return digits[:3]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 核心：載入 + 清洗 + 特徵工程
# ─────────────────────────────────────────────────────────────────────────────

def load_fcc_raw(file_path: str = None) -> pd.DataFrame:
    """
    載入 FCC 原始 CSV，進行基本清洗。

    清洗內容：
      - 過濾掉無 Caller ID Number 的記錄
      - 解析日期 / 時間
      - 過濾明顯假號碼 (000-000-0000, 555-555-5555)

    參數：
        file_path: CSV 檔案路徑，預設使用 FCC_DATASET_2017

    回傳：
        清洗後的 DataFrame
    """
    if file_path is None:
        file_path = FCC_DATASET_2017

    logger.info(f"載入 FCC 資料: {file_path}")
    df = pd.read_csv(file_path, on_bad_lines="skip")
    logger.info(f"原始記錄數: {len(df):,}")

    # 僅保留有 Caller ID 的記錄
    df = df[df["Caller ID Number"].notna()].copy()
    logger.info(f"有 Caller ID: {len(df):,}")

    # 移除明顯假號碼
    fake_numbers = {"000-000-0000", "555-555-5555", "111-111-1111", "999-999-9999"}
    df = df[~df["Caller ID Number"].isin(fake_numbers)].copy()
    logger.info(f"移除假號碼後: {len(df):,}")

    # 解析日期
    df["parsed_date"] = pd.to_datetime(df["Date of Issue"], errors="coerce")

    # 過濾極端日期（合理範圍: 2015-2018）
    date_mask = (df["parsed_date"] >= "2015-01-01") & (df["parsed_date"] <= "2018-12-31")
    df.loc[~date_mask, "parsed_date"] = pd.NaT

    # 解析小時
    df["parsed_hour"] = df["Time of Issue"].apply(_parse_hour)

    # 解析星期幾 (0=Monday ... 6=Sunday)
    df["day_of_week"] = df["parsed_date"].dt.dayofweek

    # 萃取區號
    df["area_code"] = df["Caller ID Number"].apply(_extract_area_code)

    logger.info(f"清洗完成，可用記錄: {len(df):,}")
    return df


def engineer_features_per_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    以電話號碼為單位，聚合出 20 個特徵（向量化版本）。

    每個電話號碼產生一個 20 維特徵向量，涵蓋：
      - 時間模式 (5)：上班時間/晚間/深夜/學校時間/週末
      - 通話類型 (5)：機器人/推銷/預錄語音/掛斷/真人
      - 投訴內容 (4)：色情/詐騙/多樣性/總次數
      - 號碼特徵 (3)：免費號碼/VoIP/行動電話
      - 地理行為 (3)：跨州/速率/多號碼

    參數：
        df: 經 load_fcc_raw() 清洗後的 DataFrame

    回傳：
        DataFrame，index 為電話號碼，columns 為 20 個特徵名稱
    """
    phone_col = "Caller ID Number"
    n_groups = df[phone_col].nunique()
    logger.info(f"聚合 {n_groups:,} 組電話號碼（向量化）...")

    # ── 預先計算 boolean 欄位，避免重複 str 操作 ────────────────────
    hour = df["parsed_hour"]
    dow = df["day_of_week"]

    df = df.copy()
    df["_is_business"] = (hour >= 9) & (hour < 17)
    df["_is_evening"] = (hour >= 18) & (hour < 22)
    df["_is_night"] = (hour >= 22) | (hour < 6)
    df["_is_school"] = dow.isin([0, 1, 2, 3, 4]) & (hour >= 8) & (hour < 15)
    df["_is_weekend"] = dow.isin([5, 6])
    df["_hour_valid"] = hour.notna()
    df["_dow_valid"] = dow.notna()

    issue = df["Issue"].fillna("")
    df["_is_robocall"] = issue.str.contains("Robocall", case=False, na=False)
    df["_is_telemarketing"] = issue.str.contains("Telemarketing", case=False, na=False)
    df["_is_indecency"] = issue.str.contains("Indecency", case=False, na=False)
    df["_is_cramming"] = issue.str.contains("Cramming|Slamming", case=False, na=False)
    df["_issue_notna"] = df["Issue"].notna()

    call_type = df.get("Type of Call or Messge")
    if call_type is not None:
        ct = call_type.fillna("")
        df["_is_prerecorded"] = ct == "Prerecorded Voice"
        df["_is_abandoned"] = ct == "Abandoned Calls"
        df["_is_live_voice"] = ct.isin(["Live Voice", "Autodialed Live Voice Call"])
        df["_ct_valid"] = call_type.notna()
    else:
        df["_is_prerecorded"] = False
        df["_is_abandoned"] = False
        df["_is_live_voice"] = False
        df["_ct_valid"] = False

    method = df.get("Method")
    if method is not None:
        m = method.fillna("")
        df["_is_voip_row"] = m.str.contains("VOIP", case=False, na=False)
        df["_is_wireless_row"] = m.str.contains("Wireless|cell phone", case=False, na=False)
        df["_method_valid"] = method.notna()
    else:
        df["_is_voip_row"] = False
        df["_is_wireless_row"] = False
        df["_method_valid"] = False

    adv = df.get("Advertiser Business Number")
    if adv is not None:
        df["_adv_notna"] = adv.notna()
        df["_adv_match_caller"] = (adv == df[phone_col]) & adv.notna()
    else:
        df["_adv_notna"] = False
        df["_adv_match_caller"] = False

    # ── 分組聚合（全部用 .agg + 向量操作）──────────────────────────
    g = df.groupby(phone_col, sort=False)

    agg = g.agg(
        n_records=("_hour_valid", "size"),
        # 時間
        hour_valid_sum=("_hour_valid", "sum"),
        biz_sum=("_is_business", "sum"),
        eve_sum=("_is_evening", "sum"),
        ngt_sum=("_is_night", "sum"),
        sch_sum=("_is_school", "sum"),
        dow_valid_sum=("_dow_valid", "sum"),
        wkd_sum=("_is_weekend", "sum"),
        # 通話類型
        robocall_any=("_is_robocall", "max"),
        telemarketing_any=("_is_telemarketing", "max"),
        ct_valid_sum=("_ct_valid", "sum"),
        pre_sum=("_is_prerecorded", "sum"),
        abd_sum=("_is_abandoned", "sum"),
        live_sum=("_is_live_voice", "sum"),
        # 投訴內容
        indecency_any=("_is_indecency", "max"),
        cramming_any=("_is_cramming", "max"),
        # 號碼/方法
        method_valid_sum=("_method_valid", "sum"),
        voip_sum=("_is_voip_row", "sum"),
        wireless_sum=("_is_wireless_row", "sum"),
        # 廣告商
        adv_notna_sum=("_adv_notna", "sum"),
        adv_match_sum=("_adv_match_caller", "sum"),
    )

    # Issue diversity & State uniqueness 需要 nunique
    issue_div = g["Issue"].nunique().rename("issue_diversity")
    state_uniq = g["State"].nunique().rename("unique_states")

    # 投訴速率: n_records / date_range_days
    date_min = g["parsed_date"].min()
    date_max = g["parsed_date"].max()
    date_range_days = (date_max - date_min).dt.days
    # len>=2 用 n_records/max(days,1)；len<2 用 n_records
    has_range = date_range_days.notna() & (date_range_days > 0)

    agg = agg.join(issue_div).join(state_uniq)

    # ── 計算最終 20 特徵 ──────────────────────────────────────────
    eps = 1e-9  # 避免除零
    hv = agg["hour_valid_sum"].clip(lower=eps)
    dv = agg["dow_valid_sum"].clip(lower=eps)
    cv = agg["ct_valid_sum"].clip(lower=eps)
    mv = agg["method_valid_sum"].clip(lower=eps)

    result = pd.DataFrame(index=agg.index)

    # 時間模式 (5)
    result["business_hour_ratio"] = np.where(agg["hour_valid_sum"] > 0, agg["biz_sum"] / hv, 0.5)
    result["evening_ratio"] = np.where(agg["hour_valid_sum"] > 0, agg["eve_sum"] / hv, 0.2)
    result["night_ratio"] = np.where(agg["hour_valid_sum"] > 0, agg["ngt_sum"] / hv, 0.1)
    result["school_hour_ratio"] = np.where(agg["hour_valid_sum"] > 0, agg["sch_sum"] / hv, 0.3)
    result["weekend_ratio"] = np.where(agg["dow_valid_sum"] > 0, agg["wkd_sum"] / dv, 2.0 / 7.0)

    # 通話類型 (5)
    result["robocall_flag"] = agg["robocall_any"].astype(float)
    result["telemarketing_flag"] = agg["telemarketing_any"].astype(float)
    result["prerecorded_ratio"] = np.where(agg["ct_valid_sum"] > 0, agg["pre_sum"] / cv, 0.0)
    result["abandoned_call_ratio"] = np.where(agg["ct_valid_sum"] > 0, agg["abd_sum"] / cv, 0.0)
    result["live_voice_ratio"] = np.where(agg["ct_valid_sum"] > 0, agg["live_sum"] / cv, 0.0)

    # 投訴內容 (4)
    result["indecency_flag"] = agg["indecency_any"].astype(float)
    result["cramming_flag"] = agg["cramming_any"].astype(float)
    result["issue_diversity"] = agg["issue_diversity"].fillna(0).astype(float)
    result["complaint_count_log"] = np.log1p(agg["n_records"])

    # 號碼特徵 (3)
    area_codes = pd.Series(result.index, index=result.index).apply(_extract_area_code)
    result["is_toll_free"] = area_codes.apply(lambda ac: int(ac in TOLL_FREE_PREFIXES) if ac else 0).astype(float)
    result["is_voip"] = np.where(agg["method_valid_sum"] > 0, agg["voip_sum"] / mv, 0.0)
    result["is_wireless"] = np.where(agg["method_valid_sum"] > 0, agg["wireless_sum"] / mv, 0.0)

    # 地理行為 (3)
    result["unique_states_norm"] = (agg["unique_states"].fillna(0) / 50.0).clip(upper=1.0)

    nr = agg["n_records"].astype(float)
    velocity = np.where(
        has_range,
        nr / date_range_days.clip(lower=1),
        nr,
    )
    result["complaint_velocity"] = velocity

    result["multi_caller_flag"] = np.where(
        agg["adv_notna_sum"] > 0,
        (agg["adv_match_sum"] < agg["adv_notna_sum"]).astype(float),
        0.0,
    )

    result.index.name = "phone_number"
    result = result[FEATURE_NAMES]

    logger.info(f"特徵工程完成: {result.shape[0]:,} 筆號碼 × {result.shape[1]} 特徵")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 標籤生成 + 負樣本合成
# ─────────────────────────────────────────────────────────────────────────────

def _generate_negative_samples(
    n_samples: int,
    seed: int = SEED,
    hard_negative_pool: pd.DataFrame | None = None,
    hard_negative_ratio: float = 0.03,
) -> pd.DataFrame:
    """
    生成更貼近真實世界的「非垃圾電話」負樣本（混合策略）。

    設計動機（對應 domain know-how）：
      - 多數負樣本（預設 97/100）：代表「真實正常號碼」——店家、政府機關、物流、私人號碼、通訊錄號碼。
        這些號碼通常具有 1 次左右的投訴記錄（或被外部資料源標註為正常類型），行為模式相對穩定。
        ※ 本專案不直接連外抓 Google Place API / 黃頁 / 競品網站 / 政府常用號碼清單，
           這裡用參數化分佈近似上述資料來源的行為分佈。

      - 少數負樣本（預設 3/100）：代表「hard negatives」——出現在 FCC 投訴系統中、可能被誤投訴的正常號碼。
        它們可能具備 robocall/telemarketing 訊號，且時間/地理分佈與 spam 更接近，
        讓決策邊界更貼近真實場景（預期 F1 下降到約 0.87–0.93）。

    參數：
        n_samples:             生成樣本數
        seed:                  隨機種子
        hard_negative_pool:    hard negative 候選池（若提供，會抽樣部分真實特徵作為 hard negatives）
        hard_negative_ratio:   hard negatives 比例（預設 0.03）

    依賴：
        numpy, pandas

    回傳：
        DataFrame，columns 與 FEATURE_NAMES 一致
    """
    rng = np.random.RandomState(seed)

    hard_negative_ratio = float(np.clip(hard_negative_ratio, 0.0, 0.5))
    n_hard = int(round(n_samples * hard_negative_ratio))
    n_soft = int(n_samples - n_hard)

    # ──────────────────────────────────────────────────────────────
    # (A) Soft negatives：外部已標註正常號碼的近似分佈（預設 97%）
    # ──────────────────────────────────────────────────────────────
    soft = {
        # 時間模式（正常號碼多在上班/晚間，深夜較少，但非零）
        "business_hour_ratio": rng.beta(5, 3, n_soft),
        "evening_ratio": rng.beta(3, 5, n_soft),
        "night_ratio": rng.beta(2, 18, n_soft),
        "school_hour_ratio": rng.beta(3, 6, n_soft),
        "weekend_ratio": rng.beta(2, 5, n_soft),

        # 通話類型（正常號碼仍可能被誤判/誤報為 robocall/telemarketing）
        "robocall_flag": rng.binomial(1, 0.05, n_soft).astype(float),
        "telemarketing_flag": rng.binomial(1, 0.06, n_soft).astype(float),
        "prerecorded_ratio": rng.beta(1, 12, n_soft),
        "abandoned_call_ratio": rng.beta(1, 10, n_soft),
        "live_voice_ratio": rng.beta(7, 3, n_soft),

        # 投訴內容（大多是 1 次投訴的近似：log1p(1)=0.693；加入少量波動避免完美分界）
        "indecency_flag": rng.binomial(1, 0.002, n_soft).astype(float),
        "cramming_flag": rng.binomial(1, 0.02, n_soft).astype(float),
        "issue_diversity": rng.choice([0, 1, 2], n_soft, p=[0.75, 0.22, 0.03]).astype(float),
        "complaint_count_log": (rng.normal(loc=np.log1p(1), scale=0.12, size=n_soft)).clip(0.0, 1.3),

        # 號碼特徵（店家/政府/物流可能含 toll-free，也可能是 VoIP）
        "is_toll_free": rng.binomial(1, 0.10, n_soft).astype(float),
        "is_voip": rng.beta(2, 8, n_soft),
        "is_wireless": rng.beta(4, 4, n_soft),

        # 地理行為（正常號碼偏集中，但物流/客服可能跨州：保留長尾）
        "unique_states_norm": rng.beta(2, 18, n_soft),
        "complaint_velocity": rng.exponential(0.35, n_soft).clip(0, 3),
        "multi_caller_flag": rng.binomial(1, 0.08, n_soft).astype(float),
    }
    soft_df = pd.DataFrame(soft)[FEATURE_NAMES]

    # ──────────────────────────────────────────────────────────────
    # (B) Hard negatives：FCC 內部誤投訴/邊界樣本（預設 3%）
    # ──────────────────────────────────────────────────────────────
    if n_hard <= 0:
        hard_df = pd.DataFrame(columns=FEATURE_NAMES)
    elif hard_negative_pool is not None and len(hard_negative_pool) > 0:
        hard_df = hard_negative_pool.sample(
            n=n_hard,
            replace=len(hard_negative_pool) < n_hard,
            random_state=seed,
        )[FEATURE_NAMES].copy()
    else:
        hard = {
            "business_hour_ratio": rng.beta(3, 4, n_hard),
            "evening_ratio": rng.beta(3, 4, n_hard),
            "night_ratio": rng.beta(2, 6, n_hard),
            "school_hour_ratio": rng.beta(2, 6, n_hard),
            "weekend_ratio": rng.beta(2, 4, n_hard),

            "robocall_flag": rng.binomial(1, 0.20, n_hard).astype(float),
            "telemarketing_flag": rng.binomial(1, 0.22, n_hard).astype(float),
            "prerecorded_ratio": rng.beta(1, 6, n_hard),
            "abandoned_call_ratio": rng.beta(1, 6, n_hard),
            "live_voice_ratio": rng.beta(5, 5, n_hard),

            "indecency_flag": rng.binomial(1, 0.01, n_hard).astype(float),
            "cramming_flag": rng.binomial(1, 0.05, n_hard).astype(float),
            "issue_diversity": rng.choice([0, 1, 2, 3], n_hard, p=[0.55, 0.30, 0.12, 0.03]).astype(float),
            "complaint_count_log": (rng.normal(loc=np.log1p(1), scale=0.18, size=n_hard)).clip(0.0, 1.3),

            "is_toll_free": rng.binomial(1, 0.15, n_hard).astype(float),
            "is_voip": rng.beta(3, 7, n_hard),
            "is_wireless": rng.beta(4, 4, n_hard),

            "unique_states_norm": rng.beta(3, 10, n_hard),
            "complaint_velocity": rng.exponential(0.6, n_hard).clip(0, 4),
            "multi_caller_flag": rng.binomial(1, 0.12, n_hard).astype(float),
        }
        hard_df = pd.DataFrame(hard)[FEATURE_NAMES]

    neg_df = pd.concat([soft_df, hard_df], ignore_index=True)

    # Shuffle（避免前段都是 soft / 後段都是 hard）
    shuffle_idx = rng.permutation(len(neg_df))
    neg_df = neg_df.iloc[shuffle_idx].reset_index(drop=True)
    return neg_df[FEATURE_NAMES]


def build_training_dataset(
    fcc_features: pd.DataFrame,
    negative_ratio: float = 1.0,
    hard_negative_ratio: float = 0.03,
    seed: int = SEED,
) -> tuple:
    """
    組合正/負樣本，建立 SVM 訓練用資料集。

    正樣本 (Label=1): FCC 投訴資料中投訴次數 >= 2 的電話號碼
    負樣本 (Label=0): 合成的正常電話行為模式

    參數：
        fcc_features:   由 engineer_features_per_number() 產出的 DataFrame
        negative_ratio: 負/正樣本比例 (1.0 = 平衡)
        hard_negative_ratio: hard negatives 比例（預設 0.03 = 3/100）
        seed:           隨機種子

    回傳：
        (X, y) — 特徵矩陣與標籤
    """
    # 正樣本：投訴次數 >= 2 (log1p(2) ≈ 1.1)，確保是真正的高頻投訴號碼
    complaint_threshold = np.log1p(2)
    positive = fcc_features[fcc_features["complaint_count_log"] >= complaint_threshold].copy()
    logger.info(f"正樣本 (spam): {len(positive):,} 筆")

    # 負樣本
    n_negative = int(len(positive) * negative_ratio)
    # hard negatives：從 FCC 內部抽取 complaint_count=1 的「可能被誤投訴」樣本，提升真實性
    hard_pool = fcc_features[
        (fcc_features["complaint_count_log"] > 0.0)
        & (fcc_features["complaint_count_log"] < complaint_threshold)
    ].copy()
    negative = _generate_negative_samples(
        n_negative,
        seed=seed,
        hard_negative_pool=hard_pool if len(hard_pool) > 0 else None,
        hard_negative_ratio=hard_negative_ratio,
    )
    logger.info(f"負樣本 (not spam): {len(negative):,} 筆")

    # 合併
    positive_y = pd.Series(np.ones(len(positive), dtype=int), name="label")
    negative_y = pd.Series(np.zeros(len(negative), dtype=int), name="label")

    X = pd.concat([positive.reset_index(drop=True), negative.reset_index(drop=True)], ignore_index=True)
    y = pd.concat([positive_y, negative_y], ignore_index=True)

    # Shuffle
    rng = np.random.RandomState(seed)
    shuffle_idx = rng.permutation(len(X))
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y.iloc[shuffle_idx].reset_index(drop=True)

    logger.info(f"訓練集: {X.shape[0]:,} 筆 × {X.shape[1]} 特徵")
    logger.info(f"  Spam:     {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    logger.info(f"  Not spam: {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 一站式 Public API
# ─────────────────────────────────────────────────────────────────────────────

def fcc_clean_and_prepare(
    file_path: str = None,
    negative_ratio: float = 1.0,
    hard_negative_ratio: float = 0.03,
) -> tuple:
    """
    FCC 資料完整管線：載入 → 清洗 → 20 特徵工程 → 建立訓練集。

    參數：
        file_path:      FCC CSV 路徑
        negative_ratio: 負/正樣本比例
        hard_negative_ratio: hard negatives 比例（預設 0.03 = 3/100）

    回傳：
        (X, y, fcc_features_df, raw_df)
        X: 訓練用特徵矩陣
        y: 二元標籤 (0=正常, 1=垃圾電話)
        fcc_features_df: 所有號碼的 20 維特徵 (含 lookup 用途)
        raw_df: 原始清洗後 DataFrame
    """
    print("\n" + "═" * 60)
    print("  📞 FCC Consumer Complaints 特徵工程管線")
    print("═" * 60)

    raw_df = load_fcc_raw(file_path)
    fcc_features_df = engineer_features_per_number(raw_df)
    X, y = build_training_dataset(
        fcc_features_df,
        negative_ratio=negative_ratio,
        hard_negative_ratio=hard_negative_ratio,
    )

    print(f"\n✅ FCC 管線完成")
    print(f"   號碼總數: {len(fcc_features_df):,}")
    print(f"   訓練集:   {X.shape[0]:,} × {X.shape[1]}")
    print(f"   特徵名稱: {FEATURE_NAMES}")

    return X, y, fcc_features_df, raw_df


if __name__ == "__main__":
    X, y, features, raw = fcc_clean_and_prepare()
    print(f"\nX shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")
    print(f"\nSample features (first 3 spam numbers):")
    print(features.head(3))
