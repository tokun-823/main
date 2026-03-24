"""
競輪レースIDスクレイピングスクリプト (改善版)

改善点:
- 既にバックアップに存在する月はスキップ
- 月ごとにCSV・Pickleにバックアップ追記保存
- タイムアウト・リトライ機能
- エラーハンドリング強化
"""

import pandas as pd
import re
import pickle
import os
from time import sleep
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from datetime import datetime, timedelta
import calendar


# ============================================================
# 設定
# ============================================================
START_YEAR = 2018
END_YEAR = 2026  # 2026年は含まない (2025年まで取得)
REQUEST_TIMEOUT = 30  # 秒
MAX_RETRIES = 3
RETRY_WAIT = 5  # リトライ間のウェイト(秒)
SLEEP_BETWEEN_REQUESTS = 1  # リクエスト間のウェイト(秒)
SLEEP_BETWEEN_MONTHS = 2  # 月ごとのウェイト(秒)

BACKUP_CSV_PATH = "scraped_data/csv/total_race_ids_backup.csv"
BACKUP_PKL_PATH = "scraped_data/pickle/total_race_ids_backup.pkl"


# ============================================================
# セッション作成（リトライ機能付き）
# ============================================================
def create_session():
    """リトライ機能付きのrequests.Sessionを作成"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ============================================================
# バックアップの読み込み
# ============================================================
def load_existing_ids():
    """既存のバックアップからレースIDを読み込む"""
    existing_ids = []

    # Pickleから読み込み（高速・確実）
    if os.path.exists(BACKUP_PKL_PATH):
        try:
            with open(BACKUP_PKL_PATH, "rb") as f:
                existing_ids = pickle.load(f)
            print(f"[INFO] Pickleバックアップから {len(existing_ids)} 件のレースIDを読み込みました。")
            return existing_ids
        except Exception as e:
            print(f"[WARN] Pickleの読み込みに失敗: {e}")

    # Pickle失敗時はCSVから読み込み
    if os.path.exists(BACKUP_CSV_PATH):
        try:
            df = pd.read_csv(BACKUP_CSV_PATH, header=None, dtype=str)
            existing_ids = df[0].tolist()
            print(f"[INFO] CSVバックアップから {len(existing_ids)} 件のレースIDを読み込みました。")
            return existing_ids
        except Exception as e:
            print(f"[WARN] CSVの読み込みに失敗: {e}")

    print("[INFO] 既存のバックアップが見つかりません。最初から取得します。")
    return existing_ids


def get_completed_months(existing_ids):
    """既存IDから取得済みの年月セットを取得する"""
    completed_months = set()
    for race_id in existing_ids:
        # レースIDの形式: XXYYYYYMMDDNNNNRR (XX=場所, YYYY=年, MM=月, DD=日, NNNN=開催, RR=レース)
        # 例: 1320180102010001 → 2018/01
        if len(race_id) >= 8:
            year = race_id[2:6]
            month = race_id[6:8]
            completed_months.add(f"{year}/{month}")
    return completed_months


# ============================================================
# バックアップの保存
# ============================================================
def save_backup(total_race_ids):
    """レースIDをCSVとPickleに保存"""
    # ディレクトリの作成
    os.makedirs(os.path.dirname(BACKUP_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BACKUP_PKL_PATH), exist_ok=True)

    # Pickle保存
    with open(BACKUP_PKL_PATH, "wb") as f:
        pickle.dump(total_race_ids, f)

    # CSV保存
    pd.Series(total_race_ids).to_csv(BACKUP_CSV_PATH, index=False, header=False)

    print(f"[SAVE] バックアップ保存完了 ({len(total_race_ids)} 件)")


# ============================================================
# HTTPリクエスト（タイムアウト・リトライ付き）
# ============================================================
def safe_get(session, url, max_retries=MAX_RETRIES):
    """タイムアウトとリトライ付きのGETリクエスト"""
    for attempt in range(1, max_retries + 1):
        try:
            res = session.get(url, timeout=REQUEST_TIMEOUT)
            res.encoding = "EUC-JP"
            return res
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"  [RETRY {attempt}/{max_retries}] {type(e).__name__}: {url}")
            if attempt < max_retries:
                wait_time = RETRY_WAIT * attempt
                print(f"  {wait_time}秒後にリトライします...")
                sleep(wait_time)
            else:
                print(f"  [ERROR] {max_retries}回リトライしましたが失敗しました: {url}")
                return None
        except Exception as e:
            print(f"  [ERROR] 予期しないエラー: {type(e).__name__}: {e}")
            return None
    return None


# ============================================================
# レースIDスクレイピング関数（改善版）
# ============================================================
def race_id_scrape(session, kaisai_nengetu):
    """指定された年月のレースIDを取得する（タイムアウト・リトライ対応版）"""

    ul1 = "https://keirin.kdreams.jp/gamboo/schedule/search/" + kaisai_nengetu
    res = safe_get(session, ul1)
    if res is None:
        print(f"  [ERROR] スケジュールページの取得に失敗しました: {kaisai_nengetu}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    kaisai_sches = soup.find_all("td", attrs={"class": "kaisai"})
    nen = kaisai_nengetu[0:4]
    gatu = kaisai_nengetu[5:7]
    first_day = datetime.strptime(kaisai_nengetu + "/01", "%Y/%m/%d")
    last_day = datetime.strptime(
        kaisai_nengetu + "/" + str(calendar.monthrange(int(nen), int(gatu))[1]),
        "%Y/%m/%d",
    )
    kaisai_lists = []
    kaisai_list = []

    for i in range(len(kaisai_sches)):
        if kaisai_sches[i].find("a"):
            sche_ht = str(kaisai_sches[i].find("a"))
            sche_re = re.findall(r"\d+", sche_ht)
            kaisai_lists.append(sche_re[1])
        else:
            continue

    print(f" {nen}年 {gatu}月 の解析をしました。")

    # -----------------------------------------------------------------------
    # ------------------- 開催リストから開催日数分のリストを作る ----------------
    print("開催の日数をを調べます。")

    race_id_url = (
        "https://keirin.kdreams.jp/gamboo/keirin-kaisai/race-card/result/"
    )

    for i in tqdm(range(len(kaisai_lists))):
        # ------------------------URLの生成-------------------------------
        race_id_2 = kaisai_lists[i]
        race_id_1 = kaisai_lists[i][0:10]
        race_id_3 = "01/"
        race_url = race_id_url + race_id_1 + "/" + race_id_2 + "/" + race_id_3

        # -----------------------htmlの取得-------------------------------
        race_id_res = safe_get(session, race_url)
        if race_id_res is None:
            print(f"  [SKIP] 開催日数の取得をスキップ: {race_id_2}")
            sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        race_id_soup = BeautifulSoup(race_id_res.text, "html.parser")

        # -----------------------開催日数の所得------------------------------
        day_len = len(race_id_soup.find_all("span", attrs={"class": "day"}))
        date_str = kaisai_lists[i]

        # -----------------------開催日数分のリストを追加---------------------
        for j in range(2, day_len + 1):
            date_list = list(date_str)
            date_list[11:12] = str(j)
            date_list = "".join(date_list)
            kaisai_lists.append(date_list)

        sleep(SLEEP_BETWEEN_REQUESTS)

    kaisai_lists.sort()

    # -----------------------------月始から月末までを抜き取る ----------------
    for lis in kaisai_lists:
        r_date = datetime.strptime(
            lis[2:6] + "/" + lis[6:8] + "/" + lis[8:10], "%Y/%m/%d"
        )
        ad_date = int(lis[10:12]) - 1
        r_date = r_date + timedelta(days=ad_date)

        if first_day <= r_date <= last_day:
            kaisai_list.append(lis)

    # -----------------------------------------------------------------------
    # ------------------------- すべてのレースIDを取得 ------------------------
    print("すべてのレースIDを調べます。")

    all_race_ids = []
    race_id_url = (
        "https://keirin.kdreams.jp/gamboo/keirin-kaisai/race-card/result/"
    )

    for i in tqdm(range(len(kaisai_list))):
        race_id_2 = kaisai_list[i]
        race_id_1 = kaisai_list[i][0:10]
        race_id_3 = "01/"
        race_url = race_id_url + race_id_1 + "/" + race_id_2 + "/" + race_id_3

        # -----------------------htmlの取得-------------------------------
        race_id_res = safe_get(session, race_url)
        if race_id_res is None:
            print(f"  [SKIP] レースID取得をスキップ: {race_id_2}")
            sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        race_id_soup = BeautifulSoup(race_id_res.text, "html.parser")

        sleep(SLEEP_BETWEEN_REQUESTS)

        # ------------------------レース数の取得----------------------------
        try:
            nav = race_id_soup.find(
                "div", attrs={"class": "kaisai_race_data_nav"}
            )
            if nav and nav.find_all("li"):
                race_nums = re.findall(r"\d+", nav.find_all("li")[-1].text)
                if race_nums:
                    race_max_num = int(race_nums[0])
                    # ------------------------すべてのレースIDの取得--------------------
                    for j in range(1, race_max_num + 1, 1):
                        all_race_url_num = str(j).zfill(2)
                        all_race_id = str(race_id_2 + all_race_url_num)
                        all_race_ids.append(all_race_id)
                else:
                    continue

        except AttributeError:
            print(f"  {race_id_2} 開催が中止されています")

        except IndexError:
            print("  レースが中止されています")

    print(
        f" {nen}年 {gatu}月 の全 {len(all_race_ids)} レースのIDを取得しました。"
    )

    return all_race_ids


# ============================================================
# メイン処理
# ============================================================
def main():
    print("=" * 60)
    print("競輪レースIDスクレイピング (改善版)")
    print("=" * 60)

    # 1. 既存バックアップの読み込み
    total_race_ids = load_existing_ids()
    completed_months = get_completed_months(total_race_ids)

    if completed_months:
        print(f"[INFO] 取得済み月数: {len(completed_months)} ヶ月")
        sorted_months = sorted(completed_months)
        print(f"[INFO] 取得済み範囲: {sorted_months[0]} ～ {sorted_months[-1]}")

    # 2. セッション作成
    session = create_session()

    # 3. 年月ループ
    new_ids_count = 0
    for year in range(START_YEAR, END_YEAR):
        for month in range(1, 13):
            kaisai_nengetu = f"{year}/{month:02d}"

            # 未来の月はスキップ
            now = datetime.now()
            target_date = datetime(year, month, 1)
            if target_date > now:
                print(f"\n[SKIP] {kaisai_nengetu} は未来の月のためスキップ")
                continue

            # 既に取得済みの月はスキップ
            month_key = f"{year}/{month:02d}"
            if month_key in completed_months:
                print(f"\n[SKIP] {kaisai_nengetu} は取得済みのためスキップ")
                continue

            # スクレイピング実行
            print(f"\n=== {kaisai_nengetu} の取得を開始 ===")

            try:
                monthly_ids = race_id_scrape(session, kaisai_nengetu)
            except Exception as e:
                print(f"[ERROR] {kaisai_nengetu} の取得中に予期しないエラー: {type(e).__name__}: {e}")
                print("[INFO] この月をスキップして次の月に進みます。")
                # エラー発生時も現在までのデータを保存
                save_backup(total_race_ids)
                sleep(SLEEP_BETWEEN_MONTHS)
                continue

            if monthly_ids:
                total_race_ids.extend(monthly_ids)
                new_ids_count += len(monthly_ids)

                # 月ごとにバックアップ保存
                save_backup(total_race_ids)
                print(f"[INFO] 累計レースID数: {len(total_race_ids)} (今回新規: +{len(monthly_ids)})")

            sleep(SLEEP_BETWEEN_MONTHS)

    # 4. 完了
    print("\n" + "=" * 60)
    print(f"全期間の取得が完了しました。")
    print(f"合計レースID数: {len(total_race_ids)}")
    print(f"今回新規取得数: {new_ids_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
