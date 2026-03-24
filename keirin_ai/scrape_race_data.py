import os
import re
from time import sleep
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def scrape_and_save_races(csv_file_path="scraped_data/csv/total_race_ids_backup.csv"):
    # 1. 保存先ディレクトリの作成
    base_dir = "scraped_data"
    info_dir = os.path.join(base_dir, "info_table")
    entry_dir = os.path.join(base_dir, "entry_table")
    return_dir = os.path.join(base_dir, "return_table")

    for directory in [info_dir, entry_dir, return_dir]:
        os.makedirs(directory, exist_ok=True)

    # 2. レースIDの読み込み
    try:
        # all_race_ids = pd.read_pickle(pkl_file_path)
        all_race_ids = pd.read_csv(csv_file_path, dtype=str)
        print("all_race_idsの長さ", len(all_race_ids))
        # ※all_race_idsがDataFrameの場合は、適切な列を指定してリスト化してください
        # 例: race_ids = all_race_ids['ID'].tolist()
        if isinstance(all_race_ids, pd.DataFrame):
            race_ids = all_race_ids.iloc[:, 0].tolist()
        else:
            race_ids = list(all_race_ids)
    except Exception as e:
        print(f"IDリストの読み込みに失敗しました: {e}")
        return

    # 3. スクレイピングループ
    for race_id in tqdm(race_ids):
        # 保存先パスの設定
        info_path = os.path.join(info_dir, f"{race_id}.pkl")
        entry_path = os.path.join(entry_dir, f"{race_id}.pkl")
        return_path = os.path.join(return_dir, f"{race_id}.pkl")

        # 既に3つのファイルが揃っている場合はスキップ（レジューム機能）
        if (
            os.path.exists(info_path)
            and os.path.exists(entry_path)
            and os.path.exists(return_path)
        ):
            continue

        try:
            # URL作成とHTML取得
            ur_1 = race_id[:10]
            ur_2 = race_id[:-2]
            ur_3 = race_id[-2:]
            url = f"https://keirin.kdreams.jp/gamboo/keirin-kaisai/race-card/result/{ur_1}/{ur_2}/{ur_3}"
            print("url", url)
            html = requests.get(url)
            html.encoding = ["EUC-JP"]
            soup = BeautifulSoup(html.text, "html.parser")
            p_htm = pd.read_html(url)

        except Exception as e:
            print(f"{race_id}: 読み込みエラー ({e})")
            continue

        sleep(1)

        try:
            # テーブルの取得分け
            if len(p_htm) == 8:
                pd_racecard = pd.DataFrame(p_htm[0])
                pd_raceresult = pd.DataFrame(p_htm[6])
                pd_harai = pd.DataFrame(p_htm[7])
            else:
                pd_racecard = pd.DataFrame(p_htm[0])
                pd_raceresult = pd.DataFrame(p_htm[3])
                pd_harai = pd.DataFrame(p_htm[4])

            race_grs = ["ＧＰ", "Ｇ１", "Ｇ２", "Ｇ３", "Ｆ１", "Ｆ２"]
            setcol = [
                "予 想",
                "好 気 合",
                "総 評",
                "枠 番",
                "車 番",
                "選手名 府県/年齢/期別",
                "級 班",
                "脚 質",
                "ギヤ 倍数",
                "競走得点",
                "S",
                "B",
                "逃",
                "捲",
                "差",
                "マ",
                "1 着",
                "2 着",
                "3 着",
                "着 外",
                "勝 率",
                "2連 対率",
                "3連 対率",
                "level_1",
                "level_2",
            ]
            pd_racecard.columns = setcol

            pd_racecard = pd_racecard.drop(["好 気 合", "level_1", "level_2"], axis=1)
            pd_raceresult = pd_raceresult.sort_values("車 番")
            pd_raceresult = pd_raceresult.drop(
                ["着差", "上り", "決ま り手", "S ／ B", "勝敗因"], axis=1
            )

            pd_racecard = pd.merge(pd_racecard, pd_raceresult, how="inner")
        except:
            pass

        # -----------------------選手名 府県/年齢/期別を分ける ----------------
        try:
            age = []
            gr_class = []
            sensyu = []
            for name in pd_racecard["選手名 府県/年齢/期別"]:
                if "（欠車）" in name:
                    name = name.replace("（欠車）", "")
                age.append(int(name.split("/")[1]))
                gr_class.append(int(name.split("/")[2]))
                sensyu.append(name.split(" ")[0] + " " + name.split(" ")[1])

            pd_racecard["選手名"] = sensyu
            pd_racecard["年齢"] = age
            pd_racecard["期別"] = gr_class
            pd_racecard = pd_racecard.drop(["選手名 府県/年齢/期別"], axis=1)

            pd_racecard = pd_racecard.loc[
                :,
                [
                    "予 想",
                    "着 順",
                    "総 評",
                    "枠 番",
                    "車 番",
                    "選手名",
                    "競走得点",
                    "年齢",
                    "期別",
                    "級 班",
                    "脚 質",
                    "ギヤ 倍数",
                    "S",
                    "B",
                    "逃",
                    "捲",
                    "差",
                    "マ",
                    "1 着",
                    "2 着",
                    "3 着",
                    "着 外",
                    "勝 率",
                    "2連 対率",
                    "3連 対率",
                ],
            ]
        except:
            pass

        # -----------------------ライン構成の読み取り ----------------
        try:
            line_position = soup.find(
                "div", attrs={"class": "line_position_inner"}
            ).text
            p = line_position.replace("\n", "P")
            line_n = ["mmmmmm"] * 18
            n = 0
            for i in range(len(p)):
                po_1 = p[i - 2]
                po_2 = p[i - 1]
                if re.findall(r"\d+", p[i]):
                    n += 1
                    line_n[n] = str(p[i])
                    if po_1 == po_2:
                        line_n[n] = str("mmmmmm")
                        n += 1
                        line_n[n] = str(p[i])
            line_n = "".join(line_n) + "mmmmmm"
        except:
            pass

        # ----------------------ライン構成の分析 ------------------
        pd_racecard2 = pd_racecard.copy()
        line_k = []
        kousei2 = ""
        try:
            make_line = []
            syusso_n = len(pd_racecard)
            for j in range(1, syusso_n + 1, 1):
                for k in range(len(line_n)):
                    check_str = line_n[k]
                    if str(j) == check_str:
                        if line_n[k - 1] == "m":
                            if line_n[k + 1] == "m":
                                make_line.append([j, line_n[k], 0])  # 単騎
                            else:
                                make_line.append([j, line_n[k], 1])  # 先頭
                        else:
                            if line_n[k - 2] == "m":
                                make_line.append([j, line_n[k - 1], 2])  # 番手
                            else:
                                if line_n[k - 3] == "m":
                                    make_line.append([j, line_n[k - 2], 3])  # ３番手
                                else:
                                    if line_n[k - 4] == "m":
                                        make_line.append(
                                            [j, line_n[k - 3], 4]
                                        )  # ４番手
                                    else:
                                        if line_n[-5] == "m":
                                            make_line.append(
                                                [j, line_n[k - 4], 5]
                                            )  # ５番手

            line_k = re.findall(r"\d+", line_n)
            kousei = str()
            kousei2 = str()
            for i in range(len(line_k)):
                kousei += str(len(line_k[i]))
            kousei = sorted(kousei, reverse=True)
            for i in range(len(kousei)):
                kousei2 += kousei[i] + "-"
            kousei2 = kousei2[:-1]

            # ----------------------ライン構成を追加 ----------------------------
            pd_make_line = pd.DataFrame(make_line)
            pd_make_line.columns = ["車 番", "ライン", "番手"]
            pd_racecard2 = pd.merge(pd_racecard, pd_make_line, how="inner")

            # ---------------------競走得点順位を追加 -----------------------
            toku_jyun = [i for i in range(1, len(pd_make_line) + 1)]
            pd_racecard2 = pd_racecard2.sort_values("競走得点", ascending=False)
            pd_racecard2["得点順位"] = toku_jyun
            pd_racecard2 = pd_racecard2.sort_values("車 番")
        except:
            pass

        # ----------------------レースインフォメーション ---------------------
        try:
            race_header = (
                soup.find("div", attrs={"class": "race_header"}).find("span").text
            )
            race_info_header = re.findall(r"\w+", race_header)
            race_info_day = (
                race_id[2:6]
                + "-"
                + race_info_header[2][:2]
                + "-"
                + race_info_header[2][3:5]
            )
            race_title = (
                soup.find("div", attrs={"class": "race_title_header"}).find("span").text
            )
            race_stadium = soup.find("span", attrs={"class": "velodrome"}).text
            race_name = soup.find("span", attrs={"class": "race"}).text
            race_name = "".join(race_name.split("\u3000"))
            race_gr = [
                t
                for t in race_grs
                if t in soup.find("h1", attrs={"class": "section_title"}).text
            ][0]
            race_start_time = (
                soup.find("dl", attrs={"class": "time"}).find_all("dd")[0].text
            )
            race_condition = soup.find("p", attrs={"class": "weather_info"}).find_all(
                "span"
            )
            race_condition_weather = race_condition[0].text[-1]
            race_condition_wind = race_condition[1].text[2:-1]

            # 1行のDataFrameとして作成
            pd_info = pd.DataFrame(
                {
                    "レースタイトル": [race_title],
                    "競輪場": [race_stadium],
                    "レース名": [race_name],
                    "グレード": [race_gr],
                    "開始時間": [race_start_time],
                    "天気": [race_condition_weather],
                    "風速": [race_condition_wind],
                    "レース番号": [race_info_header[0]],
                    "開催日": [race_info_day],
                    "開催番号": [race_info_header[3]],
                    "車立": [len(pd_racecard)],
                    "ライン数": [len(line_k)],
                    "ライン構成": [kousei2],
                },
                index=[race_id],
            )

            pd_racecard2.index = [race_id] * len(pd_racecard2)
            pd_harai.index = [race_id] * len(pd_harai)

            # 各テーブルをIDごとのPickleファイルとして保存
            pd_info.to_pickle(info_path)
            pd_racecard2.to_pickle(entry_path)
            pd_harai.to_pickle(return_path)

        except Exception as e:
            # print(f'{race_id}: 情報抽出・保存エラー ({e})')
            pass


# 実行
if __name__ == "__main__":
    scrape_and_save_races()
