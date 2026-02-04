"""
ボートレース予測AI メインアプリケーション
CLI インターフェース
"""
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import config, RACECOURSE_CODES, INDEX_TO_TRIFECTA

app = typer.Typer(
    name="boatrace-ai",
    help="🚤 ボートレース予測AI システム",
    add_completion=False
)

console = Console()


@app.command()
def download(
    start_year: int = typer.Option(2018, "--start", "-s", help="開始年"),
    end_year: int = typer.Option(2025, "--end", "-e", help="終了年"),
    data_type: str = typer.Option("all", "--type", "-t", help="データ種別 (lzh/scrape/all)")
):
    """データをダウンロード"""
    console.print(Panel.fit(
        f"📥 データダウンロード: {start_year}年 - {end_year}年",
        title="ボートレース予測AI"
    ))
    
    async def _download():
        if data_type in ["lzh", "all"]:
            from src.data_collection import LZHDownloader
            
            console.print("\n[bold blue]LZHファイルをダウンロード中...[/bold blue]")
            downloader = LZHDownloader()
            await downloader.download_all_data(start_year, end_year)
        
        if data_type in ["scrape", "all"]:
            from src.data_collection import CurlCffiScraper
            
            console.print("\n[bold blue]オッズ・直前情報をスクレイピング中...[/bold blue]")
            console.print("[yellow]注意: スクレイピングは時間がかかります（数週間〜数ヶ月）[/yellow]")
            
            scraper = CurlCffiScraper()
            start_date = datetime(start_year, 1, 1)
            end_date = min(datetime(end_year, 12, 31), datetime.now() - timedelta(days=1))
            scraper.scrape_date_range(start_date, end_date)
    
    asyncio.run(_download())
    console.print("\n[bold green]✅ ダウンロード完了[/bold green]")


@app.command()
def etl():
    """ETLパイプラインを実行"""
    console.print(Panel.fit("🔄 ETLパイプライン実行", title="ボートレース予測AI"))
    
    from src.etl import ETLPipeline
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("ETL処理中...", total=None)
        
        pipeline = ETLPipeline()
        pipeline.run_full_pipeline()
        
        progress.update(task, description="完了!")
    
    console.print("\n[bold green]✅ ETL完了[/bold green]")
    
    # バリデーション結果表示
    results = pipeline.validate_data()
    
    table = Table(title="データ統計")
    table.add_column("テーブル", style="cyan")
    table.add_column("レコード数", style="green")
    
    for table_name, count in results.get('record_counts', {}).items():
        table.add_row(table_name, f"{count:,}")
    
    console.print(table)


@app.command()
def train(
    optimize: bool = typer.Option(False, "--optimize", "-o", help="ハイパーパラメータ最適化"),
    start_date: str = typer.Option("20180101", "--start", "-s", help="学習開始日"),
    end_date: str = typer.Option("20241231", "--end", "-e", help="学習終了日")
):
    """モデルを学習"""
    console.print(Panel.fit("🧠 モデル学習", title="ボートレース予測AI"))
    
    from src.models import train_probability_model, train_ev_model
    
    console.print(f"\n[bold]期間: {start_date} - {end_date}[/bold]")
    console.print(f"[bold]最適化: {'有効' if optimize else '無効'}[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # 確率モデル
        task1 = progress.add_task("確率推定モデル学習中...", total=None)
        prob_model = train_probability_model(start_date, end_date, optimize)
        progress.update(task1, description="確率モデル完了!")
        
        # 期待値モデル
        task2 = progress.add_task("期待値予測モデル学習中...", total=None)
        ev_model = train_ev_model(start_date, end_date)
        progress.update(task2, description="期待値モデル完了!")
    
    console.print("\n[bold green]✅ モデル学習完了[/bold green]")


@app.command()
def predict(
    date: str = typer.Argument(..., help="予測日（YYYYMMDD形式）"),
    place: str = typer.Argument(..., help="会場コードまたは会場名"),
    race: int = typer.Argument(..., help="レース番号"),
    bankroll: int = typer.Option(100000, "--bankroll", "-b", help="資金"),
    top_n: int = typer.Option(10, "--top", "-n", help="表示する上位件数")
):
    """レース予測を実行"""
    # 会場名から会場コードへの変換
    if place in RACECOURSE_CODES.values():
        for code, name in RACECOURSE_CODES.items():
            if name == place:
                place = code
                break
    
    place_name = RACECOURSE_CODES.get(place, place)
    
    console.print(Panel.fit(
        f"🎯 予測: {date} {place_name} {race}R",
        title="ボートレース予測AI"
    ))
    
    from src.features import FeatureEngineer
    from src.models import ProbabilityModel, CombinedPredictor
    from src.betting import HorseKelly
    
    # 特徴量生成
    fe = FeatureEngineer()
    features = fe.create_race_features(date, place, race)
    
    if not features:
        console.print("[red]レースデータが見つかりません[/red]")
        return
    
    X, _ = fe.features_to_array(features)
    X = X.reshape(1, -1)
    
    # モデル読み込みと予測
    try:
        prob_model = ProbabilityModel()
        prob_model.load()
    except:
        console.print("[yellow]モデルが見つかりません。学習を実行してください。[/yellow]")
        console.print("[yellow]ダミー予測を表示します。[/yellow]")
        prob_model = None
    
    # 予測
    if prob_model:
        proba = prob_model.predict_proba(X)[0]
    else:
        # ダミー: インが有利な仮定
        proba = []
        for idx in range(120):
            combo = INDEX_TO_TRIFECTA[idx]
            base_prob = 1.0 / (combo[0] * 2)
            proba.append(base_prob)
        proba = [p / sum(proba) for p in proba]
    
    # 結果表示
    table = Table(title=f"予測結果 Top {top_n}")
    table.add_column("順位", style="cyan", justify="right")
    table.add_column("買い目", style="green")
    table.add_column("確率", justify="right")
    table.add_column("想定オッズ", justify="right")
    
    top_indices = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:top_n]
    
    for rank, idx in enumerate(top_indices, 1):
        combo = INDEX_TO_TRIFECTA[idx]
        prob = proba[idx]
        implied_odds = 1 / prob if prob > 0 else 0
        
        table.add_row(
            str(rank),
            f"{combo[0]}-{combo[1]}-{combo[2]}",
            f"{prob:.2%}",
            f"{implied_odds:.1f}"
        )
    
    console.print(table)
    
    # 購入推奨
    console.print(f"\n[bold]資金: ¥{bankroll:,}[/bold]")
    
    kelly = HorseKelly()
    predictions = [
        {
            'combination': INDEX_TO_TRIFECTA[idx],
            'probability': proba[idx],
            'prelim_odds': 1 / proba[idx] if proba[idx] > 0 else 0
        }
        for idx in range(120)
    ]
    
    plan = kelly.calculate_bets(predictions, bankroll, min_ev=1.0)
    
    if plan.bets:
        console.print("\n[bold blue]購入推奨:[/bold blue]")
        
        bet_table = Table()
        bet_table.add_column("買い目", style="green")
        bet_table.add_column("金額", justify="right")
        bet_table.add_column("期待値", justify="right")
        
        for bet in plan.bets[:10]:
            bet_table.add_row(
                f"{bet.combination[0]}-{bet.combination[1]}-{bet.combination[2]}",
                f"¥{bet.bet_amount:,}",
                f"{bet.expected_value:.2f}"
            )
        
        console.print(bet_table)
        console.print(f"\n合計: ¥{plan.total_bet:,} / 余剰: ¥{plan.remaining:,}")
    else:
        console.print("\n[yellow]購入推奨なし（期待値がプラスの買い目がありません）[/yellow]")


@app.command()
def dashboard():
    """Streamlit ダッシュボードを起動"""
    import subprocess
    
    console.print(Panel.fit("📊 ダッシュボード起動", title="ボートレース予測AI"))
    
    dashboard_path = Path(__file__).parent / "src" / "visualization" / "dashboard.py"
    
    console.print(f"\n[bold]ブラウザで http://localhost:8501 にアクセスしてください[/bold]")
    console.print("[dim]終了するには Ctrl+C を押してください[/dim]\n")
    
    subprocess.run(["streamlit", "run", str(dashboard_path)])


@app.command()
def assistant():
    """LLMアシスタントを起動"""
    console.print(Panel.fit("🤖 分析アシスタント", title="ボートレース予測AI"))
    
    from src.assistant.llm_assistant import interactive_session
    
    console.print("\n[bold]Ollama が起動している必要があります[/bold]")
    console.print("[dim]終了するには 'quit' と入力してください[/dim]\n")
    
    asyncio.run(interactive_session())


@app.command()
def discord_bot():
    """Discord Botを起動"""
    console.print(Panel.fit("🤖 Discord Bot 起動", title="ボートレース予測AI"))
    
    from src.assistant.discord_bot import run_bot
    
    console.print("\n[bold]環境変数 DISCORD_BOT_TOKEN を設定してください[/bold]")
    console.print("[dim]終了するには Ctrl+C を押してください[/dim]\n")
    
    asyncio.run(run_bot())


@app.command()
def backtest(
    start_date: str = typer.Option("20240101", "--start", "-s", help="開始日"),
    end_date: str = typer.Option("20241231", "--end", "-e", help="終了日"),
    bankroll: int = typer.Option(100000, "--bankroll", "-b", help="初期資金"),
    kelly_fraction: float = typer.Option(0.25, "--kelly", "-k", help="ケリー係数")
):
    """バックテストを実行"""
    console.print(Panel.fit(
        f"📈 バックテスト: {start_date} - {end_date}",
        title="ボートレース予測AI"
    ))
    
    from src.features import FeatureEngineer
    from src.models import ProbabilityModel
    from src.betting import HorseKelly, BettingSimulator
    from src.etl import db
    
    console.print(f"\n[bold]初期資金: ¥{bankroll:,}[/bold]")
    console.print(f"[bold]ケリー係数: {kelly_fraction}[/bold]\n")
    
    # モデル読み込み
    try:
        prob_model = ProbabilityModel()
        prob_model.load()
    except:
        console.print("[red]モデルが見つかりません。先に学習を実行してください。[/red]")
        return
    
    # レース一覧を取得
    races = db.query(f"""
        SELECT DISTINCT race_date, place_code, race_number
        FROM race_result
        WHERE race_date >= '{start_date}'
        AND race_date <= '{end_date}'
        ORDER BY race_date, place_code, race_number
    """)
    
    console.print(f"[bold]対象レース数: {len(races)}[/bold]\n")
    
    # シミュレーター
    simulator = BettingSimulator(bankroll)
    kelly = HorseKelly(kelly_fraction=kelly_fraction)
    fe = FeatureEngineer()
    
    with Progress(console=console) as progress:
        task = progress.add_task("バックテスト中...", total=len(races))
        
        for _, row in races.iterrows():
            # 特徴量生成
            features = fe.create_race_features(
                row['race_date'],
                row['place_code'],
                row['race_number']
            )
            
            if not features or not features.trifecta_result:
                progress.update(task, advance=1)
                continue
            
            X, _ = fe.features_to_array(features)
            X = X.reshape(1, -1)
            
            # 予測
            proba = prob_model.predict_proba(X)[0]
            
            predictions = [
                {
                    'combination': INDEX_TO_TRIFECTA[idx],
                    'probability': proba[idx],
                    'prelim_odds': 1 / proba[idx] if proba[idx] > 0 else 0
                }
                for idx in range(120)
            ]
            
            # 購入計画
            plan = kelly.calculate_bets(predictions, simulator.bankroll, min_ev=1.0)
            
            # シミュレーション
            if plan.bets:
                simulator.simulate_bet(plan, features.trifecta_result)
            
            progress.update(task, advance=1)
    
    # 結果表示
    stats = simulator.get_statistics()
    
    console.print("\n[bold]===== バックテスト結果 =====[/bold]\n")
    
    results_table = Table()
    results_table.add_column("指標", style="cyan")
    results_table.add_column("値", justify="right")
    
    results_table.add_row("総レース数", f"{stats.get('total_races', 0):,}")
    results_table.add_row("総投資額", f"¥{stats.get('total_bets', 0):,}")
    results_table.add_row("総払戻額", f"¥{stats.get('total_payout', 0):,}")
    
    pnl = stats.get('total_pnl', 0)
    pnl_style = "green" if pnl >= 0 else "red"
    results_table.add_row("損益", f"[{pnl_style}]¥{pnl:,}[/{pnl_style}]")
    
    results_table.add_row("的中率", f"{stats.get('hit_rate', 0):.1%}")
    
    roi = stats.get('roi', 0)
    roi_style = "green" if roi >= 1.0 else "red"
    results_table.add_row("ROI", f"[{roi_style}]{roi:.2%}[/{roi_style}]")
    
    results_table.add_row("最終資金", f"¥{stats.get('final_bankroll', 0):,}")
    results_table.add_row("最大DD", f"{stats.get('max_drawdown', 0):.1%}")
    
    console.print(results_table)


@app.command()
def info():
    """システム情報を表示"""
    console.print(Panel.fit("ℹ️ システム情報", title="ボートレース予測AI"))
    
    from src.etl import db
    
    try:
        db.connect()
        
        stats = db.query("""
            SELECT 'bangumi' as table_name, COUNT(*) as count FROM bangumi
            UNION ALL SELECT 'race_result', COUNT(*) FROM race_result
            UNION ALL SELECT 'racer_master', COUNT(*) FROM racer_master
        """)
        
        table = Table(title="データベース統計")
        table.add_column("テーブル", style="cyan")
        table.add_column("レコード数", style="green", justify="right")
        
        for _, row in stats.iterrows():
            table.add_row(row['table_name'], f"{row['count']:,}")
        
        console.print(table)
        
        # 日付範囲
        date_range = db.query("""
            SELECT MIN(race_date) as min_date, MAX(race_date) as max_date
            FROM race_result
        """)
        
        if not date_range.empty:
            console.print(f"\n[bold]データ期間:[/bold] {date_range.iloc[0]['min_date']} - {date_range.iloc[0]['max_date']}")
        
    except Exception as e:
        console.print(f"[yellow]データベース未初期化: {e}[/yellow]")
        console.print("[dim]まず `python main.py etl` を実行してください[/dim]")
    
    # モデル状態
    from src.models import ProbabilityModel
    model = ProbabilityModel()
    
    try:
        model.load()
        console.print("\n[bold green]✅ 確率モデル: 読み込み可能[/bold green]")
    except:
        console.print("\n[bold yellow]⚠️ 確率モデル: 未学習[/bold yellow]")


def main():
    """メインエントリーポイント"""
    app()


if __name__ == "__main__":
    main()
