"""
Streamlit ダッシュボード
ボートレース予測AIの分析インターフェース
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RACECOURSE_CODES, INDEX_TO_TRIFECTA
from src.etl import db
from src.features import FeatureEngineer
from src.models import ProbabilityModel, CombinedPredictor
from src.betting import HorseKelly, BettingPlan
from src.visualization.charts import (
    ProbabilityVisualizer,
    CalibrationVisualizer,
    PerformanceVisualizer
)


# ページ設定
st.set_page_config(
    page_title="ボートレース予測AI",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# スタイル
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)


def load_model():
    """モデルを読み込み"""
    model = ProbabilityModel()
    try:
        model.load()
        return model
    except:
        st.warning("モデルが見つかりません。先にモデルを学習してください。")
        return None


def main():
    st.title("🚤 ボートレース予測AI ダッシュボード")
    
    # サイドバー
    with st.sidebar:
        st.header("設定")
        
        # 日付選択
        selected_date = st.date_input(
            "日付",
            datetime.now().date()
        )
        
        # 会場選択
        place_options = {v: k for k, v in RACECOURSE_CODES.items()}
        selected_place_name = st.selectbox(
            "会場",
            list(place_options.keys())
        )
        selected_place = place_options[selected_place_name]
        
        # レース番号
        selected_race = st.selectbox(
            "レース",
            list(range(1, 13))
        )
        
        # 資金設定
        bankroll = st.number_input(
            "資金",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        # ケリー係数
        kelly_fraction = st.slider(
            "ケリー係数",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05
        )
    
    # メインコンテンツ
    tab1, tab2, tab3, tab4 = st.tabs(["予測", "分析", "バックテスト", "統計"])
    
    with tab1:
        st.header("レース予測")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 出走表
            st.subheader("出走表")
            
            try:
                race_date = selected_date.strftime("%Y%m%d")
                bangumi_df = db.query(f"""
                    SELECT * FROM bangumi
                    WHERE race_date = '{race_date}'
                    AND place_code = '{selected_place}'
                    AND race_number = {selected_race}
                    ORDER BY waku
                """)
                
                if not bangumi_df.empty:
                    # 表示用に整形
                    display_df = bangumi_df[[
                        'waku', 'racer_name', 'racer_class', 'branch',
                        'win_rate', 'two_rate', 'motor_win_rate', 'boat_win_rate'
                    ]].copy()
                    display_df.columns = ['枠', '選手名', 'クラス', '支部', '勝率', '2連率', 'モーター勝率', 'ボート勝率']
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("出走表データがありません")
            except Exception as e:
                st.error(f"データ取得エラー: {e}")
        
        with col2:
            # 予測実行ボタン
            if st.button("予測実行", type="primary"):
                with st.spinner("予測中..."):
                    model = load_model()
                    if model:
                        try:
                            # 特徴量生成
                            fe = FeatureEngineer()
                            features = fe.create_race_features(
                                race_date,
                                selected_place,
                                selected_race
                            )
                            
                            if features:
                                X, _ = fe.features_to_array(features)
                                X = X.reshape(1, -1)
                                
                                # 予測
                                proba = model.predict_proba(X)[0]
                                
                                # 上位予測を表示
                                st.subheader("予測結果")
                                
                                top_indices = np.argsort(proba)[::-1][:10]
                                pred_data = []
                                for idx in top_indices:
                                    combo = INDEX_TO_TRIFECTA[idx]
                                    pred_data.append({
                                        '買い目': f"{combo[0]}-{combo[1]}-{combo[2]}",
                                        '確率': f"{proba[idx]:.2%}",
                                        'オッズ想定': f"{1/proba[idx]:.1f}"
                                    })
                                
                                st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
                            else:
                                st.warning("特徴量を生成できませんでした")
                        except Exception as e:
                            st.error(f"予測エラー: {e}")
        
        # サンバーストチャート
        st.subheader("確率構造")
        
        # プレースホルダー（実際のデータで更新）
        viz = ProbabilityVisualizer()
        
        # ダミーデータでサンプル表示
        sample_probs = {}
        for idx in range(120):
            combo = INDEX_TO_TRIFECTA[idx]
            # イン有利の仮定
            base_prob = 1.0 / (combo[0] * 2)
            sample_probs[combo] = base_prob / 50
        
        # 正規化
        total = sum(sample_probs.values())
        sample_probs = {k: v/total for k, v in sample_probs.items()}
        
        fig = viz.create_sunburst_chart(sample_probs, f"{selected_place_name} {selected_race}R 確率構造")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("詳細分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("選手成績")
            
            racer_id = st.text_input("選手登録番号", "")
            
            if racer_id and st.button("検索"):
                try:
                    racer_stats = db.get_racer_stats(racer_id)
                    if not racer_stats.empty:
                        st.dataframe(racer_stats.head(20))
                    else:
                        st.info("選手データが見つかりません")
                except Exception as e:
                    st.error(f"エラー: {e}")
        
        with col2:
            st.subheader("会場統計")
            
            try:
                place_stats = db.get_place_stats(selected_place)
                if not place_stats.empty:
                    fig = go.Figure(go.Bar(
                        x=place_stats['waku'],
                        y=place_stats['win_rate'],
                        marker_color=['#FFFFFF', '#000000', '#FF0000', '#0000FF', '#FFFF00', '#00FF00']
                    ))
                    fig.update_layout(title=f"{selected_place_name} 枠別勝率", xaxis_title="枠番", yaxis_title="勝率(%)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"統計エラー: {e}")
    
    with tab3:
        st.header("バックテスト")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("設定")
            
            start_date = st.date_input("開始日", datetime.now().date() - timedelta(days=30))
            end_date = st.date_input("終了日", datetime.now().date())
            
            min_ev = st.slider("最小期待値", 0.8, 1.5, 1.0, 0.05)
            
            if st.button("バックテスト実行"):
                st.info("バックテスト機能は実装中です")
        
        with col2:
            st.subheader("結果")
            
            # パフォーマンス指標
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("総レース数", "-")
            with metrics_col2:
                st.metric("的中率", "-")
            with metrics_col3:
                st.metric("ROI", "-")
            with metrics_col4:
                st.metric("最大DD", "-")
    
    with tab4:
        st.header("システム統計")
        
        try:
            # データ統計
            stats = db.query("""
                SELECT
                    'bangumi' as table_name,
                    COUNT(*) as record_count,
                    MIN(race_date) as min_date,
                    MAX(race_date) as max_date
                FROM bangumi
                UNION ALL
                SELECT
                    'race_result',
                    COUNT(*),
                    MIN(race_date),
                    MAX(race_date)
                FROM race_result
            """)
            
            st.subheader("データ量")
            st.dataframe(stats, use_container_width=True)
            
        except Exception as e:
            st.error(f"統計エラー: {e}")


if __name__ == "__main__":
    main()
