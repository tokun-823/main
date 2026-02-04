"""
可視化モジュール
サンバーストチャート、キャリブレーションカーブ等の可視化
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RACECOURSE_CODES, INDEX_TO_TRIFECTA


class ProbabilityVisualizer:
    """確率の可視化"""
    
    def __init__(self):
        # 枠番の色
        self.waku_colors = {
            1: '#FFFFFF',  # 白
            2: '#000000',  # 黒
            3: '#FF0000',  # 赤
            4: '#0000FF',  # 青
            5: '#FFFF00',  # 黄
            6: '#00FF00',  # 緑
        }
    
    def create_sunburst_chart(
        self,
        probabilities: Dict[Tuple[int, int, int], float],
        title: str = "3連単確率構造"
    ) -> go.Figure:
        """
        サンバーストチャートを作成
        内側: 1着確率
        中間: 1着固定時の2着確率
        外側: 1着・2着固定時の3着確率
        """
        
        # データ準備
        ids = []
        labels = []
        parents = []
        values = []
        colors = []
        
        # 1着の確率を集計
        first_probs = {i: 0 for i in range(1, 7)}
        for (first, _, _), prob in probabilities.items():
            first_probs[first] += prob
        
        # ルート
        ids.append("total")
        labels.append("全体")
        parents.append("")
        values.append(1.0)
        colors.append("#CCCCCC")
        
        # 1着レベル
        for first in range(1, 7):
            first_id = f"{first}"
            ids.append(first_id)
            labels.append(f"{first}号艇")
            parents.append("total")
            values.append(first_probs[first])
            colors.append(self.waku_colors[first])
        
        # 2着レベル（1着固定時）
        for first in range(1, 7):
            second_probs = {i: 0 for i in range(1, 7) if i != first}
            for (f, second, _), prob in probabilities.items():
                if f == first:
                    second_probs[second] += prob
            
            for second in range(1, 7):
                if second == first:
                    continue
                second_id = f"{first}-{second}"
                ids.append(second_id)
                labels.append(f"{second}")
                parents.append(f"{first}")
                values.append(second_probs[second])
                colors.append(self.waku_colors[second])
        
        # 3着レベル（1着・2着固定時）
        for (first, second, third), prob in probabilities.items():
            third_id = f"{first}-{second}-{third}"
            ids.append(third_id)
            labels.append(f"{third}")
            parents.append(f"{first}-{second}")
            values.append(prob)
            colors.append(self.waku_colors[third])
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors, line=dict(width=1)),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>確率: %{value:.2%}<extra></extra>',
        ))
        
        fig.update_layout(
            title=title,
            width=800,
            height=800,
        )
        
        return fig
    
    def create_probability_heatmap(
        self,
        probabilities: Dict[Tuple[int, int, int], float],
        first_fixed: int = None,
        title: str = "確率ヒートマップ"
    ) -> go.Figure:
        """
        確率のヒートマップを作成
        first_fixed が指定された場合は1着固定の2着-3着確率
        """
        
        if first_fixed:
            # 1着固定の場合
            matrix = np.zeros((6, 6))
            for (first, second, third), prob in probabilities.items():
                if first == first_fixed:
                    matrix[second - 1][third - 1] = prob
            
            x_labels = [str(i) for i in range(1, 7)]
            y_labels = [str(i) for i in range(1, 7)]
            title = f"{title} ({first_fixed}号艇1着固定)"
        else:
            # 1着-2着の確率
            matrix = np.zeros((6, 6))
            for (first, second, _), prob in probabilities.items():
                matrix[first - 1][second - 1] += prob
            
            x_labels = [str(i) for i in range(1, 7)]
            y_labels = [str(i) for i in range(1, 7)]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            hovertemplate='%{y}着-%{x}着: %{z:.2%}<extra></extra>',
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="2着" if first_fixed else "2着",
            yaxis_title="3着" if first_fixed else "1着",
            width=600,
            height=600,
        )
        
        return fig
    
    def create_ev_comparison_chart(
        self,
        predictions: List[Dict],
        top_n: int = 20,
        title: str = "期待値比較"
    ) -> go.Figure:
        """期待値の比較チャート"""
        
        # 上位N件を抽出
        sorted_preds = sorted(predictions, key=lambda x: x.get('ev_mid', x.get('expected_value', 0)), reverse=True)[:top_n]
        
        combinations = [f"{p['combination'][0]}-{p['combination'][1]}-{p['combination'][2]}" for p in sorted_preds]
        ev_mid = [p.get('ev_mid', p.get('expected_value', 0)) for p in sorted_preds]
        ev_low = [p.get('ev_low', ev_mid[i]) for i, p in enumerate(sorted_preds)]
        ev_high = [p.get('ev_high', ev_mid[i]) for i, p in enumerate(sorted_preds)]
        
        fig = go.Figure()
        
        # エラーバー付きの期待値
        fig.add_trace(go.Bar(
            x=combinations,
            y=ev_mid,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ev_high[i] - ev_mid[i] for i in range(len(ev_mid))],
                arrayminus=[ev_mid[i] - ev_low[i] for i in range(len(ev_mid))]
            ),
            marker_color=['#4CAF50' if ev >= 1.0 else '#F44336' for ev in ev_mid],
            name='期待値'
        ))
        
        # 期待値=1のライン
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="EV=1.0")
        
        fig.update_layout(
            title=title,
            xaxis_title="買い目",
            yaxis_title="期待値",
            width=1000,
            height=500,
        )
        
        return fig


class CalibrationVisualizer:
    """キャリブレーションの可視化"""
    
    def create_calibration_curve(
        self,
        prob_true: np.ndarray,
        prob_pred: np.ndarray,
        title: str = "キャリブレーションカーブ"
    ) -> go.Figure:
        """キャリブレーションカーブを作成"""
        
        fig = go.Figure()
        
        # 理想線（対角線）
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='理想'
        ))
        
        # 実際のキャリブレーション
        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name='モデル',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="予測確率",
            yaxis_title="実際の確率",
            width=600,
            height=600,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )
        
        return fig


class PerformanceVisualizer:
    """パフォーマンスの可視化"""
    
    def create_bankroll_chart(
        self,
        history: List[Dict],
        initial_bankroll: int,
        title: str = "資金推移"
    ) -> go.Figure:
        """資金推移チャートを作成"""
        
        bankrolls = [initial_bankroll] + [h['bankroll'] for h in history]
        x = list(range(len(bankrolls)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=bankrolls,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#2196F3'),
            name='資金'
        ))
        
        # 初期資金のライン
        fig.add_hline(y=initial_bankroll, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=title,
            xaxis_title="レース数",
            yaxis_title="資金",
            width=1000,
            height=400,
        )
        
        return fig
    
    def create_pnl_distribution(
        self,
        history: List[Dict],
        title: str = "損益分布"
    ) -> go.Figure:
        """損益の分布を作成"""
        
        pnls = [h['pnl'] for h in history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnls,
            nbinsx=50,
            marker_color='#4CAF50',
            name='損益'
        ))
        
        # 平均線
        mean_pnl = np.mean(pnls)
        fig.add_vline(x=mean_pnl, line_dash="dash", line_color="red", 
                     annotation_text=f"平均: {mean_pnl:.0f}")
        
        fig.update_layout(
            title=title,
            xaxis_title="損益",
            yaxis_title="頻度",
            width=800,
            height=400,
        )
        
        return fig
    
    def create_roi_by_place(
        self,
        results_df: pd.DataFrame,
        title: str = "会場別ROI"
    ) -> go.Figure:
        """会場別ROIチャート"""
        
        # 会場別集計
        place_stats = results_df.groupby('place_code').agg({
            'total_bet': 'sum',
            'payout': 'sum'
        }).reset_index()
        
        place_stats['roi'] = place_stats['payout'] / place_stats['total_bet']
        place_stats['place_name'] = place_stats['place_code'].map(RACECOURSE_CODES)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=place_stats['place_name'],
            y=place_stats['roi'],
            marker_color=['#4CAF50' if roi >= 1.0 else '#F44336' for roi in place_stats['roi']],
            name='ROI'
        ))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="ROI=1.0")
        
        fig.update_layout(
            title=title,
            xaxis_title="会場",
            yaxis_title="ROI",
            width=1000,
            height=400,
        )
        
        return fig


class FeatureImportanceVisualizer:
    """特徴量重要度の可視化"""
    
    def create_importance_chart(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 30,
        title: str = "特徴量重要度"
    ) -> go.Figure:
        """特徴量重要度チャート"""
        
        df = importance_df.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['importance'],
            y=df['feature'],
            orientation='h',
            marker_color='#2196F3'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="重要度",
            yaxis_title="特徴量",
            width=800,
            height=max(400, top_n * 20),
            yaxis=dict(autorange="reversed")
        )
        
        return fig


def save_figure(fig: go.Figure, path: Path, format: str = 'html'):
    """図を保存"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'html':
        fig.write_html(str(path))
    elif format == 'png':
        fig.write_image(str(path))
    elif format == 'json':
        fig.write_json(str(path))
    
    logger.info(f"Figure saved to {path}")
