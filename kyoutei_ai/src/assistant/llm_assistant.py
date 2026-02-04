"""
LLMアシスタント
Ollama + Function Callingによる分析支援チャットボット
"""
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import httpx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, RACECOURSE_CODES
from src.etl import db


@dataclass
class FunctionCall:
    """関数呼び出し定義"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class OllamaClient:
    """Ollama APIクライアント"""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or config.llm.ollama_base_url
        self.model = model or config.llm.model_name
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(
        self,
        prompt: str,
        system: str = None,
        context: List[int] = None,
        options: Dict = None
    ) -> str:
        """テキスト生成"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system:
            payload["system"] = system
        if context:
            payload["context"] = context
        if options:
            payload["options"] = options
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"エラーが発生しました: {e}"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        options: Dict = None
    ) -> str:
        """チャット形式の生成"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return f"エラーが発生しました: {e}"
    
    async def close(self):
        """クライアントを閉じる"""
        await self.client.aclose()


class BoatRaceFunctions:
    """ボートレース分析用関数群"""
    
    def __init__(self):
        self.db = db
    
    def get_available_functions(self) -> List[FunctionCall]:
        """利用可能な関数のリスト"""
        return [
            FunctionCall(
                name="get_racer_stats",
                description="選手の過去成績を取得します",
                parameters={
                    "type": "object",
                    "properties": {
                        "racer_id": {
                            "type": "string",
                            "description": "選手の登録番号"
                        }
                    },
                    "required": ["racer_id"]
                },
                function=self.get_racer_stats
            ),
            FunctionCall(
                name="get_place_stats",
                description="会場の統計情報を取得します",
                parameters={
                    "type": "object",
                    "properties": {
                        "place_code": {
                            "type": "string",
                            "description": "会場コード（01-24）または会場名"
                        }
                    },
                    "required": ["place_code"]
                },
                function=self.get_place_stats
            ),
            FunctionCall(
                name="get_race_result",
                description="レース結果を取得します",
                parameters={
                    "type": "object",
                    "properties": {
                        "race_date": {
                            "type": "string",
                            "description": "レース日（YYYYMMDD形式）"
                        },
                        "place_code": {
                            "type": "string",
                            "description": "会場コード"
                        },
                        "race_number": {
                            "type": "integer",
                            "description": "レース番号（1-12）"
                        }
                    },
                    "required": ["race_date", "place_code", "race_number"]
                },
                function=self.get_race_result
            ),
            FunctionCall(
                name="get_start_timing_stats",
                description="選手のスタートタイミング統計を取得します",
                parameters={
                    "type": "object",
                    "properties": {
                        "racer_id": {
                            "type": "string",
                            "description": "選手の登録番号"
                        }
                    },
                    "required": ["racer_id"]
                },
                function=self.get_start_timing_stats
            ),
            FunctionCall(
                name="analyze_course_performance",
                description="進入コース別の成績を分析します",
                parameters={
                    "type": "object",
                    "properties": {
                        "racer_id": {
                            "type": "string",
                            "description": "選手の登録番号"
                        }
                    },
                    "required": ["racer_id"]
                },
                function=self.analyze_course_performance
            ),
            FunctionCall(
                name="get_winning_patterns",
                description="会場の決まり手パターンを取得します",
                parameters={
                    "type": "object",
                    "properties": {
                        "place_code": {
                            "type": "string",
                            "description": "会場コード"
                        }
                    },
                    "required": ["place_code"]
                },
                function=self.get_winning_patterns
            ),
        ]
    
    def get_racer_stats(self, racer_id: str) -> str:
        """選手の過去成績を取得"""
        try:
            df = self.db.get_racer_stats(racer_id)
            if df.empty:
                return f"選手ID {racer_id} のデータが見つかりません"
            
            # 集計
            total = len(df)
            wins = sum(df['is_first'])
            top2 = sum(df['is_top2'])
            avg_st = df['start_timing'].mean()
            
            result = f"""
選手ID: {racer_id}
選手名: {df.iloc[0]['racer_name']}
直近{total}走の成績:
- 勝率: {wins/total*100:.1f}%
- 2連対率: {top2/total*100:.1f}%
- 平均ST: {avg_st:.2f}
"""
            return result
            
        except Exception as e:
            return f"エラー: {e}"
    
    def get_place_stats(self, place_code: str) -> str:
        """会場統計を取得"""
        try:
            # 会場名から会場コードへの変換
            if place_code in RACECOURSE_CODES.values():
                for code, name in RACECOURSE_CODES.items():
                    if name == place_code:
                        place_code = code
                        break
            
            df = self.db.get_place_stats(place_code)
            if df.empty:
                return f"会場コード {place_code} のデータが見つかりません"
            
            place_name = RACECOURSE_CODES.get(place_code, place_code)
            
            result = f"{place_name}の枠別勝率:\n"
            for _, row in df.iterrows():
                result += f"- {int(row['waku'])}号艇: {row['win_rate']:.1f}%\n"
            
            return result
            
        except Exception as e:
            return f"エラー: {e}"
    
    def get_race_result(self, race_date: str, place_code: str, race_number: int) -> str:
        """レース結果を取得"""
        try:
            result_df = self.db.query(f"""
                SELECT * FROM race_result
                WHERE race_date = '{race_date}'
                AND place_code = '{place_code}'
                AND race_number = {race_number}
            """)
            
            if result_df.empty:
                return "レース結果が見つかりません"
            
            row = result_df.iloc[0]
            place_name = RACECOURSE_CODES.get(place_code, place_code)
            
            result = f"""
{place_name} {race_number}R 結果
着順: {int(row['first'])}-{int(row['second'])}-{int(row['third'])}
決まり手: {row['winning_pattern']}
3連単: {row['trifecta_combo']} ({row['trifecta_payout']:,}円)
"""
            return result
            
        except Exception as e:
            return f"エラー: {e}"
    
    def get_start_timing_stats(self, racer_id: str) -> str:
        """スタートタイミング統計"""
        try:
            df = self.db.query(f"""
                SELECT
                    b.waku,
                    r.st_{'{'}b.waku{'}'} as st,
                    r.first
                FROM bangumi b
                JOIN race_result r ON 
                    b.race_date = r.race_date 
                    AND b.place_code = r.place_code 
                    AND b.race_number = r.race_number
                WHERE b.racer_id = '{racer_id}'
                ORDER BY b.race_date DESC
                LIMIT 50
            """)
            
            if df.empty:
                return f"選手ID {racer_id} のSTデータが見つかりません"
            
            avg_st = df['st'].mean()
            std_st = df['st'].std()
            min_st = df['st'].min()
            max_st = df['st'].max()
            
            # フライングチェック
            flying = sum(df['st'] < 0)
            late = sum(df['st'] > 0.20)
            
            result = f"""
選手ID {racer_id} のST統計 (直近{len(df)}走):
- 平均ST: {avg_st:.3f}
- 標準偏差: {std_st:.3f}
- 最速ST: {min_st:.3f}
- 最遅ST: {max_st:.3f}
- フライング: {flying}回
- 出遅れ(0.20超): {late}回
"""
            return result
            
        except Exception as e:
            return f"エラー: {e}"
    
    def analyze_course_performance(self, racer_id: str) -> str:
        """進入コース別成績"""
        try:
            df = self.db.query(f"""
                SELECT
                    bi.entry_course,
                    COUNT(*) as races,
                    SUM(CASE WHEN r.first = b.waku THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN r.first = b.waku OR r.second = b.waku THEN 1 ELSE 0 END) as top2
                FROM bangumi b
                JOIN race_result r ON 
                    b.race_date = r.race_date 
                    AND b.place_code = r.place_code 
                    AND b.race_number = r.race_number
                LEFT JOIN before_info bi ON
                    b.race_date = bi.race_date 
                    AND b.place_code = bi.place_code 
                    AND b.race_number = bi.race_number
                    AND b.waku = bi.waku
                WHERE b.racer_id = '{racer_id}'
                GROUP BY bi.entry_course
                ORDER BY bi.entry_course
            """)
            
            if df.empty:
                return f"選手ID {racer_id} のコース別データが見つかりません"
            
            result = f"選手ID {racer_id} の進入コース別成績:\n"
            for _, row in df.iterrows():
                if row['entry_course'] and row['races'] > 0:
                    win_rate = row['wins'] / row['races'] * 100
                    top2_rate = row['top2'] / row['races'] * 100
                    result += f"- {int(row['entry_course'])}コース: {int(row['races'])}走 勝率{win_rate:.1f}% 2連率{top2_rate:.1f}%\n"
            
            return result
            
        except Exception as e:
            return f"エラー: {e}"
    
    def get_winning_patterns(self, place_code: str) -> str:
        """決まり手パターン"""
        try:
            if place_code in RACECOURSE_CODES.values():
                for code, name in RACECOURSE_CODES.items():
                    if name == place_code:
                        place_code = code
                        break
            
            df = self.db.query(f"""
                SELECT
                    winning_pattern,
                    COUNT(*) as count,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as rate
                FROM race_result
                WHERE place_code = '{place_code}'
                GROUP BY winning_pattern
                ORDER BY count DESC
            """)
            
            if df.empty:
                return f"会場 {place_code} の決まり手データが見つかりません"
            
            place_name = RACECOURSE_CODES.get(place_code, place_code)
            
            result = f"{place_name}の決まり手統計:\n"
            for _, row in df.iterrows():
                result += f"- {row['winning_pattern']}: {row['rate']:.1f}%\n"
            
            return result
            
        except Exception as e:
            return f"エラー: {e}"


class BoatRaceAssistant:
    """ボートレース分析アシスタント"""
    
    SYSTEM_PROMPT = """あなたはボートレース（競艇）の分析エキスパートアシスタントです。
ユーザーの質問に対して、利用可能な関数を使ってデータを取得し、分析結果を分かりやすく説明してください。

利用可能な関数:
- get_racer_stats: 選手の過去成績を取得
- get_place_stats: 会場の統計情報を取得
- get_race_result: レース結果を取得
- get_start_timing_stats: 選手のスタートタイミング統計
- analyze_course_performance: 進入コース別成績分析
- get_winning_patterns: 会場の決まり手パターン

会場コードと会場名の対応:
01=桐生, 02=戸田, 03=江戸川, 04=平和島, 05=多摩川, 06=浜名湖, 07=蒲郡, 08=常滑, 09=津, 10=三国, 11=びわこ, 12=住之江, 13=尼崎, 14=鳴門, 15=丸亀, 16=児島, 17=宮島, 18=徳山, 19=下関, 20=若松, 21=芦屋, 22=福岡, 23=唐津, 24=大村

ユーザーの質問を理解し、必要な関数を呼び出してデータを取得し、分析結果を日本語で回答してください。
"""
    
    def __init__(self):
        self.client = OllamaClient()
        self.functions = BoatRaceFunctions()
        self.conversation_history = []
    
    async def process_query(self, user_query: str) -> str:
        """ユーザーのクエリを処理"""
        
        # 関数呼び出しの判定
        function_call = await self._identify_function_call(user_query)
        
        # 関数実行
        function_result = ""
        if function_call:
            function_result = self._execute_function(function_call)
        
        # 回答生成
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        
        for msg in self.conversation_history[-5:]:  # 直近5件
            messages.append(msg)
        
        user_content = user_query
        if function_result:
            user_content += f"\n\n【取得データ】\n{function_result}"
        
        messages.append({"role": "user", "content": user_content})
        
        response = await self.client.chat(messages)
        
        # 履歴に追加
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def _identify_function_call(self, query: str) -> Optional[Dict]:
        """クエリから関数呼び出しを識別"""
        
        prompt = f"""以下のユーザークエリを分析し、どの関数を呼び出すべきか判断してください。

クエリ: {query}

利用可能な関数:
1. get_racer_stats(racer_id) - 選手成績
2. get_place_stats(place_code) - 会場統計
3. get_race_result(race_date, place_code, race_number) - レース結果
4. get_start_timing_stats(racer_id) - ST統計
5. analyze_course_performance(racer_id) - コース別成績
6. get_winning_patterns(place_code) - 決まり手パターン

JSON形式で回答してください:
{{"function": "関数名", "params": {{"パラメータ名": "値"}}}}
関数呼び出しが不要な場合は null を返してください。
"""
        
        response = await self.client.generate(prompt)
        
        try:
            # JSONを抽出
            import re
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return None
    
    def _execute_function(self, function_call: Dict) -> str:
        """関数を実行"""
        
        if not function_call:
            return ""
        
        func_name = function_call.get("function")
        params = function_call.get("params", {})
        
        available_functions = self.functions.get_available_functions()
        
        for func in available_functions:
            if func.name == func_name:
                try:
                    return func.function(**params)
                except Exception as e:
                    return f"関数実行エラー: {e}"
        
        return f"関数 {func_name} は利用できません"
    
    async def close(self):
        """クリーンアップ"""
        await self.client.close()


async def interactive_session():
    """インタラクティブセッション"""
    
    assistant = BoatRaceAssistant()
    
    print("🚤 ボートレース分析アシスタント")
    print("質問を入力してください（終了: quit）")
    print("-" * 50)
    
    try:
        while True:
            query = input("\n> ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query.strip():
                continue
            
            response = await assistant.process_query(query)
            print(f"\n{response}")
            
    finally:
        await assistant.close()


if __name__ == "__main__":
    asyncio.run(interactive_session())
