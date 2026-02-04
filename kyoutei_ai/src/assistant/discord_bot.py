"""
Discord Bot
ボートレース分析アシスタントをDiscordで利用可能に
"""
import asyncio
from pathlib import Path
import discord
from discord.ext import commands
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config
from src.assistant.llm_assistant import BoatRaceAssistant


class BoatRaceBot(commands.Bot):
    """ボートレース分析Discord Bot"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix='!',
            intents=intents,
            description='ボートレース予測AIアシスタント'
        )
        
        self.assistant = BoatRaceAssistant()
    
    async def setup_hook(self):
        """Bot起動時の処理"""
        await self.add_cog(BoatRaceCog(self))
    
    async def on_ready(self):
        """ログイン完了時"""
        logger.info(f'Logged in as {self.user.name}')
        
        # ステータス設定
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name='ボートレース予測'
        )
        await self.change_presence(activity=activity)


class BoatRaceCog(commands.Cog):
    """ボートレース関連コマンド"""
    
    def __init__(self, bot: BoatRaceBot):
        self.bot = bot
    
    @commands.command(name='ask', aliases=['q', '質問'])
    async def ask(self, ctx: commands.Context, *, question: str):
        """
        質問に回答します
        使い方: !ask 多摩川12Rの選手のスタートデータを見せて
        """
        async with ctx.typing():
            try:
                response = await self.bot.assistant.process_query(question)
                
                # 長い場合は分割
                if len(response) > 2000:
                    chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
                    for chunk in chunks:
                        await ctx.send(f"```\n{chunk}\n```")
                else:
                    await ctx.send(f"```\n{response}\n```")
                    
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                await ctx.send(f"エラーが発生しました: {e}")
    
    @commands.command(name='racer', aliases=['選手'])
    async def racer(self, ctx: commands.Context, racer_id: str):
        """
        選手情報を取得します
        使い方: !racer 4444
        """
        async with ctx.typing():
            try:
                response = await self.bot.assistant.process_query(
                    f"選手ID {racer_id} の成績を教えて"
                )
                await ctx.send(f"```\n{response}\n```")
            except Exception as e:
                await ctx.send(f"エラー: {e}")
    
    @commands.command(name='place', aliases=['会場'])
    async def place(self, ctx: commands.Context, place_name: str):
        """
        会場統計を取得します
        使い方: !place 多摩川
        """
        async with ctx.typing():
            try:
                response = await self.bot.assistant.process_query(
                    f"{place_name}の枠別勝率を教えて"
                )
                await ctx.send(f"```\n{response}\n```")
            except Exception as e:
                await ctx.send(f"エラー: {e}")
    
    @commands.command(name='result', aliases=['結果'])
    async def result(self, ctx: commands.Context, date: str, place: str, race_num: int):
        """
        レース結果を取得します
        使い方: !result 20240101 05 12
        """
        async with ctx.typing():
            try:
                response = await self.bot.assistant.process_query(
                    f"{date}の{place}の{race_num}Rの結果を教えて"
                )
                await ctx.send(f"```\n{response}\n```")
            except Exception as e:
                await ctx.send(f"エラー: {e}")
    
    @commands.command(name='help_br', aliases=['ヘルプ'])
    async def help_boatrace(self, ctx: commands.Context):
        """ヘルプを表示します"""
        help_text = """
🚤 ボートレース予測AI コマンド一覧

**質問コマンド**
`!ask <質問>` - 自由に質問できます
例: !ask 多摩川の1号艇の勝率は？

**選手情報**
`!racer <登録番号>` - 選手の成績を取得
例: !racer 4444

**会場情報**
`!place <会場名>` - 会場の統計を取得
例: !place 多摩川

**レース結果**
`!result <日付> <会場コード> <レース番号>` - レース結果を取得
例: !result 20240101 05 12

**会場コード**
01=桐生, 02=戸田, 03=江戸川, 04=平和島, 05=多摩川
06=浜名湖, 07=蒲郡, 08=常滑, 09=津, 10=三国
11=びわこ, 12=住之江, 13=尼崎, 14=鳴門, 15=丸亀
16=児島, 17=宮島, 18=徳山, 19=下関, 20=若松
21=芦屋, 22=福岡, 23=唐津, 24=大村
"""
        await ctx.send(help_text)
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """メッセージ受信時の処理"""
        # ボット自身のメッセージは無視
        if message.author.bot:
            return
        
        # メンションされた場合に反応
        if self.bot.user.mentioned_in(message):
            # メンションを除去してクエリを抽出
            content = message.content.replace(f'<@{self.bot.user.id}>', '').strip()
            content = content.replace(f'<@!{self.bot.user.id}>', '').strip()
            
            if content:
                async with message.channel.typing():
                    try:
                        response = await self.bot.assistant.process_query(content)
                        await message.reply(f"```\n{response}\n```")
                    except Exception as e:
                        await message.reply(f"エラーが発生しました: {e}")


async def run_bot():
    """Botを起動"""
    token = config.llm.discord_token
    
    if not token:
        logger.error("DISCORD_BOT_TOKEN が設定されていません")
        logger.info("環境変数 DISCORD_BOT_TOKEN を設定するか、.envファイルに追加してください")
        return
    
    bot = BoatRaceBot()
    
    try:
        await bot.start(token)
    finally:
        await bot.assistant.close()
        await bot.close()


if __name__ == "__main__":
    asyncio.run(run_bot())
