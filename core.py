import re
import time
import asyncio
from typing import Literal
from threading import Thread

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import config
from tools.log import logger
from tools.word_processing import process_sentences, opencc_converter
from tools.audio import remove_silence, play_audio
from tools.tts import tts_generation
from sentiment_analysis import multi_segment_sentiment_analysis
from VTube_Studio_API import Websocket_connect


class Chat_Core:
    def __init__(
        self
    ):
        #初始設定
        self.model_name = config.default.llm_model_name #模型型號
        self.model_path = config.llm_model.path[self.model_name]
        self.YouTube_chat_room_open = False #YouTub聊天室連接狀態
        self.VTube_Studio_API_connection_status = False #VTube_Studio_API連接狀態

        #載入LLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
        self.model_path,
        #attn_implementation="flash_attention_2",
        # safetensors=True,
        device_map="cuda:0"
        )   
        self.streamer = TextIteratorStreamer(self.tokenizer,True)
        logger.info("LLM載入成功  型號:" + self.model_name)

    #語音播放
    async def voice_playback(
        self,
        voice_queue, 
        audio_volume: int, 
        checkable_settings: list[str], 
        VTube_Studio_API: Websocket_connect,
        mode: Literal["text_generation", "sync_live2D", "timestamp_sync_live2D"] = "text_generation"
    ):
        loop = asyncio.get_event_loop()
        start_timestamp = time.time()
        
        while True:
            voice = await voice_queue.get()

            if voice is None:
                logger.info("已停止語音播放")
                break

            if mode == "timestamp_sync_live2D" and type(voice) is float:
                time_difference = (start_timestamp + voice) - time.time()

                if time_difference > 0:
                    #VTube_Studio_API.hotkey_trigger(config.default.sentiment) #觸發默認情緒的快速鍵
                    #logger.info("已觸發默認情緒的快速鍵")
                    await asyncio.sleep(time_difference)
                
                voice = await voice_queue.get()

            if ("情緒分析" in checkable_settings) and self.VTube_Studio_API_connection_status: #是否啟用情緒分析
                sentiment_label = multi_segment_sentiment_analysis(voice[1]) #情緒分析
                logger.info(f"情緒:{sentiment_label}")
                VTube_Studio_API.hotkey_trigger(sentiment_label) #觸發快速鍵

            #同步顯示文字
            if self.YouTube_chat_room_open:
                url = f"http://127.0.0.1:3840/text_to_display_add?data={voice[1]}"
                response = await loop.run_in_executor(None, requests.get(), url)
                logger.info(f"網頁即時文字狀態:{response.content}")

            audio_processing = remove_silence(voice[0], 100) #移除頭空白音訊
            await loop.run_in_executor(None, play_audio, audio_processing, audio_volume)

    #語音生成
    async def speech_generation(
        self,
        text_queue, 
        voice_queue,
        mode: Literal["text_generation", "sync_live2D", "timestamp_sync_live2D"] = "text_generation"
    ):
        loop = asyncio.get_event_loop()
        
        while True:
            text_list = await text_queue.get()

            if text_list is None:
                logger.info("已停止語音生成")
                break

            if type(text_list) is not list:
                text_list = [text_list]
            
            for text in text_list:
                if mode == "timestamp_sync_live2D":
                    search_text_timestamp = re.search(r"(?<=\<)+[0-9.]+(?=\>)", text)

                    if search_text_timestamp:
                        timestamp = float(search_text_timestamp.group(0))
                        await voice_queue.put(timestamp)

                        text = re.search(r"(?<=\d\>)(.*)", text).group(0)
                
                def blocking_generator():
                    for result in tts_generation(text, True):
                        Voice, content = result

                        if Voice is not None and content is not None:
                            asyncio.run_coroutine_threadsafe(voice_queue.put([Voice, content]), loop)
                
                await loop.run_in_executor(None, blocking_generator)
        
        logger.info("正在停止語音播放")
        await voice_queue.put(None)

    #文字生成
    async def text_generation(
        self,
        text_queue, 
        prompt: str,
        history: list[list[str | None | tuple]],
        checkable_settings: list[str]
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda") #用分詞器處理提示
        generation_kwargs = dict(
            inputs, 
            streamer=self.streamer,
            eos_token_id=2, 
            pad_token_id=2,
            max_length=4096,
            do_sample=True,
            temperature=1
        ) #設定推理參數
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs) #用多執行序文字生成
        thread.start()

        #流式輸出
        num_sentences = 1 #累計句數
        for new_text in self.streamer:
            history[-1][1] += new_text.strip(self.tokenizer.eos_token) #過濾「結束」Token
            history[-1][1] = opencc_converter.convert(history[-1][1]) #簡轉繁中用語

            if "TTS" in checkable_settings and re.search(r"[.。?？!！\n]", new_text):
                response, num_sentences = process_sentences(history, num_sentences, -1) #處理斷句
                
                if response != None: #必須要有新增的句子
                    await text_queue.put(response)
                    logger.info(f"斷句: {response}  句數: {num_sentences}")

            yield history, str([prompt + history[-1][1]]) #即時更新聊天室和日誌

        #處理最後一句
        if "TTS" in checkable_settings:
            original_num_sentences = num_sentences #原來句數
            response, num_sentences = process_sentences(history, num_sentences - 1, None) #處理斷句並包含最後一句
            logger.info(f"斷句: {response}  句數: {num_sentences}")
            offset = original_num_sentences - num_sentences - 1 #剩餘句數
            remaining_sentences = response[offset:] #剩餘句子
            await text_queue.put(remaining_sentences)
            logger.info(f"最終斷句: {remaining_sentences}  句數: {num_sentences}")
        
        logger.info(f"使用者輸入：{[history[-1][0]]}")
        logger.info(f"LLM輸出：{[history[-1][1]]}")


chat_core = Chat_Core()