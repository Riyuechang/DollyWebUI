import re
import time
import asyncio
from typing import Literal
from threading import Thread

import opencc
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import config
from tools.log import logger
from tools.word_processing import process_sentences, language_classification
from tools.audio import remove_start_silence, play_audio
from tools.tts import TTS_API
from sentiment_analysis import multi_segment_sentiment_analysis
from VTube_Studio_API import Websocket_connect


class Chat_Core:
    def __init__(
        self
    ):
        #初始設定
        self.model_name = config.default.llm_model_name #模型型號
        self.model_path = config.llm_model.path[self.model_name]
        self.converter = opencc.OpenCC('s2twp.json') #簡中轉繁中用語
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
        start_timestamp = time.time()
        
        while True:
            voice = await voice_queue.get()

            if voice is None:
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
                response = requests.get(f"http://127.0.0.1:3840/text_to_display_add?data={voice[1]}")
                logger.info(f"網頁即時文字狀態:{response.content}")

            audio_processing = remove_start_silence(voice[0]) #移除頭空白音訊
            await play_audio(audio_processing, audio_volume)

    #語音生成
    async def speech_generation(
        self,
        text_queue, 
        voice_queue,
        mode: Literal["text_generation", "sync_live2D", "timestamp_sync_live2D"] = "text_generation"
    ):
        while True:
            text_list = await text_queue.get()

            if text_list is None:
                break
            
            for text in text_list:
                if mode == "timestamp_sync_live2D":
                    search_text_timestamp = re.search(r"(?<=\<)+[0-9.]+(?=\>)", text)

                    if search_text_timestamp:
                        timestamp = float(search_text_timestamp.group(0))
                        await voice_queue.put(timestamp)

                        text = re.search(r"(?<=\d\>)(.*)", text).group(0)

                if re.search(r"[a-zA-Z\u4e00-\u9fff]", text): #檢測是否有中文或英文字母
                    text_classification = language_classification(text) #根據語言分類

                    for text_dict in text_classification: #根據語言生成TTS
                        content = text_dict["content"]
                        language = text_dict["language"]
                        
                        logger.info(f"TTS內容  語言:{language}  文字:{[content]}")

                        Voice = TTS_API(text=content, language=language) #生成人聲
                        await voice_queue.put([Voice, content])
        
        await voice_queue.put(None)

    #文字生成
    async def text_generation(
        self,
        text_queue, 
        prompt,
        history
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
            history[-1][1] = self.converter.convert(history[-1][1]) #簡轉繁中用語

            if re.search(r"[.。?？!！\n]", new_text):
                response, num_sentences = process_sentences(history, num_sentences, -1) #處理斷句
                if response != None: #必須要有新增的句子
                    await text_queue.put(response)
                    logger.info(f"斷句: {response}  句數: {num_sentences}")

            yield history, str([prompt + history[-1][1]]) #即時更新聊天室和日誌

        #處理最後一句
        original_num_sentences = num_sentences #原來句數
        response, num_sentences = process_sentences(history, num_sentences - 1, None) #處理斷句並包含最後一句
        offset = num_sentences - original_num_sentences - 1 #剩餘句數
        remaining_sentences = response[offset:] #剩餘句子

        await text_queue.put(remaining_sentences)
        logger.info(f"最終斷句: {remaining_sentences}")
        await text_queue.put(None)


chat_core = Chat_Core()