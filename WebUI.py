import re
import queue
import atexit
import asyncio
import subprocess
from ast import literal_eval
from threading import Thread

import opencc
import requests
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import config
from tools.log import logger
from tools.word_processing import prompt_process, process_sentences, language_classification
from tools.audio import get_audio_device_names, initialize_audio, remove_start_silence, change_audio_device, play_audio
from tools.tts import TTS_API, start_up_tts
from VTube_Studio_API import Websocket_connect, HotkeysInCurrentModelRequest, hotkeyID_list_processing
from sentiment_analysis import multi_segment_sentiment_analysis

#載入LLM
def load_model(model_path: str):
    global tokenizer
    global model
    global streamer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    #attn_implementation="flash_attention_2",
    # safetensors=True,
    device_map="cuda:0"
    )   
    streamer = TextIteratorStreamer(tokenizer,True)

#初始設定
model_name = config.default.llm_model_name #模型型號
model_path = config.llm_model.path[model_name]
converter = opencc.OpenCC('s2twp.json') #簡中轉繁中用語
YouTube_chat_room_open = False #YouTub聊天室連接狀態

#初始化
load_model(model_path)
logger.info("LLM載入成功  型號:" + model_name)

audio_device_names_list = get_audio_device_names() #取得音訊設備名稱清單
audio_device_name = initialize_audio(audio_device_names_list) #初始化混音器,並取得音訊設備名稱

Bert_VITS2_server = start_up_tts()

def Execute_at_the_end():
    logger.info("正在退出中...")

    Bert_VITS2_server.terminate() #關閉BertVITS2的API

    logger.info("退出完成!")

atexit.register(Execute_at_the_end) #註冊關閉事件

#使用者輸入訊息,更新聊天室
def user(
    message: str, 
    history: list[list[str | None | tuple]]
) -> tuple[str, list[list[str | tuple | None]]]:
    return "", history + [[message, ""]]

#AI回覆,更新聊天室
async def bot(
    history: list[list[str | None | tuple]], 
    audio_volume: int, 
    history_num: int,
    checkable_settings: list[str]
):
    #語音播放
    async def voice_playback(voice_queue):
        while True:
            voice = await voice_queue.get()

            if voice == "end":
                break

            if ("情緒分析" in checkable_settings) and VTube_Studio_API_connection_status: #是否啟用情緒分析
                sentiment_label = multi_segment_sentiment_analysis(voice[1]) #情緒分析
                logger.info(f"情緒:{sentiment_label}")
                VTube_Studio_API.hotkey_trigger(sentiment_label) #觸發快速鍵

            #同步顯示文字
            if YouTube_chat_room_open:
                response = requests.get(f"http://127.0.0.1:3840/text_to_display_add?data={voice[1]}")
                logger.info(f"網頁即時文字狀態:{response.content}")

            audio_processing = remove_start_silence(voice[0]) #移除頭空白音訊
            await play_audio(audio_processing, audio_volume)

    #語音生成
    async def speech_generation(text_queue, voice_queue):
        while True:
            text_list = await text_queue.get()

            if text_list == "end":
                break
            
            for text in text_list:
                if re.search(r"[a-zA-Z\u4e00-\u9fff]", text): #檢測是否有中文或英文字母
                    text_classification = language_classification(text) #根據語言分類

                    for text_dict in text_classification: #根據語言生成TTS
                        content = text_dict["content"]
                        language = text_dict["language"]
                        
                        logger.info(f"TTS內容  語言:{language}  文字:{[content]}")

                        Voice = TTS_API(text=content, language=language) #生成人聲
                        await voice_queue.put([Voice, content])
        
        await voice_queue.put("end")

    #文字生成
    async def text_generation(prompt, text_queue):
        global streamer
        global model

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda") #用分詞器處理提示
        generation_kwargs = dict(
            inputs, 
            streamer=streamer,
            eos_token_id=2, 
            pad_token_id=2,
            max_length=4096,
            do_sample=True,
            temperature=1
        ) #設定推理參數
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs) #用多執行序文字生成
        thread.start()

        #流式輸出
        num_sentences = 1 #累計句數
        for new_text in streamer:
            history[-1][1] += new_text.strip(tokenizer.eos_token) #過濾「結束」Token
            history[-1][1] = converter.convert(history[-1][1]) #簡轉繁中用語

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
        await text_queue.put("end")


    try:
        VTube_Studio_API_connection_status = False #VTube_Studio_API連接狀態
        if "情緒分析" in checkable_settings: #是否啟用情緒分析
            try:
                VTube_Studio_API = Websocket_connect(config.VTube_Studio.VTube_Studio_API_URL) #連接VTube_Studio_API
                VTube_Studio_API_connection_status = VTube_Studio_API.Connect_to_VTube_Studio_API( #嘗試與VTube_Studio_API握手
                    config.VTube_Studio.pluginName, 
                    config.VTube_Studio.pluginDeveloper
                )
            except:
                logger.info("無法連接VTube Studio API")

            if VTube_Studio_API_connection_status: #握手成功
                hotkeyID_list = VTube_Studio_API.websocket_send( #取得快速鍵列表
                    HotkeysInCurrentModelRequest()
                )
                VTube_Studio_API.hotkey_correspondence_table = hotkeyID_list_processing(hotkeyID_list) #整理快速鍵列表

        #文本生成
        text_queue = asyncio.Queue() #創建隊列
        voice_queue = asyncio.Queue() #創建隊列
        text_task = asyncio.create_task(speech_generation(text_queue, voice_queue)) #啟用異步函數
        voice_task = asyncio.create_task(voice_playback(voice_queue)) #啟用異步函數

        prompt = prompt_process(history, model_name, history_num, tokenizer)

        async for result in text_generation(prompt, text_queue):
            yield result
        
        await text_task
        await voice_task
    finally:
        try:
            VTube_Studio_API.hotkey_trigger(config.default.sentiment) #觸發默認情緒的快速鍵
            logger.info("已觸發默認情緒的快速鍵")
        except:
            logger.info("未有需要觸發的快速鍵")

        try:
            VTube_Studio_API.close()
            logger.info("已釋放VTube_Studio_API資源")
        except:
            logger.info("未有需要釋放的VTube_Studio_API資源")

#連接YouTube聊天室
async def YouTube_chat_room(
    youtube_channel_id: str, 
    history: list[list[str | None | tuple]], 
    audio_volume: int, 
    history_num: int,
    checkable_settings: list[str]
):
    global YouTube_chat_room_open
    YouTube_chat_room_open = True #設定連接狀態
    message = ""
    log = ""

    if youtube_channel_id: #YouTube帳號代碼是否為空
        response_queue = queue.Queue() #多執行序回傳用隊列
        command = ["python", "Youtube_Chat.py", "-c", youtube_channel_id] #指令和傳遞參數
        YouTube_chat_processor = subprocess.Popen(command, stdout=subprocess.PIPE) #呼叫Youtube_Chat.py

        startup_state = YouTube_chat_processor.stdout.readline().decode()
        if startup_state.split("|")[0] == "INFO": #回傳是否為"INFO"
            logger.info("已連接上YouTube直播")

            command = ["python", "web_real_time_subtitles/app.py"] #指令和傳遞參數
            web_real_time_subtitles = subprocess.Popen(command, stdout=subprocess.PIPE) #呼叫web_real_time_subtitles/app.py

            #連接狀態為True時,處理聊天室訊息
            while YouTube_chat_room_open:
                #獲取聊天室訊息
                def obtain_chat_message(q):
                    response = YouTube_chat_processor.stdout.readline().decode() #等待Youtube_Chat.py回傳訊息
                    q.put(response.lstrip("INFO|"))
                
                thread = Thread(target=obtain_chat_message, args=(response_queue,)) #防止等待訊息時卡住主程式
                thread.start()

                #等待新訊息並且連接狀態為True
                while thread.is_alive() and YouTube_chat_room_open:
                    await asyncio.sleep(0.1)

                if YouTube_chat_room_open: #連接狀態為True時,處理訊息
                    response_str = response_queue.get() #取得回覆隊列內容
                    logger.info(response_str)
                    YouTube_chat_information = literal_eval(response_str) #從字串換成字典
                    YouTube_chat_author_name = YouTube_chat_information["author_name"]
                    YouTube_chat_message = YouTube_chat_information["message"]
                    logger.info(f"聊天室【{YouTube_chat_author_name}:{YouTube_chat_message}】")
                    history += [[YouTube_chat_message, ""]] #將聊天室訊息存到歷史紀錄裡

                    response = requests.get(f"http://127.0.0.1:3840/clear")
                    logger.info(f"網頁即時文字狀態:{response}")

                    async for result in bot(history, audio_volume, history_num, checkable_settings): #文本生成
                        history, log = result
                        yield message, history, log
            
            web_real_time_subtitles.terminate()
        else:
            log_error_message = startup_state.lstrip("ERROR|").replace("\n", "")
            log += f"\n\n{log_error_message}"
            logger.error(f"{log_error_message}")
        
        logger.info("已停止獲取聊天室訊息")
        log += "\n\n已停止獲取聊天室訊息"
        YouTube_chat_processor.terminate()
        yield message, history, log
    else:
        logger.error("YouTube帳號代碼不可空白!!")
        log += "\n\nYouTube帳號代碼不可空白!!"
        yield message, history, log
    
    YouTube_chat_room_open = False #設定連接狀態

#關閉連接YouTube聊天室
def close_YouTube_chat_room():
    global YouTube_chat_room_open
    YouTube_chat_room_open = False
    logger.info("中止連接YouTube直播")


with gr.Blocks() as demo:
    gr.Markdown("Dolly WebUI v0.0.1")

    with gr.Tab("控制面板"):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="聊天室", bubble_full_width=False,)

                with gr.Row():
                    with gr.Column(scale=6):
                        message = gr.Textbox(label="訊息")

                    with gr.Column(scale=1, min_width=0):
                        clear = gr.Button("清除聊天紀錄", variant="primary", scale=1)

                log = gr.Textbox(label="日誌")

            with gr.Column(scale=1, min_width=0):
                youtube_channel_id = gr.Textbox(label="YouTube帳號代碼", value=config.default.YouTubeChannelID)

                with gr.Row():
                    with gr.Column(min_width=0):
                        connect_chat = gr.Button("連接YouTube聊天室", variant="primary", scale=1)

                    with gr.Column(min_width=0):
                        stop_connect_chat = gr.Button("中止連接", variant="primary", scale=1)
                
                checkable_settings_checkboxgroup = gr.CheckboxGroup(
                    ["情緒分析"], 
                    label="可選項",
                    value=config.default.checkable_settings
                )
                audio_device_dropdown = gr.Dropdown(
                    ["預設"] + audio_device_names_list, 
                    label="音訊裝置", 
                    value=audio_device_name,
                    filterable=False
                )
                audio_volume_slider = gr.Slider(
                    label="音量", 
                    value=config.default.audio.audio_volume, 
                    minimum=0, 
                    maximum=100, 
                    step=1
                )
                history_num_slider = gr.Slider(
                    label=f"上下文數量  模式:{config.default.history.history_mode}", 
                    value=config.default.history.history_num, 
                    minimum=1 if config.default.history.history_mode == "rounds" else 50, 
                    maximum=config.default.history.history_num_max, 
                    step=1 if config.default.history.history_mode == "rounds" else 50
                )

    with gr.Tab("設定"):
        model_name_Radio = gr.Radio(
            config.llm_model.name, 
            label="LLM Model", 
            value=config.default.llm_model_name
        )
        language_Radio = gr.Radio(
            ["ZH", "EN", "AUTO"], 
            label="語言", 
            value=config.default.TTS_Language
        )

    message.submit( #使用者輸入,然後AI回覆
        user,
        [message,chatbot],
        [message,chatbot]
    ).then(
        bot,
        [chatbot, audio_volume_slider, history_num_slider, checkable_settings_checkboxgroup],
        [chatbot,log]
    )
    clear.click(lambda : None,None,chatbot) #清空聊天室
    connect_chat.click( #連接YouTube聊天室
        YouTube_chat_room,
        [youtube_channel_id, chatbot, audio_volume_slider, history_num_slider, checkable_settings_checkboxgroup],
        [message,chatbot,log]
    )
    stop_connect_chat.click(close_YouTube_chat_room) #關閉連接YouTube聊天室
    audio_device_dropdown.change(change_audio_device, audio_device_dropdown) #改變音訊設備

demo.launch() #啟用WebUI