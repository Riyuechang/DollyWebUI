import queue
import asyncio
import subprocess
from typing import Literal
from ast import literal_eval
from threading import Thread

import requests
import gradio as gr

from config import config
from tools.log import logger
from tools.tts import tts_generation
from tools.word_processing import prompt_process, sentence_break, processing_timestamps
from tools.audio import get_audio_device_names, initialize_audio, change_audio_device
from VTube_Studio_API import Websocket_connect, HotkeysInCurrentModelRequest, hotkeyID_list_processing
from core import chat_core

#初始化
audio_device_names_list = get_audio_device_names() #取得音訊設備名稱清單
audio_device_name = initialize_audio(audio_device_names_list) #初始化混音器,並取得音訊設備名稱

#使用者輸入訊息,更新聊天室
def user(
    message: str, 
    history: list[list[str | None | tuple]]
) -> tuple[str, list[list[str | tuple | None]]]:
    return "", history + [[message, ""]]

#AI回覆,更新聊天室
async def bot(
    question_input: list[list[str | None | tuple]] | str, 
    audio_volume: int, 
    history_num: int | None,
    checkable_settings: list[str],
    mode: Literal["text_generation", "sync_live2D", "timestamp_sync_live2D"] = "text_generation"
):
    try:
        chat_core.VTube_Studio_API_connection_status = False #VTube_Studio_API連接狀態
        if "情緒分析" in checkable_settings: #是否啟用情緒分析
            try:
                VTube_Studio_API = Websocket_connect(config.VTube_Studio.VTube_Studio_API_URL) #連接VTube_Studio_API
                chat_core.VTube_Studio_API_connection_status = VTube_Studio_API.Connect_to_VTube_Studio_API( #嘗試與VTube_Studio_API握手
                    config.VTube_Studio.pluginName, 
                    config.VTube_Studio.pluginDeveloper
                )
            except:
                logger.info("無法連接VTube Studio API")

            if chat_core.VTube_Studio_API_connection_status: #握手成功
                hotkeyID_list = VTube_Studio_API.websocket_send( #取得快速鍵列表
                    HotkeysInCurrentModelRequest()
                )
                VTube_Studio_API.hotkey_correspondence_table = hotkeyID_list_processing(hotkeyID_list) #整理快速鍵列表

        text_queue = asyncio.Queue() #創建隊列
        voice_queue = asyncio.Queue() #創建隊列
        text_task = asyncio.create_task(
            chat_core.speech_generation( #啟用異步函數
                text_queue, 
                voice_queue,
                mode
            )
        )
        voice_task = asyncio.create_task( #啟用異步函數
            chat_core.voice_playback(
                voice_queue, 
                audio_volume,
                checkable_settings,
                VTube_Studio_API if chat_core.VTube_Studio_API_connection_status else None,
                mode
            )
        )

        match mode:
            case "text_generation":
                prompt = prompt_process(
                    question_input, 
                    chat_core.model_name, 
                    history_num, 
                    chat_core.tokenizer,
                    "hypnotic"
                )

                #文本生成
                async for result in chat_core.text_generation(
                    text_queue, 
                    prompt,
                    question_input,
                    checkable_settings
                ):
                    yield result

            case "sync_live2D":
                sentence_list = sentence_break(question_input)
                await text_queue.put(sentence_list)
            
            case "timestamp_sync_live2D":
                timestamp_text_list = processing_timestamps(question_input)
                await text_queue.put(timestamp_text_list)
                
        
        await text_queue.put(None)
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

#重新生成
async def regenerate(
    history: list[list[str | None | tuple]], 
    *args
):
    history[-1][1] = ""
    
    async for result in bot( #文本生成
        question_input=history, 
        *args
    ):
        yield result

#連接YouTube聊天室
async def YouTube_chat_room(
    youtube_channel_id: str, 
    history: list[list[str | None | tuple]], 
    *args
):
    chat_core.YouTube_chat_room_open = True #設定連接狀態
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
            while chat_core.YouTube_chat_room_open:
                #獲取聊天室訊息
                def obtain_chat_message(q):
                    response = YouTube_chat_processor.stdout.readline().decode() #等待Youtube_Chat.py回傳訊息
                    q.put(response.lstrip("INFO|"))
                
                thread = Thread(target=obtain_chat_message, args=(response_queue,)) #防止等待訊息時卡住主程式
                thread.start()

                #等待新訊息並且連接狀態為True
                while thread.is_alive() and chat_core.YouTube_chat_room_open:
                    await asyncio.sleep(0.1)

                if chat_core.YouTube_chat_room_open: #連接狀態為True時,處理訊息
                    response_str = response_queue.get() #取得回覆隊列內容
                    logger.info(response_str)

                    YouTube_chat_information = literal_eval(response_str) #從字串換成字典
                    YouTube_chat_author_name = YouTube_chat_information["author_name"]
                    YouTube_chat_message = YouTube_chat_information["message"]
                    logger.info(f"聊天室【{YouTube_chat_author_name}:{YouTube_chat_message}】")

                    response = requests.get(f"http://127.0.0.1:3840/clear")
                    logger.info(f"網頁即時文字狀態:{response}")

                    history += [[YouTube_chat_message, ""]] #將聊天室訊息存到歷史紀錄裡
                    async for result in bot( #文本生成
                        history, 
                        *args
                    ):
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
    
    chat_core.YouTube_chat_room_open = False #設定連接狀態

#關閉連接YouTube聊天室
def close_YouTube_chat_room():
    chat_core.YouTube_chat_room_open = False
    logger.info("中止連接YouTube直播")

#TTS處理
def tts_processing(
    text: str, 
    audio
) -> bytes | None:
    Voice_bytes = tts_generation(text)

    if Voice_bytes is None:
        return audio
    else:
        return Voice_bytes


async def sync_live2D(
    text: str, 
    audio_volume: int, 
    checkable_settings: list[str]
):
    async for _ in bot( #文本生成
        question_input=text, 
        audio_volume=audio_volume, 
        history_num=None, 
        checkable_settings=checkable_settings,
        mode="sync_live2D"
    ):
        pass


async def timestamp_sync_live2D(
    text: str, 
    audio_volume: int, 
    checkable_settings: list[str]
):
    async for _ in bot( #文本生成
        question_input=text, 
        audio_volume=audio_volume, 
        history_num=None, 
        checkable_settings=checkable_settings,
        mode="timestamp_sync_live2D"
    ):
        pass


def update_and_reload_config(hypnotic_prompt: str):
    with config.updata_config_yml() as config_yml:
        config_yml.updata(
            key="hypnotic_prompt",
            old=config.default.hypnotic_prompt,
            new=hypnotic_prompt,
            str_mode=True
        )

    config.load_config()
    logger.info("更新並重新載入設定檔成功")


with gr.Blocks() as demo:
    gr.Markdown("Dolly WebUI v1.0.0")

    with gr.Tab("控制面板"):
        with gr.Row():
            with gr.Column(scale=4):
                chat_room_chatbot = gr.Chatbot(label="聊天室", bubble_full_width=False, show_copy_button=True)

                with gr.Row():
                    with gr.Column(scale=5):
                        user_message_textbox = gr.Textbox(label="訊息")

                    with gr.Column(scale=1, min_width=0):
                        regenerate_button = gr.Button("重新生成  ↻", variant="primary", scale=1)
                        clear_button = gr.Button("清除聊天紀錄", variant="primary", scale=1)

                log = gr.Textbox(label="日誌")

            with gr.Column(scale=1, min_width=0):
                with gr.Row():
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
                    checkable_settings_checkboxgroup = gr.CheckboxGroup(
                        ["TTS", "情緒分析"], 
                        label="可選項",
                        value=config.default.checkable_settings
                    )

            with gr.Column(scale=2, min_width=0):
                youtube_channel_id = gr.Textbox(label="YouTube帳號代碼", value=config.default.YouTubeChannelID)

                with gr.Row():
                    with gr.Column(min_width=0):
                        connect_chat = gr.Button("連接YouTube聊天室", variant="primary", scale=1)

                    with gr.Column(min_width=0):
                        stop_connect_chat = gr.Button("中止連接", variant="primary", scale=1)

                TTS_textarea = gr.TextArea(
                    label="TTS",
                    placeholder="使用\"根據時間戳同步Live2D\"功能時\n要在對應文字前加上時間戳\n\n例：\n<3.5>你好\n<5.5>我是XXX\n<7>很高興認識你\n\n時間戳單位為秒"
                )

                with gr.Row():
                    with gr.Column(min_width=0):
                        tts_generation_button = gr.Button("生成", variant="primary")

                    with gr.Column(min_width=0):
                        sync_live2D_button = gr.Button("同步Live2D", variant="primary")

                timestamp_sync_live2D_button = gr.Button("根據時間戳同步Live2D", variant="primary")
                voice_audio = gr.Audio(label="聲音", interactive=False)

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
        hypnotic_prompt_textarea = gr.TextArea(
            label="催眠提示詞", 
            value=config.default.hypnotic_prompt
        )
        update_and_reload_config_button = gr.Button("更新並重新載入設定檔", variant="primary")

    user_message_textbox.submit( #使用者輸入,然後AI回覆
        user,
        [user_message_textbox,chat_room_chatbot],
        [user_message_textbox,chat_room_chatbot]
    ).then(
        bot,
        [chat_room_chatbot, audio_volume_slider, history_num_slider, checkable_settings_checkboxgroup],
        [chat_room_chatbot,log]
    )
    regenerate_button.click(
        regenerate,
        [chat_room_chatbot, audio_volume_slider, history_num_slider, checkable_settings_checkboxgroup],
        [chat_room_chatbot,log]
    )
    clear_button.click(lambda : None,None,chat_room_chatbot) #清空聊天室
    connect_chat.click( #連接YouTube聊天室
        YouTube_chat_room,
        [youtube_channel_id, chat_room_chatbot, audio_volume_slider, history_num_slider, checkable_settings_checkboxgroup],
        [user_message_textbox,chat_room_chatbot,log]
    )
    stop_connect_chat.click(close_YouTube_chat_room) #關閉連接YouTube聊天室
    audio_device_dropdown.change(change_audio_device, audio_device_dropdown) #改變音訊設備
    tts_generation_button.click(tts_processing, [TTS_textarea,voice_audio], voice_audio)
    sync_live2D_button.click(sync_live2D, [TTS_textarea,audio_volume_slider,checkable_settings_checkboxgroup])
    timestamp_sync_live2D_button.click(timestamp_sync_live2D, [TTS_textarea,audio_volume_slider,checkable_settings_checkboxgroup])
    update_and_reload_config_button.click(update_and_reload_config, hypnotic_prompt_textarea)

demo.launch() #啟用WebUI