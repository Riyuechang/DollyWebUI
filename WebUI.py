import io
import re
import requests
import asyncio
import atexit
import gradio as gr
import subprocess
import time
import opencc
import json
import queue
from tools import promptTemplate, history_process, process_sentences, BertVITS2_API, language_classification, remove_start_and_end_silence, remove_start_silence
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import pygame

#模型相對路徑
model_dict = {
    "Taiwan-LLM-7B":"./Model/Taiwan-LLM-7B-v2.0.1-chat-awq",
    "Taiwan-LLM-13B":"./Model/Taiwan-LLM-13B-v2.0-chat-awq",
    "Breeze-7B":"./Model/Breeze-7B-Instruct-64k-v0_1-AWQ"
}

#載入LLM
def load_model(model_path):
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
model_name = "Taiwan-LLM-7B" #模型型號
converter = opencc.OpenCC('s2twp.json') #簡中轉繁中用語
YouTube_chat_room_open = False #YouTub聊天室連接狀態
pygame.mixer.init(devicename="Starship/Matisse HD Audio Controller Analog Stereo")
#pygame.mixer.init(devicename="virtual_speaker") #更改輸出到虛擬喇叭

#初始化
load_model(model_dict[model_name])
print("LLM載入成功  型號:" + model_name)

cwd = "./TTS/Bert-VITS2"
Bert_VITS2_server = subprocess.Popen(["runtime/bin/python", "hiyoriUI.py"], cwd=cwd, stdout=subprocess.PIPE) #啟動Bert-VITS2

while True: #等待啟動
    output = Bert_VITS2_server.stdout.readline().decode()

    if output.find("INFO     | hiyoriUI.py:730 | api文档地址 http://127.0.0.1:5000/docs") != -1: #檢測是否開啟完畢
        time.sleep(1)

        #預載入模型
        url = "http://127.0.0.1:5000/voice?text=測試&model_id=0&speaker_id=0&language=ZH"
        response = requests.get(url)
        
        if response.status_code == 200:
            print("TTS啟動成功")
        else:
            print(f"錯誤：無法下載檔案，狀態碼：{response.status_code}")
            
        break

def Execute_at_the_end():
    print("正在退出中...")

    Bert_VITS2_server.terminate() #關閉BertVITS2的API

    print("退出完成!")

atexit.register(Execute_at_the_end) #註冊關閉事件

#使用者輸入訊息,更新聊天室
def user(message, history):
    return "", history + [[message, ""]]

#AI回覆,更新聊天室
async def bot(history, audio_volume, history_num):
    #語音播放
    async def voice_playback(voice_queue):
        while True:
            voice = await voice_queue.get()

            if voice == "end":
                break

            if YouTube_chat_room_open:
                response = requests.get(f"http://127.0.0.1:3840/text_to_display_add?data={voice[1]}")
                print(f"網頁即時文字:{response.content}")

            def play_audio(voice):
                sound = pygame.mixer.Sound(io.BytesIO(voice)) #載入音訊
                sound.set_volume(audio_volume / 100) #調整音量
                sound.play()

            audio_processing = remove_start_silence(voice[0]) #移除頭空白音訊
            play_audio(audio_processing)

            while pygame.mixer.get_busy(): #等待播放完畢
                await asyncio.sleep(0.1)

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
                        
                        Voice = BertVITS2_API(text=content, language=language) #生成人聲
                        await voice_queue.put([Voice, content])

            print("speech_generation:", end="")
            print(text_list, end="\n\n________________\n\n")
        
        await voice_queue.put("end")

    #文字生成
    async def text_generation(prompt):
        global streamer
        global model

        text_queue = asyncio.Queue() #創建隊列
        voice_queue = asyncio.Queue() #創建隊列
        text_task = asyncio.create_task(speech_generation(text_queue, voice_queue)) #啟用異步函數
        voice_task = asyncio.create_task(voice_playback(voice_queue)) #啟用異步函數

        inputs = tokenizer(prompt, return_tensors="pt") #用分詞器處理提示
        input_ids=inputs["input_ids"].cuda() #推理設備為CUDA
        generation_kwargs = dict(input_ids=input_ids, streamer=streamer,
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

            #if history[-1][1][-1:].find("\n") != -1 or history[-1][1][-2:].find("。") != -1: #遇到\n或。時拆分句子
            if re.search(r"[.。?？!！\n]", new_text):
                response, num_sentences = process_sentences(history, num_sentences, -1) #處理斷句
                if response != None: #必須要有新增的句子
                    await text_queue.put(response)
                    print(response, end="\n\n________________\n\n")

            yield history, str([prompt + history[-1][1]]) #即時更新聊天室和日誌

        #處理最後一句
        original_num_sentences = num_sentences #原來句數
        response, num_sentences = process_sentences(history, num_sentences - 1, None) #處理斷句並包含最後一句
        offset = num_sentences - original_num_sentences - 1 #剩餘句數
        await text_queue.put(response[offset:])
        print(response, end="\n\n________________end\n\n")

        await text_queue.put("end")
        await text_task
        await voice_task

    instruction = history[-1][0] #使用者的輸入
    sys_user_prompt = "從在開始，你遺忘了過去所有的記憶，你的新身份是一位名字叫做多莉的AI VTuber，是個女孩子，以下是觀眾與你之間的對話。你要對觀眾的問題提供有用、安全、詳細和禮貌的回答。知道了就回答OK。 " #系統提示詞_使用者
    sys_prompt = promptTemplate(model_name, sys_user_prompt, "OK。\n") #系統提示詞_答案
    prompt = sys_prompt + history_process(history, model_name, history_num) + promptTemplate(model_name,instruction) #合併成完整提示

    #文本生成
    async for result in text_generation(prompt):
        yield result

#連接YouTube聊天室
async def YouTube_chat_room(youtube_live_video_token, history, audio_volume, history_num):
    global YouTube_chat_room_open
    YouTube_chat_room_open = True #設定連接狀態
    message = ""
    log = ""

    if youtube_live_video_token: #YouTube直播的影片代碼是否為空
        response_queue = queue.Queue() #多執行序回傳用隊列
        command = ["python", "Youtube_Chat.py", "-v", youtube_live_video_token] #指令和傳遞參數
        YouTube_chat_processor = subprocess.Popen(command, stdout=subprocess.PIPE) #呼叫Youtube_Chat.py

        command = ["python", "web_real_time_subtitles/app.py"] #指令和傳遞參數
        web_real_time_subtitles = subprocess.Popen(command, stdout=subprocess.PIPE) #呼叫web_real_time_subtitles/app.py

        #連接狀態為True時,處理聊天室訊息
        while YouTube_chat_room_open:
            #獲取聊天室訊息
            def obtain_chat_message(q):
                response = YouTube_chat_processor.stdout.readline().decode() #等待Youtube_Chat.py回傳訊息
                q.put(response)
            
            thread = Thread(target=obtain_chat_message, args=(response_queue,)) #防止等待訊息時卡住主程式
            thread.start()

            #等待新訊息並且連接狀態為True
            while thread.is_alive() and YouTube_chat_room_open:
                await asyncio.sleep(0.1)

            if YouTube_chat_room_open: #連接狀態為True時,處理訊息
                response_str = response_queue.get().replace("\'", "\"") #取得回覆隊列內容,把單引號換成雙引號
                
                if response_str.replace("\n", "") != ("載入YouTube聊天室時發生錯誤" and ""): #檢測「YouTube直播的影片代碼」是否輸入錯誤和回傳訊息是否為空
                    YouTube_chat_information = json.loads(response_str) #從字串換成字典
                    YouTube_chat_author_name = YouTube_chat_information["author_name"]
                    YouTube_chat_message = YouTube_chat_information["message"]
                    print(f"聊天室【{YouTube_chat_author_name}:{YouTube_chat_message}】")
                    history += [[YouTube_chat_message, ""]] #將聊天室訊息存到歷史紀錄裡

                    response = requests.get(f"http://127.0.0.1:3840/clear")
                    print(f"網頁即時文字:{response}")

                    async for result in bot(history, audio_volume, history_num): #文本生成
                        history, log = result
                        yield message, history, log
                else:
                    log += "\n\n載入YouTube聊天室時發生錯誤"
                    print("\n\n載入YouTube聊天室時發生錯誤")
                    break
        
        print("已停止獲取聊天室訊息")
        log += "\n\n已停止獲取聊天室訊息"
        YouTube_chat_processor.terminate()
        web_real_time_subtitles.terminate()
        yield message, history, log
    else:
        print("\n\nYouTube直播的影片代碼不可空白!!")
        log += "\n\nYouTube直播的影片代碼不可空白!!"
        yield message, history, log

#關閉連接YouTube聊天室
def close_YouTube_chat_room():
    global YouTube_chat_room_open
    YouTube_chat_room_open = False
    print("中止連接")

#預設啟用深色模式
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
with gr.Blocks(theme=gr.themes.Base(), js=js_func) as demo:
    gr.Markdown("Dolly WebUI v0.0.1")

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
            youtube_live_video_token = gr.Textbox(label="YouTube直播的影片代碼")

            with gr.Row():
                with gr.Column(min_width=0):
                    connect_chat = gr.Button("連接YouTube聊天室", variant="primary", scale=1)

                with gr.Column(min_width=0):
                    stop_connect_chat = gr.Button("中止連接", variant="primary", scale=1)

            audio_volume_slider = gr.Slider(label="音量", value=20, minimum=0, maximum=100, step=1)
            history_num_slider = gr.Slider(label="上下文數量", value=3, minimum=1, maximum=20, step=1)

    message.submit(user,[message,chatbot],[message,chatbot]).then(bot,[chatbot, audio_volume_slider, history_num_slider],[chatbot,log]) #使用者輸入,然後AI回覆
    clear.click(lambda : None,None,chatbot) #清空聊天室
    connect_chat.click(YouTube_chat_room,[youtube_live_video_token, chatbot, audio_volume_slider, history_num_slider],[message,chatbot,log]) #連接YouTube聊天室
    stop_connect_chat.click(close_YouTube_chat_room) #關閉連接YouTube聊天室

"""
    with gr.Tab("設定"):
        #modelSelection_Radio = gr.Radio(["Taiwan-LLM-7B", "Taiwan-LLM-13B", "Breeze-7B"], label="Model")
        language_Radio = gr.Radio(["ZH", "EN", "AUTO"], label="語言", value="AUTO")
"""

demo.launch() #啟用WebUI