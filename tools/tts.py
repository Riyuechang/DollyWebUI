import re
import time
import atexit
import subprocess
from urllib.parse import quote

import requests
from emoji import emoji_count

from config import config
from tools.log import logger
from tools.audio import remove_silence, Splicing_audio
from tools.word_processing import language_classification, unicode_emoji_parse

#啟動TTS
cwd = config.TTS.path
venv = config.TTS.venv
api_file_name = config.TTS.api_file_name
command = [f"{venv}/bin/python", f"{api_file_name}.py"]
Bert_VITS2_server = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE) #啟動Bert-VITS2

while True: #等待啟動
    output = Bert_VITS2_server.stdout.readline().decode()

    if output.find("api文档地址 http://127.0.0.1:5000/docs") != -1: #檢測是否開啟完畢
        logger.info("TTS啟動成功")
        time.sleep(1)

        #預載入模型
        for language in config.default.TTS_loading_language:
            url = f"http://127.0.0.1:5000/voice?text=測試test&model_id=0&speaker_id=0&language={language}"
            response = requests.get(url)
            
            if response.status_code == 200:
                logger.info(f"成功載入TTS語言：{language}")
            else:
                logger.error(f"錯誤：無法下載檔案，錯誤碼：{response.status_code}")
            
        break

@atexit.register #註冊關閉事件
def Execute_at_the_end():
    logger.info("正在退出中...")

    Bert_VITS2_server.terminate() #關閉BertVITS2的API

    logger.info("退出完成!")

#TTS API
def TTS_API(
    text: str="Voice test", 
    language: str="EN", 
    TTS_config: dict[dict[str, str | int]]=config.TTS.config
) -> bytes | None:
    # 合併成完整url
    match config.TTS.TTS_type:
        case "BertVITS2":
            BertVITS2_config = TTS_config["BertVITS2"]
            audio_url = f"http://{BertVITS2_config['BertVITS2IP']}/voice?"
            audio_url += f"text={quote(text)}&"
            audio_url += f"model_id={BertVITS2_config['model_id']}&"
            audio_url += f"speaker_id=0&"
            audio_url += f"sdp_ratio={BertVITS2_config['sdp_ratio']}&"
            audio_url += f"noise={BertVITS2_config['noise']}&"
            audio_url += f"noisew={BertVITS2_config['noisew']}&"
            audio_url += f"length={BertVITS2_config['length']}&"
            audio_url += f"language={language}&"
            audio_url += f"auto_translate={BertVITS2_config['auto_translate']}&"
            audio_url += f"auto_split={BertVITS2_config['auto_split']}&"
            audio_url += f"style_text={BertVITS2_config['style_text']}&"
            audio_url += f"style_weight={BertVITS2_config['style_weight']}"

    # 呼叫API
    logger.info(f"正在調用TTS_API   URL:{audio_url}")
    response = requests.get(audio_url)

    if response.status_code == 200:
        logger.info(f"TTS語音生成完畢")
        return response.content
    else:
        logger.error(f"錯誤：無法下載檔案，狀態碼：{response.status_code}")

#TTS生成
def tts_generation(
    text: str, 
    streamer: bool = False
):
    Voice_list: list[bytes] = []

    if re.search(r"[a-zA-Z\u4e00-\u9fff]", text) or emoji_count(text): #檢測是否有中文或英文字母或表情符號
        processed_emoji_text = unicode_emoji_parse(text) #處理表情符號
        text_classification = language_classification(processed_emoji_text) #根據語言分類

        for text_dict in text_classification: #根據語言生成TTS
            content = text_dict["content"]
            language = text_dict["language"]
            
            logger.info(f"TTS內容  語言:{language}  文字:{[content]}")

            Voice = TTS_API(text=content, language=language) #生成人聲

            if streamer:
                yield Voice, content
            else:
                Voice_list.append(remove_silence(Voice, 100))
    
        if not streamer:
            if Voice_list:
                Voice_bytes = Splicing_audio(Voice_list)
            else:
                Voice_bytes = None

            yield Voice_bytes
    else:
        yield None, None