import requests
from urllib.parse import quote

from tools.log import logger
from config import config

#TTS API
def BertVITS2_API(
    text: str="Voice test", 
    language: str="EN", 
    style_text: str="", 
    config: dict[str, str | int]=config.BertVITS2
) -> bytes | None:
    # 合併成完整url
    audio_url = f"http://{config['BertVITS2IP']}/voice?"
    audio_url += f"text={quote(text)}&"
    audio_url += f"model_id={config['model_id']}&"
    audio_url += f"speaker_id=0&"
    audio_url += f"sdp_ratio={config['sdp_ratio']}&"
    audio_url += f"noise={config['noise']}&"
    audio_url += f"noisew={config['noisew']}&"
    audio_url += f"length={config['length']}&"
    audio_url += f"language={language}&"
    audio_url += f"auto_translate={config['auto_translate']}&"
    audio_url += f"auto_split={config['auto_split']}&"
    audio_url += f"style_text={style_text}&"
    audio_url += f"style_weight={config['style_weight']}"

    # 呼叫API
    #logger.info(audio_url)
    response = requests.get(audio_url)

    if response.status_code == 200:
        logger.info(f"TTS語音生成完畢")
        return response.content
    else:
        logger.error(f"錯誤：無法下載檔案，狀態碼：{response.status_code}")