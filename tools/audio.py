import time
from io import BytesIO
from typing import Literal

from pygame import mixer as pygame_mixer
from pygame._sdl2 import audio as sdl2_audio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import config
from tools.log import logger

#取得音訊設備名稱
def get_audio_device_names() -> list[str]:
    pygame_mixer.init() #初始化混音器
    audio_device_names = sdl2_audio.get_audio_device_names(False) #取得音訊設備名稱
    pygame_mixer.quit() #退出混音器
    return audio_device_names

#重新取得音訊設備名稱
def get_audio_device_name_again():
    return sdl2_audio.get_audio_device_names(False) #取得音訊設備名稱

#初始化音訊
def initialize_audio(audio_device_names_list: list) -> str:
    default_audio_device = config.default.audio.audio_device_name

    if default_audio_device in audio_device_names_list: #判斷默認音訊設備是否在"音訊設備名稱清單"裡
        pygame_mixer.init(devicename=default_audio_device) #初始化混音器,並指定音訊設備
        logger.info(f"音訊初始化成功  設備名稱:{default_audio_device}")
        return default_audio_device
    else: #找不到默認音訊設備,則不指定音訊設備
        pygame_mixer.init() #初始化混音器
        logger.info("音訊初始化成功  設備名稱:預設音訊裝置")
        return "預設"

#改變音訊設備
def change_audio_device(audio_device_name: str):
    pygame_mixer.quit() #退出混音器

    if audio_device_name == "預設":
        pygame_mixer.init() #初始化混音器
        logger.info("改變音訊設備成功  設備名稱:預設音訊裝置")
    else:
        pygame_mixer.init(devicename=audio_device_name) #初始化混音器,並指定音訊設備
        logger.info(f"改變音訊設備成功  設備名稱:{audio_device_name}")

#播放音訊
def play_audio(
    voice: bytes, 
    audio_volume: int
):
    sound = pygame_mixer.Sound(BytesIO(voice)) #載入音訊
    sound.set_volume(audio_volume / 100) #調整音量
    sound.play()

    while pygame_mixer.get_busy(): #等待播放完畢
        time.sleep(0.1)

#把pydub格式轉成bytes
def pydub_format_to_bytes(pydub_format):
    audio_wav = pydub_format.export(None, format="wav") #轉換成.wav格式
    audio_bytes = audio_wav.read() #讀取成bytes

    return audio_bytes

#移除空白音訊
def remove_silence(
    audio: bytes, 
    reserved_length: int = 0,
    mode: Literal["start", "end", "start_and_end"] = "start_and_end"
) -> bytes:
    audio_pydub = AudioSegment.from_wav(BytesIO(audio)) #轉換成pydhb可以讀的格式

    if "start" in mode:
        start_silence = detect_leading_silence(audio_pydub) #計算頭空白音訊範圍
        start_silence_index = max(start_silence - reserved_length, 0)
    else:
        start_silence_index = None

    if "end" in mode:
        end_silence = detect_leading_silence(audio_pydub.reverse()) #計算尾空白音訊範圍
        audio_duration = len(audio_pydub) #音訊總長度
        end_silence_index = min(audio_duration - end_silence + reserved_length, audio_duration)
    else:
        end_silence_index = None

    after_trim_audio = audio_pydub[start_silence_index:end_silence_index] #只取扣除頭尾空白的部份
    audio_bytes = pydub_format_to_bytes(after_trim_audio)

    return audio_bytes

#拼接音訊
def Splicing_audio(bytes_list: list[bytes]) -> bytes:
    for index,file in enumerate(bytes_list):
        if index == 0:
            audio_pydub = AudioSegment.from_wav(BytesIO(file))
        else:
            audio_pydub += AudioSegment.from_wav(BytesIO(file))
    
    audio_bytes = pydub_format_to_bytes(audio_pydub)

    return audio_bytes