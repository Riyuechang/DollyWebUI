import io
import asyncio

import pygame
import pygame._sdl2.audio as sdl2_audio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from config import config
from tools.log import logger

#取得音訊設備名稱
def get_audio_device_names() -> list[str]:
    pygame.mixer.init() #初始化混音器
    audio_device_names = sdl2_audio.get_audio_device_names(False) #取得音訊設備名稱
    pygame.mixer.quit() #退出混音器
    return audio_device_names

#初始化音訊
def initialize_audio(audio_device_names_list: list) -> str:
    default_audio_device = config.default.audio.audio_device_name

    if default_audio_device in audio_device_names_list: #判斷默認音訊設備是否在"音訊設備名稱清單"裡
        pygame.mixer.init(devicename=default_audio_device) #初始化混音器,並指定音訊設備
        logger.info(f"音訊初始化成功  設備名稱:{default_audio_device}")
        return default_audio_device
    else: #找不到默認音訊設備,則不指定音訊設備
        pygame.mixer.init() #初始化混音器
        logger.info("音訊初始化成功  設備名稱:預設音訊裝置")
        return "預設"

#改變音訊設備
def change_audio_device(audio_device_name: str):
    pygame.mixer.quit() #退出混音器

    if audio_device_name == "預設":
        pygame.mixer.init() #初始化混音器
        logger.info("改變音訊設備成功  設備名稱:預設音訊裝置")
    else:
        pygame.mixer.init(devicename=audio_device_name) #初始化混音器,並指定音訊設備
        logger.info(f"改變音訊設備成功  設備名稱:{audio_device_name}")

#播放音訊
async def play_audio(
    voice: bytes, 
    audio_volume: int
):
    sound = pygame.mixer.Sound(io.BytesIO(voice)) #載入音訊
    sound.set_volume(audio_volume / 100) #調整音量
    sound.play()

    while pygame.mixer.get_busy(): #等待播放完畢
        await asyncio.sleep(0.1)

#移除頭尾空白音訊
def remove_start_and_end_silence(audio: bytes) -> bytes:
    audio_pydub = AudioSegment.from_wav(io.BytesIO(audio)) #轉換成pydhb可以讀的格式
    start_silence = detect_leading_silence(audio_pydub) #計算頭空白音訊範圍
    end_silence = detect_leading_silence(audio_pydub.reverse()) #計算尾空白音訊範圍
    audio_duration = len(audio_pydub) #音訊總長度
    after_trim_audio = audio_pydub[start_silence:audio_duration - end_silence] #只取扣除頭尾空白的部份
    audio_wav = after_trim_audio.export(None, format="wav") #轉換成.wav格式
    audio_bytes = audio_wav.read() #讀取成bytes

    return audio_bytes

#移除頭空白音訊
def remove_start_silence(audio: bytes) -> bytes:
    audio_pydub = AudioSegment.from_wav(io.BytesIO(audio))
    start_silence = detect_leading_silence(audio_pydub)
    after_trim_audio = audio_pydub[start_silence:]
    audio_wav = after_trim_audio.export(None, format="wav")
    audio_bytes = audio_wav.read()

    return audio_bytes