import yaml
from typing import Literal


class Audio_Config:
    def __init__(
        self,
        audio_volume: int,
        audio_device_name: str
    ):
        self.audio_volume: int = audio_volume
        self.audio_device_name: str = audio_device_name


class History_Config:
    def __init__(
        self,
        history_mode: Literal["rounds","word","token"],
        history_num: int,
        history_num_max: int,
    ):
        self.history_mode: Literal["rounds","word","token"] = history_mode
        self.history_num: int = history_num
        self.history_num_max: int = history_num_max


class Default_Config:
    def __init__(
        self, 
        YouTubeChannelID: str,
        TTS_Language: str,
        llm_model_name: str,
        sentiment_analysis_model_name: str,
        sentiment: str,
        checkable_settings: list[str],
        hypnotic_prompt: str,
        audio: dict[str | int],
        history: dict[str | int],
    ):
        self.YouTubeChannelID: str = YouTubeChannelID
        self.TTS_Language: str = TTS_Language
        self.llm_model_name: str = llm_model_name
        self.sentiment_analysis_model_name: str = sentiment_analysis_model_name
        self.sentiment: str = sentiment
        self.checkable_settings: list[str] = checkable_settings
        self.hypnotic_prompt: str = hypnotic_prompt
        self.audio: Audio_Config = Audio_Config(**audio)
        self.history: History_Config = History_Config(**history)


class LLM_Model_Config:
    def __init__(
        self,
        name: list[str],
        sys_prompt: dict[str, str],
        promptTemplate: dict[str, str],
        eos_token: dict[str, str],
        path: dict[str, str]
    ):
        self.name: list[str] = name
        self.sys_prompt: dict[str, str] = sys_prompt
        self.promptTemplate: dict[str, str] = promptTemplate
        self.eos_token: dict[str, str] = eos_token
        self.path: dict[str, str] = path


class VTube_Studio_Config:
    def __init__(
        self,
        pluginName: str,
        pluginDeveloper: str,
        VTube_Studio_API_URL: str,
        sentiment_analysis: list
    ):
        self.pluginName: str = pluginName
        self.pluginDeveloper: str = pluginDeveloper
        self.VTube_Studio_API_URL: str = VTube_Studio_API_URL
        self.sentiment_analysis: list = sentiment_analysis


class TTS_Config:
    def __init__(
        self,
        TTS_type: str,
        path: str,
        venv: str,
        api_file_name: str,
        config: dict[dict[str, str | int]]
    ):
        self.TTS_type: str = TTS_type
        self.path: str = path
        self.venv: str = venv
        self.api_file_name: str = api_file_name
        self.config: dict[dict[str, str | int]] = config


class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as file:
            config_data: dict = yaml.safe_load(file)
        
        self.default: Default_Config = Default_Config(**config_data["default"])
        self.llm_model: LLM_Model_Config = LLM_Model_Config(**config_data["llm_model"])
        self.TTS: TTS_Config = TTS_Config(**config_data["TTS"])
        self.VTube_Studio: VTube_Studio_Config = VTube_Studio_Config(**config_data["VTube_Studio"])


config_path = "config.yml"
config = Config(config_path)