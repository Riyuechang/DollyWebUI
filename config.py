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

    @classmethod
    def from_dict(cls, config_data: dict):
        return cls(**config_data)


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
    
    @classmethod
    def from_dict(cls, config_data: dict):
        return cls(**config_data)


class Default_Config:
    def __init__(
        self, 
        YouTubeChannelID: str,
        TTS_Language: str,
        llm_model_name: str,
        sentiment_analysis_model_name: str,
        sentiment: str,
        checkable_settings: list[str],
        audio: dict[str | int],
        history: dict[str | int],
    ):
        self.YouTubeChannelID: str = YouTubeChannelID
        self.TTS_Language: str = TTS_Language
        self.llm_model_name: str = llm_model_name
        self.sentiment_analysis_model_name: str = sentiment_analysis_model_name
        self.sentiment: str = sentiment
        self.checkable_settings: list[str] = checkable_settings
        self.audio: Audio_Config = Audio_Config.from_dict(audio)
        self.history: History_Config = History_Config.from_dict(history)
    
    @classmethod
    def from_dict(cls, config_data: dict):
        return cls(**config_data)


class LLM_Model_Config:
    def __init__(
        self,
        name: list[str],
        promptTemplate: dict[str, str],
        eos_token: dict[str, str],
        path: dict[str, str]
    ):
        self.name: list[str] = name
        self.promptTemplate: dict[str, str] = promptTemplate
        self.eos_token: dict[str, str] = eos_token
        self.path: dict[str, str] = path
    
    @classmethod
    def from_dict(cls, config_data: dict):
        return cls(**config_data)


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
    
    @classmethod
    def from_dict(cls, config_data: dict):
        return cls(**config_data)

"""
class BertVITS2_Config:
    def __init__(
        self,
        BertVITS2IP: str,
        model_id: int,
        sdp_ratio: int,
        noise: int,
        noisew: int,
        length: int,
        auto_translate: str,
        auto_split: str,
        style_weight: int,
        path: str
    ):
        self.BertVITS2IP: str = BertVITS2IP
        self.model_id: int = model_id
        self.sdp_ratio: int = sdp_ratio
        self.noise: int = noise
        self.noisew: int = noisew
        self.length: int = length
        self.auto_translate: str = auto_translate
        self.auto_split: str = auto_split
        self.style_weight: int = style_weight
        self.path: str = path
    
    @classmethod
    def from_dict(cls, config_data: dict):
        return cls(**config_data)
"""

class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as file:
            config_data: dict = yaml.safe_load(file)
        
        self.default: Default_Config = Default_Config.from_dict(config_data["default"])
        self.llm_model: LLM_Model_Config = LLM_Model_Config.from_dict(config_data["llm_model"])
        #self.BertVITS2: BertVITS2_Config = BertVITS2_Config.from_dict(config_data["BertVITS2"])
        self.BertVITS2: dict = config_data["BertVITS2"]
        self.VTube_Studio: VTube_Studio_Config = VTube_Studio_Config.from_dict(config_data["VTube_Studio"])


config_path = "config.yml"
config = Config(config_path)