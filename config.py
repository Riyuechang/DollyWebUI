import yaml


class Default_Config:
    def __init__(
        self, 
        audio_volume: int,
        audio_device_name: str,
        TTS_Language: str,
        history_num: int,
        model_name: str,
        YouTubeChannelID: str
    ):
        self.audio_volume: int = audio_volume
        self.audio_device_name: str = audio_device_name
        self.TTS_Language: str = TTS_Language
        self.history_num: int = history_num
        self.model_name: str = model_name
        self.YouTubeChannelID: str = YouTubeChannelID
    
    @classmethod
    def from_dict(cls, config_data: dict):
        return cls(**config_data)


class Model_Config:
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
        self.model: Model_Config = Model_Config.from_dict(config_data["model"])
        #self.BertVITS2: BertVITS2_Config = BertVITS2_Config.from_dict(config_data["BertVITS2"])
        self.BertVITS2: dict = config_data["BertVITS2"]


config_path = "config.yml"
config = Config(config_path)