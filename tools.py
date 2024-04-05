import io
import re
import spacy
import requests
from urllib.parse import quote
from pydub import AudioSegment
from pydub.silence import detect_leading_silence


nlp = spacy.load("zh_core_web_sm")


def promptTemplate(model_name,instruction,response="",rounds=1):
    match model_name:
        case "Taiwan-LLM-7B":
            prompt = f"USER: {instruction} ASSISTANT:{response}"
        case "Taiwan-LLM-13B":
            prompt = f"USER: {instruction} ASSISTANT:{response}"
        case "Breeze-7B":
            prompt = f"[INST] {instruction} [/INST] RESPONSE{rounds} [INST]{response}"
    return prompt

#處理聊天紀錄
def history_process(history, model_name, num=9999):
    process = ""
    len_history = len(history)

    for index,layer_1 in enumerate(reversed(history)):
        if index > num:
            break

        if index != 0:
            instruction = layer_1[0]
            response = layer_1[1]
            process = promptTemplate(model_name,instruction,response,len_history - index) + "\n" + process
    
    return process

#處理句子
def process_sentences(history, num_sentences, end=None, nlp=nlp):
    split_n = history[-1][1].split("\n")
    text = [i for i in split_n if i != ""]
    
    sentences = []
    for i in text:
        doc = nlp(i)
        sentences += [sent.text for sent in doc.sents]

    response = None
    if len(sentences) > num_sentences: #句數大於前值時,處理增加的句子
        num = len(sentences) - num_sentences
        num_sentences = len(sentences)

        response = sentences[-(num + 1):end]
    
    return response, num_sentences


#初始設定
config = {
    "BertVITS2IP": "127.0.0.1:5000",  # Bert-VITS2伺服器IP
    "model_id" : "0",  # 模型ID  默認即可
    "sdp_ratio" : "0.5",  # SDP/DP混合比
    "noise" : "0.6",  # 感情
    "noisew" : "0.9",  # 音素長度
    "length" : "1",  # 語速
    "auto_translate" : "false",  # 自動翻譯
    "auto_split" : "false",  # 自動切分
    "style_weight" : "0.7"  # 風格權重
}

def BertVITS2_API(text="Voice test", language="EN", style_text="", config=config):
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
    print(audio_url)
    response = requests.get(audio_url)

    if response.status_code == 200:
        print(f"檔案下載成功")
        #with open("./_cache/audio.wav", 'wb') as file:
        #    file.write(response.content)
        return response.content
    else:
        print(f"錯誤：無法下載檔案，狀態碼：{response.status_code}")

#語言分類
def language_classification(content):
    language_result = ""
    language_dict = {}
    language_list = []
    first_character = ""
    for text in content:
        if re.search(r"[\u4e00-\u9fff]", text):
            if language_result == "ZH":
                language_dict["content"] += text
            else:
                if language_dict != {}:
                    language_list.append(language_dict)
                
                language_dict = {}
                language_dict["language"] = "ZH"
                language_dict["content"] = text
                language_result = "ZH"

        elif re.search(r"[a-zA-Z]", text):
            if language_result == "EN":
                language_dict["content"] += text
            else:
                if language_dict != {}:
                    language_list.append(language_dict)

                language_dict = {}
                language_dict["language"] = "EN"
                language_dict["content"] = text
                language_result = "EN"
        else:
            if language_dict == {}:
                first_character += text
            else:
                language_dict["content"] += text

    language_list.append(language_dict)
    language_list[0]["content"] = first_character + language_list[0]["content"]

    return language_list

#移除頭尾空白音訊
def remove_start_and_end_silence(audio):
    audio_pydub = AudioSegment.from_wav(io.BytesIO(audio))
    start_silence = detect_leading_silence(audio_pydub)
    end_silence = detect_leading_silence(audio_pydub.reverse())
    audio_duration = len(audio_pydub)
    after_trim_audio = audio_pydub[start_silence:audio_duration - end_silence]
    audio_wav = after_trim_audio.export(None, format="wav")
    audio_bytes = audio_wav.read()

    return audio_bytes

#移除頭空白音訊
def remove_start_silence(audio):
    audio_pydub = AudioSegment.from_wav(io.BytesIO(audio))
    start_silence = detect_leading_silence(audio_pydub)
    after_trim_audio = audio_pydub[start_silence:]
    audio_wav = after_trim_audio.export(None, format="wav")
    audio_bytes = audio_wav.read()

    return audio_bytes