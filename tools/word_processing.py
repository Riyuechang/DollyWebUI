import re
from typing import Literal

from config import config

#提示模板
def promptTemplate(
    model_name: str,
    instruction: str,
    response: str="",
    rounds: int=1
) -> str:
    model_config: dict = config.llm_model.promptTemplate
    prompt: str = model_config[model_name]
    prompt = prompt.replace("{instruction}", instruction)
    prompt = prompt.replace("{response}", response)
    prompt = prompt.replace("{rounds}", str(rounds))
    return prompt

#處理聊天紀錄
def history_process(
    history: list[list[str | None | tuple]], 
    model_name: str, 
    num: int=9999,
    mode: Literal["rounds","word","token"]="rounds",
    tokenizer: None=None
) -> str:
    process = ""
    rounds_num = 0
    len_history = len(history)

    def Processed_by_rounds(rounds_num: int) -> str:
        text = ""

        if rounds_num > len_history - 1: #"輪數"大於"歷史紀錄輪數",則"輪數"等於"歷史紀錄輪數",防止提示輪數超過總輪數
            rounds_num = len_history - 1
        
        for index,layer_text in enumerate(reversed(history)):
            if index > rounds_num: #超過輪數上限時,跳出迴圈
                break

            if index != 0:
                instruction = layer_text[0]
                response = layer_text[1]
                Correction = len_history - index - (len_history - rounds_num - 1)
                text = promptTemplate(model_name,instruction,response,Correction) + "\n" + text
                
        return text

    match mode:
        case "rounds": #根據回合數處理
            process = Processed_by_rounds(num)

        case "word": #根據字數處理
            text = ""

            for index,layer_text in enumerate(reversed(history)):
                text += str(layer_text)

                if len(text) > num: #超過字數上限時,跳出迴圈
                    break
                else: #沒有則加輪數1
                    if index != 0: #不是第一輪才+1
                        rounds_num += 1
            
            process = Processed_by_rounds(rounds_num)
            
        case "token":
            text = ""

            for index,layer_text in enumerate(reversed(history)):
                text += str(layer_text)

                if len(tokenizer.tokenize(text)) > num: #超過Token數上限時,跳出迴圈
                    break
                else: #沒有則加輪數1
                    if index != 0: #不是第一輪才+1
                        rounds_num += 1
            
            process = Processed_by_rounds(rounds_num)
            
    return process

#斷句
def sentence_break(content: str) -> list[str]:
    previous_text_type = "" #上一個文字的類型
    text = "" #文字暫存
    sentences = [] #句子列表
    character_to_wrap = r"[.。?？!！\n]" #要換行的字元

    for char in content:
        if re.match(character_to_wrap, char): #是換行字元就暫存
            text += char
            previous_text_type = "newline_character" #更新文字類型
        else:
            if previous_text_type == "newline_character": #上一個是換行字元就增加新句子
                sentences.append(text)
                text = char
                previous_text_type = "Other" #更新文字類型
            else:
                text += char #不是則暫存

    if text != "": #如果暫存不是空的就增加新句子
        sentences.append(text)

    return sentences

#處理句子
def process_sentences(
    history: list[list[str | None | tuple]], 
    num_sentences: int, 
    end: int | None=None
) -> tuple[list[str], int]:
    content = history[-1][1]
    sentences = sentence_break(content) #斷句
    
    response = None
    if len(sentences) > num_sentences: #句數大於前值時,處理增加的句子
        num = len(sentences) - num_sentences
        num_sentences = len(sentences)
        response = sentences[-(num + 1):end]
    
    return response, num_sentences

#語言分類
def language_classification(content: str) -> list[dict[str, str]]:
    language_result = "" #上一次的語言
    language_dict = {} #分類用字典
    language_list = [] #儲存用列表
    first_character = "" #一開始是符號儲存的地方
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

    language_list.append(language_dict) #儲存最後一個
    language_list[0]["content"] = first_character + language_list[0]["content"] #加上一開始的符號

    return language_list