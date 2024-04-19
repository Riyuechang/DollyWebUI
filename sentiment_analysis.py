from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from config import config
from tools.word_processing import extract_chinese_characters, most_common

mpdel_path = "./nlp_model/" + config.default.sentiment_analysis_model_name
tokenizer = AutoTokenizer.from_pretrained(mpdel_path)
model = AutoModelForSequenceClassification.from_pretrained(mpdel_path)
nlp = pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)

#情緒分析
def sentiment_analysis(text: str) -> int:
    nlp_output = nlp(text)
    #star = int(nlp_output[0][0]["label"].lstrip("star "))
    label = nlp_output[0]["label"]

    return label

#多段情緒分析
def multi_segment_sentiment_analysis(text: str) -> str:
    text_chinese_list = extract_chinese_characters(text) #過濾出中文

    if text_chinese_list: #有中文
        sentiment_list = []

        for word in text_chinese_list: #情緒分析
            sentiment = sentiment_analysis(word)
            sentiment_list.append(sentiment)
        
        most_common_sentiment_list = most_common(sentiment_list, 2) #找出出現最多次的前兩個

        if len(most_common_sentiment_list) == 1 or not "none" in most_common_sentiment_list: #只有一個或沒有"none"
            return most_common_sentiment_list[0]
        else:
            return [i for i in most_common_sentiment_list if i != "none"][0] #過濾出"none"
    else:
        return "none" #沒有中文則是"none"  未來增加英文情緒分析
